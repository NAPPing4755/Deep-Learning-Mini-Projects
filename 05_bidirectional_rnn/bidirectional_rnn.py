"""
Project 5 - Bidirectional RNN for Named Entity Recognition

Goal: show WHY processing a sequence in both directions helps, on a task
where it should matter - NER. A unidirectional LSTM tagging "Washington" only
sees words BEFORE it; a bidirectional one also sees what comes AFTER, which
often disambiguates the entity type ("Washington played..." = PERSON vs.
"...arrived in Washington yesterday" = LOCATION).

Dataset: CoNLL-2003 (HuggingFace `datasets`) - the standard NER benchmark,
9 BIO tags: O, B-PER/I-PER, B-ORG/I-ORG, B-LOC/I-LOC, B-MISC/I-MISC.

Architecture, unidirectional vs. bidirectional (only this part differs):
  Unidirectional: Embedding -> LSTM(->)              -> TimeDistributed Dense(9, softmax)
  Bidirectional:  Embedding -> [LSTM(->); LSTM(<-)]  -> TimeDistributed Dense(9, softmax)
                              concatenate forward & backward hidden state at EVERY token

Evaluated with entity-level F1 (seqeval), not just per-token accuracy -
per-token accuracy is misleading here because ~83% of tokens are "O" (not
part of any entity), so a model that predicts "O" for everything already
scores ~83% token accuracy while being useless for the actual task.

RUN WITH:
    python bidirectional_rnn.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset
from seqeval.metrics import f1_score, classification_report

np.random.seed(42)
tf.random.set_seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LEN = 60
EMBED_DIM = 64
HIDDEN_SIZE = 64
EPOCHS = 12
BATCH_SIZE = 32


def load_ner_data():
    ds = load_dataset('conll2003', trust_remote_code=True)
    tag_names = ds['train'].features['ner_tags'].feature.names

    # Build a word vocabulary from the training set only (standard practice -
    # never let validation/test tokens leak into vocabulary construction)
    word_to_ix = {'<PAD>': 0, '<UNK>': 1}
    for tokens in ds['train']['tokens']:
        for tok in tokens:
            key = tok.lower()
            if key not in word_to_ix:
                word_to_ix[key] = len(word_to_ix)

    def encode_split(split):
        X = [[word_to_ix.get(tok.lower(), word_to_ix['<UNK>']) for tok in tokens]
             for tokens in ds[split]['tokens']]
        y = ds[split]['ner_tags']
        X = pad_sequences(X, maxlen=MAX_LEN, padding='post', value=0)
        y = pad_sequences(y, maxlen=MAX_LEN, padding='post', value=0)
        raw_tokens = ds[split]['tokens']
        raw_tags = ds[split]['ner_tags']
        return X, y, raw_tokens, raw_tags

    X_train, y_train, _, _ = encode_split('train')
    X_val, y_val, val_tokens, val_tags = encode_split('validation')

    print(f"Vocab size: {len(word_to_ix)}, num tags: {len(tag_names)}")
    print(f"Train sentences: {len(X_train)}, val sentences: {len(X_val)}")
    return (X_train, y_train), (X_val, y_val, val_tokens, val_tags), word_to_ix, tag_names


def build_model(bidirectional, vocab_size, num_tags):
    """
    The ONLY architectural difference between the two models being compared:
    whether the LSTM layer is wrapped in Bidirectional() or not. Everything
    else - embedding size, hidden size, optimizer, epochs, data - is identical,
    isolating exactly what bidirectionality contributes.
    """
    recurrent = layers.LSTM(HIDDEN_SIZE, return_sequences=True)
    if bidirectional:
        recurrent = layers.Bidirectional(recurrent)

    model = models.Sequential([
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(vocab_size, EMBED_DIM, mask_zero=True),
        recurrent,
        layers.TimeDistributed(layers.Dense(num_tags, activation='softmax')),
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def predictions_to_tag_sequences(pred_ids, true_tokens, true_tags_raw, tag_names):
    """
    Trims padding back off using each sentence's REAL length (from the raw,
    unpadded token lists) and converts tag ids -> tag name strings, which is
    the format seqeval expects for entity-level (not token-level) scoring.
    """
    pred_seqs, true_seqs = [], []
    for i, tokens in enumerate(true_tokens):
        # Cap at MAX_LEN too: sentences longer than MAX_LEN got truncated by
        # padding before the model ever saw them, so predictions only exist
        # for the first MAX_LEN tokens - comparing against the full untruncated
        # true-tag list would silently mismatch lengths for those sentences.
        length = min(len(tokens), MAX_LEN)
        pred_seqs.append([tag_names[t] for t in pred_ids[i][:length]])
        true_seqs.append([tag_names[t] for t in true_tags_raw[i][:length]])
    return pred_seqs, true_seqs


def main():
    (X_train, y_train), (X_val, y_val, val_tokens, val_tags), word_to_ix, tag_names = load_ner_data()
    vocab_size = len(word_to_ix)
    num_tags = len(tag_names)

    results = {}
    for bidir in [False, True]:
        label = 'Bidirectional LSTM' if bidir else 'Unidirectional LSTM'
        print(f"\n{'='*70}\nTraining: {label}\n{'='*70}")
        model = build_model(bidir, vocab_size, num_tags)
        history = model.fit(
            X_train, np.expand_dims(y_train, -1),
            validation_data=(X_val, np.expand_dims(y_val, -1)),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
        )

        pred_probs = model.predict(X_val, verbose=0)
        pred_ids = np.argmax(pred_probs, axis=-1)
        pred_seqs, true_seqs = predictions_to_tag_sequences(pred_ids, val_tokens, val_tags, tag_names)

        entity_f1 = f1_score(true_seqs, pred_seqs)
        report = classification_report(true_seqs, pred_seqs, digits=3)
        print(f"\n{label} - entity-level F1: {entity_f1:.4f}")
        print(report)

        results[label] = dict(
            history=history.history,
            entity_f1=entity_f1,
            report=report,
            params=model.count_params(),
        )

    # --- Plot: validation accuracy/loss curves for both models ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, r in results.items():
        axes[0].plot(r['history']['val_accuracy'], label=label, linewidth=2)
        axes[1].plot(r['history']['val_loss'], label=label, linewidth=2)
    axes[0].set_title('Validation token accuracy (misleading alone - see entity F1 below)')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_title('Validation loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
    print("\nSaved: training_curves.png")

    # --- Plot: the metric that actually matters - entity-level F1 ---
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    labels = list(results.keys())
    f1s = [results[l]['entity_f1'] for l in labels]
    bars = ax2.bar(labels, f1s, color=['#d62728', '#1f77b4'])
    ax2.set_ylabel('Entity-level F1 (seqeval)')
    ax2.set_title('Unidirectional vs. Bidirectional LSTM - NER entity F1')
    ax2.set_ylim(0, 1)
    for bar, f1 in zip(bars, f1s):
        ax2.text(bar.get_x() + bar.get_width() / 2, f1 + 0.02, f'{f1:.3f}', ha='center', fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'entity_f1_comparison.png'), dpi=150)
    print("Saved: entity_f1_comparison.png")

    with open(os.path.join(OUTPUT_DIR, 'classification_reports.txt'), 'w', encoding='utf-8') as f:
        for label, r in results.items():
            f.write(f"{'='*70}\n{label} (params={r['params']:,}, entity F1={r['entity_f1']:.4f})\n{'='*70}\n")
            f.write(r['report'] + "\n\n")
    print("Saved: classification_reports.txt")

    print("\nFinal comparison:")
    for label, r in results.items():
        print(f"  {label:<22} entity F1={r['entity_f1']:.4f}  params={r['params']:,}")


if __name__ == "__main__":
    main()
