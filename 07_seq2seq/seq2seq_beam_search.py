"""
Project 7 - Seq2Seq: Greedy vs. Beam Search Decoding

Project 6 already built and trained the encoder-decoder with teacher forcing.
The piece that project deliberately left at its simplest was DECODING:
greedy search, which picks the single highest-probability token at every
step and never reconsiders that choice. That's a real limitation - the
single best token right now isn't always part of the best FULL sentence,
because an early greedy mistake can force worse tokens later with no way
back.

This project trains the same architecture (so the comparison is fair) and
then decodes the SAME trained model three ways:
  1. Greedy search           (always take the single best next token)
  2. Beam search, width 3    (keep the 3 best partial sentences at each step)
  3. Beam search, width 5    (keep the 5 best partial sentences at each step)

and measures whether keeping more candidate sentences alive actually
produces better translations (BLEU) - and shows concrete examples where
beam search recovers from a mistake greedy search commits to.

Dataset: Multi30k, same as Project 6 (English -> German), so results are
directly comparable.

RUN WITH:
    python seq2seq_beam_search.py
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

np.random.seed(42)
tf.random.set_seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LEN = 16
MAX_VOCAB = 6000
EMBED_DIM = 128
HIDDEN_SIZE = 256
EPOCHS = 20
BATCH_SIZE = 64
NUM_EXAMPLES = 15000   # same as Project 6, for a fair comparison


def tokenize(text):
    text = text.lower().strip()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-ZäöüÄÖÜß.,!?' ]+", " ", text)
    return text.split()


class Vocab:
    def __init__(self, tokenized_sentences, max_vocab):
        freq = {}
        for sent in tokenized_sentences:
            for tok in sent:
                freq[tok] = freq.get(tok, 0) + 1
        most_common = sorted(freq.items(), key=lambda x: -x[1])[:max_vocab - 4]
        self.itos = ['<pad>', '<sos>', '<eos>', '<unk>'] + [w for w, _ in most_common]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, tokens, max_len):
        ids = [self.stoi.get(t, self.stoi['<unk>']) for t in tokens]
        ids = [self.stoi['<sos>']] + ids[:max_len - 2] + [self.stoi['<eos>']]
        return ids

    def __len__(self):
        return len(self.itos)


def load_data():
    ds = load_dataset('bentrevett/multi30k')
    train = ds['train'].select(range(NUM_EXAMPLES))
    val = ds['validation']

    src_train = [tokenize(x) for x in train['en']]
    tgt_train = [tokenize(x) for x in train['de']]
    src_val = [tokenize(x) for x in val['en']]
    tgt_val = [tokenize(x) for x in val['de']]

    src_vocab = Vocab(src_train, MAX_VOCAB)
    tgt_vocab = Vocab(tgt_train, MAX_VOCAB)
    print(f"English vocab: {len(src_vocab)}, German vocab: {len(tgt_vocab)}")

    def encode_pairs(src_list, tgt_list):
        src_ids = pad_sequences([src_vocab.encode(s, MAX_LEN) for s in src_list], maxlen=MAX_LEN, padding='post', value=0)
        tgt_ids = pad_sequences([tgt_vocab.encode(t, MAX_LEN) for t in tgt_list], maxlen=MAX_LEN, padding='post', value=0)
        return src_ids, tgt_ids

    X_train, y_train = encode_pairs(src_train, tgt_train)
    X_val, y_val = encode_pairs(src_val, tgt_val)
    print(f"Train pairs: {len(X_train)}, val pairs: {len(X_val)}")
    return (X_train, y_train), (X_val, y_val, src_val, tgt_val), src_vocab, tgt_vocab


def build_training_model(src_vocab_size, tgt_vocab_size):
    encoder_input = layers.Input(shape=(MAX_LEN,), name='encoder_input')
    enc_emb = layers.Embedding(src_vocab_size, EMBED_DIM, mask_zero=True)(encoder_input)
    _, state_h, state_c = layers.LSTM(HIDDEN_SIZE, return_state=True)(enc_emb)

    decoder_input = layers.Input(shape=(MAX_LEN,), name='decoder_input')
    dec_emb_layer = layers.Embedding(tgt_vocab_size, EMBED_DIM, mask_zero=True)
    dec_emb = dec_emb_layer(decoder_input)
    decoder_lstm = layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
    decoder_dense = layers.Dense(tgt_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    train_model = models.Model([encoder_input, decoder_input], decoder_outputs)
    train_model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

    encoder_model = models.Model(encoder_input, [state_h, state_c])

    dec_state_input_h = layers.Input(shape=(HIDDEN_SIZE,))
    dec_state_input_c = layers.Input(shape=(HIDDEN_SIZE,))
    dec_single_input = layers.Input(shape=(1,))
    dec_single_emb = dec_emb_layer(dec_single_input)
    dec_single_out, dec_h, dec_c = decoder_lstm(dec_single_emb, initial_state=[dec_state_input_h, dec_state_input_c])
    dec_single_pred = decoder_dense(dec_single_out)
    decoder_model = models.Model(
        [dec_single_input, dec_state_input_h, dec_state_input_c],
        [dec_single_pred, dec_h, dec_c],
    )
    return train_model, encoder_model, decoder_model


def greedy_decode(src_ids, encoder_model, decoder_model, tgt_vocab):
    h, c = encoder_model.predict(src_ids, verbose=0)
    tgt_tok = np.array([[tgt_vocab.stoi['<sos>']]])
    output_ids = []
    for _ in range(MAX_LEN):
        pred, h, c = decoder_model.predict([tgt_tok, h, c], verbose=0)
        next_id = int(np.argmax(pred[0, -1, :]))
        if next_id == tgt_vocab.stoi['<eos>']:
            break
        output_ids.append(next_id)
        tgt_tok = np.array([[next_id]])
    return output_ids


def beam_search_decode(src_ids, encoder_model, decoder_model, tgt_vocab, beam_width):
    """
    Keeps the `beam_width` most probable PARTIAL sequences alive at every
    step, instead of committing to the single best token like greedy search.

    Each beam entry is (token_ids_so_far, cumulative_log_prob, h, c, finished).
    At each step, every live beam is expanded with every possible next token,
    all (beam_width * vocab_size) candidates are scored, and only the top
    `beam_width` overall survive to the next step. This lets the search
    recover from an early choice that looked good in isolation but leads to
    a worse full sentence - something greedy search can never do, since it
    never looks back.

    Log-probabilities (not raw probabilities) are summed because multiplying
    many probabilities < 1 underflows to zero very fast; summing logs is the
    numerically stable equivalent of multiplying probabilities.
    """
    h0, c0 = encoder_model.predict(src_ids, verbose=0)
    sos_id, eos_id = tgt_vocab.stoi['<sos>'], tgt_vocab.stoi['<eos>']

    beams = [([sos_id], 0.0, h0, c0, False)]

    for _ in range(MAX_LEN):
        all_candidates = []
        for tokens, log_prob, h, c, finished in beams:
            if finished:
                all_candidates.append((tokens, log_prob, h, c, True))
                continue
            last_tok = np.array([[tokens[-1]]])
            pred, h_new, c_new = decoder_model.predict([last_tok, h, c], verbose=0)
            probs = pred[0, -1, :]
            top_ids = np.argsort(probs)[-beam_width:]
            for tid in top_ids:
                p = max(probs[tid], 1e-12)
                new_tokens = tokens + [int(tid)]
                new_log_prob = log_prob + np.log(p)
                is_finished = (tid == eos_id)
                all_candidates.append((new_tokens, new_log_prob, h_new, c_new, is_finished))

        # Length-normalized score: without this, beam search systematically
        # prefers SHORTER sentences (each extra token can only add a negative
        # log-prob, so cumulative log-prob mechanically shrinks with length).
        all_candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
        beams = all_candidates[:beam_width]

        if all(b[4] for b in beams):
            break

    best_tokens = beams[0][0]
    return [t for t in best_tokens if t not in (sos_id, eos_id)]


def ids_to_words(ids, vocab):
    return [vocab.itos[i] for i in ids]


def main():
    (X_train, y_train), (X_val, y_val, src_val_raw, tgt_val_raw), src_vocab, tgt_vocab = load_data()

    decoder_input_train = np.zeros_like(y_train)
    decoder_input_train[:, 1:] = y_train[:, :-1]
    decoder_input_train[:, 0] = tgt_vocab.stoi['<sos>']
    decoder_input_val = np.zeros_like(y_val)
    decoder_input_val[:, 1:] = y_val[:, :-1]
    decoder_input_val[:, 0] = tgt_vocab.stoi['<sos>']

    train_model, encoder_model, decoder_model = build_training_model(len(src_vocab), len(tgt_vocab))
    history = train_model.fit(
        [X_train, decoder_input_train], np.expand_dims(y_train, -1),
        validation_data=([X_val, decoder_input_val], np.expand_dims(y_val, -1)),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    fig1, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history.history['loss'], label='train'); axes[0].plot(history.history['val_loss'], label='val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history.history['accuracy'], label='train'); axes[1].plot(history.history['val_accuracy'], label='val')
    axes[1].set_title('Per-token accuracy (teacher-forced)'); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
    print("Saved: training_curves.png")

    # --- Decode the SAME validation sentences 3 ways: greedy, beam-3, beam-5 ---
    print("\nDecoding validation sample 3 ways (greedy / beam-3 / beam-5) - this is the slow part...")
    N_EVAL = 200
    smoother = SmoothingFunction().method1
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(src_val_raw), size=min(N_EVAL, len(src_val_raw)), replace=False)

    bleu_greedy, bleu_beam3, bleu_beam5 = [], [], []
    interesting_examples = []

    for idx in sample_idx:
        src_tokens = src_val_raw[idx]
        tgt_tokens = tgt_val_raw[idx]
        src_ids = pad_sequences([src_vocab.encode(src_tokens, MAX_LEN)], maxlen=MAX_LEN, padding='post', value=0)

        greedy_ids = greedy_decode(src_ids, encoder_model, decoder_model, tgt_vocab)
        beam3_ids = beam_search_decode(src_ids, encoder_model, decoder_model, tgt_vocab, beam_width=3)
        beam5_ids = beam_search_decode(src_ids, encoder_model, decoder_model, tgt_vocab, beam_width=5)

        greedy_words = ids_to_words(greedy_ids, tgt_vocab)
        beam3_words = ids_to_words(beam3_ids, tgt_vocab)
        beam5_words = ids_to_words(beam5_ids, tgt_vocab)

        b_greedy = sentence_bleu([tgt_tokens], greedy_words, smoothing_function=smoother)
        b_beam3 = sentence_bleu([tgt_tokens], beam3_words, smoothing_function=smoother)
        b_beam5 = sentence_bleu([tgt_tokens], beam5_words, smoothing_function=smoother)

        bleu_greedy.append(b_greedy)
        bleu_beam3.append(b_beam3)
        bleu_beam5.append(b_beam5)

        if b_beam5 > b_greedy + 0.15 and len(interesting_examples) < 6:
            interesting_examples.append((src_tokens, tgt_tokens, greedy_words, beam3_words, beam5_words,
                                          b_greedy, b_beam3, b_beam5))

    print(f"\nMean BLEU over {N_EVAL} validation sentences:")
    print(f"  Greedy search   : {np.mean(bleu_greedy):.4f}")
    print(f"  Beam search (3) : {np.mean(bleu_beam3):.4f}")
    print(f"  Beam search (5) : {np.mean(bleu_beam5):.4f}")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    labels = ['Greedy', 'Beam (width=3)', 'Beam (width=5)']
    means = [np.mean(bleu_greedy), np.mean(bleu_beam3), np.mean(bleu_beam5)]
    bars = ax2.bar(labels, means, color=['#d62728', '#ff7f0e', '#1f77b4'])
    for bar, m in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, m + 0.003, f'{m:.3f}', ha='center', fontweight='bold')
    ax2.set_ylabel('Mean BLEU score')
    ax2.set_title('Decoding strategy comparison - same trained model')
    ax2.grid(alpha=0.3, axis='y')
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'decoding_comparison.png'), dpi=150)
    print("Saved: decoding_comparison.png")

    with open(os.path.join(OUTPUT_DIR, 'beam_vs_greedy_examples.txt'), 'w', encoding='utf-8') as f:
        f.write("Examples where beam search (width=5) clearly beat greedy search:\n\n")
        for src, tgt, gw, b3w, b5w, bg, b3, b5 in interesting_examples:
            block = (
                f"EN:          {' '.join(src)}\n"
                f"DE (true):   {' '.join(tgt)}\n"
                f"Greedy      (BLEU={bg:.3f}): {' '.join(gw)}\n"
                f"Beam-3      (BLEU={b3:.3f}): {' '.join(b3w)}\n"
                f"Beam-5      (BLEU={b5:.3f}): {' '.join(b5w)}\n\n"
            )
            f.write(block)
            print(block)
    print("Saved: beam_vs_greedy_examples.txt")


if __name__ == "__main__":
    main()
