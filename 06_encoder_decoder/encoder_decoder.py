"""
Project 6 - Encoder-Decoder Sequence Models (English -> German translation)

Goal: understand the core encoder-decoder idea - split "understand the
input" and "generate the output" into two separate RNNs connected by ONE
fixed-size vector - and see its main weakness directly: that fixed-size
"bottleneck" vector has to represent an entire sentence, however long, and
translation quality measurably degrades as source sentences get longer.
This is the exact problem Project 8 (Attention) is built to fix, so this
project's final plot is the setup for that one.

Dataset: Multi30k (HuggingFace `bentrevett/multi30k`) - 29,000 real
English-German sentence pairs (image captions, short and clean), the same
dataset Projects 7 (Seq2Seq) and 8 (Attention) will reuse so all three are
directly comparable.

Architecture:
  ENCODER: Embedding -> LSTM, we keep ONLY its final (h, c) states.
           Every word of the source sentence has been compressed into this
           one fixed-size vector pair - regardless of whether the sentence
           was 3 words or 30.
  DECODER: Embedding -> LSTM initialized with the encoder's final (h, c)
           -> Dense(vocab, softmax), generating the target sentence one
           token at a time. During training, teacher forcing feeds the
           REAL previous target token as input (not the model's own
           guess) - this is standard and is exactly what Project 7 will
           name and study directly.

RUN WITH:
    python encoder_decoder.py
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

MAX_LEN = 16          # tokens, including <sos>/<eos> - keeps training fast on CPU
MAX_VOCAB = 6000
EMBED_DIM = 128
HIDDEN_SIZE = 256
EPOCHS = 20
BATCH_SIZE = 64
NUM_EXAMPLES = 15000   # subset of the 29k pairs - plenty for a from-scratch demo, fast on CPU


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

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.itos[i]
            if w == '<eos>':
                break
            if w not in ('<pad>', '<sos>'):
                words.append(w)
        return words

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
        src_ids = [src_vocab.encode(s, MAX_LEN) for s in src_list]
        tgt_ids = [tgt_vocab.encode(t, MAX_LEN) for t in tgt_list]
        src_lens = [len(s) for s in src_ids]  # kept for the length-vs-quality analysis later
        src_ids = pad_sequences(src_ids, maxlen=MAX_LEN, padding='post', value=0)
        tgt_ids = pad_sequences(tgt_ids, maxlen=MAX_LEN, padding='post', value=0)
        return src_ids, tgt_ids, src_lens

    X_train, y_train, _ = encode_pairs(src_train, tgt_train)
    X_val, y_val, val_src_lens = encode_pairs(src_val, tgt_val)

    print(f"Train pairs: {len(X_train)}, val pairs: {len(X_val)}")
    return (X_train, y_train), (X_val, y_val, val_src_lens, src_val, tgt_val), src_vocab, tgt_vocab


def build_training_model(src_vocab_size, tgt_vocab_size):
    """
    Two inputs during training:
      encoder_input:      the full source sentence
      decoder_input:      the target sentence SHIFTED RIGHT by one
                           (teacher forcing - decoder sees the true previous
                           token, not its own earlier prediction)
    One output: the target sentence, predicted one token at a time.
    """
    encoder_input = layers.Input(shape=(MAX_LEN,), name='encoder_input')
    enc_emb = layers.Embedding(src_vocab_size, EMBED_DIM, mask_zero=True)(encoder_input)
    _, state_h, state_c = layers.LSTM(HIDDEN_SIZE, return_state=True)(enc_emb)
    # state_h, state_c together ARE the bottleneck - the entire source
    # sentence's meaning, compressed into two fixed-length vectors.

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

    # --- Separate inference-time encoder/decoder, sharing the SAME trained weights ---
    # At inference there's no ground-truth target to feed the decoder, so it
    # must be run one step at a time, feeding each step's own prediction back
    # in as the next step's input - this is why encoder and decoder are split
    # into two callable models here, even though training uses one combined graph.
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


def greedy_translate(sentence_tokens, encoder_model, decoder_model, src_vocab, tgt_vocab):
    src_ids = src_vocab.encode(sentence_tokens, MAX_LEN)
    src_ids = pad_sequences([src_ids], maxlen=MAX_LEN, padding='post', value=0)
    h, c = encoder_model.predict(src_ids, verbose=0)

    tgt_seq = np.array([[tgt_vocab.stoi['<sos>']]])
    output_ids = []
    for _ in range(MAX_LEN):
        pred, h, c = decoder_model.predict([tgt_seq, h, c], verbose=0)
        next_id = int(np.argmax(pred[0, -1, :]))
        if next_id == tgt_vocab.stoi['<eos>']:
            break
        output_ids.append(next_id)
        tgt_seq = np.array([[next_id]])
    return [tgt_vocab.itos[i] for i in output_ids]


def main():
    (X_train, y_train), (X_val, y_val, val_src_lens, src_val_raw, tgt_val_raw), src_vocab, tgt_vocab = load_data()

    decoder_input_train = np.zeros_like(y_train)
    decoder_input_train[:, 1:] = y_train[:, :-1]
    decoder_input_train[:, 0] = tgt_vocab.stoi['<sos>']

    decoder_input_val = np.zeros_like(y_val)
    decoder_input_val[:, 1:] = y_val[:, :-1]
    decoder_input_val[:, 0] = tgt_vocab.stoi['<sos>']

    train_model, encoder_model, decoder_model = build_training_model(len(src_vocab), len(tgt_vocab))
    train_model.summary()

    history = train_model.fit(
        [X_train, decoder_input_train], np.expand_dims(y_train, -1),
        validation_data=([X_val, decoder_input_val], np.expand_dims(y_val, -1)),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    # --- Plot: training curves ---
    fig1, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history.history['loss'], label='train')
    axes[0].plot(history.history['val_loss'], label='val')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history.history['accuracy'], label='train')
    axes[1].plot(history.history['val_accuracy'], label='val')
    axes[1].set_title('Per-token accuracy (teacher-forced)'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
    print("Saved: training_curves.png")

    # --- Greedy-decode a sample of validation sentences, compute per-example BLEU ---
    print("\nGreedy-decoding a sample of validation sentences (this takes a bit - one token at a time)...")
    N_EVAL = 300
    smoother = SmoothingFunction().method1
    bleu_scores = []
    eval_src_lens = []
    example_outputs = []

    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(src_val_raw), size=min(N_EVAL, len(src_val_raw)), replace=False)

    for idx in sample_idx:
        src_tokens = src_val_raw[idx]
        tgt_tokens = tgt_val_raw[idx]
        pred_tokens = greedy_translate(src_tokens, encoder_model, decoder_model, src_vocab, tgt_vocab)
        bleu = sentence_bleu([tgt_tokens], pred_tokens, smoothing_function=smoother)
        bleu_scores.append(bleu)
        eval_src_lens.append(len(src_tokens))
        if len(example_outputs) < 8:
            example_outputs.append((src_tokens, tgt_tokens, pred_tokens))

    bleu_scores = np.array(bleu_scores)
    eval_src_lens = np.array(eval_src_lens)
    print(f"\nMean BLEU over {N_EVAL} validation sentences: {bleu_scores.mean():.4f}")

    # --- THE KEY PLOT: does the fixed-size bottleneck hurt longer sentences? ---
    bins = [0, 5, 8, 11, 14, 100]
    bin_labels = ['1-5', '6-8', '9-11', '12-14', '15+']
    bucket_means = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (eval_src_lens > lo) & (eval_src_lens <= hi)
        bucket_means.append(bleu_scores[mask].mean() if mask.sum() > 0 else np.nan)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(bin_labels, bucket_means, color='#1f77b4')
    ax2.set_xlabel('Source sentence length (tokens, including <sos>/<eos>)')
    ax2.set_ylabel('Mean BLEU score')
    ax2.set_title('Fixed-bottleneck encoder-decoder: quality vs. sentence length')
    ax2.grid(alpha=0.3, axis='y')
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'bleu_vs_length.png'), dpi=150)
    print("Saved: bleu_vs_length.png")

    with open(os.path.join(OUTPUT_DIR, 'example_translations.txt'), 'w', encoding='utf-8') as f:
        for src, tgt, pred in example_outputs:
            line = f"EN:   {' '.join(src)}\nDE (true): {' '.join(tgt)}\nDE (pred): {' '.join(pred)}\n\n"
            f.write(line)
            print(line)
    print("Saved: example_translations.txt")


if __name__ == "__main__":
    main()
