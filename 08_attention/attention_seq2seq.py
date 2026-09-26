"""
Project 8 - Attention Mechanism (Bahdanau-style)

Project 6 measured the exact problem attention exists to solve: a fixed-size
encoder context vector, and BLEU dropping as source sentences get longer
because that one vector has to hold more and more meaning. Attention's fix
is simple to state: stop forcing everything through one vector. Instead,
keep ALL of the encoder's per-token hidden states, and let the decoder look
back at all of them at every generation step, weighting each one by how
relevant it is right now.

Architecture:
  ENCODER: Embedding -> LSTM(return_sequences=True) - keeps the hidden state
           at EVERY source position, not just the last one.

  ATTENTION (Bahdanau/additive, computed fresh at every decoder step t):
      score_i = v^T * tanh(W1 . encoder_state_i + W2 . decoder_state_{t-1})
      alpha   = softmax(score)      <- one weight per source position,
                                        summing to 1
      context_t = sum_i alpha_i * encoder_state_i
    "alpha" is literally an attention MAP: which source words the decoder is
    reading from right now. We visualize this directly (a heatmap) at the
    end - this is the part textbook explanations describe but rarely show
    computed from a real trained model.

  DECODER: at step t, concatenate [context_t, target_embedding_t], feed to
           LSTM, then Dense(vocab, softmax).

Same dataset, same size budget, same evaluation (BLEU vs. source sentence
length) as Project 6, so the two plots can be put side by side and the
claim "attention fixes the long-sentence problem" is measured, not assumed.

RUN WITH:
    python attention_seq2seq.py
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
EPOCHS = 10   # fewer than Projects 6/7's 20: this custom per-step attention
              # decoder is inherently slower per step than a native Keras
              # LSTM, and val loss in both prior projects had already
              # plateaued by epoch 5-6, so this trades a bit of training
              # budget for a much more reasonable wall-clock time
BATCH_SIZE = 64
NUM_EXAMPLES = 15000   # same as Projects 6 & 7, for a fair comparison


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
        src_ids = [src_vocab.encode(s, MAX_LEN) for s in src_list]
        tgt_ids = [tgt_vocab.encode(t, MAX_LEN) for t in tgt_list]
        src_lens = [len(s) for s in src_ids]
        src_ids = pad_sequences(src_ids, maxlen=MAX_LEN, padding='post', value=0)
        tgt_ids = pad_sequences(tgt_ids, maxlen=MAX_LEN, padding='post', value=0)
        return src_ids, tgt_ids, src_lens

    X_train, y_train, _ = encode_pairs(src_train, tgt_train)
    X_val, y_val, val_src_lens = encode_pairs(src_val, tgt_val)
    print(f"Train pairs: {len(X_train)}, val pairs: {len(X_val)}")
    return (X_train, y_train), (X_val, y_val, val_src_lens, src_val, tgt_val), src_vocab, tgt_vocab


class BahdanauAttention(layers.Layer):
    """
    Additive attention. Returns BOTH the context vector (what the decoder
    actually uses) and the raw attention weights (what we plot as a heatmap) -
    a real Keras Attention layer would hide the weights from you, which
    defeats the point of this project.
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W1 = layers.Dense(units)   # applied to every encoder state
        self.W2 = layers.Dense(units)   # applied to the current decoder state
        self.V = layers.Dense(1)        # collapses to a single relevance score per position

    def call(self, decoder_state, encoder_states):
        # decoder_state: (batch, hidden)      encoder_states: (batch, src_len, hidden)
        decoder_state_expanded = tf.expand_dims(decoder_state, 1)  # (batch, 1, hidden)
        score = self.V(tf.nn.tanh(self.W1(encoder_states) + self.W2(decoder_state_expanded)))  # (batch, src_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch, src_len, 1) - sums to 1 over source positions
        context = tf.reduce_sum(attention_weights * encoder_states, axis=1)  # (batch, hidden)
        return context, tf.squeeze(attention_weights, axis=-1)  # weights: (batch, src_len)


class AttentionDecoder(layers.Layer):
    """
    One Layer, one internal Python loop over time steps, traced ONCE by
    Keras/tf.function when the model is built. This replaces an earlier
    version of this script that called 16 separate Keras functional-API
    layers in an outer Python loop - that worked, but each of those 80+
    layer() calls carries its own Python-side bookkeeping (KerasTensor
    wrapping, graph-node registration, shape inference), which made even
    building the training graph take several minutes. Looping INSIDE one
    Layer's call() keeps all of that bookkeeping to a single call.
    """
    def __init__(self, hidden_size, vocab_size, embed_dim, max_len, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.embedding = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.attention = BahdanauAttention(hidden_size)
        self.lstm_cell = layers.LSTMCell(hidden_size)
        self.dense = layers.Dense(vocab_size, activation='softmax')

    def call(self, decoder_input_seq, encoder_states_seq, h, c):
        dec_emb_seq = self.embedding(decoder_input_seq)  # (batch, max_len, embed)
        all_preds, all_attn = [], []
        for t in range(self.max_len):
            dec_emb_t = dec_emb_seq[:, t, :]                          # (batch, embed)
            context, attn_w = self.attention(h, encoder_states_seq)   # (batch, hidden), (batch, src_len)
            lstm_input = tf.concat([dec_emb_t, context], axis=-1)
            output, (h, c) = self.lstm_cell(lstm_input, states=[h, c])
            pred = self.dense(output)                                  # (batch, vocab)
            all_preds.append(pred)
            all_attn.append(attn_w)
        preds = tf.stack(all_preds, axis=1)      # (batch, max_len, vocab)
        attn_stack = tf.stack(all_attn, axis=1)  # (batch, max_len, src_len)
        return preds, attn_stack


def build_models(src_vocab_size, tgt_vocab_size):
    # --- Encoder: keep every hidden state, not just the last ---
    encoder_input = layers.Input(shape=(MAX_LEN,), name='encoder_input')
    enc_emb = layers.Embedding(src_vocab_size, EMBED_DIM, mask_zero=True)(encoder_input)
    encoder_states_seq, enc_h, enc_c = layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)(enc_emb)
    encoder_model = models.Model(encoder_input, [encoder_states_seq, enc_h, enc_c])

    decoder = AttentionDecoder(HIDDEN_SIZE, tgt_vocab_size, EMBED_DIM, MAX_LEN)
    return encoder_model, decoder


def build_train_model(encoder_model, decoder, src_vocab_size, tgt_vocab_size):
    encoder_input = layers.Input(shape=(MAX_LEN,), name='encoder_input_train')
    decoder_input = layers.Input(shape=(MAX_LEN,), name='decoder_input_train')

    enc_states_seq, h, c = encoder_model(encoder_input)
    preds, _ = decoder(decoder_input, enc_states_seq, h, c)

    model = models.Model([encoder_input, decoder_input], preds)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def greedy_translate_with_attention(src_tokens, encoder_model, decoder, src_vocab, tgt_vocab):
    src_ids = src_vocab.encode(src_tokens, MAX_LEN)
    src_ids = pad_sequences([src_ids], maxlen=MAX_LEN, padding='post', value=0)
    enc_states_seq, h, c = encoder_model.predict(src_ids, verbose=0)
    enc_states_seq = tf.convert_to_tensor(enc_states_seq)
    h, c = tf.convert_to_tensor(h), tf.convert_to_tensor(c)

    tok = tf.constant([[tgt_vocab.stoi['<sos>']]])
    output_ids, attn_history = [], []
    for _ in range(MAX_LEN):
        dec_emb_t = decoder.embedding(tok)[:, 0, :]
        context, attn_w = decoder.attention(h, enc_states_seq)
        lstm_input = tf.concat([dec_emb_t, context], axis=-1)
        output, (h, c) = decoder.lstm_cell(lstm_input, states=[h, c])
        pred = decoder.dense(output)
        attn_history.append(attn_w.numpy()[0])
        next_id = int(np.argmax(pred.numpy()[0]))
        if next_id == tgt_vocab.stoi['<eos>']:
            break
        output_ids.append(next_id)
        tok = tf.constant([[next_id]])
    return output_ids, np.array(attn_history)


def main():
    (X_train, y_train), (X_val, y_val, val_src_lens, src_val_raw, tgt_val_raw), src_vocab, tgt_vocab = load_data()

    decoder_input_train = np.zeros_like(y_train)
    decoder_input_train[:, 1:] = y_train[:, :-1]
    decoder_input_train[:, 0] = tgt_vocab.stoi['<sos>']
    decoder_input_val = np.zeros_like(y_val)
    decoder_input_val[:, 1:] = y_val[:, :-1]
    decoder_input_val[:, 0] = tgt_vocab.stoi['<sos>']

    encoder_model, decoder = build_models(len(src_vocab), len(tgt_vocab))
    train_model = build_train_model(encoder_model, decoder, len(src_vocab), len(tgt_vocab))
    train_model.summary()

    ckpt_path = os.path.join(OUTPUT_DIR, 'checkpoint.weights.h5')
    initial_epoch = 0
    if os.path.exists(ckpt_path):
        print(f"Found checkpoint at {ckpt_path} - resuming instead of retraining from scratch.")
        train_model.load_weights(ckpt_path)
        if os.path.exists(ckpt_path + '.epoch'):
            initial_epoch = int(open(ckpt_path + '.epoch').read().strip())
            print(f"Resuming from epoch {initial_epoch}")

    class SaveEveryEpoch(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            self.model.save_weights(ckpt_path)
            with open(ckpt_path + '.epoch', 'w') as f:
                f.write(str(epoch + 1))
            print(f"  (checkpoint saved after epoch {epoch + 1})")

    history = train_model.fit(
        [X_train, decoder_input_train], np.expand_dims(y_train, -1),
        validation_data=([X_val, decoder_input_val], np.expand_dims(y_val, -1)),
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=[SaveEveryEpoch()],
    )

    fig1, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history.history['loss'], label='train'); axes[0].plot(history.history['val_loss'], label='val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history.history['accuracy'], label='train'); axes[1].plot(history.history['val_accuracy'], label='val')
    axes[1].set_title('Per-token accuracy (teacher-forced)'); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
    print("Saved: training_curves.png")

    print("\nGreedy-decoding validation sample with attention (this takes a bit)...")
    N_EVAL = 300
    smoother = SmoothingFunction().method1
    bleu_scores, eval_src_lens = [], []
    best_attn_example = None
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(src_val_raw), size=min(N_EVAL, len(src_val_raw)), replace=False)

    for idx in sample_idx:
        src_tokens = src_val_raw[idx]
        tgt_tokens = tgt_val_raw[idx]
        pred_ids, attn_hist = greedy_translate_with_attention(src_tokens, encoder_model, decoder, src_vocab, tgt_vocab)
        pred_words = [tgt_vocab.itos[i] for i in pred_ids]
        bleu = sentence_bleu([tgt_tokens], pred_words, smoothing_function=smoother)
        bleu_scores.append(bleu)
        eval_src_lens.append(len(src_tokens))
        if best_attn_example is None and 8 <= len(src_tokens) <= 12 and len(pred_words) >= 4:
            best_attn_example = (src_tokens, pred_words, attn_hist)

    bleu_scores = np.array(bleu_scores)
    eval_src_lens = np.array(eval_src_lens)
    print(f"\nMean BLEU (attention model) over {N_EVAL} validation sentences: {bleu_scores.mean():.4f}")

    bins = [0, 5, 8, 11, 14, 100]
    bin_labels = ['1-5', '6-8', '9-11', '12-14', '15+']
    bucket_means = []
    for label, lo, hi in zip(bin_labels, bins[:-1], bins[1:]):
        mask = (eval_src_lens > lo) & (eval_src_lens <= hi)
        bucket_means.append(bleu_scores[mask].mean() if mask.sum() > 0 else np.nan)
        print(f"  length {label}: mean BLEU = {bucket_means[-1]:.4f}  (n={mask.sum()})")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(bin_labels, bucket_means, color='#2ca02c')
    ax2.set_xlabel('Source sentence length (tokens, including <sos>/<eos>)')
    ax2.set_ylabel('Mean BLEU score')
    ax2.set_title('WITH attention: quality vs. sentence length (compare to Project 6)')
    ax2.grid(alpha=0.3, axis='y')
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'bleu_vs_length_with_attention.png'), dpi=150)
    print("Saved: bleu_vs_length_with_attention.png")

    # --- The signature attention visualization: which source words did the
    #     decoder actually look at for each output word? ---
    if best_attn_example is not None:
        src_tokens, pred_words, attn_hist = best_attn_example
        src_len_actual = len(src_tokens) + 2  # + <sos>/<eos>
        attn_matrix = attn_hist[:len(pred_words), :src_len_actual]

        fig3, ax3 = plt.subplots(figsize=(1 + 0.6 * src_len_actual, 1 + 0.6 * len(pred_words)))
        im = ax3.imshow(attn_matrix, cmap='viridis', aspect='auto')
        ax3.set_xticks(range(src_len_actual))
        ax3.set_xticklabels(['<sos>'] + src_tokens + ['<eos>'], rotation=90)
        ax3.set_yticks(range(len(pred_words)))
        ax3.set_yticklabels(pred_words)
        ax3.set_xlabel('Source (English)')
        ax3.set_ylabel('Generated (German)')
        ax3.set_title('Attention weights: which English words each German word attended to')
        fig3.colorbar(im, ax=ax3, label='Attention weight')
        fig3.tight_layout()
        fig3.savefig(os.path.join(OUTPUT_DIR, 'attention_heatmap.png'), dpi=150)
        print("Saved: attention_heatmap.png")
        print(f"\nExample: {' '.join(src_tokens)}\n-> {' '.join(pred_words)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
