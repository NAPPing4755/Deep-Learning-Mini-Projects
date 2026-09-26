"""
Project 9 - Scaled Dot-Product Attention & Multi-Head Attention
(the direct bridge from Project 8's Bahdanau attention to Transformers)

Project 8's attention was RECURRENT: at each decoder time step, the
(single) decoder hidden state asked "which encoder states matter right now?"
using a small learned MLP. That has two limits: (1) it needs a sequential
decoder loop (slow - Project 8's whole runtime problem), and (2) one
attention head can only learn one notion of "relevant."

Scaled dot-product attention replaces the MLP scoring function with a single
matrix multiplication (Q @ K^T), which is parallelizable across the WHOLE
sequence at once - no step-by-step loop needed. Multi-head attention runs
several of these in parallel subspaces, so different heads can specialize
(e.g. one head tracks subject-verb agreement, another tracks negation).
This combination (multi-head self-attention + position-wise feedforward,
no recurrence at all) is the core building block of the Transformer.

PART 1 - anatomy (pure NumPy, forward pass only, same style as Projects 1/3/4):
  Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
  computed on a small toy example with the math fully exposed.

PART 2 - real training (Keras): a minimal Transformer encoder (multi-head
self-attention + feedforward + residual/layernorm, NO recurrence anywhere)
trained on real sentiment classification data, with the actual learned
per-head attention weights visualized for a real sentence.

Dataset: rotten_tomatoes (HuggingFace) - 8,530 real movie review sentences,
binary sentiment (positive/negative).

RUN WITH:
    python multihead_attention.py
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

np.random.seed(42)
tf.random.set_seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LEN = 40
MAX_VOCAB = 8000
EMBED_DIM = 64
NUM_HEADS = 4
FF_DIM = 128
EPOCHS = 12
BATCH_SIZE = 32


# ============================================================
# PART 1 - ANATOMY: scaled dot-product & multi-head attention from scratch
# ============================================================

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # numerically stable softmax
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    """
    Q, K, V: (seq_len, d_k). One matrix multiply computes EVERY query's
    similarity to EVERY key simultaneously (Q @ K^T is (seq_len, seq_len)) -
    contrast with Project 8, where getting one row of this matrix required
    one full decoder step.

    Dividing by sqrt(d_k) keeps the dot products from growing large as d_k
    grows (large dot products push softmax into a near-one-hot regime with
    vanishing gradients almost everywhere else) - this is the ONE piece of
    the formula that isn't just "attention, but with a matmul."
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)         # (seq_len, seq_len) - "how relevant is key_j to query_i"
    weights = softmax(scores, axis=-1)      # each ROW sums to 1 - a probability distribution over keys, per query
    output = weights @ V                     # (seq_len, d_v) - weighted blend of values
    return output, weights


def multi_head_attention_numpy(X, num_heads, d_model):
    """
    Splits the model dimension into `num_heads` equal chunks, runs scaled
    dot-product attention independently in EACH chunk (a different learned
    subspace), then concatenates the results back together. Each head gets
    its own random projection here (a stand-in for learned W_q/W_k/W_v
    weights) purely to demonstrate the mechanics - real training learns
    these projections, which Part 2 does.
    """
    seq_len, _ = X.shape
    d_k = d_model // num_heads
    all_head_outputs = []
    all_head_weights = []
    for head in range(num_heads):
        Wq = np.random.randn(d_model, d_k) * 0.1
        Wk = np.random.randn(d_model, d_k) * 0.1
        Wv = np.random.randn(d_model, d_k) * 0.1
        Q, K, V = X @ Wq, X @ Wk, X @ Wv
        out, weights = scaled_dot_product_attention(Q, K, V)
        all_head_outputs.append(out)
        all_head_weights.append(weights)
    concatenated = np.concatenate(all_head_outputs, axis=-1)  # (seq_len, d_model) again
    return concatenated, all_head_weights


def run_anatomy_demo():
    print("=" * 70)
    print("PART 1: Scaled dot-product & multi-head attention - mechanics")
    print("=" * 70)

    seq_len, d_model, num_heads = 6, 16, 4
    X = np.random.randn(seq_len, d_model) * 0.5  # a toy "sequence" of 6 token embeddings

    Wq = np.random.randn(d_model, d_model) * 0.1
    Wk = np.random.randn(d_model, d_model) * 0.1
    Wv = np.random.randn(d_model, d_model) * 0.1
    Q, K, V = X @ Wq, X @ Wk, X @ Wv

    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"\nSingle-head attention: input {X.shape} -> output {output.shape}")
    print(f"Attention weight matrix (one row per query, sums to 1 across keys):\n{np.round(weights, 3)}")

    mh_output, head_weights = multi_head_attention_numpy(X, num_heads, d_model)
    print(f"\nMulti-head attention ({num_heads} heads): output shape {mh_output.shape} "
          f"(same as input - each head sees a {d_model // num_heads}-dim slice, concatenated back to {d_model})")
    for i, w in enumerate(head_weights):
        print(f"  Head {i}: attention weights for token 0 -> {np.round(w[0], 3)}")
    print()


# ============================================================
# PART 2 - REAL TRAINING: a minimal Transformer encoder for sentiment classification
# ============================================================

def tokenize(text):
    text = text.lower().strip()
    text = re.sub(r"([.,!?'])", r" \1 ", text)
    text = re.sub(r"[^a-z.,!?' ]+", " ", text)
    return text.split()


class Vocab:
    def __init__(self, tokenized_sentences, max_vocab):
        freq = {}
        for sent in tokenized_sentences:
            for tok in sent:
                freq[tok] = freq.get(tok, 0) + 1
        most_common = sorted(freq.items(), key=lambda x: -x[1])[:max_vocab - 2]
        self.itos = ['<pad>', '<unk>'] + [w for w, _ in most_common]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, tokens, max_len):
        ids = [self.stoi.get(t, self.stoi['<unk>']) for t in tokens][:max_len]
        return ids

    def __len__(self):
        return len(self.itos)


def load_data():
    ds = load_dataset('rotten_tomatoes')
    train_tokens = [tokenize(x) for x in ds['train']['text']]
    val_tokens = [tokenize(x) for x in ds['validation']['text']]

    vocab = Vocab(train_tokens, MAX_VOCAB)
    print(f"Vocab size: {len(vocab)}")

    X_train = pad_sequences([vocab.encode(t, MAX_LEN) for t in train_tokens], maxlen=MAX_LEN, padding='post')
    X_val = pad_sequences([vocab.encode(t, MAX_LEN) for t in val_tokens], maxlen=MAX_LEN, padding='post')
    y_train = np.array(ds['train']['label'])
    y_val = np.array(ds['validation']['label'])

    print(f"Train: {len(X_train)}, val: {len(X_val)}")
    return (X_train, y_train), (X_val, y_val, val_tokens), vocab


class PositionalEncoding(layers.Layer):
    """
    Self-attention has NO built-in notion of word order (it's just a
    weighted average over the whole sequence regardless of position) -
    unlike an RNN, which processes tokens one at a time and so implicitly
    knows their order. Positional encoding adds a fixed, position-dependent
    pattern to each embedding so the model can still tell "the dog bit the
    man" apart from "the man bit the dog."
    """
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        positions = np.arange(max_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
        angles = positions * angle_rates
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pos_encoding = tf.constant(angles[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]


class TransformerEncoderBlock(layers.Layer):
    """
    The core Transformer building block, with NO recurrence anywhere:
      1. Multi-head self-attention (query=key=value=the same sequence -
         every token directly looks at every other token, in one matmul,
         regardless of how far apart they are - contrast with Project 3's
         RNN, where distant tokens could only interact through many
         sequential hidden-state updates, which is exactly what caused the
         vanishing-gradient problem in the first place)
      2. Residual connection + LayerNorm
      3. Position-wise feedforward network
      4. Residual connection + LayerNorm
    """
    def __init__(self, d_model, num_heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, x, return_attention_scores=False):
        attn_output, attn_scores = self.attention(x, x, return_attention_scores=True)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        if return_attention_scores:
            return x, attn_scores
        return x


def build_model(vocab_size):
    inputs = layers.Input(shape=(MAX_LEN,))
    x = layers.Embedding(vocab_size, EMBED_DIM, mask_zero=True)(inputs)
    x = PositionalEncoding(MAX_LEN, EMBED_DIM)(x)
    encoder_block = TransformerEncoderBlock(EMBED_DIM, NUM_HEADS, FF_DIM)
    x = encoder_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model, encoder_block


def main():
    run_anatomy_demo()

    (X_train, y_train), (X_val, y_val, val_tokens), vocab = load_data()
    model, encoder_block = build_model(len(vocab))
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    fig1, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history.history['loss'], label='train'); axes[0].plot(history.history['val_loss'], label='val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history.history['accuracy'], label='train'); axes[1].plot(history.history['val_accuracy'], label='val')
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
    print("Saved: training_curves.png")

    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\nFinal validation accuracy: {final_val_acc:.4f}")

    # --- Extract real per-head self-attention weights for one actual sentence ---
    sample_idx = 3
    sample_tokens = val_tokens[sample_idx][:MAX_LEN]
    sample_ids = pad_sequences([vocab.encode(sample_tokens, MAX_LEN)], maxlen=MAX_LEN, padding='post')

    embed_layer = model.layers[1]
    pos_layer = model.layers[2]
    x_embedded = pos_layer(embed_layer(sample_ids))
    _, attn_scores = encoder_block(x_embedded, return_attention_scores=True)
    attn_scores = attn_scores.numpy()[0]  # (num_heads, seq_len, seq_len)

    seq_len_actual = len(sample_tokens)
    fig2, axes2 = plt.subplots(1, NUM_HEADS, figsize=(4.2 * NUM_HEADS, 4.5))
    for head in range(NUM_HEADS):
        ax = axes2[head]
        im = ax.imshow(attn_scores[head, :seq_len_actual, :seq_len_actual], cmap='viridis')
        ax.set_xticks(range(seq_len_actual))
        ax.set_xticklabels(sample_tokens, rotation=90, fontsize=8)
        ax.set_yticks(range(seq_len_actual))
        ax.set_yticklabels(sample_tokens, fontsize=8)
        ax.set_title(f'Head {head}')
    fig2.suptitle(f'Self-attention per head: "{" ".join(sample_tokens)}"')
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'multihead_self_attention.png'), dpi=150)
    print("Saved: multihead_self_attention.png")
    print(f"\nExample sentence analyzed: {' '.join(sample_tokens)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
