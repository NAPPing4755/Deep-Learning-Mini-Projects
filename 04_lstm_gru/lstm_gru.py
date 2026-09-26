"""
Project 4 - LSTM and GRU: gates, cell state, and why they beat a vanilla RNN

Two parts:

PART 1 - "Anatomy" demo (pure NumPy, forward pass only):
  Runs ONE LSTM cell step and ONE GRU cell step with every gate computed
  and printed explicitly, so you can see exactly what a "gate" is: a sigmoid
  output in [0,1] that multiplies (gates) some other vector element-wise.
  This is the same style as Project 3's hand-written RNN cell, just with the
  extra gating machinery made visible.

PART 2 - Real training comparison (Keras):
  SimpleRNN vs LSTM vs GRU, same hidden size, same data, same optimizer,
  trained on the SAME character-level next-character task as Project 3
  (tiny_shakespeare). This directly answers "do gates actually help?" with
  a measured loss curve instead of just trusting the theory.

RUN WITH:
    python lstm_gru.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from datasets import load_dataset

np.random.seed(42)
tf.random.set_seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEXT_CHARS_USED = 40000
# 25 characters turned out too SHORT a window for vanilla RNN's vanishing
# gradient to actually bite (Project 3's own BPTT plot showed decay, but not
# enough over only 25 steps to hurt training). 100 characters is long enough
# that a vanilla RNN struggles to connect a character to context 100 steps
# back, which is exactly the regime LSTM/GRU's gating is designed to fix -
# so this is where the comparison should actually show a real gap.
SEQ_LEN = 100
HIDDEN_SIZE = 64
EPOCHS = 15
BATCH_SIZE = 128


# ============================================================
# PART 1 - ANATOMY: one LSTM step and one GRU step, gates made explicit
# ============================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def lstm_cell_step(x_t, h_prev, c_prev, params):
    """
    One LSTM time step. Four gates, each its OWN small feedforward layer
    over [h_prev, x_t]:

      forget gate  f_t = sigmoid(Wf . [h_prev, x_t] + bf)   -> how much of the
                                                                OLD cell state to keep
      input gate   i_t = sigmoid(Wi . [h_prev, x_t] + bi)   -> how much of the
                                                                NEW candidate to let in
      candidate    g_t = tanh(Wg . [h_prev, x_t] + bg)      -> proposed new content
      output gate  o_t = sigmoid(Wo . [h_prev, x_t] + bo)   -> how much of the
                                                                cell state to expose as h_t

      c_t = f_t * c_prev + i_t * g_t     <- the cell state "conveyor belt":
                                             this sum (not a repeated matrix
                                             multiply like vanilla RNN's h_t)
                                             is WHY gradients don't vanish as
                                             fast - the path from c_t back to
                                             c_{t-1} has a direct additive term.
      h_t = o_t * tanh(c_t)
    """
    concat = np.concatenate([h_prev, x_t], axis=0)
    f_t = sigmoid(params['Wf'] @ concat + params['bf'])
    i_t = sigmoid(params['Wi'] @ concat + params['bi'])
    g_t = np.tanh(params['Wg'] @ concat + params['bg'])
    o_t = sigmoid(params['Wo'] @ concat + params['bo'])

    c_t = f_t * c_prev + i_t * g_t
    h_t = o_t * np.tanh(c_t)
    return h_t, c_t, dict(f=f_t, i=i_t, g=g_t, o=o_t)


def gru_cell_step(x_t, h_prev, params):
    """
    One GRU time step. GRU merges LSTM's forget+input gates into a single
    "update gate" and has no separate cell state - h_t IS the state:

      update gate  z_t = sigmoid(Wz . [h_prev, x_t] + bz)   -> how much of
                                                                h_prev to keep
      reset gate   r_t = sigmoid(Wr . [h_prev, x_t] + br)   -> how much of
                                                                h_prev to use when
                                                                computing the candidate
      candidate    h~_t = tanh(Wh . [r_t*h_prev, x_t] + bh)
      h_t = (1 - z_t) * h_prev + z_t * h~_t

    Fewer gates -> fewer parameters than LSTM (roughly 3/4 as many), often
    trains faster with comparable accuracy on smaller datasets - which is
    exactly what we measure in Part 2.
    """
    concat = np.concatenate([h_prev, x_t], axis=0)
    z_t = sigmoid(params['Wz'] @ concat + params['bz'])
    r_t = sigmoid(params['Wr'] @ concat + params['br'])

    concat_reset = np.concatenate([r_t * h_prev, x_t], axis=0)
    h_candidate = np.tanh(params['Wh'] @ concat_reset + params['bh'])

    h_t = (1 - z_t) * h_prev + z_t * h_candidate
    return h_t, dict(z=z_t, r=r_t, h_candidate=h_candidate)


def run_anatomy_demo():
    print("=" * 70)
    print("PART 1: LSTM / GRU cell anatomy - one time step, gates exposed")
    print("=" * 70)

    input_size, hidden_size = 8, 5
    x_t = np.random.randn(input_size, 1) * 0.5
    h_prev = np.zeros((hidden_size, 1))
    c_prev = np.zeros((hidden_size, 1))

    def init_gate(out_dim, in_dim):
        return np.random.randn(out_dim, in_dim) * 0.1

    concat_dim = hidden_size + input_size
    lstm_params = {
        'Wf': init_gate(hidden_size, concat_dim), 'bf': np.zeros((hidden_size, 1)),
        'Wi': init_gate(hidden_size, concat_dim), 'bi': np.zeros((hidden_size, 1)),
        'Wg': init_gate(hidden_size, concat_dim), 'bg': np.zeros((hidden_size, 1)),
        'Wo': init_gate(hidden_size, concat_dim), 'bo': np.zeros((hidden_size, 1)),
    }
    gru_params = {
        'Wz': init_gate(hidden_size, concat_dim), 'bz': np.zeros((hidden_size, 1)),
        'Wr': init_gate(hidden_size, concat_dim), 'br': np.zeros((hidden_size, 1)),
        'Wh': init_gate(hidden_size, concat_dim), 'bh': np.zeros((hidden_size, 1)),
    }

    h_lstm, c_lstm, gates_lstm = lstm_cell_step(x_t, h_prev, c_prev, lstm_params)
    print("\nLSTM step:")
    print(f"  forget gate f_t (mean={gates_lstm['f'].mean():.3f}) - close to 1 means 'keep old cell state'")
    print(f"  input  gate i_t (mean={gates_lstm['i'].mean():.3f}) - close to 1 means 'accept new candidate'")
    print(f"  output gate o_t (mean={gates_lstm['o'].mean():.3f}) - close to 1 means 'expose full cell state as h_t'")
    print(f"  resulting cell state c_t: {c_lstm.ravel()}")
    print(f"  resulting hidden state h_t: {h_lstm.ravel()}")

    h_gru, gates_gru = gru_cell_step(x_t, h_prev, gru_params)
    print("\nGRU step:")
    print(f"  update gate z_t (mean={gates_gru['z'].mean():.3f}) - close to 1 means 'favor the new candidate'")
    print(f"  reset  gate r_t (mean={gates_gru['r'].mean():.3f}) - close to 0 means 'ignore h_prev when forming candidate'")
    print(f"  resulting hidden state h_t: {h_gru.ravel()}")
    print()


# ============================================================
# PART 2 - REAL TRAINING COMPARISON: SimpleRNN vs LSTM vs GRU
# ============================================================

def load_char_dataset():
    ds = load_dataset('tiny_shakespeare', trust_remote_code=True)
    text = ds['train']['text'][0][:TEXT_CHARS_USED]
    chars = sorted(set(text))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    # Build (X, y) pairs: X = 25 previous char indices, y = next char index
    ix_text = [char_to_ix[ch] for ch in text]
    X, y = [], []
    for i in range(len(ix_text) - SEQ_LEN):
        X.append(ix_text[i:i + SEQ_LEN])
        y.append(ix_text[i + SEQ_LEN])
    X = np.array(X)
    y = np.array(y)

    split = int(0.9 * len(X))
    return (X[:split], y[:split]), (X[split:], y[split:]), vocab_size


def build_model(cell_type, vocab_size):
    """
    Identical architecture except for the recurrent cell type, so the
    comparison isolates the gating mechanism's effect - same idea as
    Project 2's "identical starting weights" trick, applied to architecture
    choice instead of optimizer choice.
    """
    RecurrentLayer = {'SimpleRNN': layers.SimpleRNN, 'LSTM': layers.LSTM, 'GRU': layers.GRU}[cell_type]
    model = models.Sequential([
        layers.Input(shape=(SEQ_LEN,)),
        layers.Embedding(vocab_size, 32),
        RecurrentLayer(HIDDEN_SIZE),
        layers.Dense(vocab_size, activation='softmax'),
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run_training_comparison():
    print("=" * 70)
    print("PART 2: SimpleRNN vs LSTM vs GRU - same task as Project 3")
    print("=" * 70)
    (X_train, y_train), (X_val, y_val), vocab_size = load_char_dataset()
    print(f"Vocab size: {vocab_size}, train sequences: {len(X_train)}, val sequences: {len(X_val)}")

    histories = {}
    param_counts = {}
    for cell_type in ['SimpleRNN', 'LSTM', 'GRU']:
        print(f"\n--- Training {cell_type} ---")
        model = build_model(cell_type, vocab_size)
        param_counts[cell_type] = model.count_params()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
        )
        histories[cell_type] = history.history

    # --- Plot: validation loss + accuracy, all three cells on the same axes ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for cell_type, hist in histories.items():
        axes[0].plot(hist['val_loss'], label=f"{cell_type} ({param_counts[cell_type]:,} params)", linewidth=2)
        axes[1].plot(hist['val_accuracy'], label=cell_type, linewidth=2)

    axes[0].set_title('Validation loss - next-character prediction')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-entropy loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title('Validation accuracy - next-character prediction')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_name = f'rnn_vs_lstm_vs_gru_seqlen{SEQ_LEN}.png'
    fig.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=150)
    print(f"\nSaved: {out_name}")

    print("\nFinal validation accuracy:")
    for cell_type, hist in histories.items():
        print(f"  {cell_type:<10} acc={hist['val_accuracy'][-1]:.4f}  loss={hist['val_loss'][-1]:.4f}  params={param_counts[cell_type]:,}")


if __name__ == "__main__":
    run_anatomy_demo()
    run_training_comparison()
