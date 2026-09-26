"""
Project 3 - Vanilla RNN: Forward Propagation Through Time + BPTT (pure NumPy)

Goal: see exactly how an RNN processes a SEQUENCE (not a single vector like
Project 1's ANN), and how Backpropagation Through Time (BPTT) accumulates
gradients across time steps - including watching the vanishing-gradient
problem happen live, which is the entire motivation for Project 4 (LSTM/GRU).

Dataset: tiny_shakespeare (HuggingFace `datasets`) - real text, character-level
language modeling: given characters seen so far, predict the next character.

Architecture (one RNN cell, unrolled across a sequence of length T):
    h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
    y_t = W_hy @ h_t + b_y   (softmax -> distribution over next character)

The SAME weights (W_xh, W_hh, W_hy) are reused at every time step - that
weight sharing across time is what makes it a "recurrent" network, exactly
like a CNN's filter is shared across space (Project 2).

RUN WITH:
    python rnn_bptt.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset

np.random.seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 25          # BPTT is truncated to this many time steps per training step
HIDDEN_SIZE = 100
LEARNING_RATE = 0.1
TRAIN_STEPS = 3000
TEXT_CHARS_USED = 20000   # slice of tiny_shakespeare - keeps this a quick, real run


def load_text():
    ds = load_dataset('tiny_shakespeare', trust_remote_code=True)
    text = ds['train']['text'][0][:TEXT_CHARS_USED]
    chars = sorted(set(text))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    print(f"Loaded {len(text)} characters, vocab size {len(chars)}")
    return text, char_to_ix, ix_to_char


class VanillaRNN:
    """
    Every weight matrix is shared across ALL time steps - this is what lets
    an RNN handle variable-length sequences with a fixed number of parameters.
    """
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

        # Adagrad accumulators (adaptive per-parameter learning rate - keeps
        # training stable without needing careful LR tuning by hand)
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)

    def forward(self, inputs, h_prev):
        """
        FORWARD PROPAGATION THROUGH TIME.
        inputs: list of one-hot vectors, one per time step.
        Returns per-step activations - all needed later by BPTT, exactly
        like Project 1's forward() stored every layer's activation for backprop,
        except here the "layers" are TIME STEPS sharing one set of weights.
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        for t in range(len(inputs)):
            xs[t] = inputs[t]
            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t - 1] + self.bh)
            ys[t] = self.Why @ hs[t] + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # softmax -> next-char distribution
        return xs, hs, ps

    def bptt(self, xs, hs, ps, targets):
        """
        BACKPROPAGATION THROUGH TIME.

        Same chain rule as Project 1's backward(), but now the gradient at
        each time step ALSO depends on the gradient from the NEXT time step
        (dh_next), because h_t feeds into h_{t+1}. This is the recurrent
        dependency that makes RNNs different from a plain feedforward stack:
        gradients get multiplied by Whh repeatedly as they flow backward
        through time. If Whh's eigenvalues are < 1, repeated multiplication
        shrinks the gradient toward zero the further back in time you go -
        this IS the vanishing gradient problem, and we measure it below.
        """
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])

        grad_norms_by_timestep = []  # tracks how the gradient magnitude decays going backward

        for t in reversed(range(len(xs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # softmax + cross-entropy gradient simplifies to (p - target_onehot)

            dWhy += dy @ hs[t].T
            dby += dy

            # chain rule: error flowing back from the OUTPUT at time t,
            # PLUS error flowing back from the NEXT hidden state (recurrence)
            dh = self.Why.T @ dy + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh   # tanh derivative
            dbh += dh_raw
            dWxh += dh_raw @ xs[t].T
            dWhh += dh_raw @ hs[t - 1].T

            dh_next = self.Whh.T @ dh_raw    # this becomes "dh_next" for time step t-1
            grad_norms_by_timestep.append(np.linalg.norm(dh_raw))

        for dparam in (dWxh, dWhh, dWhy, dbh, dby):
            np.clip(dparam, -5, 5, out=dparam)  # gradient clipping - standard RNN stabilizer

        return dWxh, dWhh, dWhy, dbh, dby, list(reversed(grad_norms_by_timestep))

    def adagrad_step(self, grads, lr):
        dWxh, dWhh, dWhy, dbh, dby = grads
        for param, dparam, mem in zip(
            [self.Wxh, self.Whh, self.Why, self.bh, self.by],
            [dWxh, dWhh, dWhy, dbh, dby],
            [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby],
        ):
            mem += dparam * dparam
            param += -lr * dparam / np.sqrt(mem + 1e-8)

    def sample(self, h, seed_ix, n, char_to_ix):
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for _ in range(n):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes


def one_hot(ix, vocab_size):
    v = np.zeros((vocab_size, 1))
    v[ix] = 1
    return v


def main():
    text, char_to_ix, ix_to_char = load_text()
    vocab_size = len(char_to_ix)
    rnn = VanillaRNN(vocab_size, HIDDEN_SIZE)

    losses = []
    grad_decay_snapshots = []  # gradient-norm-by-timestep, sampled a few times during training
    smooth_loss = -np.log(1.0 / vocab_size) * SEQ_LEN  # initial expected loss for random guessing

    n, p = 0, 0
    h_prev = np.zeros((HIDDEN_SIZE, 1))

    for step in range(TRAIN_STEPS):
        if p + SEQ_LEN + 1 >= len(text) or n == 0:
            h_prev = np.zeros((HIDDEN_SIZE, 1))  # reset hidden state at start of each "epoch" over the text
            p = 0

        inputs_ix = [char_to_ix[ch] for ch in text[p:p + SEQ_LEN]]
        targets_ix = [char_to_ix[ch] for ch in text[p + 1:p + SEQ_LEN + 1]]
        inputs = [one_hot(ix, vocab_size) for ix in inputs_ix]

        xs, hs, ps = rnn.forward(inputs, h_prev)
        loss = sum(-np.log(ps[t][targets_ix[t], 0] + 1e-12) for t in range(SEQ_LEN))
        *grads, grad_norms = rnn.bptt(xs, hs, ps, targets_ix)

        rnn.adagrad_step(grads, LEARNING_RATE)
        h_prev = hs[SEQ_LEN - 1]

        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if step % 50 == 0:
            losses.append((step, smooth_loss))
        if step in (0, TRAIN_STEPS // 4, TRAIN_STEPS // 2, TRAIN_STEPS - 1):
            grad_decay_snapshots.append((step, grad_norms))
        if step % 500 == 0:
            print(f"step {step:5d}  smooth_loss={smooth_loss:.3f}")
            sample_ix = rnn.sample(h_prev, inputs_ix[0], 120, char_to_ix)
            sample_text = ''.join(ix_to_char[ix] for ix in sample_ix)
            print(f"  sample: {sample_text!r}")

        p += SEQ_LEN
        n += 1

    # --- Plot 1: training loss curve ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(*zip(*losses), linewidth=2, color='#1f77b4')
    ax1.set_xlabel('Training step')
    ax1.set_ylabel('Smoothed cross-entropy loss')
    ax1.set_title('Vanilla RNN (BPTT) training loss - tiny_shakespeare, character-level')
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, 'training_loss.png'), dpi=150)
    print("Saved: training_loss.png")

    # --- Plot 2: the vanishing gradient, made visible ---
    # For each snapshot, grad_norms[0] is the gradient at the FIRST time step
    # of the truncated window (furthest back in time) and grad_norms[-1] is
    # the LAST (most recent). If training is behaving like classic vanilla-RNN
    # theory, the norm should shrink noticeably from recent -> distant steps.
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for step, norms in grad_decay_snapshots:
        ax2.plot(range(len(norms)), norms, marker='o', label=f'step {step}')
    ax2.set_xlabel('Time step within the 25-step BPTT window (0 = oldest)')
    ax2.set_ylabel('Gradient norm ||dL/dh_t||')
    ax2.set_title('Vanishing gradient across BPTT window (vanilla RNN)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'vanishing_gradient.png'), dpi=150)
    print("Saved: vanishing_gradient.png")

    # --- Final generated sample, saved to a text file for the README/portfolio ---
    h = np.zeros((HIDDEN_SIZE, 1))
    sample_ix = rnn.sample(h, char_to_ix[text[0]], 400, char_to_ix)
    sample_text = ''.join(ix_to_char[ix] for ix in sample_ix)
    with open(os.path.join(OUTPUT_DIR, 'final_sample.txt'), 'w', encoding='utf-8') as f:
        f.write(sample_text)
    print("Saved: final_sample.txt")
    print("\nFinal generated text sample:\n" + sample_text)


if __name__ == "__main__":
    main()
