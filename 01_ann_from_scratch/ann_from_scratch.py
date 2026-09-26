"""
Project 1 - ANN from scratch (pure NumPy, no frameworks)

Goal: SEE the chain rule and backprop happen, instead of trusting a library.
Dataset: two interleaving moons (2D, binary classification) - small enough
to visualize the decision boundary the network learns.

Architecture: Input(2) -> Dense(8, activation) -> Dense(8, activation) -> Dense(1, sigmoid)

What this script demonstrates side by side:
  1. Forward pass (the math each layer computes)
  2. Backward pass (chain rule applied layer by layer, right to left)
  3. Three activation functions (sigmoid / tanh / relu) and how their
     derivatives change the gradient signal
  4. Plain SGD vs SGD with momentum - same architecture, same data,
     different update rule, different convergence speed

RUN WITH:
    python ann_from_scratch.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe: save PNGs without needing a GUI backend
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

np.random.seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# ACTIVATION FUNCTIONS + THEIR DERIVATIVES
# The derivative is what actually flows backward during backprop -
# this is the "local gradient" each layer contributes to the chain rule.
# ============================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(a):
    # a = sigmoid(z) already computed in forward pass, so reuse it
    return a * (1 - a)

def tanh(z):
    return np.tanh(z)

def tanh_deriv(a):
    return 1 - a ** 2

def relu(z):
    return np.maximum(0, z)

def relu_deriv(a):
    return (a > 0).astype(float)

ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_deriv),
    'tanh': (tanh, tanh_deriv),
    'relu': (relu, relu_deriv),
}


# ============================================================
# THE NETWORK
# 3 layers: Dense -> activation -> Dense -> activation -> Dense -> sigmoid
# Every weight matrix W_i and bias b_i is a parameter learned by gradient
# descent. Forward pass computes z_i = a_{i-1} @ W_i + b_i, a_i = f(z_i).
# ============================================================

class ANNFromScratch:
    def __init__(self, layer_sizes=(2, 8, 8, 1), hidden_activation='relu'):
        self.layer_sizes = layer_sizes
        self.act_fn, self.act_deriv = ACTIVATIONS[hidden_activation]

        # He-style initialization: keeps activations from exploding/vanishing
        # early on, especially important for relu.
        self.W = []
        self.b = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            self.W.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / fan_in))
            self.b.append(np.zeros((1, layer_sizes[i + 1])))

        # Momentum buffers (velocity), same shape as each parameter.
        # Zero when momentum is unused (plain SGD), so one code path serves both.
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(bb) for bb in self.b]

    def forward(self, X):
        """
        Returns all intermediate activations - we need them later for backprop
        because the chain rule needs each layer's local input/output.
        """
        activations = [X]
        zs = []
        a = X
        n_layers = len(self.W)
        for i in range(n_layers):
            z = a @ self.W[i] + self.b[i]
            zs.append(z)
            if i < n_layers - 1:
                a = self.act_fn(z)          # hidden layers: relu/tanh/sigmoid
            else:
                a = sigmoid(z)               # output layer: always sigmoid (binary classification)
            activations.append(a)
        return activations, zs

    def backward(self, activations, y):
        """
        THE CHAIN RULE, WRITTEN OUT EXPLICITLY.

        Loss: binary cross-entropy  L = -[y*log(yhat) + (1-y)*log(1-yhat)]
        For a sigmoid output, dL/dz_out simplifies beautifully to (yhat - y)
        - this is the starting point of the backward pass ("output error").

        For every earlier layer, the chain rule says:
            dL/dz_i = (dL/dz_{i+1} @ W_{i+1}^T) * f'(a_i)
        i.e. "how much this layer's output affected the next layer's input"
        (the W^T term) times "how much this layer's own activation
        function damps/passes that signal" (the f'(a_i) term).

        Once we have dL/dz_i for a layer, the parameter gradients fall out:
            dL/dW_i = a_{i-1}^T @ dL/dz_i
            dL/db_i = sum over batch of dL/dz_i
        """
        m = y.shape[0]
        n_layers = len(self.W)
        grads_W = [None] * n_layers
        grads_b = [None] * n_layers

        # --- Output layer: dL/dz = yhat - y (derived from BCE + sigmoid combo) ---
        yhat = activations[-1]
        dz = (yhat - y) / m

        for i in reversed(range(n_layers)):
            a_prev = activations[i]  # input to this layer
            grads_W[i] = a_prev.T @ dz
            grads_b[i] = np.sum(dz, axis=0, keepdims=True)

            if i > 0:
                # propagate error backward through W_i, then through this
                # layer's OWN activation derivative (chain rule, one more link)
                da_prev = dz @ self.W[i].T
                dz = da_prev * self.act_deriv(activations[i])

        return grads_W, grads_b

    def step(self, grads_W, grads_b, lr, momentum=0.0):
        """
        Plain SGD:            v = 0 always -> param -= lr * grad
        SGD with momentum:    v = momentum*v + grad -> param -= lr * v
        Momentum accumulates a moving average of past gradients, so it keeps
        moving in a consistent direction and dampens oscillation - that's why
        it usually converges faster on ravine-shaped loss surfaces.
        """
        for i in range(len(self.W)):
            self.vW[i] = momentum * self.vW[i] + grads_W[i]
            self.vb[i] = momentum * self.vb[i] + grads_b[i]
            self.W[i] -= lr * self.vW[i]
            self.b[i] -= lr * self.vb[i]

    def predict_proba(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

    def loss(self, X, y):
        yhat = self.predict_proba(X)
        eps = 1e-8
        return -np.mean(y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps))


def train(net, X, y, epochs, lr, momentum, label):
    losses = []
    for epoch in range(epochs):
        activations, _ = net.forward(X)
        grads_W, grads_b = net.backward(activations, y)
        net.step(grads_W, grads_b, lr, momentum)
        if epoch % 20 == 0 or epoch == epochs - 1:
            l = net.loss(X, y)
            losses.append((epoch, l))
            print(f"  [{label}] epoch {epoch:4d}  loss={l:.4f}")
    return losses


def plot_decision_boundary(ax, net, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = net.predict_proba(grid).reshape(xx.shape)

    ax.contourf(xx, yy, probs, levels=25, cmap='RdBu', alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdBu', edgecolors='k', s=20)
    ax.set_title(title)


def main():
    X, y = make_moons(n_samples=400, noise=0.2, random_state=42)
    y = y.reshape(-1, 1).astype(float)

    EPOCHS = 400
    LR = 0.5

    print("Training with PLAIN SGD (momentum=0.0)...")
    net_sgd = ANNFromScratch(hidden_activation='relu')
    losses_sgd = train(net_sgd, X, y, EPOCHS, LR, momentum=0.0, label="plain SGD")

    print("\nTraining with SGD + MOMENTUM (momentum=0.9)...")
    net_mom = ANNFromScratch(hidden_activation='relu')
    losses_mom = train(net_mom, X, y, EPOCHS, LR, momentum=0.9, label="SGD+momentum")

    # --- Plot 1: loss curves side by side ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(*zip(*losses_sgd), label='Plain SGD', linewidth=2)
    ax1.plot(*zip(*losses_mom), label='SGD + Momentum (0.9)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary Cross-Entropy Loss')
    ax1.set_title('Plain SGD vs. SGD with Momentum - same net, same data')
    ax1.legend()
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, 'loss_curves_sgd_vs_momentum.png'), dpi=150)
    print("\nSaved: loss_curves_sgd_vs_momentum.png")

    # --- Plot 2: decision boundaries learned by each ---
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_decision_boundary(axes[0], net_sgd, X, y, 'Decision boundary - Plain SGD')
    plot_decision_boundary(axes[1], net_mom, X, y, 'Decision boundary - SGD + Momentum')
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'decision_boundaries.png'), dpi=150)
    print("Saved: decision_boundaries.png")

    # --- Bonus: compare activation functions (relu vs tanh vs sigmoid) using plain SGD ---
    print("\nComparing hidden-layer activations (plain SGD, same LR)...")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for act_name in ['relu', 'tanh', 'sigmoid']:
        net = ANNFromScratch(hidden_activation=act_name)
        losses = train(net, X, y, EPOCHS, LR, momentum=0.0, label=act_name)
        ax3.plot(*zip(*losses), label=act_name, linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Binary Cross-Entropy Loss')
    ax3.set_title('Effect of hidden activation function on convergence (plain SGD)')
    ax3.legend()
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, 'activation_comparison.png'), dpi=150)
    print("Saved: activation_comparison.png")


if __name__ == "__main__":
    main()
