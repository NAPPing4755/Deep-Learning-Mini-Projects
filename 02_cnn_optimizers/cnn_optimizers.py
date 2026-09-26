"""
Project 2 — CNN architecture + Optimizer comparison

Goal: understand a CNN's building blocks (Conv -> Activation -> Pool,
stacked, then Flatten -> Dense -> Softmax) and see how the CHOICE OF
OPTIMIZER changes training dynamics on the exact same architecture.

Dataset: Fashion-MNIST (28x28 grayscale, 10 classes) — small enough to
train several optimizer variants in one sitting on CPU.

Optimizers compared (same CNN, same init, same epochs):
  - SGD (plain)
  - SGD + momentum
  - RMSprop
  - Adam

RUN WITH:
    python cnn_optimizers.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe: save PNGs without needing a GUI backend
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, datasets

tf.random.set_seed(42)
np.random.seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 8
BATCH_SIZE = 128


def load_data():
    (X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()
    # Normalize to [0,1] and add the channel dimension CNNs expect (H, W, C)
    X_train = X_train.astype('float32')[..., None] / 255.0
    X_test = X_test.astype('float32')[..., None] / 255.0
    return X_train, y_train, X_test, y_test


def build_cnn():
    """
    Architecture, read top to bottom:
      Conv(32, 3x3) -> ReLU -> MaxPool(2x2)   # learns edges/simple textures
      Conv(64, 3x3) -> ReLU -> MaxPool(2x2)   # combines edges into shapes
      Flatten -> Dense(128) -> ReLU -> Dropout
      Dense(10) -> Softmax                    # class probabilities

    Each Conv layer slides small filters across the image (weight sharing —
    the same filter is reused at every spatial position, which is what makes
    CNNs far more parameter-efficient than a fully-connected net on images).
    MaxPool shrinks the spatial size while keeping the strongest activations,
    giving some translation invariance and reducing compute for later layers.
    """
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax'),
    ])
    return model


def get_fresh_model_with_fixed_init(reference_weights):
    """Build a new model and load IDENTICAL starting weights, so every
    optimizer starts from the exact same point — a fair comparison."""
    model = build_cnn()
    model.set_weights(reference_weights)
    return model


OPTIMIZER_FACTORY = {
    'SGD (plain)': lambda: optimizers.SGD(learning_rate=0.01, momentum=0.0),
    'SGD + momentum': lambda: optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'RMSprop': lambda: optimizers.RMSprop(learning_rate=0.001),
    'Adam': lambda: optimizers.Adam(learning_rate=0.001),
}


def main():
    print("Loading Fashion-MNIST...")
    X_train, y_train, X_test, y_test = load_data()

    # Use a subset for speed — this is a learning exercise, not a leaderboard run.
    X_train, y_train = X_train[:12000], y_train[:12000]
    X_test, y_test = X_test[:2000], y_test[:2000]

    # Freeze one reference set of initial weights so every optimizer starts identically.
    reference_model = build_cnn()
    reference_weights = reference_model.get_weights()

    histories = {}
    for name, opt_factory in OPTIMIZER_FACTORY.items():
        print(f"\n{'='*60}\nTraining with optimizer: {name}\n{'='*60}")
        model = get_fresh_model_with_fixed_init(reference_weights)
        model.compile(optimizer=opt_factory(),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
        )
        histories[name] = history.history

    # --- Plot: validation accuracy per optimizer ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, hist in histories.items():
        axes[0].plot(hist['val_accuracy'], label=name, linewidth=2)
        axes[1].plot(hist['val_loss'], label=name, linewidth=2)

    axes[0].set_title('Validation Accuracy by Optimizer')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title('Validation Loss by Optimizer')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'optimizer_comparison.png'), dpi=150)
    print("\nSaved: optimizer_comparison.png")

    print("\nFinal validation accuracy per optimizer:")
    for name, hist in histories.items():
        print(f"  {name:<18} {hist['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()
