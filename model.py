"""
STEP 2: Lightweight CNN Age Prediction Model
=============================================
Architecture overview (designed for 64×64 input on CPU):

  Input (64×64×1 grayscale)
      │
  Conv2D(16 filters, 3×3) → ReLU → MaxPool(2×2)   [output: 32×32×16]
      │
  Conv2D(32 filters, 3×3) → ReLU → MaxPool(2×2)   [output: 16×16×32]
      │
  Conv2D(64 filters, 3×3) → ReLU → MaxPool(2×2)   [output:  8×8×64]
      │
  Flatten                                           [output: 4096]
      │
  Dense(64) → ReLU → Dropout(0.3)
      │
  Dense(1)  → Linear (regression output = predicted age)

Why so small?
  • Only 3 conv layers → trains in seconds on CPU
  • 64×64 input → tiny feature maps, low memory
  • Single Dense head → regression (age is a continuous number)

Loss: Mean Absolute Error (MAE)
  → Penalises actual year differences, easy to interpret
  → "MAE=6" means the model is off by ~6 years on average
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TensorFlow warnings

import tensorflow as tf
from tensorflow.keras import layers, models, Input


def build_age_cnn(input_size: int = 64) -> tf.keras.Model:
    """
    Builds and returns the compiled lightweight CNN.

    Parameters
    ----------
    input_size : int  — image height = width (default 64)

    Returns
    -------
    tf.keras.Model — compiled, ready to train
    """
    inp = Input(shape=(input_size, input_size, 1), name="skull_input")

    # ── Block 1 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="conv1")(inp)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # ── Block 2 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # ── Block 3 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # ── Fully Connected Head ─────────────────────────────────────────────────
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(64, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)           # reduce overfitting

    # ── Output: single neuron, no activation (regression) ────────────────────
    out = layers.Dense(1, activation="linear", name="age_output")(x)

    model = models.Model(inputs=inp, outputs=out, name="SkullAgeCNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mean_absolute_error",   # MAE: intuitive "years off" metric
        metrics=["mae"]
    )

    return model


def load_model(path: str) -> tf.keras.Model:
    """Loads a saved Keras model from disk."""
    return tf.keras.models.load_model(path)


# ── Quick sanity check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_age_cnn()
    model.summary()

    total_params = model.count_params()
    print(f"\n[INFO] Total parameters: {total_params:,}")
    print("[INFO] Model is lightweight and ready for CPU training.")
