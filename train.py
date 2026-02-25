"""
STEP 3: Training the CNN Age Predictor
=======================================
Pipeline:
  1. Load reconstructed skull images from data/masked/ (after inpainting)
     OR use the raw full images if you prefer (cleaner signal)
  2. Normalise pixel values to [0, 1]
  3. Load age labels from data/labels.json
  4. Split 80 % train / 20 % validation
  5. Train for 5 epochs (fast on CPU)
  6. Save model to models/skull_age_cnn.h5

Estimated training time on Intel i3: ~2–4 minutes for 200 samples × 5 epochs.
"""

import os, json, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from preprocess import (
    generate_dataset,
    RAW_DIR, MASKED_DIR, MASK_DIR, LABELS_FILE, IMG_SIZE
)
from inpaint import inpaint_skull
from model import build_age_cnn

MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "skull_age_cnn.h5")
EPOCHS     = 50
BATCH_SIZE = 16


# ─────────────────────────────────────────────────────────────────────────────
def load_and_reconstruct_dataset():
    """
    For each sample in MASKED_DIR:
      1. Load the masked (incomplete) skull image
      2. Load its corresponding binary mask
      3. Run Telea inpainting → reconstructed image
      4. Resize to IMG_SIZE × IMG_SIZE and normalise to [0,1]

    Returns
    -------
    X : np.ndarray, shape (N, IMG_SIZE, IMG_SIZE, 1)  — float32 [0,1]
    y : np.ndarray, shape (N,)                        — float32 ages
    """
    with open(LABELS_FILE, "r") as f:
        labels = json.load(f)

    X, y = [], []
    fnames = sorted(os.listdir(MASKED_DIR))

    print(f"[INFO] Loading and inpainting {len(fnames)} images...")
    t0 = time.time()

    for fname in fnames:
        if not fname.endswith(".png"):
            continue

        # Load masked image (grayscale)
        masked_path = os.path.join(MASKED_DIR, fname)
        mask_path   = os.path.join(MASK_DIR,   fname)

        masked_img  = cv2.imread(masked_path, cv2.IMREAD_GRAYSCALE)
        mask_img    = cv2.imread(mask_path,   cv2.IMREAD_GRAYSCALE)

        if masked_img is None or mask_img is None:
            continue

        # Inpaint → reconstruct
        recon = inpaint_skull(masked_img, mask_img, method="telea")

        # Resize (already 64×64, but ensures consistency)
        recon = cv2.resize(recon, (IMG_SIZE, IMG_SIZE))

        # Normalise [0,255] → [0,1]
        recon_norm = recon.astype(np.float32) / 255.0

        # Add channel dimension: (64,64) → (64,64,1)
        X.append(recon_norm[..., np.newaxis])
        y.append(float(labels[fname]))

    elapsed = time.time() - t0
    print(f"[✓] Loaded {len(X)} samples in {elapsed:.1f}s")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
def normalise_labels(y: np.ndarray):
    """
    Scales ages to [0,1] range for easier optimisation.
    We use min-max normalisation: y_norm = (y - 18) / (80 - 18)

    The model outputs a value in [0,1] which we scale back to years.
    """
    y_min, y_max = 18.0, 80.0
    return (y - y_min) / (y_max - y_min), y_min, y_max


# ─────────────────────────────────────────────────────────────────────────────
def train():
    # ── 0. Generate dataset if not already present ─────────────────────────
    if not os.path.exists(LABELS_FILE):
        print("[INFO] Dataset not found. Generating now...")
        generate_dataset()

    # ── 1. Load data ────────────────────────────────────────────────────────
    X, y = load_and_reconstruct_dataset()
    print(f"[INFO] X shape: {X.shape} | y range: {y.min():.0f} – {y.max():.0f} years")

    # ── 2. Normalise labels ─────────────────────────────────────────────────
    y_norm, y_min, y_max = normalise_labels(y)

    # ── 3. Train / validation split ─────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_norm, test_size=0.2, random_state=42
    )
    print(f"[INFO] Train: {len(X_train)} | Val: {len(X_val)}")

    # ── 4. Build model ──────────────────────────────────────────────────────
    model = build_age_cnn(input_size=IMG_SIZE)
    model.summary()

    # ── 5. Train ────────────────────────────────────────────────────────────
    print(f"\n[INFO] Training for {EPOCHS} epochs on CPU...")
    t0 = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    elapsed = time.time() - t0
    print(f"\n[✓] Training complete in {elapsed/60:.1f} minutes")

    # ── 6. Report final MAE in years ────────────────────────────────────────
    final_val_mae_norm = history.history["val_mae"][-1]
    final_val_mae_yrs  = final_val_mae_norm * (y_max - y_min)
    print(f"[RESULT] Validation MAE ≈ {final_val_mae_yrs:.1f} years")
    print("  (Demo model — accuracy improves with real skull X-ray data)")

    # ── 7. Save model + label scale info ────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)

    scale_info = {"y_min": y_min, "y_max": y_max}
    with open(os.path.join(MODEL_DIR, "label_scale.json"), "w") as f:
        json.dump(scale_info, f)

    print(f"[✓] Model saved to '{MODEL_PATH}'")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
