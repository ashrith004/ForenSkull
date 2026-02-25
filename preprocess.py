"""
STEP 1 & 2: Preprocessing & Synthetic Dataset Generation
=========================================================
Since we don't have a real skull dataset, we:
1. Generate synthetic "skull-like" grayscale images programmatically
2. Simulate aging features (texture, density, shape)
3. Apply random masks to simulate incomplete skulls
4. Save images + age labels for training

For a REAL project: replace generate_skull_image() with your actual skull X-ray images.
"""

import cv2
import numpy as np
import os
import json
import random

# ── Configuration ──────────────────────────────────────────────────────────────
IMG_SIZE    = 64          # Resize all images to 64x64 (keeps model tiny & fast)
NUM_SAMPLES = 1000         # Generate 1000 synthetic samples (enough for demo)
DATA_DIR    = "data"
RAW_DIR     = os.path.join(DATA_DIR, "raw")        # Full (complete) skull images
MASKED_DIR  = os.path.join(DATA_DIR, "masked")     # Incomplete skull images
MASK_DIR    = os.path.join(DATA_DIR, "masks")      # Binary masks of missing region
LABELS_FILE = os.path.join(DATA_DIR, "labels.json") # {filename: age} mapping

os.makedirs(RAW_DIR,    exist_ok=True)
os.makedirs(MASKED_DIR, exist_ok=True)
os.makedirs(MASK_DIR,   exist_ok=True)


def generate_skull_image(age: int, size: int = IMG_SIZE) -> np.ndarray:
    """
    Creates a synthetic grayscale skull-like image.

    How aging is simulated:
    - Younger skulls (< 25): smoother texture, brighter bone density
    - Middle age (25–55):    moderate texture + noise
    - Older skulls (> 55):   rougher surface, lower bone density, more pitting

    Returns: uint8 numpy array, shape (size, size)
    """
    img = np.zeros((size, size), dtype=np.float32)

    # ── Base oval skull shape ──────────────────────────────────────────────────
    cx, cy = size // 2, size // 2
    rx, ry = int(size * 0.38), int(size * 0.45)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)

    # ── Bone density: young = bright, old = dimmer ─────────────────────────────
    age_norm  = np.clip(age / 80.0, 0.1, 1.0)
    base_val  = 0.95 - 0.25 * age_norm          # range ~0.70 – 0.95
    img      *= base_val

    # ── Texture / noise: increases with age ────────────────────────────────────
    noise_std = 0.05 + 0.15 * age_norm           # more noise = older
    noise     = np.random.normal(0, noise_std, (size, size)).astype(np.float32)
    img       = np.clip(img + noise * (img > 0), 0, 1)

    # ── Eye socket (dark oval near upper region) ───────────────────────────────
    eye_cx = int(cx - size * 0.12)
    eye_cy = int(cy - size * 0.08)
    cv2.ellipse(img, (eye_cx, eye_cy), (int(size*0.09), int(size*0.07)), 0, 0, 360, 0.0, -1)

    # ── Nasal opening (inverted triangle) ──────────────────────────────────────
    pts = np.array([
        [cx,           cy + int(size*0.05)],
        [cx - int(size*0.05), cy + int(size*0.18)],
        [cx + int(size*0.05), cy + int(size*0.18)],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], 0.0)

    # ── Age-related pitting: random dark spots for older skulls ────────────────
    if age > 40:
        num_pits = int((age - 40) / 5)
        for _ in range(num_pits):
            px = random.randint(cx - rx + 5, cx + rx - 5)
            py = random.randint(cy - ry + 5, cy + ry - 5)
            pit_r = random.randint(1, 3)
            cv2.circle(img, (px, py), pit_r, 0.0, -1)

    # Convert to uint8 [0, 255]
    img_uint8 = (img * 255).astype(np.uint8)
    return img_uint8


def create_random_mask(size: int = IMG_SIZE) -> np.ndarray:
    """
    Creates a binary mask (0=keep, 255=missing) that simulates
    a broken or missing skull region.

    Returns: uint8 mask array, same size as image
    """
    mask = np.zeros((size, size), dtype=np.uint8)

    # Random rectangle (simulates a fragment missing from skull)
    x1 = random.randint(10, size // 2)
    y1 = random.randint(10, size // 2)
    x2 = random.randint(x1 + 8, size - 5)
    y2 = random.randint(y1 + 8, size - 5)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # Sometimes add a second smaller gap
    if random.random() > 0.5:
        cx2 = random.randint(5, size - 10)
        cy2 = random.randint(5, size - 10)
        cv2.circle(mask, (cx2, cy2), random.randint(4, 8), 255, -1)

    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Blacks-out the masked region on the skull image.
    This simulates a fragment of skull that is missing/broken.
    """
    masked = image.copy()
    masked[mask == 255] = 0
    return masked


def generate_dataset():
    """
    Main function: generates NUM_SAMPLES synthetic skull images,
    applies random masks, and saves everything to disk.
    """
    labels = {}
    print(f"[INFO] Generating {NUM_SAMPLES} synthetic skull samples...")

    for i in range(NUM_SAMPLES):
        # Random age between 18 and 80
        age = random.randint(18, 80)

        # Generate full skull
        full_img = generate_skull_image(age, IMG_SIZE)

        # Create a mask for the missing region
        mask = create_random_mask(IMG_SIZE)

        # Apply mask → incomplete skull
        incomplete_img = apply_mask(full_img, mask)

        # Save files
        fname = f"skull_{i:04d}.png"
        cv2.imwrite(os.path.join(RAW_DIR,    fname), full_img)
        cv2.imwrite(os.path.join(MASKED_DIR, fname), incomplete_img)
        cv2.imwrite(os.path.join(MASK_DIR,   fname), mask)
        labels[fname] = age

    # Save labels
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"[✓] Dataset saved to '{DATA_DIR}/' with {NUM_SAMPLES} samples.")
    print(f"[✓] Labels saved to '{LABELS_FILE}'")


if __name__ == "__main__":
    generate_dataset()
