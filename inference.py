"""
STEP 4: Inference — Predict Age from a Single Skull Image
==========================================================
Given a new (potentially incomplete) skull image:
  1. Inpaint missing regions
  2. Preprocess (resize, normalise)
  3. Run CNN forward pass
  4. Scale output back to years
  5. Return age prediction + confidence range
"""

import os, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np

from inpaint import reconstruct_from_upload, generate_mask_from_dark_region, inpaint_skull

MODEL_DIR        = "models"
MODEL_PATH       = os.path.join(MODEL_DIR, "skull_age_cnn.h5")
LABEL_SCALE_PATH = os.path.join(MODEL_DIR, "label_scale.json")
IMG_SIZE         = 64


def load_inference_model():
    """
    Loads the saved Keras model and label scale parameters.
    Call once at app startup to avoid reloading on every prediction.
    """
    import tensorflow as tf

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Please run `python train.py` first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABEL_SCALE_PATH, "r") as f:
        scale = json.load(f)

    return model, scale["y_min"], scale["y_max"]


def preprocess_for_inference(image_bgr: np.ndarray) -> tuple:
    """
    Full preprocessing pipeline for a user-uploaded image.

    Steps:
      1. Convert to grayscale
      2. Resize to 64×64
      3. Auto-detect missing region
      4. Inpaint
      5. Normalise [0,1]
      6. Add batch + channel dims → (1, 64, 64, 1)

    Returns
    -------
    (model_input, reconstructed_bgr, mask)
      model_input      : np.ndarray shape (1,64,64,1) ready for model.predict()
      reconstructed_bgr: np.ndarray for display in Streamlit
      mask             : np.ndarray binary mask that was detected
    """
    # Step 1: grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Step 2: resize
    gray_resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # Step 3: detect missing region
    mask = generate_mask_from_dark_region(gray_resized)

    # Step 4: inpaint
    reconstructed_gray = inpaint_skull(gray_resized, mask, method="telea")

    # Step 5: normalise
    recon_norm = reconstructed_gray.astype(np.float32) / 255.0

    # Step 6: shape for model
    model_input = recon_norm[np.newaxis, ..., np.newaxis]   # (1,64,64,1)

    # For display: convert back to BGR
    reconstructed_bgr = cv2.cvtColor(reconstructed_gray, cv2.COLOR_GRAY2BGR)

    return model_input, reconstructed_bgr, mask


def predict_age(model, y_min: float, y_max: float, model_input: np.ndarray) -> dict:
    """
    Runs the model and converts normalised output → years.

    Returns a dict:
      {
        "predicted_age"  : float  — point estimate in years,
        "age_range_low"  : float  — optimistic lower bound (±5 yrs),
        "age_range_high" : float  — conservative upper bound (±5 yrs),
        "age_group"      : str    — e.g. "Young Adult (18–35)"
      }
    """
    raw_output = float(model.predict(model_input, verbose=0)[0][0])

    # Clip to valid [0,1] range, then scale back
    raw_clipped  = np.clip(raw_output, 0.0, 1.0)
    predicted_age = raw_clipped * (y_max - y_min) + y_min

    # ±5 year forensic estimation band (standard in forensic anthropology)
    low  = max(y_min, predicted_age - 5)
    high = min(y_max, predicted_age + 5)

    # Age group classification
    if predicted_age < 25:
        group = "Young Adult (18–25)"
    elif predicted_age < 40:
        group = "Early Adult (25–40)"
    elif predicted_age < 55:
        group = "Middle Age (40–55)"
    elif predicted_age < 70:
        group = "Mature Adult (55–70)"
    else:
        group = "Senior (70+)"

    return {
        "predicted_age"  : round(predicted_age, 1),
        "age_range_low"  : round(low, 1),
        "age_range_high" : round(high, 1),
        "age_group"      : group
    }


# ── Command-line quick test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_skull_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not load image: {img_path}")
        sys.exit(1)

    print("[INFO] Loading model...")
    model, y_min, y_max = load_inference_model()

    print("[INFO] Preprocessing & inpainting...")
    model_input, recon_bgr, mask = preprocess_for_inference(img)

    print("[INFO] Predicting age...")
    result = predict_age(model, y_min, y_max, model_input)

    print("\n" + "="*40)
    print(f"  Predicted Age : {result['predicted_age']} years")
    print(f"  Age Range     : {result['age_range_low']} – {result['age_range_high']} years")
    print(f"  Age Group     : {result['age_group']}")
    print("="*40)

    cv2.imwrite("reconstructed_output.png", recon_bgr)
    print("[✓] Reconstructed skull saved to 'reconstructed_output.png'")
