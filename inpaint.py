"""
STEP 1: Skull Completion via OpenCV Inpainting
===============================================
Inpainting = filling in missing/damaged parts of an image using surrounding pixels.

Two methods available:
  - Telea (cv2.INPAINT_TELEA): Fast Marching Method — usually cleaner results
  - Navier-Stokes (cv2.INPAINT_NS): Based on fluid dynamics — better for textures

We default to Telea for speed on low-power hardware.
"""

import cv2
import numpy as np


def inpaint_skull(
    incomplete_img: np.ndarray,
    mask: np.ndarray,
    method: str = "telea",
    inpaint_radius: int = 5
) -> np.ndarray:
    """
    Reconstructs missing skull regions using OpenCV inpainting.

    Parameters
    ----------
    incomplete_img : np.ndarray
        Grayscale or BGR image with missing (zeroed-out) regions.
    mask : np.ndarray
        Binary mask — 255 where pixels are missing, 0 elsewhere.
        Must be same H×W as incomplete_img.
    method : str
        'telea' (default) or 'ns' (Navier-Stokes).
    inpaint_radius : int
        How far (in pixels) the algorithm looks to fill each missing pixel.
        Larger = smoother but slower.

    Returns
    -------
    np.ndarray : Reconstructed image (same type/shape as incomplete_img).
    """
    # Choose algorithm flag
    flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS

    # OpenCV inpainting needs the mask to be uint8 with values 0 or 255
    mask_uint8 = mask.astype(np.uint8)

    # Run inpainting
    reconstructed = cv2.inpaint(incomplete_img, mask_uint8, inpaint_radius, flag)
    return reconstructed


def generate_mask_from_dark_region(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Auto-detects the missing region from an incomplete skull image
    by finding very dark (near-black) pixels inside the skull area.

    Useful when the user uploads their own image and we don't have
    the original mask file.

    Parameters
    ----------
    image     : Grayscale image (np.ndarray)
    threshold : Pixels darker than this value (0–255) are treated as missing.

    Returns
    -------
    np.ndarray : Binary mask (255 = missing, 0 = present)
    """
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Morphological closing to fill small holes and connect regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def reconstruct_from_upload(image_bgr: np.ndarray, method: str = "telea") -> tuple:
    """
    High-level function for the Streamlit app.

    1. Converts to grayscale
    2. Auto-detects the missing region
    3. Runs inpainting
    4. Returns (reconstructed_bgr, mask)

    Parameters
    ----------
    image_bgr : np.ndarray  — uploaded image in BGR format
    method    : str         — 'telea' or 'ns'

    Returns
    -------
    (reconstructed_bgr, mask)
    """
    gray          = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask          = generate_mask_from_dark_region(gray)
    reconstructed = inpaint_skull(gray, mask, method=method)

    # Convert grayscale back to BGR for display
    reconstructed_bgr = cv2.cvtColor(reconstructed, cv2.COLOR_GRAY2BGR)
    return reconstructed_bgr, mask


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocess import generate_skull_image, create_random_mask, apply_mask

    age      = 45
    full     = generate_skull_image(age)
    mask     = create_random_mask()
    broken   = apply_mask(full, mask)
    fixed    = inpaint_skull(broken, mask, method="telea")

    cv2.imwrite("test_full.png",  full)
    cv2.imwrite("test_broken.png", broken)
    cv2.imwrite("test_fixed.png",  fixed)
    print("[✓] Inpainting test saved: test_full.png, test_broken.png, test_fixed.png")
