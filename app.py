"""
STEP 5: Streamlit Web Interface
================================
Run with:  streamlit run app.py

Features:
  • Upload an incomplete skull image
  • Choose inpainting method (Telea / Navier-Stokes)
  • See side-by-side: original | mask | reconstructed
  • View predicted biological age + age group classification
  • First-run button to train the model if not yet trained
"""

import os, json, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ForenSkull · Age Estimator",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS: dark forensic theme ──────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background: #0d0f14;
    color: #c8d8e8;
  }
  .stApp { background: #0d0f14; }

  /* Title block */
  .title-block {
    background: linear-gradient(135deg, #0f1922 0%, #12232e 100%);
    border: 1px solid #1e4060;
    border-radius: 8px;
    padding: 24px 32px;
    margin-bottom: 24px;
    font-family: 'Share Tech Mono', monospace;
  }
  .title-block h1 { color: #4dd9f0; font-size: 2.4rem; margin: 0; letter-spacing: 3px; }
  .title-block p  { color: #6a8fa8; margin: 4px 0 0; font-size: 1rem; }

  /* Metric cards */
  .metric-card {
    background: #12232e;
    border: 1px solid #1e4060;
    border-radius: 8px;
    padding: 20px 24px;
    text-align: center;
  }
  .metric-card .label { color: #4a7090; font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase; }
  .metric-card .value { color: #4dd9f0; font-family: 'Share Tech Mono', monospace; font-size: 2.4rem; }
  .metric-card .sub   { color: #6a8fa8; font-size: 0.85rem; }

  /* Section headers */
  .section-header {
    color: #4dd9f0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid #1e4060;
    padding-bottom: 6px;
    margin: 20px 0 12px;
  }

  /* Result badge */
  .age-group-badge {
    display: inline-block;
    background: #0a2535;
    border: 1px solid #4dd9f0;
    color: #4dd9f0;
    border-radius: 4px;
    padding: 4px 12px;
    font-size: 0.9rem;
    font-family: 'Share Tech Mono', monospace;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: #0a1520 !important; }
  section[data-testid="stSidebar"] * { color: #c8d8e8 !important; }

  /* Buttons */
  .stButton>button {
    background: #0a2535;
    border: 1px solid #4dd9f0;
    color: #4dd9f0;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 1px;
    border-radius: 4px;
    transition: all 0.2s;
  }
  .stButton>button:hover {
    background: #4dd9f0;
    color: #0d0f14;
  }

  /* Warning / info boxes */
  .stAlert { border-radius: 6px !important; }

  /* Image captions */
  .img-caption {
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    color: #4a7090;
    text-transform: uppercase;
    margin-top: 6px;
  }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def get_model():
    """Loads model once and caches it for the session."""
    from inference import load_inference_model
    return load_inference_model()


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image → OpenCV BGR array."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR → PIL Image."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def mask_to_pil(mask: np.ndarray) -> Image.Image:
    """Convert binary mask → heatmap-style PIL image for display."""
    # Apply a cyan-ish colour to missing regions
    h, w = mask.shape[:2]
    vis  = np.zeros((h, w, 3), dtype=np.uint8)
    vis[mask == 0]   = [20, 35, 50]        # dark = present
    vis[mask == 255] = [77, 217, 240]      # cyan = missing/masked
    return Image.fromarray(vis)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    method = st.radio(
        "Inpainting Method",
        ["Telea (Fast Marching)", "Navier-Stokes (Fluid)"],
        help=(
            "Telea: faster, sharper edges — recommended for bone imagery.\n\n"
            "Navier-Stokes: smoother fill — better for curved surfaces."
        )
    )
    inpaint_method = "telea" if "Telea" in method else "ns"

    inpaint_radius = st.slider(
        "Inpaint Radius (px)", min_value=2, max_value=15, value=5,
        help="How far (pixels) the algorithm looks to fill each missing pixel."
    )

    st.markdown("---")
    st.markdown("### 🏋️ Train Model")
    st.caption("Only needed once. Takes ~3–5 mins on Intel i3.")

    if st.button("▶ Train / Re-Train Model"):
        with st.spinner("Training CNN… please wait"):
            t0 = time.time()
            from train import train
            train()
            elapsed = time.time() - t0
        st.success(f"Training complete in {elapsed/60:.1f} min ✓")
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "Demo forensic system for skull age estimation.\n\n"
        "Pipeline: Inpainting (OpenCV) → CNN Regression (TensorFlow)\n\n"
        "⚠️ For demonstration only — not for clinical use."
    )


# ── Main UI ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="title-block">
  <h1>🦴 FORENSKULL</h1>
  <p>Forensic Skull Reconstruction & Biological Age Estimation System</p>
</div>
""", unsafe_allow_html=True)

# ── Check model exists ────────────────────────────────────────────────────────
model_exists = os.path.exists(os.path.join("models", "skull_age_cnn.h5"))

if not model_exists:
    st.warning(
        "⚠️ **No trained model found.**  "
        "Click **▶ Train / Re-Train Model** in the sidebar to train first."
    )
    st.stop()

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📂 Upload Skull Image</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload an incomplete skull X-ray or image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    help="Regions with missing bone should appear as dark/black areas."
)

if uploaded is None:
    # Show a demo sample from the dataset
    sample_path = os.path.join("data", "masked", "skull_0000.png")
    if os.path.exists(sample_path):
        st.info("💡 No image uploaded. Showing a sample from the demo dataset.")
        sample_pil = Image.open(sample_path).convert("RGB")
        # Scale up for visibility
        sample_pil = sample_pil.resize((256, 256), Image.NEAREST)
        col_demo, _ = st.columns([1, 3])
        with col_demo:
            st.image(sample_pil, caption="Sample masked skull", use_container_width=True)
    else:
        st.info("Upload a skull image to begin, or train the model first to generate samples.")
    st.stop()


# ── Process uploaded image ────────────────────────────────────────────────────
pil_img = Image.open(uploaded).convert("RGB")
bgr_img = pil_to_bgr(pil_img)

# Scale up for display (original is 64×64, hard to see)
DISPLAY_SIZE = 256

st.markdown('<div class="section-header">🔬 Processing Pipeline</div>', unsafe_allow_html=True)

with st.spinner("Reconstructing skull and predicting age…"):
    from inference import preprocess_for_inference, predict_age

    # Override default inpaint params using sidebar choices
    from inpaint import generate_mask_from_dark_region, inpaint_skull

    gray        = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gray_small  = cv2.resize(gray, (64, 64))
    mask        = generate_mask_from_dark_region(gray_small)
    recon_gray  = inpaint_skull(gray_small, mask, method=inpaint_method, inpaint_radius=inpaint_radius)
    recon_bgr   = cv2.cvtColor(recon_gray, cv2.COLOR_GRAY2BGR)

    # Model input
    recon_norm  = recon_gray.astype(np.float32) / 255.0
    model_input = recon_norm[np.newaxis, ..., np.newaxis]

    model, y_min, y_max = get_model()
    result = predict_age(model, y_min, y_max, model_input)


# ── Display images ────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

original_display = cv2.resize(
    cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB),
    (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST
)
mask_display = cv2.resize(
    np.array(mask_to_pil(mask)),
    (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST
)
recon_display = cv2.resize(
    cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2RGB),
    (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST
)

with col1:
    st.image(original_display, caption="", use_container_width=True)
    st.markdown('<div class="img-caption">① Uploaded Image</div>', unsafe_allow_html=True)

with col2:
    st.image(mask_display, caption="", use_container_width=True)
    st.markdown('<div class="img-caption">② Detected Missing Region (cyan)</div>', unsafe_allow_html=True)

with col3:
    st.image(recon_display, caption="", use_container_width=True)
    st.markdown('<div class="img-caption">③ Reconstructed Skull</div>', unsafe_allow_html=True)


# ── Age prediction results ────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Age Estimation Results</div>', unsafe_allow_html=True)

r1, r2, r3 = st.columns(3)

with r1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">Estimated Age</div>
      <div class="value">{result['predicted_age']}</div>
      <div class="sub">years</div>
    </div>""", unsafe_allow_html=True)

with r2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">Probable Range</div>
      <div class="value">{result['age_range_low']}–{result['age_range_high']}</div>
      <div class="sub">years (±5 yr forensic band)</div>
    </div>""", unsafe_allow_html=True)

with r3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">Age Group</div>
      <div class="value" style="font-size:1.3rem; margin-top:8px;">{result['age_group']}</div>
      <div class="sub">classification</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "⚠️ **Disclaimer:** This is a demonstration system trained on synthetic data. "
    "Predictions should NOT be used for real forensic analysis. "
    "Replace synthetic data with real skull X-ray datasets for meaningful results."
)
