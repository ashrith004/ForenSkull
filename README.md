# 🦴 ForenSkull — Forensic Skull Reconstruction & Age Estimation

A lightweight end-to-end demo system for forensic skull analysis.

---

## 📁 Project Structure

```
skull_forensics/
├── preprocess.py     ← Dataset generation & masking
├── inpaint.py        ← OpenCV skull reconstruction
├── model.py          ← Lightweight CNN architecture
├── train.py          ← Full training pipeline
├── inference.py      ← Single-image prediction
├── app.py            ← Streamlit web interface
├── requirements.txt  ← Python dependencies
└── README.md
```

---

## 🚀 Quick Start (5 Steps)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic dataset (takes ~30 seconds)
```bash
python preprocess.py
```

### 3. Train the CNN model (takes ~3–5 min on Intel i3)
```bash
python train.py
```

### 4. Test on a single image (optional)
```bash
python inference.py data/masked/skull_0000.png
```

### 5. Launch the Streamlit app
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

## 🧠 How It Works

### Step 1 — Preprocessing (`preprocess.py`)
- Generates 200 synthetic skull images with aging features
- Each image = 64×64 grayscale
- Applies a random rectangular/circular mask to simulate missing skull fragments
- Saves: full skull, masked skull, binary mask, age labels

### Step 2 — Inpainting (`inpaint.py`)
- Uses **OpenCV `cv2.inpaint()`** to fill in missing regions
- **Telea method**: Fast Marching — sharp, clean edges (recommended)
- **Navier-Stokes**: Fluid simulation — smoother fills
- Auto-detects dark (missing) regions from uploaded images

### Step 3 — CNN Model (`model.py`)
```
Input: 64×64×1 grayscale
  → Conv2D(16) + MaxPool
  → Conv2D(32) + MaxPool
  → Conv2D(64) + MaxPool
  → Flatten → Dense(64) → Dropout(0.3)
  → Dense(1) [predicted age, normalised]
```
- **Loss**: Mean Absolute Error (MAE in years)
- **Total params**: ~260K (tiny, fast on CPU)

### Step 4 — Training (`train.py`)
- Inpaints all training images first
- Normalises ages to [0,1] range
- 80/20 train-validation split
- 5 epochs, batch size 16
- Saves model + label scale to `models/`

### Step 5 — Streamlit App (`app.py`)
- Upload incomplete skull image
- Choose inpainting method in sidebar
- See: original → mask → reconstructed
- Displays: predicted age, ±5 year range, age group

---

## 🔧 Customisation

### Use real skull X-ray images
Replace `generate_skull_image()` in `preprocess.py` with code to load your own images:
```python
# Instead of generating:
full_img = cv2.imread("my_skull_xray.jpg", cv2.IMREAD_GRAYSCALE)
full_img = cv2.resize(full_img, (64, 64))
```

### Increase model capacity
In `model.py`, add more filters or a 4th conv block:
```python
x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
x = layers.MaxPooling2D((2,2))(x)
```

### Train for more epochs
In `train.py`, change:
```python
EPOCHS = 10  # or 20 for better accuracy
```

---

## ⚠️ Disclaimer
This is a **demonstration project** built on **synthetic data**.
Predictions are NOT suitable for real forensic investigations.
For production use, replace synthetic data with validated skull X-ray datasets.

---

## 📚 References
- Telea Inpainting: "An Image Inpainting Technique Based on the Fast Marching Method" (2004)
- Navier-Stokes: "Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting" (Bertalmio et al., 2001)
- Forensic Age Estimation: Uses bone density, suture fusion, and morphological features
