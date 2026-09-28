"""
Deep Learning Breast Cancer Detection - Web App

Research prototype only - not a diagnostic tool. Do not use for real clinical decisions.
"""
# Imports
import os
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf

# Paths and settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.keras")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
IMG_SIZE = 224
# Decision threshold
THRESHOLD = 0.591

# Browser tab title and icon
st.set_page_config(page_title="Breast Cancer Detection", page_icon="🎗️", layout="centered")


# Loads the trained model from disk once and caches it across sessions.
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(file_bytes):
    buf = np.frombuffer(file_bytes, dtype=np.uint8)
    gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Could not decode image - please upload a JPEG or PNG.")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)  # [0, 255]
    return rgb


def list_samples():
    """Return the sample image filenames (sorted, so benign_ files come before malignant_)."""
    if not os.path.isdir(SAMPLES_DIR):
        return []
    return sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))


# FrontPage Title
st.title("Deep Learning Breast Cancer Detection - Mammogram Classifier")
# Instructions, explaining what the app does, what input it expects, and which model the classification is based on.
st.markdown(
    "Upload a mammogram mass image (JPEG/PNG), or choose one of the sample images, "
    "to classify it as Benign or Malignant using a fine-tuned EfficientNetB0 model "
    "trained on the CBIS-DDSM dataset."
)
# Disclaimer
st.warning(
    "**Research prototype only - not a diagnostic tool.** Sensitivity on "
    "held-out test data is well below clinical benchmarks; do not use this "
    "for real clinical decisions."
)

# Loads the model
model = load_model()

# Image source: sample picker or own upload
source = st.radio("Image source", ["Use a sample image", "Upload my own"], horizontal=True, index=None)

file_bytes, name, true_label = None, None, None

if source == "Use a sample image":
    samples = list_samples()
    if samples:
        name = st.selectbox("Sample mammogram (CBIS-DDSM test set)", samples, index=None, placeholder="Choose a sample image...")
        if name is not None:
            with open(os.path.join(SAMPLES_DIR, name), "rb") as f:
                file_bytes = f.read()
            # The true label is encoded in the filename (benign_... / malignant_...)
            true_label = "Malignant" if name.lower().startswith("malignant") else "Benign"
        st.caption("Sample images: CBIS-DDSM (Lee et al., 2017), CC BY-SA 3.0.")
    else:
        st.info("No sample images found in the samples/ folder.")
else:
    # File upload widget to only use jpg, jpeg and png
    uploaded_file = st.file_uploader("Choose a mammogram image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        name = uploaded_file.name

if file_bytes is not None:
    # Run the image through the same preprocessing pipeline used at training
    try:
        processed = preprocess(file_bytes)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Show the original and the preprocessed image side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(file_bytes, caption=f"Original: {name}", width=300)
    with col2:
        st.image(processed.astype(np.uint8), caption="Preprocessed (CLAHE, 224×224)", width=300)

    # Add a batch dimension and run inference
    batch = processed[np.newaxis, ...]
    prob = float(model.predict(batch, verbose=0)[0, 0])

    # Apply the validation-selected threshold to convert the probability into a class label
    label = "Malignant" if prob >= THRESHOLD else "Benign"

    # red/error for a malignant result, green/success for a benign result
    if label == "Malignant":
        st.error(f"Prediction: **{label}**")
    else:
        st.success(f"Prediction: **{label}**")
    # Show the probability and the threshold used
    st.caption(f"P(malignant) = {prob:.4f}  (decision threshold = {THRESHOLD})")

    # For sample images, compare the prediction with the known label
    if true_label is not None:
        if label == true_label:
            st.write(f"True label: **{true_label}** ✅ the model's prediction is correct.")
        else:
            st.write(f"True label: **{true_label}** ❌ the model's prediction is incorrect.")