"""
Deep Learning Breast Cancer Detection - Web App

Research prototype only - not a diagnostic tool. Do not use for real clinical decisions.
"""
# Imports
import os
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import tensorflow as tf

# Trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.keras")
IMG_SIZE = 224
# Browser tab title and icon
st.set_page_config(page_title="Breast Cancer Detection", page_icon="🎗️", layout="centered")

# Loads the trained model from disk once and caches it across sessions.
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess(image_rgb):
    """Mirrors load_and_preprocess() from the training."""
# Training images loaded with cv2.IMREAD_GRAYSCALE
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
# CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
# Resize the model's input size which is 224
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)  # [0, 255]
    return rgb

# FrontPage Title
st.title("Deep Learning Breast Cancer Detection - Mammogram Classifier")
# Instructions, explaining what the app does, what input it expects, and which model the classification is based on.
st.markdown(
    "Upload a mammogram mass image (JPEG/PNG) to classify it as Benign or "
    "Malignant, using a fine-tuned EfficientNetB0 model trained on the "
    "CBIS-DDSM dataset."
)
# Disclaimer
st.warning(
    "**Research prototype only - not a diagnostic tool.** Sensitivity on "
    "held-out test data is well below clinical benchmarks; do not use this "
    "for real clinical decisions."
)

# Loads the model
model = load_model()

# File upload widget to only use jpg, jpeg and png
uploaded_file = st.file_uploader("Choose a mammogram image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)

# Run the image through the same preprocessing pipeline used at training and Display the result
    processed = preprocess(image_rgb)
    st.image(processed.astype(np.uint8), caption="Preprocessed image", width=300)

# Add a batch dimension (the model expects shape (batch, H, W, 3)) and
# run inference, extracting the single sigmoid output as a plain float.
    batch = processed[np.newaxis, ...]
    prob = float(model.predict(batch, verbose=0)[0, 0])
# Apply the standard 0.5 decision threshold to convert the probability into a class label, and compute confidence relative to that label
    label = "Malignant" if prob >= 0.5 else "Benign"
    confidence = prob if label == "Malignant" else 1 - prob

# red/error for a malignant result
# green/success for benign result
    if label == "Malignant":
        st.error(f"Prediction: **{label}**  ({confidence * 100:.1f}% confidence)")
    else:
        st.success(f"Prediction: **{label}**  ({confidence * 100:.1f}% confidence)")
# Show the probability
    st.caption(f"P(malignant) = {prob:.4f}")