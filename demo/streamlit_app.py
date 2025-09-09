import streamlit as st
import numpy as np
import yaml
import cv2
import tensorflow as tf
from PIL import Image


# Page configuration
st.set_page_config(page_title="ASL Demo", layout="centered")
st.title("ASL Letter Recognition Demo")

# Load configuration
cfg = yaml.safe_load(open("configs/config.yaml"))
classes = cfg["training"]["classes"]
img_size = tuple(cfg["training"]["img_size"])

# Inputs
weights_path = st.text_input(
    "Model weights (.h5)", value="outputs/baseline_cnn_best.h5"
)
uploaded = st.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False
)


def preprocess(pil_img):
    """Convert PIL image to normalized array for model input."""
    arr = np.array(pil_img.convert("L"))
    arr = cv2.resize(arr, img_size).astype("float32") / 255.0
    return arr[..., None][None, ...]


if uploaded and weights_path:
    # Display the uploaded image
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input", use_container_width=True)
    try:
        # Perform prediction
        model = tf.keras.models.load_model(weights_path)
        x = preprocess(image)
        proba = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(proba))
        st.success(
            f"Prediction: **{classes[idx]}**  (confidence {proba[idx]:.3f})"
        )
        st.json({c: float(p) for c, p in zip(classes, proba)})
    except Exception as e:
        st.error(str(e))