import gradio as gr
import numpy as np
import yaml
import cv2
import tensorflow as tf
from PIL import Image


# Load configuration
cfg = yaml.safe_load(open("configs/config.yaml"))
classes = cfg["training"]["classes"]
img_size = tuple(cfg["training"]["img_size"])


def predict(image: Image.Image, weights: str = "outputs/baseline_cnn_best.h5"):
    """Predict the ASL letter given an input image and weights path."""
    # Load model weights
    model = tf.keras.models.load_model(weights)
    # Preprocess image
    gray = image.convert("L")
    arr = cv2.resize(np.array(gray), img_size).astype("float32") / 255.0
    x = arr[..., None][None, ...]
    # Predict probabilities
    proba = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(proba))
    # Return per-class probabilities and top-1 result
    return (
        {c: float(p) for c, p in zip(classes, proba)},
        f"{classes[idx]} ({proba[idx]:.3f})",
    )


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload image"),
        gr.Textbox(value="outputs/baseline_cnn_best.h5", label="Weights path"),
    ],
    outputs=[gr.Label(label="Class probabilities"), gr.Textbox(label="Top-1")],
    title="ASL Letter Recognition",
)


if __name__ == "__main__":
    demo.launch()