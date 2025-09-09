import argparse
import yaml
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path


def load_and_preprocess(img_path, img_size=(64, 64)):
    """Load an image from disk and preprocess for inference."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    # Resize and normalize
    img = cv2.resize(img, img_size).astype("float32") / 255.0
    # Add channel dimension
    return img[..., None]


def main() -> None:
    """Run single-image ASL letter inference."""
    parser = argparse.ArgumentParser(
        description="Infer an ASL letter from a single image using a trained model."
    )
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to YAML config file"
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the .h5 model weights (e.g., outputs/baseline_cnn_best.h5)",
    )
    parser.add_argument(
        "--image", required=True, help="Path to the image to classify"
    )
    args = parser.parse_args()

    # Load configuration and model
    cfg = yaml.safe_load(open(args.config))
    classes = cfg["training"]["classes"]
    img_size = tuple(cfg["training"]["img_size"])

    # Preprocess the image
    x = load_and_preprocess(args.image, img_size=img_size)[None, ...]

    # Load model and perform inference
    model = tf.keras.models.load_model(args.weights)
    proba = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(proba))

    # Print prediction result
    print({"pred_class": classes[idx], "confidence": float(proba[idx])})


if __name__ == "__main__":
    main()