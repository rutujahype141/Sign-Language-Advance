
import yaml, json
from pathlib import Path
import numpy as np, cv2
import tensorflow as tf
from sklearn.metrics import accuracy_score
from src.data.preprocess import load_images_split

# augmentations

def augment_brightness(img, factor):
    out = np.clip(img.astype('float32') * factor, 0, 255).astype('uint8')
    return out

def add_noise(img, sigma=15):
    noise = np.random.randn(*img.shape) * sigma
    out = np.clip(img + noise, 0, 255).astype('uint8')
    return out

def main(cfg, weights):
    out_dir = Path(cfg["paths"]["outputs"]); out_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_images_split(cfg)
    model = tf.keras.models.load_model(weights)
    scenarios = {
        "low_light": lambda x: augment_brightness((x*255).astype('uint8'), 0.6),
        "bright":    lambda x: augment_brightness((x*255).astype('uint8'), 1.4),
        "noise":     lambda x: add_noise((x*255).astype('uint8'), 15),
    }
    accs = {}
    for name, fx in scenarios.items():
        X2 = np.array([fx(x.squeeze(-1)) for x in X])[...,None]/255.0
        y_pred = np.argmax(model.predict(X2, verbose=0), axis=1)
        accs[name] = float(accuracy_score(y, y_pred))
    (out_dir/"robustness.json").write_text(json.dumps(accs, indent=2))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--weights", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg, args.weights)
