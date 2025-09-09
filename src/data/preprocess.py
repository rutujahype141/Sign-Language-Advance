
import os, yaml, numpy as np, cv2
from pathlib import Path

def load_images_split(cfg):
    data_root = Path(cfg["paths"]["data_root"])
    img_size = tuple(cfg["training"]["img_size"])
    classes = cfg["training"]["classes"]
    X, y = [], []
    for idx, cls in enumerate(classes):
        for split in ["train", "val", "test"]:
            d = data_root / split / cls
            if not d.exists():
                continue
            for p in d.glob("*.png"):
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(idx)
    X = np.array(X)[..., None]/255.0
    y = np.array(y)
    return X, y

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    X, y = load_images_split(cfg)
    print("Loaded:", X.shape, y.shape)
