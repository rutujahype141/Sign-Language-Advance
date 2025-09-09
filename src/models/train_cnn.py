
import yaml, os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.data.preprocess import load_images_split
from src.models.cnn_baseline import build_cnn
from src.eval.utils import set_seed

def main(cfg):
    set_seed(cfg.get("seed",42))
    out_dir = Path(cfg["paths"]["outputs"]); out_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_images_split(cfg)
    n_classes = len(cfg["training"]["classes"])
    model = build_cnn((cfg["training"]["img_size"][0], cfg["training"]["img_size"][1], 1), n_classes)
    model.compile(optimizer=SGD(learning_rate=cfg["training"]["lr"], momentum=0.9),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    ckpt = ModelCheckpoint(out_dir/"baseline_cnn_best.h5", save_best_only=True, monitor="val_accuracy", mode="max")
    es = EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy", mode="max")
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    tr = int(0.8*len(idx))
    Xtr, Xva = X[idx[:tr]], X[idx[tr:]]
    ytr, yva = y[idx[:tr]], y[idx[tr:]]
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=cfg["training"]["epochs"], batch_size=cfg["training"]["batch_size"],
              callbacks=[ckpt, es], verbose=2)
    model.save(out_dir/"baseline_cnn_last.h5")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
