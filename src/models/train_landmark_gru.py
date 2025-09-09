
import yaml, numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from src.eval.utils import set_seed

def main(cfg):
    set_seed(cfg.get("seed",42))
    out_dir = Path(cfg["paths"]["outputs"]); out_dir.mkdir(parents=True, exist_ok=True)
    n = 200
    seq_len = cfg["landmarks"]["seq_len"]
    n_classes = len(cfg["training"]["classes"])
    X = np.random.randn(n, seq_len, 63).astype("float32") * 0.1
    y = np.random.randint(0, n_classes, size=(n,))
    inp = layers.Input(shape=(seq_len, 63))
    x = layers.GRU(cfg['landmarks']['hidden'], return_sequences=False)(inp)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, validation_split=0.2, epochs=3, batch_size=16, verbose=2)
    model.save(out_dir/"landmark_gru.h5")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
