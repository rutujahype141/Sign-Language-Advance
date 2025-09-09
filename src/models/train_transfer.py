
import yaml, numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
from tensorflow.keras.optimizers import Adam
from src.data.preprocess import load_images_split
from src.eval.utils import set_seed

def main(cfg, backbone):
    set_seed(cfg.get("seed",42))
    out_dir = Path(cfg["paths"]["outputs"]); out_dir.mkdir(parents=True, exist_ok=True)
    img_size = tuple(cfg["transfer"]["img_size"])
    classes = cfg["training"]["classes"]
    Xg, y = load_images_split(cfg)
    X = np.stack([tf.image.resize(Xg, img_size).numpy().squeeze(-1)]*3, axis=-1)
    n_classes = len(classes)
    if backbone.lower() == "resnet50":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=(img_size[0],img_size[1],3))
        pre = resnet_pre
    else:
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(img_size[0],img_size[1],3))
        pre = eff_pre
    inp = layers.Input(shape=(img_size[0],img_size[1],3))
    x = pre(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    idx = np.arange(len(y)); np.random.shuffle(idx)
    tr = int(0.8*len(idx))
    Xtr, Xva = X[idx[:tr]], X[idx[tr:]]
    ytr, yva = y[idx[:tr]], y[idx[tr:]]
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=2, batch_size=16, verbose=2)
    model.save(out_dir/f"{backbone}_finetuned.h5")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50","efficientnetb0"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg, args.backbone)
