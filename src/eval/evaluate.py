
import yaml, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt
from src.data.preprocess import load_images_split
from src.eval.utils import set_seed

def main(cfg, weights):
    set_seed(cfg.get("seed",42))
    out_dir = Path(cfg["paths"]["outputs"]); out_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_images_split(cfg)
    model = tf.keras.models.load_model(weights)
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
    macro = f1_score(y, y_pred, average='macro')
    micro = f1_score(y, y_pred, average='micro')
    kappa = cohen_kappa_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    (out_dir/"results.json").write_text(json.dumps({"macro_f1":macro, "micro_f1":micro, "kappa":kappa}, indent=2))
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir/"confusion_matrix.png")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--weights", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg, args.weights)
