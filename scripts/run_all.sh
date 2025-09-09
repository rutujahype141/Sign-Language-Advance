
#!/usr/bin/env bash
set -e
python -m src.models.train_cnn --config configs/config.yaml
python -m src.eval.evaluate --config configs/config.yaml --weights outputs/baseline_cnn_best.h5
python -m src.eval.robustness --config configs/config.yaml --weights outputs/baseline_cnn_best.h5
