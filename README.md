
# Advancing Sign Language Interpretation with Transfer Learning and Multimodal Features

- Baseline CNN for sign language gesture recognition
- Transfer learning using ResNet-50 and EfficientNet-B0
- Hand landmark extraction via MediaPipe (GRU-based)
- Comprehensive evaluation: accuracy, macro/micro F1, Cohen's kappa, confusion matrix
- Robustness testing (lighting, noise)

The repository includes:

- Scripts for training and evaluation.
- Sample outputs (results.json, plots).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train baseline CNN on mini-sample
python -m src.models.train_cnn --config configs/config.yaml
python -m src.eval.evaluate --config configs/config.yaml --weights outputs/baseline_cnn_best.h5
python -m src.eval.robustness --config configs/config.yaml --weights outputs/baseline_cnn_best.h5

# Transfer Learning
python -m src.models.train_transfer --config configs/config.yaml --backbone resnet50
python -m src.models.train_transfer --config configs/config.yaml --backbone efficientnetb0

# Landmark + GRU (stub example)
python -m src.models.train_landmark_gru --config configs/config.yaml

## Demo

### Streamlit (browser)
To run a simple browser-based demo using Streamlit:
```bash
pip install -r requirements.txt
streamlit run demo/streamlit_app.py
```
Upload a hand image (PNG or JPEG) and optionally adjust the weights path. The app displays the predicted letter and confidence as well as per-class probabilities.

### Gradio (browser)
To run a Gradio demo:
```bash
pip install -r requirements.txt
python demo/gradio_app.py
```
This serves a web interface where you can upload an image and see the probability distribution across classes and the top prediction.

### Webcam (local)
To run a live webcam demo (requires a webcam):
```bash
python scripts/webcam_demo.py
```

