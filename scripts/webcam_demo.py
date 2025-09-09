import cv2
import yaml
import numpy as np
import tensorflow as tf


# Load configuration
cfg = yaml.safe_load(open("configs/config.yaml"))
classes = cfg["training"]["classes"]
img_size = tuple(cfg["training"]["img_size"])

# Load trained model weights
model = tf.keras.models.load_model("outputs/baseline_cnn_best.h5")


def main() -> None:
    """Run a live webcam demo showing predicted ASL letters."""
    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Convert to grayscale and resize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(gray, img_size)[..., None].astype("float32") / 255.0
        # Predict letter
        pred = model.predict(roi[None, ...], verbose=0)[0]
        idx = int(np.argmax(pred))
        # Overlay prediction
        txt = f"{classes[idx]} ({pred[idx]:.2f})"
        cv2.putText(
            frame,
            txt,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("ASL Demo", frame)
        # Break on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()