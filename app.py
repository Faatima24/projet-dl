"""Webcam en direct : YOLO détecte les visages, le CNN prédit l'expression toutes les 2 secondes."""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from tensorflow import keras
from ultralytics import YOLO

# opencv-python fixe ce chemin vers ses propres plugins Qt, incompatibles avec PyQt5
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
CNN_PATH = BASE_DIR / "models" / "cnn_v1.keras"
YOLO_PATH = BASE_DIR / "models" / "yolov8n-face.pt"

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
IMAGE_SIZE = (48, 48)
INFERENCE_PERIOD = 2.0  # secondes entre deux prédictions du CNN
CAMERA_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # python app.py 1 pour changer de caméra


def preprocess_face(image):
    """Image de visage (gris ou BGR, taille quelconque) -> tableau (48, 48, 1) float32 dans [0, 1]."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[:2] != IMAGE_SIZE:
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image = image.astype("float32") / 255.0
    return image[..., np.newaxis]


def center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Expressions faciales")
        self.view = QLabel(alignment=Qt.AlignCenter)
        self.setCentralWidget(self.view)
        self.resize(960, 720)

        self.detector = YOLO(str(YOLO_PATH))
        self.cnn = keras.models.load_model(CNN_PATH)
        self.capture = cv2.VideoCapture(CAMERA_INDEX)

        self.predictions = []  # [(centre de la boîte, label, probabilité)]
        self.last_inference = 0.0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ok, frame = self.capture.read()
        if not ok:
            return

        result = self.detector(frame, verbose=False)[0]
        boxes = [tuple(map(int, box)) for box in result.boxes.xyxy.cpu().numpy()]

        if boxes and time.time() - self.last_inference >= INFERENCE_PERIOD:
            self.classify(frame, boxes)
            self.last_inference = time.time()

        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = self.label_for(box)
            if label:
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.view.setPixmap(QPixmap.fromImage(image).scaled(self.view.size(), Qt.KeepAspectRatio))

    def classify(self, frame, boxes):
        crops = [frame[max(y1, 0):y2, max(x1, 0):x2] for x1, y1, x2, y2 in boxes]
        valid = [(box, crop) for box, crop in zip(boxes, crops) if crop.size]
        if not valid:
            return
        batch = np.stack([preprocess_face(crop) for _, crop in valid])
        probs = self.cnn(batch, training=False).numpy()
        self.predictions = [
            (center(box), CLASS_NAMES[p.argmax()], float(p.max())) for (box, _), p in zip(valid, probs)
        ]

    def label_for(self, box):
        """Associe la boîte à la dernière prédiction la plus proche (le visage bouge entre deux inférences)."""
        if not self.predictions:
            return None
        c = center(box)
        _, name, prob = min(self.predictions, key=lambda pred: np.linalg.norm(pred[0] - c))
        return f"{name} {prob:.0%}"

    def closeEvent(self, event):
        self.timer.stop()
        self.capture.release()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
