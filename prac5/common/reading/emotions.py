import cv2
import numpy as np
import pandas as pd
import torch

from ..emotions.data import EMOTIONS
from ..emotions.model import ResNetEmotion, TorchEmotionPredictor
from ..experiment import ROOT, WEIGHTS, load_config

CFG = load_config(ROOT / "emotion_detection" / "emotions.yaml")
EVERY_S = 0.2
SMOOTHING = 0.6
MIN_FACE = 24
CROP_MARGIN = 0.1


def load_model():
    net = ResNetEmotion(CFG.resnet.img_size, CFG.resnet.dropout, pretrained=False)
    net.load_state_dict(torch.load(WEIGHTS / "emotions" / "resnet18_ferplus.pt", map_location="cpu"))
    model = TorchEmotionPredictor(net.to("cuda" if torch.cuda.is_available() else "cpu"))
    model.predict_proba(np.zeros((1, 48, 48), np.uint8))
    return model


def face_crop(gray, landmarks):
    height, width = gray.shape
    xs, ys = np.array([p.x for p in landmarks]) * width, np.array([p.y for p in landmarks]) * height
    side = max(np.ptp(xs), np.ptp(ys)) * (1 + CROP_MARGIN)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    x1, y1 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    x2, y2 = int(min(cx + side / 2, width)), int(min(cy + side / 2, height))
    if min(x2 - x1, y2 - y1) < MIN_FACE:
        return None
    return cv2.resize(gray[y1:y2, x1:x2], (48, 48), interpolation=cv2.INTER_AREA)


class Emotions:
    def __init__(self):
        self.model = load_model()
        self.current, self.next, self.log = None, 0.0, []

    def update(self, t, frame, result):
        if t < self.next or not result.face_landmarks:
            return
        self.next = t + EVERY_S
        face = face_crop(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), result.face_landmarks[0])
        if face is None:
            return
        probabilities = dict(zip(EMOTIONS, self.model.predict_proba(face[None])[0]))
        self.log.append({"t": t} | probabilities)
        self.current = probabilities if self.current is None else {
            name: SMOOTHING * self.current[name] + (1 - SMOOTHING) * p for name, p in probabilities.items()}

    def dominant(self):
        return None if self.current is None else max(self.current, key=self.current.get)

    def table(self, step):
        samples = pd.DataFrame(self.log, columns=["t", *EMOTIONS])
        shares = pd.crosstab((samples["t"] // step) * step, samples[list(EMOTIONS)].idxmax(axis=1), normalize="index")
        return shares.reindex(columns=list(EMOTIONS), fill_value=0.0)
