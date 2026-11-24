import joblib
import numpy as np
import pandas as pd

from ..drowsiness.landmarks import COLUMNS
from ..drowsiness.videos import CLASSES
from ..drowsiness.windows import WINDOW_FEATURES, calibration, default_baseline, normalise, window_features
from ..experiment import ROOT, WEIGHTS, load_config

CFG = load_config(ROOT / "drowsiness_detection" / "drowsiness.yaml")
MODEL = "random_forest"
CALIBRATION_S = 40
STEP_S = 10
WINDOWS = (30, 60, 120)


def load_models():
    return {seconds: joblib.load(WEIGHTS / "drowsiness" / f"calibrated_3class_{seconds}s_{MODEL}.joblib")[0]["model"]
            for seconds in WINDOWS}


class Drowsiness:
    def __init__(self):
        self.models = load_models()
        self.fps = CFG.windows.fps
        self.times, self.frames, self.log = [], [], []
        self.baseline, self.next_window = None, CALIBRATION_S + WINDOWS[0]

    def update(self, t, features):
        self.times.append(t)
        self.frames.append(features)
        if self.baseline is None and t >= CALIBRATION_S:
            self.baseline = self.calibrate()
        if t >= self.next_window:
            seconds = max(w for w in WINDOWS if w <= self.next_window - CALIBRATION_S)
            self.next_window += STEP_S
            self.predict(t, seconds)

    def signals(self, start, end):
        target = start + np.arange(int(round((end - start) * self.fps))) / self.fps
        index = np.clip(np.searchsorted(self.times, target, side="right") - 1, 0, len(self.times) - 1)
        return dict(zip(COLUMNS, np.array(self.frames, np.float32)[index].T))

    def calibrate(self):
        signals = self.signals(0, CALIBRATION_S)
        if not (signals["face"] == 1).any():
            return default_baseline(CFG.calibration.default_open_ear)
        return calibration(signals, self.fps, CALIBRATION_S, CFG.calibration.open_percentile)

    def predict(self, t, seconds):
        window = normalise(self.signals(t - seconds, t), self.baseline)
        if 1 - window["face"].mean() > CFG.windows.max_no_face:
            return
        features = pd.DataFrame([window_features(window, self.fps, CFG.events)])[WINDOW_FEATURES]
        self.log.append({"t": t} | dict(zip(CLASSES, self.models[seconds].predict_proba(features)[0])))

    def table(self):
        states = pd.DataFrame(self.log, columns=["t", *CLASSES]).set_index("t")
        return states.rolling(3, center=True, min_periods=1).mean()
