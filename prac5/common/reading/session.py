import subprocess
import time

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
import pandas as pd

from ..drowsiness.landmarks import face_landmarker, frame_features
from .drowsiness import CALIBRATION_S, Drowsiness
from .emotions import Emotions
from .overlay import draw

SAMPLE_RATE = 24000


def load_audio(path):
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", str(path),
               "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    return np.frombuffer(subprocess.run(command, capture_output=True, check=True).stdout, np.float32)


def clock(t):
    return "Calibrando" if t < CALIBRATION_S else f"{int(t // 60)}:{int(t % 60):02d}"


class FaceLandmarks:
    def __init__(self):
        self.model = face_landmarker()
        self.timestamp = 0

    def __call__(self, bgr):
        self.timestamp = max(self.timestamp + 1, int(time.perf_counter() * 1000))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return self.model.detect_for_video(image, self.timestamp)


class ReadingSession:
    def __init__(self, audio_path, sentences_path):
        self.audio = load_audio(audio_path)
        self.duration = len(self.audio) / SAMPLE_RATE
        self.sentences = pd.read_csv(sentences_path)
        self.face = FaceLandmarks()
        self.drowsiness, self.emotions = Drowsiness(), Emotions()
        self.elapsed = 0.0

    def sentence(self, t):
        current = self.sentences[(self.sentences["start"] <= t) & (self.sentences["end"] > t)]
        return current["text"].iloc[0] if len(current) else ""

    def update(self, frame, t):
        self.elapsed = t
        result = self.face(frame)
        self.drowsiness.update(t, frame_features(result, frame.shape))
        self.emotions.update(t, frame, result)

    def draw(self, frame, t):
        return draw(frame, clock(t), self.emotions.dominant(), self.sentence(t))
