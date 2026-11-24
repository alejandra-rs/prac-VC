import time

import cv2
import mediapipe as mp
import numpy as np

from ..weights import model_bytes

UPPER_LIP, LOWER_LIP, LEFT_CORNER, RIGHT_CORNER = 13, 14, 61, 291
CROP_MARGIN = 0.3


def to_mediapipe(bgr):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def to_pixels(landmarks, shape):
    return np.array([(p.x * shape[1], p.y * shape[0]) for p in landmarks])


def base_options(model):
    return mp.tasks.BaseOptions(model_asset_buffer=model_bytes(model))


def face_crop(bgr, box):
    x1, y1, x2, y2 = box
    margin = CROP_MARGIN * (x2 - x1)
    x1, y1 = int(max(0, x1 - margin)), int(max(0, y1 - margin))
    x2, y2 = int(min(bgr.shape[1], x2 + margin)), int(min(bgr.shape[0], y2 + margin))
    return np.ascontiguousarray(bgr[y1:y2, x1:x2]), np.array([x1, y1])


class Hands:
    def __init__(self):
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(base_options=base_options("hand_landmarker.task"),
                                               running_mode=vision.RunningMode.VIDEO, num_hands=2)
        self.model = vision.HandLandmarker.create_from_options(options)
        self.timestamp = 0

    def __call__(self, bgr):
        self.timestamp = max(self.timestamp + 1, int(time.perf_counter() * 1000))
        result = self.model.detect_for_video(to_mediapipe(bgr), self.timestamp)
        return [to_pixels(hand, bgr.shape) for hand in result.hand_landmarks]


class MouthOpening:
    def __init__(self):
        vision = mp.tasks.vision
        options = vision.FaceLandmarkerOptions(base_options=base_options("face_landmarker.task"), num_faces=1)
        self.model = vision.FaceLandmarker.create_from_options(options)
        self.lips = []

    def __call__(self, bgr, face):
        crop, origin = face_crop(bgr, face.box)
        result = self.model.detect(to_mediapipe(crop))
        if not result.face_landmarks:
            self.lips = []
            return None
        points = to_pixels(result.face_landmarks[0], crop.shape)
        self.lips = points[[UPPER_LIP, LOWER_LIP, LEFT_CORNER, RIGHT_CORNER]] + origin
        gap = np.linalg.norm(points[UPPER_LIP] - points[LOWER_LIP])
        return gap / np.linalg.norm(points[LEFT_CORNER] - points[RIGHT_CORNER])


def draw_landmarks(frame, face, hands, lips):
    groups = [(hand, (255, 255, 0)) for hand in hands]
    if face is not None:
        x1, y1, x2, y2 = map(int, face.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        groups += [(face.points or [], (0, 255, 255)), (lips, (255, 0, 255))]
    for points, color in groups:
        for x, y in points:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
