import json

import cv2
import numpy as np

MAX_TILT = 40
EMBER_COLOR = (40, 140, 255)


def load_sprite(png):
    points = json.loads(png.with_suffix(".json").read_text())
    return {"image": cv2.imread(str(png), cv2.IMREAD_UNCHANGED), **{name: np.array(p) for name, p in points.items()}}


def overlay(frame, rgba, x, y):
    height, width = rgba.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + width, frame.shape[1]), min(y + height, frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    patch = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = patch[..., 3:] / 255
    frame[y0:y1, x0:x1] = (frame[y0:y1, x0:x1] * (1 - alpha) + patch[..., :3] * alpha).astype(np.uint8)


class Pipe:
    def __init__(self, sprite, grip, length):
        self.sprite = sprite
        self.grip = np.array(grip, np.float32)
        self.length = length
        self.angle = 0.0
        self.flipped = False
        self.held = False

    @property
    def scale(self):
        return self.length / self.sprite["image"].shape[1]

    def transform(self):
        cos, sin = np.cos(np.radians(self.angle)), np.sin(np.radians(self.angle))
        flip = np.diag([-1.0 if self.flipped else 1.0, 1.0])
        matrix = self.scale * np.array([[cos, -sin], [sin, cos]]) @ flip
        return matrix, self.grip - matrix @ self.sprite["grip"]

    def point(self, point):
        matrix, offset = self.transform()
        return matrix @ point + offset

    def aim(self, target, dt):
        direction = target - self.grip
        if abs(direction[0]) > 0.15 * self.length:
            self.flipped = direction[0] < 0
        stem = (self.sprite["tip"] - self.sprite["grip"]) * [-1 if self.flipped else 1, 1]
        angle = np.degrees(np.arctan2(direction[1], direction[0]) - np.arctan2(stem[1], stem[0]))
        angle = np.clip((angle + 180) % 360 - 180, -MAX_TILT, MAX_TILT)
        self.angle += (angle - self.angle) * min(1.0, 8 * dt)

    def straighten(self, dt):
        self.angle -= self.angle * min(1.0, 3 * dt)

    def draw(self, frame, ember):
        matrix, offset = self.transform()
        height, width = self.sprite["image"].shape[:2]
        corners = np.array([[0, 0], [width, 0], [0, height], [width, height]]) @ matrix.T + offset
        x0, y0 = np.floor(corners.min(axis=0)).astype(int)
        x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
        warp = np.hstack([matrix, (offset - [x0, y0])[:, None]])
        overlay(frame, cv2.warpAffine(self.sprite["image"], warp, (x1 - x0, y1 - y0)), x0, y0)
        if ember >= 0.05:
            glow = frame.copy()
            center = tuple(self.point(self.sprite["ember"][:2]).astype(int))
            axes = (int(self.sprite["ember"][2] * self.scale), max(1, int(self.sprite["ember"][3] * self.scale)))
            cv2.ellipse(glow, center, axes, self.angle, 0, 360, EMBER_COLOR, -1, cv2.LINE_AA)
            cv2.addWeighted(glow, ember, frame, 1 - ember, 0, frame)
