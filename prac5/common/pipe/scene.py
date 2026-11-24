import numpy as np

from .pipe import Pipe
from .smoke import Smoke
from .smoking import Smoking

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_BASE = 0, 4, 8, 9
GRAB_RATIO = 0.35
RELEASE_RATIO = 0.55
GRAB_REACH = 0.6
LOST_TIME = 0.4

PIPE_TO_FACE = 1.3
MOUTH_OPEN = 0.1
FACE_MEMORY = 0.5


def pinch(hand):
    size = np.linalg.norm(hand[WRIST] - hand[MIDDLE_BASE])
    return np.linalg.norm(hand[THUMB_TIP] - hand[INDEX_TIP]) / size, (hand[THUMB_TIP] + hand[INDEX_TIP]) / 2


class PinchGrab:
    def __init__(self):
        self.lost = 0.0

    def update(self, pipe, hands, dt):
        pinches = [pinch(hand) for hand in hands]
        if not pipe.held:
            for ratio, point in pinches:
                if ratio < GRAB_RATIO and np.linalg.norm(point - pipe.grip) < GRAB_REACH * pipe.length:
                    pipe.held, pipe.grip, self.lost = True, point.astype(np.float32), 0.0
            return
        nearby = [(ratio, point) for ratio, point in pinches if np.linalg.norm(point - pipe.grip) < pipe.length]
        if not nearby:
            self.lost += dt
            pipe.held = self.lost < LOST_TIME
            return
        ratio, point = min(nearby, key=lambda nearest: np.linalg.norm(nearest[1] - pipe.grip))
        self.lost = 0.0
        pipe.held = ratio < RELEASE_RATIO
        pipe.grip += (point - pipe.grip) * (1 - np.exp(-20 * dt))


class PipeScene:
    def __init__(self, sprite, shape):
        self.sprite = sprite
        self.height, self.width = shape[:2]
        self.reset()

    def reset(self):
        self.pipe = Pipe(self.sprite, (0.3 * self.width, 0.68 * self.height), 0.3 * self.width)
        self.grab = PinchGrab()
        self.smoke = Smoke()
        self.smoking = Smoking(self.smoke)
        self.face, self.face_age = None, np.inf
        self.mouth_open = False

    def update(self, face, openness, hands, dt):
        self.see_face(face, openness, dt)
        self.grab.update(self.pipe, hands, dt)
        if self.pipe.held and self.face is not None:
            self.pipe.aim(np.array(self.face.mouth), dt)
        elif not self.pipe.held:
            self.pipe.straighten(dt)
        self.smoking.update(self.pipe, self.face, self.mouth_open, dt)
        self.smoke.update(dt, rise=0.23 * self.pipe.length, wind=0.37 * self.pipe.length)

    def see_face(self, face, openness, dt):
        if face is not None:
            self.face, self.face_age = face, 0.0
            self.pipe.length += (PIPE_TO_FACE * face.width - self.pipe.length) * min(1.0, 4 * dt)
        else:
            self.face_age += dt
            self.face = self.face if self.face_age < FACE_MEMORY else None
        if openness is not None:
            self.mouth_open = openness > MOUTH_OPEN

    def draw(self, frame):
        self.pipe.draw(frame, self.smoking.ember)
        self.smoke.draw(frame)
