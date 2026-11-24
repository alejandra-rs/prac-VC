import numpy as np

MOUTH_RADIUS = 0.25
INHALE_TIME = 4.0
EXHALE_TIME = 8.0
BUBBLE_EVERY = 0.4
BUBBLE_COST = 0.06
EMBER_FADE = 8.0


class Smoking:
    def __init__(self, smoke):
        self.smoke = smoke
        self.lungs = self.ember = self.next_bubble = 0.0
        self.smoking = self.exhaling = False

    def update(self, pipe, face, mouth_open, dt):
        self.smoking = self.exhaling = False
        if face is not None:
            mouth = np.array(face.mouth)
            self.smoking = np.linalg.norm(pipe.point(pipe.sprite["tip"]) - mouth) < MOUTH_RADIUS * face.width
            self.exhaling = mouth_open and self.lungs > 0

        if self.smoking and not self.exhaling:
            self.lungs = min(1.0, self.lungs + dt / INHALE_TIME)
            self.ember = min(1.0, self.ember + 1.5 * dt)
        else:
            self.ember = max(0.0, self.ember - dt / EMBER_FADE)

        count = self.smoke.rng.poisson((30 + 40 * self.smoking) * self.ember * dt)
        self.smoke.emit(pipe.point(pipe.sprite["bowl"]), count, (0, -0.23 * pipe.length), 12, 0.017 * pipe.length, 2.8, 0.3)
        if self.exhaling:
            self.blow(pipe, face, mouth, dt)
        else:
            self.next_bubble = 0.0

    def blow(self, pipe, face, mouth, dt):
        side = 1 if pipe.point(pipe.sprite["bowl"])[0] < mouth[0] else -1
        velocity = np.array([side, -0.35]) * 2.2 * face.width
        self.lungs = max(0.0, self.lungs - dt / EXHALE_TIME)
        count = self.smoke.rng.poisson((60 * self.lungs + 10) * dt)
        self.smoke.emit(mouth, count, velocity, 16, 0.05 * face.width, 2.2, 0.3)
        self.next_bubble -= dt
        if self.next_bubble <= 0 and self.lungs >= BUBBLE_COST:
            self.smoke.emit(mouth, 1, 0.6 * velocity, 0, 0.25 * face.width, 2.5, 1.0)
            self.lungs -= BUBBLE_COST
            self.next_bubble = BUBBLE_EVERY
