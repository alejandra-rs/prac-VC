import cv2
import numpy as np

X, Y, VX, VY, AGE, LIFE, SIZE, DENSITY = range(8)
SMOKE_COLOR = np.array([214, 214, 208], np.float32)
BLUR_SIZES = np.array([2, 4, 8, 16], np.float32)
MAX_PARTICLES = 900


class Smoke:
    def __init__(self):
        self.particles = np.zeros((0, 8), np.float32)
        self.rng = np.random.default_rng(0)

    def emit(self, origin, count, velocity, spread, size, life, density):
        angle = np.arctan2(velocity[1], velocity[0]) + self.rng.normal(0, np.radians(spread), count)
        speed = np.hypot(*velocity) * self.rng.uniform(0.6, 1.2, count)
        new = np.column_stack([np.full(count, origin[0]), np.full(count, origin[1]), np.cos(angle) * speed,
                               np.sin(angle) * speed, np.zeros(count), life * self.rng.uniform(0.7, 1.3, count),
                               size * self.rng.uniform(0.7, 1.3, count), np.full(count, density)])
        self.particles = np.vstack([self.particles, new])[-MAX_PARTICLES:]

    def update(self, dt, rise, wind):
        p = self.particles
        p[:, AGE] += dt
        p = p[p[:, AGE] < p[:, LIFE]]
        p[:, [VX, VY]] *= 0.97 ** (dt * 60)
        p[:, VY] -= rise * dt
        p[:, VX] += np.sin(3 * p[:, AGE] + p[:, X] / 50) * wind * dt
        p[:, [X, Y]] += p[:, [VX, VY]] * dt
        self.particles = p

    def draw(self, frame):
        height, width = frame.shape[:2]
        density = cv2.resize(self.density_layer(height // 2, width // 2), (width, height))
        alpha = 1 - np.exp(-density)[..., None]
        frame[:] = (frame + (SMOKE_COLOR - frame) * alpha).astype(np.uint8)

    def density_layer(self, height, width):
        layer = np.zeros((height, width), np.float32)
        p = self.particles
        t = p[:, AGE] / p[:, LIFE]
        density = p[:, DENSITY] * (1 - t) ** 1.5 * np.minimum(1, 4 * p[:, AGE])
        size = np.maximum(1, p[:, SIZE] * (1 + 3.5 * t) / 2)
        x, y = (p[:, X] / 2).astype(int), (p[:, Y] / 2).astype(int)
        inside = (x >= 0) & (x < layer.shape[1]) & (y >= 0) & (y < layer.shape[0])
        group = np.abs(np.log(size[:, None] / BLUR_SIZES)).argmin(axis=1)
        for i, blur in enumerate(BLUR_SIZES):
            selected = inside & (group == i)
            points = np.zeros_like(layer)
            np.add.at(points, (y[selected], x[selected]), density[selected] * 2 * np.pi * blur ** 2)
            layer += cv2.GaussianBlur(points, (0, 0), float(blur))
        return layer
