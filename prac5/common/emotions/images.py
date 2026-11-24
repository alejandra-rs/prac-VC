import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

FLIP = 0.5
ROTATION = math.radians(10)
SCALE = (0.9, 1.1)
SHIFT = 0.16
BRIGHTNESS = 0.2
CONTRAST = 0.2


def to_unit(images):
    return images.float().div(255).unsqueeze(1)


def uniform(x, low, high, shape=()):
    return torch.empty((len(x),) + shape, device=x.device).uniform_(low, high)


def random_flip(x):
    return torch.where(uniform(x, 0, 1, (1, 1, 1)) < FLIP, x.flip(-1), x)


def random_affine(x):
    angle, scale, shift = uniform(x, -ROTATION, ROTATION), uniform(x, *SCALE), uniform(x, -SHIFT, SHIFT, (2,))
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([cos, -sin, shift[:, 0], sin, cos, shift[:, 1]], 1).view(-1, 2, 3)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False)


def random_light(x):
    mean = x.mean((1, 2, 3), keepdim=True)
    contrast = uniform(x, 1 - CONTRAST, 1 + CONTRAST, (1, 1, 1))
    brightness = uniform(x, 1 - BRIGHTNESS, 1 + BRIGHTNESS, (1, 1, 1))
    return (((x - mean) * contrast + mean) * brightness).clamp(0, 1)


@torch.no_grad()
def augment(x):
    return random_light(random_affine(random_flip(x)))


def blur(images):
    return np.array([cv2.GaussianBlur(image, (5, 5), 0) for image in images])


def low_resolution(images):
    return np.array([cv2.resize(cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA), (48, 48))
                     for image in images])


def noise(images):
    noisy = images + np.random.default_rng(0).normal(0, 20, images.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def dark(images):
    return (images * 0.5).astype(np.uint8)


def bright(images):
    return np.clip(images.astype(int) + 80, 0, 255).astype(np.uint8)


def low_contrast(images):
    mean = images.mean((1, 2), keepdims=True)
    return (mean + (images - mean) * 0.4).astype(np.uint8)


def eyes_covered(images):
    covered = images.copy()
    covered[:, 12:24, 4:44] = 0
    return covered


def mouth_covered(images):
    covered = images.copy()
    covered[:, 30:42, 12:36] = 0
    return covered


PERTURBATIONS = {
    "Original": lambda images: images,
    "Desenfoque": blur,
    "Baja resolución": low_resolution,
    "Ruido": noise,
    "Oscura": dark,
    "Sobreexpuesta": bright,
    "Bajo contraste": low_contrast,
    "Ojos tapados": eyes_covered,
    "Boca tapada": mouth_covered,
}
