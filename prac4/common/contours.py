import math

import cv2
import numpy as np

WORK_WIDTH = 640
PLATE_KINDS = {"wide": {"ratio": (2.318, 5.909), "rows": 1, "chars": (4, 10)},
               "square": {"ratio": (1.031, 2.1), "rows": 2, "chars": (2, 6)}}


class Candidate:
    def __init__(self, rect, source):
        self.rect, self.source, self.features, self.score = rect, source, {}, 0.0

    @property
    def corners(self):
        return rect_corners(self.rect)

    @property
    def box(self):
        return list(cv2.boundingRect(self.corners))

    @property
    def ratio(self):
        w, h = self.rect[1]
        return w / h

    @property
    def kind(self):
        return next((k for k, v in PLATE_KINDS.items() if v["ratio"][0] <= self.ratio <= v["ratio"][1]), None)


def preprocess(bgr):
    scale = WORK_WIDTH / bgr.shape[1]
    work = cv2.resize(bgr, (WORK_WIDTH, round(bgr.shape[0] * scale)), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    gray = cv2.equalizeHist(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY))
    return {"bgr": work, "gray": gray, "white": white_mask(cv2.cvtColor(work, cv2.COLOR_BGR2HSV), gray), "scale": scale}


def generate_candidates(crop):
    masks = {"edges": edge_mask(crop), "white": white_region_mask(crop), "hough": hough_line_mask(crop)}
    return [Candidate(rect, name) for name, mask in masks.items() for rect in find_rotated_rects(mask)]


def white_mask(hsv, gray, saturation_max=60, value_percentile=60):
    low_saturation = cv2.inRange(hsv[..., 1], 0, saturation_max)
    bright = cv2.inRange(gray, int(np.percentile(gray, value_percentile)), 255)
    return cv2.bitwise_and(low_saturation, bright)


def edge_mask(crop):
    return cv2.dilate(canny_edges(crop), rect_kernel(3))


def canny_edges(crop, sigma=0.33):
    smooth = cv2.bilateralFilter(crop["gray"], 9, 75, 75)
    median = float(np.median(smooth))
    return cv2.Canny(smooth, int(max(0, (1 - sigma) * median)), int(min(255, (1 + sigma) * median)))


def rect_kernel(ksize):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))


def white_region_mask(crop):
    opened = cv2.morphologyEx(crop["white"], cv2.MORPH_OPEN, rect_kernel(5))
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, rect_kernel(5))


def hough_line_mask(crop):
    horizontal, vertical = split_segments(hough_segments(crop))
    return draw_segments(crop["gray"].shape, np.vstack([horizontal, vertical]))


def hough_segments(crop):
    lines = cv2.HoughLinesP(canny_edges(crop), 1, np.pi / 180, 15, minLineLength=10, maxLineGap=6)
    return np.zeros((0, 4), np.float32) if lines is None else lines[:, 0].astype(np.float32)


def split_segments(segments, max_tilt=15):
    angles = np.degrees(np.arctan2(segments[:, 3] - segments[:, 1], segments[:, 2] - segments[:, 0]))
    tilt = np.abs((angles + 90) % 180 - 90)
    return segments[tilt <= max_tilt], segments[tilt >= 90 - max_tilt]


def draw_segments(shape, segments, extend=6):
    mask = np.zeros(shape, np.uint8)
    for x1, y1, x2, y2 in segments:
        dx, dy = np.array([x2 - x1, y2 - y1]) / max(math.hypot(x2 - x1, y2 - y1), 1e-6) * extend
        cv2.line(mask, (round(x1 - dx), round(y1 - dy)), (round(x2 + dx), round(y2 + dy)), 255, 2)
    return mask


def normalize_rect(rect):
    (cx, cy), (w, h), angle = rect
    while angle > 45:
        w, h, angle = h, w, angle - 90
    while angle <= -45:
        w, h, angle = h, w, angle + 90
    return (cx, cy), (w, h), angle


def rect_corners(rect):
    (cx, cy), (w, h), angle = rect
    theta = math.radians(angle)
    u = np.array([math.cos(theta), math.sin(theta)]) * w / 2
    v = np.array([-math.sin(theta), math.cos(theta)]) * h / 2
    center = np.array([cx, cy])
    return np.float32([center - u - v, center + u - v, center + u + v, center - u + v])


def find_rotated_rects(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = (normalize_rect(cv2.minAreaRect(c)) for c in contours)
    return [r for r in rects if min(r[1]) >= 2]
