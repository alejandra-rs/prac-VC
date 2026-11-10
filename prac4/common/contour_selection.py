import math

import cv2
import numpy as np
from scipy.signal import find_peaks

from .contours import PLATE_KINDS, rect_corners

SEARCH_TOP_EXCLUDED = 0.2
RELATIVE_WIDTH_RANGE = (0.05, 0.5)
NMS_IOU = 0.75

FEATURE_NAMES = ("white", "vertical profile", "horizontal profile", "lower half", "width")


def apply_filters(candidates, crop):
    height, width = crop["gray"].shape
    return [c for c in candidates
            if c.kind is not None
            and c.rect[0][1] >= SEARCH_TOP_EXCLUDED * height
            and RELATIVE_WIDTH_RANGE[0] <= c.rect[1][0] / width <= RELATIVE_WIDTH_RANGE[1]]


def remove_duplicates(candidates, max_iou=NMS_IOU):
    boxes = [c.box for c in candidates]
    areas = [c.rect[1][0] * c.rect[1][1] for c in candidates]
    return [candidates[i] for i in np.array(cv2.dnn.NMSBoxes(boxes, areas, 0.0, max_iou)).flatten()]


def score_candidate(candidate, crop):
    interior = candidate_interior(candidate, crop)
    height, width = crop["gray"].shape
    if min(interior["gray"].shape) < 6:
        candidate.features = dict.fromkeys(FEATURE_NAMES, 0.0)
    else:
        scores = (white_ratio_score(interior["white"]), vertical_profile_score(interior["gray"], candidate.kind),
                  horizontal_profile_score(interior["gray"], candidate.kind), lower_half_score(candidate, height),
                  width_score(candidate, width))
        candidate.features = dict(zip(FEATURE_NAMES, scores))
    candidate.score = float(np.mean(list(candidate.features.values())))
    return candidate


def candidate_interior(candidate, crop):
    return {"gray": shrink(warp_rect(crop["gray"], candidate.rect)),
            "white": shrink(warp_rect(crop["white"], candidate.rect, cv2.INTER_NEAREST))}


def warp_rect(img, rect, interpolation=cv2.INTER_LINEAR):
    width, height = (max(round(side), 1) for side in rect[1])
    target = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    transform = cv2.getPerspectiveTransform(rect_corners(rect), target)
    return cv2.warpPerspective(img, transform, (width, height), flags=interpolation)


def shrink(img, fraction=0.08):
    height, width = img.shape[:2]
    dy, dx = round(height * fraction), round(width * fraction)
    return img[dy:height - dy, dx:width - dx]


def white_ratio_score(white):
    return min(cv2.countNonZero(white) / white.size / 0.5, 1.0)


def profile(gray, axis):
    mean = cv2.reduce(gray, axis, cv2.REDUCE_AVG, dtype=cv2.CV_32F)
    return cv2.GaussianBlur(mean.reshape(-1, 1), (1, 5), 0).ravel()


def find_valleys(values):
    return find_peaks(-values, prominence=10, width=1)


def vertical_profile_score(gray, kind):
    valleys, props = find_valleys(profile(gray, 1))
    if len(valleys) == 0:
        return 0.0
    count = 1.0 if len(valleys) == PLATE_KINDS[kind]["rows"] else 0.5
    return (count + min(props["prominences"].max() / 60, 1.0)) / 2


def character_rows(gray, kind):
    _, props = find_valleys(profile(gray, 1))
    strongest = np.argsort(-props["prominences"])[:PLATE_KINDS[kind]["rows"]]
    rows = sorted((int(props["left_ips"][i]), int(math.ceil(props["right_ips"][i])) + 1) for i in strongest)
    return rows or [(0, gray.shape[0])]


def range_score(value, value_range):
    low, high = value_range
    if value < low:
        return value / low
    return 1.0 if value <= high else max(0.0, 1 - (value - high) / high)


def horizontal_profile_score(gray, kind):
    counts = [len(find_valleys(profile(gray[top:bottom], 0))[0]) for top, bottom in character_rows(gray, kind)]
    return float(np.mean([range_score(n, PLATE_KINDS[kind]["chars"]) for n in counts]))


def lower_half_score(candidate, height):
    return 1.0 if candidate.rect[0][1] >= 0.5 * height else 0.0


def width_score(candidate, width, min_width=0.15):
    return 1.0 if candidate.rect[1][0] >= min_width * width else 0.0
