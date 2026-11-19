import numpy as np


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def iou(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0
    intersection = width * height
    return intersection / (area(a) + area(b) - intersection)


def match_image(dets, boxes, min_iou=0.5):
    found_by = [-np.inf] * len(boxes)
    results = []
    for det in sorted(dets, key=lambda det: -det[4]):
        free = [i for i in range(len(boxes)) if found_by[i] == -np.inf]
        best = max(free, key=lambda i: iou(det, boxes[i]), default=None)
        hit = best is not None and iou(det, boxes[best]) >= min_iou
        if hit:
            found_by[best] = det[4]
        results.append((det[4], hit))
    return results, found_by


def average_precision(recall, precision):
    best_precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(recall, prepend=0) * best_precision))


def match_all(all_dets, samples):
    results, found_by = [], []
    for dets, sample in zip(all_dets, samples):
        image_results, image_found_by = match_image(dets, sample.boxes)
        results += image_results
        found_by += image_found_by
    results.sort(key=lambda result: -result[0])
    scores = np.array([score for score, _ in results])
    hits = np.array([hit for _, hit in results])
    return scores, hits, np.array(found_by)


def precision_recall_curve(hits, n_faces):
    recall = np.cumsum(hits) / n_faces
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return recall, precision


def f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall + 1e-9)


def evaluate(all_dets, samples, threshold):
    scores, hits, found_by = match_all(all_dets, samples)
    recall, precision = precision_recall_curve(hits, len(found_by))

    kept = scores >= threshold
    kept_precision = (kept & hits).sum() / max(kept.sum(), 1)
    kept_recall = (found_by >= threshold).mean()
    return {"ap": average_precision(recall, precision), "precision": kept_precision, "recall": kept_recall,
            "f1": f1_score(kept_precision, kept_recall), "fp_per_image": (kept & ~hits).sum() / len(samples),
            "curve": (recall, precision, scores), "found_by": found_by}


def best_f1_threshold(metrics):
    recall, precision, scores = metrics["curve"]
    return float(scores[f1_score(precision, recall).argmax()])
