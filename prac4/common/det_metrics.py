import json
import time
from collections import defaultdict

import torch
from faster_coco_eval import COCO, COCOeval_faster

MAP_KEYS = ("mAP50-95", "mAP50")
PRF_KEYS = ("precision", "recall", "accuracy")


def coco_map(annotations_json, dets, classes):
    if not dets:
        zeros = dict.fromkeys(MAP_KEYS, 0.0)
        return {**zeros, **{name: zeros for name in classes.values()}}

    annotations = COCO(str(annotations_json))
    predictions = annotations.loadRes(dets)
    results = class_map(annotations, predictions, list(classes))
    for cat, name in classes.items():
        results[name] = class_map(annotations, predictions, [cat])
    return results


def get_metrics(annotations_json, dets, classes, conf=0.5, min_iou=0.5):
    data = json.load(open(annotations_json, encoding="utf-8"))
    true_boxes, crowd_boxes = group_annotations(data["annotations"])
    det_boxes = group_detections(dets, conf)

    results, total = {}, [0, 0, 0]
    for cat, name in classes.items():
        counts = [0, 0, 0]
        for im in data["images"]:
            key = (im["id"], cat)
            counts = [c + n for c, n in zip(counts, match_image(det_boxes[key], true_boxes[key], crowd_boxes[key], min_iou))]
        results[name] = counts_to_scores(*counts)
        total = [t + c for t, c in zip(total, counts)]
    results.update(counts_to_scores(*total))
    return results


def class_map(annotations, predictions, cat_ids):
    ev = COCOeval_faster(annotations, predictions, "bbox")
    ev.params.catIds = cat_ids
    ev.params.imgIds = sorted(annotations.getImgIds())
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {"mAP50-95": float(ev.stats[0]), "mAP50": float(ev.stats[1])}


def iou(a, b):
    inter_w = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    inter_h = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    intersection = inter_w * inter_h
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union > 0 else 0.0


def match_image(det_boxes, true_boxes, crowd_boxes, min_iou):
    taken = [False] * len(true_boxes)
    tp = fp = 0
    for det in det_boxes:
        free = [j for j in range(len(true_boxes)) if not taken[j]]
        best = max(free, key=lambda j: iou(det, true_boxes[j]), default=None)
        if best is not None and iou(det, true_boxes[best]) >= min_iou:
            taken[best] = True
            tp += 1
        elif any(iou(det, crowd) >= min_iou for crowd in crowd_boxes):
            continue
        else:
            fp += 1
    return tp, fp, taken.count(False)


def counts_to_scores(tp, fp, fn):
    return {"TP": tp, "FP": fp, "FN": fn,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "accuracy": tp / (tp + fp + fn) if (tp + fp + fn) else 0.0}


def group_annotations(annotations):
    true_boxes, crowd_boxes = defaultdict(list), defaultdict(list)
    for a in annotations:
        key = (a["image_id"], a["category_id"])
        (crowd_boxes if a.get("iscrowd") else true_boxes)[key].append(a["bbox"])
    return true_boxes, crowd_boxes


def group_detections(dets, conf):
    det_boxes = defaultdict(list)
    for d in sorted(dets, key=lambda d: -d["score"]):
        if d["score"] >= conf:
            det_boxes[(d["image_id"], d["category_id"])].append(d["bbox"])
    return det_boxes



def to_coco_dets(image_id, boxes, scores, cats, score_min=0.0):
    return [{"image_id": image_id, "category_id": int(c), "score": float(s),
             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]}
            for (x1, y1, x2, y2), s, c in zip(boxes, scores, cats) if s >= score_min]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_inference(detect, images, read_image, warmup=10):
    for im in images[:warmup]:
        detect(read_image(im))
    sync()

    results, total = [], 0.0
    for im in images:
        bgr = read_image(im)
        sync()
        t0 = time.perf_counter()
        results.append(detect(bgr))
        sync()
        total += time.perf_counter() - t0
    return results, total
