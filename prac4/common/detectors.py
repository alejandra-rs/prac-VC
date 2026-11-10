import gc

import numpy as np
import torch
import torchvision
from ultralytics import YOLO

from . import contour_selection, contours
from .detr import FAMILIES, boxes_to_pixels, build_model, coco_config, get_weights, imagenet_norm, preprocess


class Detector:
    def __init__(self, name):
        self.name = name
        self.n_params = 0
        self.model = None

    def close(self):
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class YoloDetector(Detector):
    def __init__(self, name, weights, classes=None, image_size=640, score_min=0.001, top_k=300):
        super().__init__(name)
        self.model = YOLO(str(weights))
        self.n_params = sum(p.numel() for p in self.model.model.parameters())
        self.args = {"imgsz": image_size, "conf": score_min, "max_det": top_k, "classes": classes, "verbose": False}

    def __call__(self, bgr):
        boxes = self.model.predict(bgr, **self.args)[0].boxes
        return boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)


class DetrDetector(Detector):
    def __init__(self, name, family, config, weights, device="cuda", keep=(0,), top_k=300):
        super().__init__(name)
        self.device, self.top_k = torch.device(device), top_k
        self.keep = torch.tensor(list(keep), device=self.device)
        self.model, self.size_hw = build_model(family, config, weights, self.device)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.norm = imagenet_norm(self.device) if FAMILIES[family]["normalize"] else None

    def predict(self, bgr):
        height, width = bgr.shape[:2]
        outputs = self.model(preprocess(bgr, self.size_hw, self.device, self.norm))
        return outputs["pred_logits"][0][:, self.keep].sigmoid(), boxes_to_pixels(outputs["pred_boxes"][0], width, height)

    @torch.inference_mode()
    def __call__(self, bgr):
        probs, boxes = self.predict(bgr)
        scores, index = probs.flatten().topk(min(self.top_k, probs.numel()))
        queries, classes = index // probs.shape[1], index % probs.shape[1]
        return boxes[queries].cpu().numpy(), scores.cpu().numpy(), self.keep[classes].cpu().numpy()


class ObjectDetector(DetrDetector):
    def __init__(self, labels, conf=0.5, nms_iou=0.7, device="cuda", family="rtdetrv4", size="l"):
        super().__init__(f"{family}-{size}", family, coco_config(family, size), get_weights(family, size), device, labels)
        self.conf, self.nms_iou = conf, nms_iou

    @torch.inference_mode()
    def __call__(self, bgr):
        probs, boxes = self.predict(bgr)
        scores, classes = probs.max(-1)
        found = scores >= self.conf
        boxes, scores, labels = boxes[found], scores[found], self.keep[classes[found]]
        kept = torchvision.ops.nms(boxes, scores, self.nms_iou)
        return boxes[kept].cpu().numpy(), scores[kept].cpu().numpy(), labels[kept].cpu().numpy()


class ContourPlateDetector(Detector):
    def __init__(self, name="contours"):
        super().__init__(name)

    def run(self, bgr):
        crop = contours.preprocess(bgr)
        candidates = contours.generate_candidates(crop)
        survivors = contour_selection.remove_duplicates(contour_selection.apply_filters(candidates, crop))
        scored = sorted((contour_selection.score_candidate(c, crop) for c in survivors), key=lambda c: -c.score)
        return {"crop": crop, "candidates": candidates, "survivors": survivors, "scored": scored,
                "best": scored[0] if scored else None}

    def __call__(self, bgr):
        trace = self.run(bgr)
        if trace["best"] is None:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
        height, width = bgr.shape[:2]
        x, y, w, h = trace["best"].box
        box = np.clip(np.float32([x, y, x + w, y + h]) / trace["crop"]["scale"], 0, [width, height, width, height])
        return box[None], np.float32([trace["best"].score]), np.zeros(1, int)
