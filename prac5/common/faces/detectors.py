import cv2
import mediapipe as mp
import numpy as np
from mtcnn import MTCNN
from retinaface import RetinaFace

from ..weights import get_model, model_bytes


def middle(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


class Face:
    def __init__(self, box, score, points=None, mouth=None):
        self.box = box
        self.score = score
        self.points = points or []
        self.mouth = mouth

    @property
    def width(self):
        return self.box[2] - self.box[0]


class FaceDetector:
    label = ""
    threshold = 0.5
    min_score = 0.05
    has_mouth = False

    def __call__(self, bgr, threshold=None):
        threshold = self.threshold if threshold is None else threshold
        return [face for face in self.find(bgr) if face.score >= threshold]

    def close(self):
        pass


class ViolaJones(FaceDetector):
    label = "Viola-Jones"
    threshold, min_score = 5, 1

    def __init__(self):
        self.model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def find(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        boxes, neighbors = self.model.detectMultiScale2(gray, scaleFactor=1.1, minNeighbors=self.min_score,
                                                        minSize=(30, 30))
        return [Face((x, y, x + w, y + h), float(n)) for (x, y, w, h), n in zip(boxes, neighbors)]


class ResNetSSD(FaceDetector):
    label = "DNN ResNet-10"

    def __init__(self):
        self.model = cv2.dnn.readNetFromCaffe(str(get_model("deploy.prototxt.txt")),
                                              str(get_model("res10_300x300_ssd_iter_140000.caffemodel")))

    def find(self, bgr):
        height, width = bgr.shape[:2]
        self.model.setInput(cv2.dnn.blobFromImage(bgr, 1.0, (300, 300), (104.0, 117.0, 123.0)))
        detections = self.model.forward()[0, 0]
        faces = []
        for detection in detections[detections[:, 2] >= self.min_score]:
            score = detection[2]
            x1, y1, x2, y2 = detection[3:7] * [width, height, width, height]
            if x2 > x1 and y2 > y1:
                faces.append(Face((x1, y1, x2, y2), float(score)))
        return faces


class MTCNNDetector(FaceDetector):
    label = "MTCNN"
    threshold = 0.8
    has_mouth = True

    def __init__(self):
        self.model = MTCNN(device="CPU:0")

    def find(self, bgr):
        faces = []
        for result in self.model.detect_faces(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), threshold_onet=self.min_score):
            x, y, w, h = result["box"]
            points = result["keypoints"]
            faces.append(Face((x, y, x + w, y + h), float(result["confidence"]), list(points.values()),
                              middle(points["mouth_left"], points["mouth_right"])))
        return faces


class BlazeFace(FaceDetector):
    label = "BlazeFace"
    has_mouth = True

    def __init__(self):
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=model_bytes("blaze_face_short_range.tflite")),
            min_detection_confidence=self.min_score, min_suppression_threshold=0.3)
        self.model = mp.tasks.vision.FaceDetector.create_from_options(options)

    def find(self, bgr):
        height, width = bgr.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        faces = []
        for detection in self.model.detect(image).detections:
            box = detection.bounding_box
            points = [(p.x * width, p.y * height) for p in detection.keypoints]
            faces.append(Face((box.origin_x, box.origin_y, box.origin_x + box.width, box.origin_y + box.height),
                              detection.categories[0].score, points, points[3]))
        return faces

    def close(self):
        self.model.close()


class YuNet(FaceDetector):
    label = "YuNet"
    threshold = 0.9
    has_mouth = True

    def __init__(self):
        self.model = cv2.FaceDetectorYN.create(str(get_model("face_detection_yunet_2023mar.onnx")), "", (320, 320),
                                               self.min_score, 0.3, 5000)

    def find(self, bgr):
        self.model.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, detections = self.model.detect(bgr)
        faces = []
        for detection in detections if detections is not None else []:
            x, y, w, h = detection[:4]
            points = detection[4:14].reshape(5, 2)
            score = detection[14]
            faces.append(Face((x, y, x + w, y + h), float(score), points.tolist(), middle(points[3], points[4])))
        return faces


class RetinaFaceDetector(FaceDetector):
    label = "RetinaFace"
    threshold, min_score = 0.9, 0.1
    has_mouth = True

    def __init__(self):
        self.model = RetinaFace.build_model()

    def find(self, bgr):
        results = RetinaFace.detect_faces(bgr, threshold=self.min_score, model=self.model)
        faces = []
        for result in results.values() if isinstance(results, dict) else []:
            points = result["landmarks"]
            faces.append(Face(tuple(result["facial_area"]), float(result["score"]), list(points.values()),
                              middle(points["mouth_left"], points["mouth_right"])))
        return faces


def largest_face(faces):
    return max(faces, key=lambda face: face.width, default=None)


def draw_faces(image, faces, color, points=True):
    for face in faces:
        x1, y1, x2, y2 = map(int, face.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        for x, y in face.points if points else []:
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), -1)
    return image


DETECTORS = {"vj": ViolaJones, "dnn": ResNetSSD, "mtcnn": MTCNNDetector, "blazeface": BlazeFace, "yunet": YuNet,
             "retinaface": RetinaFaceDetector}
