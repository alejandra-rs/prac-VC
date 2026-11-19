import urllib.request

from .paths import WEIGHTS

COURSE = "https://raw.githubusercontent.com/otsedom/otsedom.github.io/main/VC/P5/"
MEDIAPIPE = "https://storage.googleapis.com/mediapipe-models/"

URLS = {
    "deploy.prototxt.txt": COURSE + "deploy.prototxt.txt",
    "res10_300x300_ssd_iter_140000.caffemodel": COURSE + "res10_300x300_ssd_iter_140000.caffemodel",
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "blaze_face_short_range.tflite":
        MEDIAPIPE + "face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "hand_landmarker.task": MEDIAPIPE + "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "face_landmarker.task": MEDIAPIPE + "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}


def get_model(name):
    path = WEIGHTS / name
    if not path.exists():
        WEIGHTS.mkdir(exist_ok=True)
        print(f"Descargando {name}...")
        urllib.request.urlretrieve(URLS[name], path)
    return path


def model_bytes(name):
    return get_model(name).read_bytes()
