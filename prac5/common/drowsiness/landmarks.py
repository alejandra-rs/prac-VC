import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from ..weights import model_bytes

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 82, 312, 291, 317, 87]
BLENDSHAPES = ["eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight", "jawOpen"]
COLUMNS = ["face", "ear_left", "ear_right", "mar", "pitch", "yaw", "roll"] + BLENDSHAPES


def face_landmarker():
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=model_bytes("face_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def video_fps(path):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return fps


def read_frames(path, start_ms=0):
    capture = cv2.VideoCapture(str(path))
    if start_ms:
        capture.set(cv2.CAP_PROP_POS_MSEC, start_ms)
    try:
        while (frame := capture.read()[1]) is not None:
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def to_image(frame):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)


def pixel_points(landmarks, ids, shape):
    height, width = shape[:2]
    return np.array([(landmarks[i].x * width, landmarks[i].y * height) for i in ids])


def aspect_ratio(points):
    corner_a, top_a, top_b, corner_b, bottom_b, bottom_a = points
    opening = np.linalg.norm(top_a - bottom_a) + np.linalg.norm(top_b - bottom_b)
    return opening / (2 * np.linalg.norm(corner_a - corner_b))


def head_angles(matrix):
    rotation = matrix[:3, :3] / np.linalg.norm(matrix[:3, :3], axis=0)
    pitch = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arcsin(np.clip(-rotation[2, 0], -1, 1))
    roll = np.arctan2(rotation[1, 0], rotation[0, 0])
    return list(np.degrees([pitch, yaw, roll]))


def frame_features(result, shape):
    if not result.face_landmarks:
        return [0] + [np.nan] * (len(COLUMNS) - 1)
    landmarks = result.face_landmarks[0]
    ratios = [aspect_ratio(pixel_points(landmarks, ids, shape)) for ids in (LEFT_EYE, RIGHT_EYE, MOUTH)]
    scores = {category.category_name: category.score for category in result.face_blendshapes[0]}
    return ([1] + ratios + head_angles(np.array(result.facial_transformation_matrixes[0]))
            + [scores[name] for name in BLENDSHAPES])


def extract_video(path):
    fps = video_fps(path)
    with face_landmarker() as landmarker:
        rows = [frame_features(landmarker.detect_for_video(to_image(frame), int(1000 * i / fps)), frame.shape)
                for i, frame in enumerate(read_frames(path))]
    return {"frames": np.array(rows, dtype=np.float32), "fps": fps}


def frames_table(data):
    seconds = np.arange(len(data["frames"])) / data["fps"]
    return pd.DataFrame(data["frames"], columns=COLUMNS, index=pd.Index(seconds, name="segundos"))
