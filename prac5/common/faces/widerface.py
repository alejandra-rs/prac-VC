import io
import urllib.request
import zipfile
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from ..experiment import DATA

FOLDER = DATA / "wider_face"
URL = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/"
INVALID_COLUMN = 7


class Sample:
    def __init__(self, file, scale, boxes):
        self.file = file
        self.scale = scale
        self.boxes = boxes


def download(name):
    path = FOLDER / name
    if not path.exists():
        FOLDER.mkdir(parents=True, exist_ok=True)
        print(f"Descargando {name}...")
        urllib.request.urlretrieve(URL + name, path)
    return path


def download_widerface():
    download("wider_face_split.zip")
    download("WIDER_val.zip")


@lru_cache(maxsize=1)
def images_zip():
    return zipfile.ZipFile(FOLDER / "WIDER_val.zip")


def read_image(file):
    return images_zip().read(f"WIDER_val/images/{file}")


def read_annotations():
    with zipfile.ZipFile(FOLDER / "wider_face_split.zip") as annotations_zip:
        lines = annotations_zip.read("wider_face_split/wider_face_val_bbx_gt.txt").decode().splitlines()
    annotations, i = {}, 0
    while i < len(lines):
        file, n = lines[i], int(lines[i + 1])
        faces = [line.split() for line in lines[i + 2:i + 2 + n]]
        annotations[file] = np.array(faces, dtype=float).reshape(-1, 10)
        i += 2 + max(n, 1)
    return annotations


def build_subset(width, min_face, n_images, max_faces, seed):
    samples = []
    for file, faces in sorted(read_annotations().items()):
        if not 1 <= len(faces) <= max_faces:
            continue
        scale = width / Image.open(io.BytesIO(read_image(file))).width
        x, y, w, h = (faces[:, :4] * scale).T
        if (h >= min_face).all() and (w > 0).all() and not faces[:, INVALID_COLUMN].any():
            samples.append(Sample(file, scale, np.column_stack([x, y, x + w, y + h])))
    chosen = np.random.default_rng(seed).choice(len(samples), size=n_images, replace=False)
    return [samples[i] for i in sorted(chosen)]


def load_image(sample):
    image = cv2.imdecode(np.frombuffer(read_image(sample.file), np.uint8), cv2.IMREAD_COLOR)
    height, width = image.shape[:2]
    return cv2.resize(image, (round(width * sample.scale), round(height * sample.scale)), interpolation=cv2.INTER_AREA)
