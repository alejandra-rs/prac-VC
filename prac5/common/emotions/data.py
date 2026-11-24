import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..experiment import DATA

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
FER_DIR = DATA / "fer2013"
FERPLUS_URL = "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"
FERPLUS_VOTES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral", "contempt", "unknown", "NF"]


def load_fer2013():
    cache = FER_DIR / "fer2013_arrays.npz"
    if not cache.exists():
        table = pd.read_csv(FER_DIR / "fer2013.csv")
        images = np.array(" ".join(table["pixels"]).split(), dtype=np.uint8).reshape(-1, 48, 48)
        np.savez(cache, images=images, labels=table["emotion"].to_numpy(), usage=table["Usage"].to_numpy(str))
    arrays = np.load(cache)
    return arrays["images"], arrays["labels"], arrays["usage"]


def load_ferplus_labels():
    path = FER_DIR / "fer2013new.csv"
    if not path.exists():
        urllib.request.urlretrieve(FERPLUS_URL, path)
    winner = pd.read_csv(path)[FERPLUS_VOTES].to_numpy().argmax(1)
    return np.where(winner < len(EMOTIONS), winner, -1)


def compute_class_weights(labels):
    counts = np.bincount(labels, minlength=len(EMOTIONS))
    weights = np.cbrt(counts.sum() / (len(EMOTIONS) * counts))
    return weights / weights[labels].mean()


def fold_splits(labels, n_folds, seed):
    return list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(labels, labels))
