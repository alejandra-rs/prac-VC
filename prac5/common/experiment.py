import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import yaml

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"

DETECTOR_RESULTS = RESULTS / "face_detectors"
PIPE_RESULTS = RESULTS / "pipe"
READING_RESULTS = RESULTS / "reading"


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.loads(json.dumps(yaml.safe_load(f)), object_hook=lambda d: SimpleNamespace(**d))


def cached(path, compute):
    if not path.exists():
        joblib.dump(compute(), path)
    return joblib.load(path)
