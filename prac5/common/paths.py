from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"

DETECTOR_RESULTS = RESULTS / "face_detectors"
PIPE_RESULTS = RESULTS / "pipe"
