from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
RUNS = ROOT / "runs"

DATASET = ROOT.parent / "dataset"
PLATES_DATA = ROOT / "data" / "vehicles"
PLATES_RUNS = RUNS / "plates"
PLATES_MODULE = ROOT / "plates_module2"
