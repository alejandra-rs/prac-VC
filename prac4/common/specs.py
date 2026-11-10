import importlib.metadata
import json
import platform
import shutil
import subprocess
import threading
import time

import psutil
import torch

OPENCV_PACKAGES = ("opencv-python", "opencv-contrib-python", "opencv-python-headless")


def command_output(cmd):
    if shutil.which(cmd[0]) is None:
        return None
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def nvidia_smi(field):
    output = command_output(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"])
    return output.splitlines()[0] if output else None


def gpu_memory_mb():
    used = nvidia_smi("memory.used")
    return int(used) if used else 0


class VramMonitor:
    def __init__(self, interval_s=0.5):
        self.interval_s = interval_s
        self.baseline = self.peak = 0
        self.running = False
        self.thread = None

    def sample(self):
        while self.running:
            self.peak = max(self.peak, gpu_memory_mb())
            time.sleep(self.interval_s)

    def __enter__(self):
        self.baseline = self.peak = gpu_memory_mb()
        self.running = True
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.running = False
        self.thread.join()

    @property
    def peak_mb(self):
        return self.peak - self.baseline


def package_version(name):
    candidates = OPENCV_PACKAGES if name in OPENCV_PACKAGES else (name,)  # cualquier paquete de OpenCV instala el mismo módulo cv2
    for candidate in candidates:
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def system_info(packages=(), **extra):
    return {"os": platform.platform(),
            "cpu": platform.processor(),
            "cpu_cores": psutil.cpu_count(logical=False),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "python": platform.python_version(),
            "gpu": nvidia_smi("name"),
            "gpu_driver": nvidia_smi("driver_version"),
            "cuda": torch.version.cuda,
            **extra,
            "packages": {name: package_version(name) for name in packages}}


def save_system_info(path, packages=(), **extra):
    info = system_info(packages, **extra)
    json.dump(info, open(path, "w"), indent=1)
    return info
