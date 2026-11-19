import importlib.metadata
import json
import platform

import psutil


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def save_system_info(path, packages):
    info = {"os": platform.platform(),
            "cpu": platform.processor(),
            "cpu_cores": psutil.cpu_count(logical=False),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "python": platform.python_version(),
            "packages": {name: package_version(name) for name in packages}}
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(info, open(path, "w"), indent=1)
    return info
