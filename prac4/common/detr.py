import importlib
import subprocess
import sys

import gdown
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from .paths import ROOT, WEIGHTS

FAMILIES = {
    "rtdetrv4": {"url":       "https://github.com/RT-DETRs/RT-DETRv4.git",
                 "dir":       "RT-DETRv4",
                 "root":      "RT-DETRv4",
                 "config":    "configs/rtv4/rtv4_hgnetv2_{size}_coco.yml",
                 "weights":   {"s": "gdrive:1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z",
                               "l": "gdrive:1shO9EzZvXZyKedE2urLsN4dwEv8Jqa_8"},
                 "normalize": False},
    "deimv2":   {"url":       "https://github.com/Intellindust-AI-Lab/DEIMv2.git",
                 "dir":       "DEIMv2",
                 "root":      "DEIMv2",
                 "config":    "configs/deimv2/deimv2_dinov3_{size}_coco.yml",
                 "weights":   {"s": "gdrive:1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL",
                               "l": "gdrive:1dRJfVHr9HtpdvaHlnQP460yPVHynMray"},
                 "normalize": True},
    "ec":       {"url":       "https://github.com/Intellindust-AI-Lab/EdgeCrafter.git",
                 "dir":       "EdgeCrafter",
                 "root":      "EdgeCrafter/ecdetseg",
                 "config":    "configs/ecdet/ecdet_{size}.yml",
                 "weights":   {"s": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_s.pth",
                               "l": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_l.pth"},
                 "normalize": True},
}

INFERENCE_OVERRIDES = {
    "HGNetv2":    {"pretrained": False},
    "DINOv3STAs": {"weights_path": None},
    "ViTAdapter": {"skip_load_backbone": True},
}


def get_repo(family):
    family_info = FAMILIES[family]
    if not (ROOT / family_info["dir"]).exists():
        subprocess.run(["git", "clone", "--depth", "1", family_info["url"], str(ROOT / family_info["dir"])], check=True)
    return ROOT / family_info["root"]


def get_weights(family, size):
    dest = WEIGHTS / f"{family}_{size}.pth"
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    source = FAMILIES[family]["weights"][size]
    if source.startswith("gdrive:"):
        file_id = source.removeprefix("gdrive:")
        gdown.download(id=file_id, output=str(dest), quiet=False)
        if not dest.exists():
            raise RuntimeError(f"Error al descargar {family}-{size}. Se debe descargar manualmente "
                               f"de https://drive.google.com/file/d/{file_id} y guardar en {dest}")
    else:
        torch.hub.download_url_to_file(source, str(dest))
    return dest


def coco_config(family, size):
    return get_repo(family) / FAMILIES[family]["config"].format(size=size)


def use_repo_engine(family):
    root = str(get_repo(family))
    for module in [m for m in sys.modules if m == "engine" or m.startswith("engine.")]:
        del sys.modules[module]

    others = {str(ROOT / f["root"]) for f in FAMILIES.values()} - {root}
    sys.path[:] = [p for p in sys.path if p not in others and p != root]
    sys.path.insert(0, root)
    importlib.invalidate_caches()


def load_config(family, config_path, weights):
    use_repo_engine(family)
    from engine.core import YAMLConfig
    
    cfg = YAMLConfig(str(config_path), resume=str(weights))
    for section, values in INFERENCE_OVERRIDES.items():
        if section in cfg.yaml_cfg:
            cfg.yaml_cfg[section].update(values)
    return cfg


def load_ema_weights(model, weights):
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"])


def move_pos_embeds(model, device):
    for module in model.modules():
        for attr, value in list(vars(module).items()):
            if attr.startswith("pos_embed") and isinstance(value, torch.Tensor):
                setattr(module, attr, value.to(device))


def build_model(family, config_path, weights, device):
    cfg = load_config(family, config_path, weights)
    load_ema_weights(cfg.model, weights)
    model = cfg.model.deploy().to(device).eval()
    move_pos_embeds(model, device)
    return model, tuple(cfg.yaml_cfg.get("eval_spatial_size", [640, 640]))


def imagenet_norm(device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return mean, std


def preprocess(bgr, size_hw, device, norm=None):
    x = torch.from_numpy(np.ascontiguousarray(bgr)).to(device)[..., [2, 1, 0]].permute(2, 0, 1)[None].float()
    x = F.interpolate(x, size=size_hw, mode="bilinear", antialias=True, align_corners=False) / 255
    return x if norm is None else (x - norm[0]) / norm[1]


def boxes_to_pixels(boxes, width, height):
    boxes = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
    return boxes * torch.tensor([width, height, width, height], device=boxes.device)
