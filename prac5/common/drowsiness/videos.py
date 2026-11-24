import re
import subprocess

import cv2
import imageio_ffmpeg
import pandas as pd

LABELS = (0, 5, 10)
CLASSES = ("alerta", "poco atento", "somnoliento")
PUBLISHABLE = {1, 2, 4, 9, 10, 11, 13, 16, 17, 19, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 33, 34, 42, 43, 46, 47, 48,
               49, 50, 51, 52, 53, 55, 57, 59, 60}
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
VIDEO_NAME = re.compile(r"Fold(\d)(?:_part\d)?[\\/](\d+)[\\/]((\d+)(?:_\d)?)\.\w+$")


def list_videos(folder):
    rows = []
    for path in sorted(folder.rglob("*")):
        match = VIDEO_NAME.search(str(path))
        if match:
            fold, participant, name, label = match.groups()
            rows.append({"video": f"{int(participant):02d}_{name}", "fold": int(fold), "participant": int(participant),
                         "label": int(label), "path": path})
    return pd.DataFrame(rows)


def parse_duration(text):
    hours, minutes, seconds = re.search(r"Duration: (\d+):(\d+):([\d.]+)", text).groups()
    return 3600 * int(hours) + 60 * int(minutes) + float(seconds)


def parse_size(text, stream):
    width, height = map(int, re.search(r" (\d{2,5})x(\d{2,5})", stream).groups())
    rotation = re.search(r"rotation of (-?[\d.]+)", text)
    turned = rotation and abs(float(rotation[1])) % 180 == 90
    return (height, width) if turned else (width, height)


def probe(path):
    text = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stderr
    stream = next(line for line in text.splitlines() if "Video:" in line)
    fps = float(re.search(r"([\d.]+) fps", stream)[1])
    width, height = parse_size(text, stream)
    return {"duration": parse_duration(text), "fps": fps, "width": width, "height": height,
            "mb": path.stat().st_size / 1e6}


def scale_filter(info, max_short_side):
    side = min(info["width"], info["height"], max_short_side)
    return f"scale={side}:-2" if info["height"] > info["width"] else f"scale=-2:{side}"


def compress(source, target, settings):
    info = probe(source)
    video_filter = f"fps={min(settings.max_fps, info['fps'])},{scale_filter(info, settings.max_short_side)}"
    partial = target.with_suffix(".part.mp4")
    partial.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(source), "-vf", video_filter, "-c:v", "libx264",
                    "-crf", str(settings.crf), "-preset", settings.preset, "-an", str(partial)], check=True)
    partial.replace(target)


def opens(path):
    capture = cv2.VideoCapture(str(path))
    readable = capture.read()[0]
    capture.release()
    return readable


def reduced_path(folder, video):
    return folder / f"Fold{video.fold}" / f"{video.participant:02d}" / f"{video.path.stem}.mp4"


def is_valid(target, before, after, settings):
    return (opens(target) and abs(before["duration"] - after["duration"]) <= settings.duration_tolerance
            and abs(after["fps"] - min(settings.max_fps, before["fps"])) < 0.1)


def reduce_video(video, folder, settings):
    target = reduced_path(folder, video)
    if not target.exists():
        compress(video.path, target, settings)
    before, after = probe(video.path), probe(target)
    return {"video": video.video, "original": str(video.path), "reducido": str(target),
            "MB antes": before["mb"], "MB después": after["mb"], "duración (s)": after["duration"],
            "fps antes": before["fps"], "fps después": after["fps"],
            "resolución antes": f"{before['width']}x{before['height']}",
            "resolución después": f"{after['width']}x{after['height']}",
            "verificado": is_valid(target, before, after, settings)}
