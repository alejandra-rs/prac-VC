import numpy as np
import pandas as pd

from .videos import LABELS
from .landmarks import COLUMNS

EARS = ["ear_left", "ear_right"]
POSE = ["pitch", "yaw", "roll"]
WINDOW_FEATURES = ["perclos", "blinks_per_min", "blink_mean", "blink_std", "blink_interval", "long_blinks", "ear_mean",
                   "ear_std", "yawns", "pitch_std", "nods", "no_face"]


def as_columns(frames, fps, target_fps):
    index = np.arange(int(len(frames) * target_fps / fps)) * fps / target_fps
    frames = frames[np.minimum(index.astype(int), len(frames) - 1)]
    return dict(zip(COLUMNS, frames.T))


def calibration(frames, fps, seconds, percentile):
    first = slice(0, int(seconds * fps))
    found = frames["face"][first] == 1
    ear = (frames["ear_left"][first] + frames["ear_right"][first])[found] / 2
    return {"ear": np.percentile(ear, percentile), "pose": [np.median(frames[name][first][found]) for name in POSE]}


def default_baseline(open_ear):
    return {"ear": open_ear, "pose": [0, 0, 0]}


def normalise(frames, baseline):
    frames = dict(frames)
    for name in EARS:
        frames[name] = frames[name] / baseline["ear"]
    for name, value in zip(POSE, baseline["pose"]):
        frames[name] = frames[name] - value
    return frames


def runs(mask):
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def blinks(closed, fps, min_frames):
    starts, ends = runs(closed)
    long_enough = ends - starts >= min_frames
    return starts[long_enough], (ends - starts)[long_enough] / fps


def eye_features(face, ear, fps, events):
    closed = face & (ear < events.closed_fraction)
    starts, durations = blinks(closed, fps, events.min_blink_frames)
    return {"perclos": closed.sum() / face.sum(),
            "blinks_per_min": len(durations) / (face.sum() / fps / 60),
            "blink_mean": durations.mean() if len(durations) else 0,
            "blink_std": durations.std() if len(durations) else 0,
            "blink_interval": np.diff(starts).mean() / fps if len(durations) > 1 else len(face) / fps,
            "long_blinks": (durations > events.long_blink).sum(),
            "ear_mean": ear[face].mean(),
            "ear_std": ear[face].std()}


def yawn_features(face, mar, fps, events):
    starts, ends = runs(face & (mar > events.yawn_mar))
    return {"yawns": (ends - starts >= events.yawn_seconds * fps).sum()}


def head_features(face, pitch, events):
    nods, _ = runs(face & (np.abs(pitch - np.median(pitch[face])) > events.nod_degrees))
    return {"pitch_std": pitch[face].std(), "nods": len(nods)}


def window_features(window, fps, events):
    face = window["face"] == 1
    ear = (window["ear_left"] + window["ear_right"]) / 2
    return {**eye_features(face, ear, fps, events), **yawn_features(face, window["mar"], fps, events),
            **head_features(face, window["pitch"], events), "no_face": 1 - face.mean()}


def sequence(window):
    return np.nan_to_num(np.stack([window[name] for name in COLUMNS], axis=1))


def pad(sequences):
    padded = np.zeros((len(sequences), max(map(len, sequences)), sequences[0].shape[1] + 1), np.float32)
    for i, steps in enumerate(sequences):
        padded[i, :len(steps), :-1], padded[i, :len(steps), -1] = steps, 1
    return padded


def video_windows(video, frames, seconds, skip_seconds, cfg):
    fps, length = cfg.windows.fps, len(frames["face"])
    first = int(skip_seconds * fps)
    size = length - first if seconds is None else int(seconds * fps)
    rows, sequences = [], []
    for start in range(first, length - size + 1, int(cfg.windows.step * fps)):
        window = {name: values[start:start + size] for name, values in frames.items()}
        if 1 - window["face"].mean() > cfg.windows.max_no_face:
            continue
        rows.append({"video": video.video, "fold": video.fold, "participant": video.participant,
                     "label": LABELS.index(video.label), "start": start / fps}
                    | window_features(window, fps, cfg.events))
        sequences.append(sequence(window))
    return rows, sequences


def build_windows(videos, frames_by_video, baselines, seconds, cfg):
    rows, sequences = [], []
    for video in videos.itertuples():
        frames = normalise(frames_by_video[video.video], baselines[video.participant])
        skip = cfg.calibration.seconds if video.label == 0 else 0
        video_rows, video_sequences = video_windows(video, frames, seconds, skip, cfg)
        rows += video_rows
        sequences += video_sequences
    return pd.DataFrame(rows), pad(sequences) if seconds is None else np.array(sequences, dtype=np.float32)
