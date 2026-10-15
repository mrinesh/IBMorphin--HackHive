"""Evaluation-only window labels from simulation truth.

Used to train and score models, never by feature extraction or replay
inference. A window gets a class label only when it lies entirely inside one
segment of the truth label timeline; otherwise it is "transition".
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

TRANSITION = "transition"


def load_truth(truth_dir: Path, run_id: str) -> dict[str, Any]:
    return json.loads((Path(truth_dir) / f"{run_id}_events.json").read_text(encoding="utf-8"))


def window_labels(truth: dict[str, Any], t_end_s: np.ndarray, window_s: float) -> list[str]:
    segments = truth["label_timeline"]
    if isinstance(segments, dict):  # MATLAB encodes a one-element struct array as an object
        segments = [segments]
    labels = []
    for t in t_end_s:
        start = t - window_s
        label = TRANSITION
        for segment in segments:
            if segment["start_s"] <= start and t <= segment["end_s"]:
                label = segment["label"]
                break
        labels.append(label)
    return labels
