"""Evaluation-only window labels from simulation truth.

Used to train and score models, never by feature extraction or replay
inference. A window gets a class label only when it lies entirely inside one
segment of the truth label timeline; otherwise it is "transition". In
multi-fault runs the timeline labels name the active set (e.g.
"knock_like+misfire"). Per-fault labels give each fault its own 1/0/NaN.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

TRANSITION = "transition"
FAULTS = ("misfire", "knock_like", "valve_clearance_like")
FAULT_COLUMNS = [f"y_{f}" for f in FAULTS]
CHANGE_MARGIN_S = 0.5  # matches the 1 s transition around each change in the timeline


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


def _as_intervals(raw) -> list[tuple[float, float]]:
    raw = raw or []
    if raw and not isinstance(raw[0], list):  # one interval encoded without nesting
        raw = [raw]
    return [(float(a), float(b)) for a, b in raw]


def fault_intervals(truth: dict[str, Any]) -> dict[str, list[tuple[float, float]]]:
    """Active intervals per fault; older single-fault truth has no `faults` list."""
    faults = truth.get("faults")
    if faults is None:
        if truth["condition"] == "normal":
            return {}
        return {truth["condition"]: _as_intervals(truth.get("active_intervals"))}
    if isinstance(faults, dict):
        faults = [faults]
    return {f["condition"]: _as_intervals(f.get("active_intervals")) for f in faults}


def fault_cylinders(truth: dict[str, Any]) -> dict[str, int]:
    """Affected cylinder per fault (localization labels); older truth has one fault."""
    faults = truth.get("faults")
    if faults is None:
        if truth["condition"] == "normal" or truth.get("affected_cylinder") is None:
            return {}
        return {truth["condition"]: int(truth["affected_cylinder"])}
    if isinstance(faults, dict):
        faults = [faults]
    return {f["condition"]: int(f["affected_cylinder"]) for f in faults
            if f.get("affected_cylinder") is not None}


def duration_s(truth: dict[str, Any]) -> float:
    segments = truth["label_timeline"]
    if isinstance(segments, dict):
        segments = [segments]
    return float(segments[-1]["end_s"])


def fault_window_labels(truth: dict[str, Any], t_end_s: np.ndarray,
                        window_s: float) -> dict[str, np.ndarray]:
    """Per fault: 1 inside an active interval, 0 clear of every interval, NaN between.

    The margin around each change matches the label timeline's transitions, so
    a single-fault run's labels agree with window_labels.
    """
    duration = duration_s(truth)
    intervals = fault_intervals(truth)
    t_end = np.asarray(t_end_s, dtype=float)
    start = t_end - window_s
    out = {}
    for fault in FAULTS:
        y = np.zeros(len(t_end))
        near = np.zeros(len(t_end), dtype=bool)
        inside = np.zeros(len(t_end), dtype=bool)
        for a, b in intervals.get(fault, []):
            lo = a + CHANGE_MARGIN_S if a > 0 else a
            hi = b - CHANGE_MARGIN_S if b < duration else b
            inside |= (start >= lo) & (t_end <= hi)
            near |= (t_end > a - CHANGE_MARGIN_S) & (start < b + CHANGE_MARGIN_S)
        y[inside] = 1
        y[near & ~inside] = np.nan
        out[fault] = y
    return out
