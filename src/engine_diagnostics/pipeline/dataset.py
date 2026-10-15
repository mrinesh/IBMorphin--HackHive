"""Assemble the labeled window feature table for training and scoring.

Bookkeeping columns (run_id, engine_id, split, subset, t_end_s, label, quality,
and the per-fault labels y_<fault>) sit beside the feature columns; models must
select `feature_names()` only. `subset` is "challenge" for reserved-condition
test runs, otherwise "core".
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from engine_diagnostics.data.run_io import find_runs, load_run, validate_run
from engine_diagnostics.pipeline.features import FEATURE_VERSION, extract_run
from engine_diagnostics.pipeline.labels import (
    FAULT_COLUMNS,
    FAULTS,
    fault_window_labels,
    load_truth,
    window_labels,
)

BOOKKEEPING = ["run_id", "engine_id", "split", "subset", "t_end_s", "label", "quality",
               *FAULT_COLUMNS]


def engine_splits(batch_config_path: Path) -> dict[str, str]:
    """Map engine_id to its partition from the batch configuration."""
    batch = json.loads(Path(batch_config_path).read_text(encoding="utf-8"))
    mapping = {}
    for split, engines in batch["engine_splits"].items():
        for engine in engines:
            if engine in mapping:
                raise ValueError(f"{engine} assigned to both {mapping[engine]} and {split}")
            mapping[engine] = split
    return mapping


def build_feature_table(raw_dir: Path, truth_dir: Path, batch_config_path: Path,
                        config: dict[str, Any]) -> pd.DataFrame:
    splits = engine_splits(batch_config_path)
    frames = []
    for run_dir in find_runs(raw_dir):
        bundle = load_run(run_dir)
        issues = validate_run(bundle, config)
        if issues:
            raise ValueError("invalid run:\n" + "\n".join(issues))
        engine = bundle.metadata["engine_id"]
        if engine not in splits:
            raise ValueError(f"{bundle.run_id}: {engine} has no partition")
        features = extract_run(bundle, config)
        truth = load_truth(truth_dir, bundle.run_id)
        t_end = features["t_end_s"].to_numpy()
        window_s = config["analysis"]["window_s"]
        per_fault = fault_window_labels(truth, t_end, window_s)
        for fault, column in zip(reversed(FAULTS), reversed(FAULT_COLUMNS), strict=True):
            features.insert(0, column, per_fault[fault])
        features.insert(0, "label", window_labels(truth, t_end, window_s))
        features.insert(0, "subset", "challenge" if truth.get("challenge") else "core")
        features.insert(0, "split", splits[engine])
        features.insert(0, "engine_id", engine)
        features.insert(0, "run_id", bundle.run_id)
        frames.append(features)
    if not frames:
        raise ValueError(f"no runs found in {raw_dir}")
    table = pd.concat(frames, ignore_index=True)
    table.attrs["feature_version"] = FEATURE_VERSION
    return table
