"""Validate shared configuration before MATLAB generation or Python analysis."""

import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def load_config(config_path: Path, schema_path: Path) -> dict[str, Any]:
    """Read a workspace configuration and enforce structural and numeric contracts."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(config))
    if errors:
        details = "\n".join(
            f"  {'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ValueError(f"Configuration schema validation failed:\n{details}")
    _validate_relationships(config)
    return config


def _validate_relationships(config: dict[str, Any]) -> None:
    engine = config["engine"]
    analysis = config["analysis"]
    dataset = config["dataset"]

    if engine["rpm_min"] >= engine["rpm_max"]:
        raise ValueError("engine.rpm_min must be below engine.rpm_max")
    if engine["load_min"] >= engine["load_max"]:
        raise ValueError("engine.load_min must be below engine.load_max")
    if analysis["step_s"] > analysis["window_s"]:
        raise ValueError("analysis.step_s must not exceed analysis.window_s")
    if analysis["stft_hop_samples"] > analysis["stft_segment_samples"]:
        raise ValueError("STFT hop must not exceed its segment length")
    if sum(dataset["split_engines"].values()) != dataset["engine_count"]:
        raise ValueError("Engine split counts must sum to dataset.engine_count")
    if dataset["runs_per_engine"] % len(config["conditions"]) != 0:
        raise ValueError("runs_per_engine must divide evenly across the condition classes")
    for duration in ("integration_duration_s", "core_duration_s"):
        if dataset[duration] < analysis["window_s"]:
            raise ValueError(f"dataset.{duration} must cover a complete analysis window")

    for name in ("audio", "vibration"):
        sample_rate = config["streams"][name]["sample_rate_hz"]
        if not analysis[f"{name}_low_hz"] < analysis["high_hz"] < sample_rate / 2:
            raise ValueError(f"{name} analysis band must be ordered and below Nyquist")
        for setting in ("window_s", "step_s"):
            samples = analysis[setting] * sample_rate
            if not math.isclose(samples, round(samples), rel_tol=0, abs_tol=1e-8):
                raise ValueError(f"{name}: {setting} must produce an integer sample count")
        window_samples = round(analysis["window_s"] * sample_rate)
        largest_segment = max(analysis["welch_segment_samples"], analysis["stft_segment_samples"])
        if largest_segment > window_samples:
            raise ValueError(f"{name}: spectral segments must fit within the analysis window")
