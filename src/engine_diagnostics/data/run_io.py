"""Load and validate exported run bundles (docs/DATA_CONTRACT.md).

Only public run files are read: waveforms.mat, telemetry.csv, metadata.json.
Evaluation truth lives elsewhere and is never opened here. Contract 1.1 adds
the cam phase reference `cam_sync_s` (one pulse per 720 crank degrees at a
fixed angle after cylinder 1's firing TDC, on the vibration clock); 1.0
bundles have none and are still read.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

WAVEFORM_STREAMS = ("audio", "vibration")
TELEMETRY_UNITS = {
    "rpm": "rpm",
    "load": "percent",
    "throttle": "percent",
    "coolant_temp": "degC",
    "short_term_fuel_trim": "percent",
    "long_term_fuel_trim": "percent",
}
TELEMETRY_COLUMNS = ["t_s", "signal", "value", "unit", "valid"]
PUBLIC_FILES = ("waveforms.mat", "telemetry.csv", "metadata.json")
REQUIRED_METADATA = (
    "schema_version", "run_id", "engine_id", "batch_id", "generator_version",
    "config_schema_version", "created_utc", "synthetic", "duration_s", "streams",
    "telemetry_signals", "clock", "waveform_summary",
)
RUN_ID = re.compile(r"^run_\d{5}$")
# Missing waveform samples (NaN) are a declared data-quality problem, allowed up to
# this fraction of a run; windows that contain them are flagged by the feature path.
MAX_RUN_MISSING_FRACTION = 0.05
ENGINE_ID = re.compile(r"^engine_\d{3}$")
SCHEMA_VERSIONS = ("1.0", "1.1")
CAM_SYNC_SCHEMA = "1.1"  # first data contract version with the cam phase reference
CAM_SYNC_INFO = {"unit": "s", "clock": "vibration", "reference": "cylinder_1_firing_tdc"}
CAM_LOCAL_SPACINGS = 2  # neighbours on each side that set a pulse gap's local cycle period


@dataclass
class RunBundle:
    run_id: str
    audio: np.ndarray
    vibration: np.ndarray
    audio_fs_hz: float
    vibration_fs_hz: float
    audio_t0_s: float
    vibration_t0_s: float
    telemetry: pd.DataFrame
    metadata: dict[str, Any]
    cam_sync_s: np.ndarray | None = None  # pulse times on the vibration clock (contract 1.1)


def load_run(run_dir: Path) -> RunBundle:
    """Read one run bundle, requiring MATLAB column vectors of class single."""
    run_dir = Path(run_dir)
    # mat_dtype=True reports the MATLAB class; MATLAB may store integer-valued
    # doubles in a compact integer form on disk.
    mat = loadmat(run_dir / "waveforms.mat", mat_dtype=True)
    waveforms = {}
    for name in WAVEFORM_STREAMS:
        if name not in mat:
            raise ValueError(f"{run_dir.name}: waveforms.mat has no '{name}'")
        array = mat[name]
        if array.ndim != 2 or array.shape[1] != 1:
            raise ValueError(f"{run_dir.name}: {name} must be an (N, 1) column vector, "
                             f"got shape {array.shape}")
        if array.dtype != np.float32:
            raise ValueError(f"{run_dir.name}: {name} must be single (float32), got {array.dtype}")
        waveforms[name] = array[:, 0]

    scalars = {}
    for name in ("audio_fs_hz", "vibration_fs_hz", "audio_t0_s", "vibration_t0_s"):
        if name not in mat or mat[name].size != 1:
            raise ValueError(f"{run_dir.name}: waveforms.mat needs scalar '{name}'")
        scalars[name] = float(mat[name].item())
    cam_sync_s = None
    if "cam_sync_s" in mat:  # contract 1.1; MATLAB stores an empty vector as 0x0 or 0x1
        cam = mat["cam_sync_s"]
        if cam.size and (cam.ndim != 2 or cam.shape[1] != 1):
            raise ValueError(f"{run_dir.name}: cam_sync_s must be an (N, 1) column vector, "
                             f"got shape {cam.shape}")
        cam_sync_s = cam.astype(np.float64).reshape(-1)

    telemetry = pd.read_csv(
        run_dir / "telemetry.csv",
        dtype={"t_s": "float64", "signal": "string", "value": "float64",
               "unit": "string", "valid": "int64"},
        keep_default_na=False, na_values={"value": [""]},
    )
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    return RunBundle(
        run_id=run_dir.name,
        audio=waveforms["audio"],
        vibration=waveforms["vibration"],
        telemetry=telemetry,
        metadata=metadata,
        cam_sync_s=cam_sync_s,
        **scalars,
    )


def validate_run(bundle: RunBundle, config: dict[str, Any]) -> list[str]:
    """Return contract violations for a loaded run; an empty list means valid."""
    issues = _check_metadata(bundle)
    if not issues:
        duration = bundle.metadata["duration_s"]
        try:
            for name in WAVEFORM_STREAMS:
                issues += _check_waveform(bundle, name, duration, config)
            issues += _check_telemetry(bundle.telemetry, bundle.metadata, duration, config)
            issues += _check_cam_sync(bundle, duration, config)
        except KeyError as error:
            issues.append(f"metadata field missing: {error}")
    return [f"{bundle.run_id}: {issue}" for issue in issues]


def _check_metadata(bundle: RunBundle) -> list[str]:
    metadata = bundle.metadata
    missing = [key for key in REQUIRED_METADATA if key not in metadata]
    if missing:
        return [f"metadata missing {', '.join(missing)}"]
    issues = []
    if metadata["run_id"] != bundle.run_id or not RUN_ID.match(bundle.run_id):
        issues.append(f"run_id '{metadata['run_id']}' does not match directory '{bundle.run_id}'")
    if not ENGINE_ID.match(str(metadata["engine_id"])):
        issues.append(f"engine_id '{metadata['engine_id']}' is not engine_NNN")
    if metadata["synthetic"] is not True:
        issues.append("metadata.synthetic must be true")
    if metadata["schema_version"] not in SCHEMA_VERSIONS:
        issues.append(f"schema_version '{metadata['schema_version']}' is not one of "
                      f"{', '.join(SCHEMA_VERSIONS)}")
    return issues


def cam_cycle_steps(spacing: np.ndarray) -> np.ndarray:
    """Cycles spanned by each cam pulse gap (1, or more where pulses are missing).

    Each gap is compared with the median of its neighbouring gaps
    (CAM_LOCAL_SPACINGS on each side), not the run-wide median, so a speed
    ramp (cycle period changing smoothly, up to several-fold over a run) is
    never mistaken for a missing pulse, while one lost pulse (a gap twice its
    neighbours) is.
    """
    spacing = np.asarray(spacing, dtype=np.float64)
    if spacing.size < 3:
        return np.ones(spacing.size)
    k = CAM_LOCAL_SPACINGS
    local = np.array([np.median(np.delete(spacing[max(0, i - k):i + k + 1], min(i, k)))
                      for i in range(spacing.size)])
    return np.maximum(1, np.round(spacing / local))


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(text).split("."))


def _check_cam_sync(bundle: RunBundle, duration: float, config: dict) -> list[str]:
    """The cam phase reference: required from contract 1.1, consistent with metadata."""
    declared = bundle.metadata["streams"].get("cam_sync")
    pulses = bundle.cam_sync_s
    if _version(bundle.metadata["schema_version"]) < _version(CAM_SYNC_SCHEMA) and declared is None:
        return [] if pulses is None else ["cam_sync_s present but metadata has no streams.cam_sync"]
    if declared is None or pulses is None:
        return ["cam_sync: contract 1.1 needs both waveforms.mat cam_sync_s and "
                "metadata.streams.cam_sync"]
    issues = [f"cam_sync: {key} '{declared.get(key)}', expected '{value}'"
              for key, value in CAM_SYNC_INFO.items() if declared.get(key) != value]
    if declared.get("angle_deg") != config["engine"]["cam_sync_angle_deg"]:
        issues.append(f"cam_sync: angle_deg {declared.get('angle_deg')} differs from config "
                      f"{config['engine']['cam_sync_angle_deg']}")
    if declared.get("n_pulses") != pulses.size:
        issues.append(f"cam_sync: {pulses.size} pulses, metadata n_pulses "
                      f"{declared.get('n_pulses')}")
    if not np.all(np.isfinite(pulses)):
        return issues + ["cam_sync: non-finite pulse times"]
    if ((pulses < 0) | (pulses >= duration)).any():
        issues.append(f"cam_sync: pulse times outside [0, {duration})")
    spacing = np.diff(pulses)
    # A 720-degree cycle cannot be shorter than at twice the engine's maximum speed.
    if spacing.size and spacing.min() < 0.5 * 120 / config["engine"]["rpm_max"]:
        issues.append("cam_sync: pulses not strictly increasing at a feasible cycle period")
    return issues


def _check_waveform(bundle: RunBundle, name: str, duration: float, config: dict) -> list[str]:
    issues = []
    x = getattr(bundle, name)
    fs = getattr(bundle, f"{name}_fs_hz")
    t0 = getattr(bundle, f"{name}_t0_s")
    stream = bundle.metadata["streams"][name]
    expected_fs = config["streams"][name]["sample_rate_hz"]
    if fs != expected_fs or stream["fs_hz"] != expected_fs:
        issues.append(f"{name}: sample rate {fs} (metadata {stream['fs_hz']}), "
                      f"config {expected_fs}")
    if t0 != stream["t0_s"]:
        issues.append(f"{name}: t0_s {t0} differs from metadata {stream['t0_s']}")
    if stream["unit"] != config["streams"][name]["unit"]:
        issues.append(f"{name}: unit '{stream['unit']}' differs from config")
    expected_n = round(duration * expected_fs)
    if x.size != expected_n:
        issues.append(f"{name}: {x.size} samples, expected {expected_n}")
    missing = np.isnan(x)
    if np.isinf(x).any():
        issues.append(f"{name}: infinite samples")
        return issues
    if missing.mean() > MAX_RUN_MISSING_FRACTION:
        issues.append(f"{name}: non-finite samples ({missing.mean():.4f} missing)")
        return issues

    summary = bundle.metadata["waveform_summary"][name]
    x64 = x.astype(np.float64)
    if summary["n_samples"] != x.size:
        issues.append(f"{name}: summary n_samples {summary['n_samples']} != {x.size}")
    elif summary.get("n_missing", 0) != int(missing.sum()):
        issues.append(f"{name}: {int(missing.sum())} missing samples, "
                      f"MATLAB summary {summary.get('n_missing', 0)}")
    elif x.size:
        if not (_same_sample(summary["first"], x[0]) and _same_sample(summary["last"], x[-1])):
            issues.append(f"{name}: first/last samples differ from MATLAB summary")
        for key, value in (("sum", np.nansum(x64)), ("sum_sq", np.nansum(np.square(x64)))):
            if not math.isclose(value, summary[key], rel_tol=1e-9, abs_tol=1e-6):
                issues.append(f"{name}: {key} {value!r} differs from MATLAB {summary[key]!r}")
    return issues


def _same_sample(recorded, sample) -> bool:
    if recorded is None:  # MATLAB writes NaN as null
        return bool(np.isnan(sample))
    return bool(np.float32(recorded) == sample)


def run_warnings(bundle: RunBundle, config: dict[str, Any]) -> list[str]:
    """Data-quality findings that do not break the contract (clipping, gaps)."""
    warnings = []
    for name in WAVEFORM_STREAMS:
        x = getattr(bundle, name)
        stream = bundle.metadata["streams"][name]
        clipped = float(np.mean(np.abs(x[np.isfinite(x)]) >= stream["full_scale"]))
        if clipped > config["runtime"]["max_clipped_fraction"]:
            warnings.append(f"{bundle.run_id}: {name} clipped fraction {clipped:.4f}")
        missing = float(np.isnan(x).mean())
        if missing:
            warnings.append(f"{bundle.run_id}: {name} missing fraction {missing:.4f}")
    if bundle.cam_sync_s is not None and bundle.cam_sync_s.size > 3:
        gaps = int(np.sum(cam_cycle_steps(np.diff(bundle.cam_sync_s)) - 1))
        if gaps:
            warnings.append(f"{bundle.run_id}: cam_sync has {gaps} gaps (missing pulses)")
    invalid = int((bundle.telemetry["valid"] == 0).sum())
    if invalid:
        warnings.append(f"{bundle.run_id}: {invalid} invalid telemetry rows")
    return warnings


def _check_telemetry(telemetry: pd.DataFrame, metadata: dict, duration: float,
                     config: dict) -> list[str]:
    if list(telemetry.columns) != TELEMETRY_COLUMNS:
        return [f"telemetry columns {list(telemetry.columns)}, expected {TELEMETRY_COLUMNS}"]
    issues = []
    ordered = telemetry.sort_values(["t_s", "signal"], kind="stable").reset_index(drop=True)
    if not ordered.equals(telemetry.reset_index(drop=True)):
        issues.append("telemetry rows not sorted by t_s, signal")
    if ((telemetry["t_s"] < 0) | (telemetry["t_s"] >= duration)).any():
        issues.append(f"telemetry timestamps outside [0, {duration})")
    if not telemetry["valid"].isin([0, 1]).all():
        issues.append("telemetry valid must be 0 or 1")
    if (telemetry["valid"].eq(1) & telemetry["value"].isna()).any():
        issues.append("telemetry rows marked valid have empty values")

    rates = config["streams"]["telemetry_rates_hz"]
    for signal, unit in TELEMETRY_UNITS.items():
        rows = telemetry[telemetry["signal"] == signal]
        if rows.empty:
            issues.append(f"telemetry signal '{signal}' missing")
            continue
        units = set(rows["unit"])
        if units != {unit}:
            issues.append(f"telemetry '{signal}' unit {sorted(units)}, expected '{unit}'")
        declared = metadata["telemetry_signals"].get(signal, {})
        if declared != {"rate_hz": rates[signal], "unit": unit}:
            issues.append(f"metadata telemetry_signals.{signal} disagrees with config")
        expected = duration * rates[signal]
        if abs(len(rows) - expected) > 0.05 * expected:
            issues.append(f"telemetry '{signal}' has {len(rows)} rows, expected ~{expected:g}")
    unknown = set(telemetry["signal"]) - set(TELEMETRY_UNITS)
    if unknown:
        issues.append(f"unknown telemetry signals {sorted(unknown)}")
    return issues


def find_runs(raw_dir: Path) -> list[Path]:
    return sorted(p for p in Path(raw_dir).iterdir() if p.is_dir() and RUN_ID.match(p.name))


def write_manifest(raw_dir: Path, run_dirs: list[Path]) -> Path:
    """Record SHA-256 and size of every public file; identifies a locked dataset."""
    runs = {}
    for run_dir in run_dirs:
        runs[run_dir.name] = {
            name: {
                "sha256": hashlib.sha256((run_dir / name).read_bytes()).hexdigest(),
                "bytes": (run_dir / name).stat().st_size,
            }
            for name in PUBLIC_FILES
        }
    path = Path(raw_dir) / "manifest.json"
    path.write_text(json.dumps({"schema_version": "1.0", "runs": runs}, indent=2) + "\n",
                    encoding="utf-8")
    return path
