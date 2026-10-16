"""Per-cylinder exhaust gas (EGT) and head (CHT) temperature cross-check.

Slow (seconds) but hard to fool: a thermocouple does not hear road noise. At
each decision time ThermalPath reads the latest valid reading of every
egt_cyl<N> / cht_cyl<N> signal before t (data contract 1.2), then:

  relative   each cylinder minus the median over cylinders, so engine-wide
             changes with speed and load cancel;
  change     minus that cylinder's own relative level learned over the first
             baseline_s of the recording (fixed per-cylinder sensor and
             mixture offsets, as an aircraft engine monitor's "lean find"
             reference); before the baseline is complete the path is
             warming up;
  z          change divided by the healthy spread calibrated on training runs
             without faults.

Patterns, each confirmed and cleared like an alert:
  cold      EGT and CHT both clearly low on one cylinder: it is not firing
            (misfire);
  head_hot  CHT high while EGT is not: detonation heating the head (knock);
  exhaust_hot  EGT high while CHT is not: hot gas past an exhaust valve
            (valve_clearance_like in this simulator).
The mapping from pattern to fault is a physical prior stated here, not
learned from labels. Nothing in this module reads evaluation truth.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engine_diagnostics.runtime.alerts import AlertState

THERMAL_MODEL_VERSION = "thermal-1.0.0"
PATTERN_FAULT = {"cold": "misfire", "head_hot": "knock_like", "exhaust_hot": "valve_clearance_like"}
KINDS = ("egt", "cht")


@dataclass
class ThermalCalibration:
    """Healthy spreads and pattern thresholds (artifacts/models/thermal_monitor.json)."""

    spread_c: dict[str, float]  # healthy sd of each signal's change from baseline, degC
    thresholds: dict[str, float]  # z thresholds: cold, hot, not_hot
    baseline_s: float = 5.0
    max_age_s: dict[str, float] = field(default_factory=lambda: {"egt": 0.5, "cht": 1.0})
    confirm_updates: int = 4
    clear_updates: int = 6
    model_version: str = THERMAL_MODEL_VERSION
    training: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ThermalCalibration":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def latest_readings(telemetry: pd.DataFrame, t: float, cylinders: int,
                    max_age_s: dict[str, float]) -> dict[str, np.ndarray]:
    """Latest valid value of every cylinder temperature strictly before t (NaN if stale)."""
    out = {kind: np.full(cylinders, np.nan) for kind in KINDS}
    if telemetry.empty:
        return out
    before = telemetry[(telemetry["t_s"] < t) & (telemetry["valid"] == 1)]
    for kind in KINDS:
        for c in range(1, cylinders + 1):
            rows = before[before["signal"] == f"{kind}_cyl{c}"]
            if not rows.empty and t - rows["t_s"].iloc[-1] <= max_age_s[kind]:
                out[kind][c - 1] = float(rows["value"].iloc[-1])
    return out


def relative(values: np.ndarray) -> np.ndarray:
    if not np.isfinite(values).all():
        return np.full(values.shape, np.nan)
    return values - np.median(values)


class ThermalPath:
    """Thermal cross-check for one replay."""

    def __init__(self, calibration: ThermalCalibration, config: dict[str, Any]):
        self.calibration = calibration
        self.cylinders = config["engine"]["cylinders"]
        self.alerts = {p: AlertState(confirm=calibration.confirm_updates,
                                     clear=calibration.clear_updates) for p in PATTERN_FAULT}
        self.reset()

    def reset(self) -> None:
        for state in self.alerts.values():
            state.reset()
        self.samples = {kind: [] for kind in KINDS}
        self.baseline: dict[str, np.ndarray] | None = None
        self.cylinder: dict[str, int | None] = dict.fromkeys(PATTERN_FAULT)

    def update(self, t: float, telemetry: pd.DataFrame) -> dict[str, Any]:
        cal = self.calibration
        readings = latest_readings(telemetry, t, self.cylinders, cal.max_age_s)
        rel = {kind: relative(readings[kind]) for kind in KINDS}
        quality = "ok" if all(np.isfinite(rel[k]).all() for k in KINDS) else "stale"
        record: dict[str, Any] = {
            "egt_c": _rounded(readings["egt"]), "cht_c": _rounded(readings["cht"]),
            "quality": quality, "status": "warming_up", "patterns": {}, "alerts": [],
        }
        if quality != "ok":
            record["status"] = "unavailable" if all(
                not np.isfinite(readings[k]).any() for k in KINDS) else "stale"
            return record
        if self.baseline is None:
            for kind in KINDS:
                self.samples[kind].append(rel[kind])
            if t >= cal.baseline_s:
                self.baseline = {k: np.median(np.vstack(self.samples[k]), axis=0) for k in KINDS}
            else:
                return record
        z = {k: (rel[k] - self.baseline[k]) / cal.spread_c[k] for k in KINDS}
        cold, hot = cal.thresholds["cold"], cal.thresholds["hot"]
        not_hot = cal.thresholds["not_hot"]
        candidates = {
            "cold": (z["egt"] <= -cold) & (z["cht"] <= -cold / 2),
            "head_hot": (z["cht"] >= hot) & (z["egt"] < not_hot),
            "exhaust_hot": (z["egt"] >= hot) & (z["cht"] < not_hot),
        }
        strength = {"cold": -z["egt"], "head_hot": z["cht"], "exhaust_hot": z["egt"]}
        for pattern, hit in candidates.items():
            decision = "fault" if hit.any() else "normal"
            cylinder = int(np.argmax(np.where(hit, strength[pattern], -np.inf))) + 1 \
                if hit.any() else None
            active = self.alerts[pattern].update(decision)
            if active and cylinder is not None:
                self.cylinder[pattern] = cylinder
            elif not active:
                self.cylinder[pattern] = None
            record["patterns"][pattern] = {
                "fault": PATTERN_FAULT[pattern], "candidate": cylinder,
                "alert": bool(active), "cylinder": self.cylinder[pattern] if active else None}
            if active:
                record["alerts"].append(PATTERN_FAULT[pattern])
        record.update(status="alert" if record["alerts"] else "normal",
                      egt_z=_rounded(z["egt"], 2), cht_z=_rounded(z["cht"], 2))
        return record


def _rounded(values: np.ndarray, digits: int = 1) -> list[float | None]:
    return [None if not np.isfinite(v) else round(float(v), digits) for v in values]
