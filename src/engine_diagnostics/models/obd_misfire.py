"""Simulated OBD-II misfire monitor: the benchmark an ECU would give.

Modelled on the standard crankshaft-speed misfire monitor of production
engine controllers (as required by OBD-II):

  per stroke   a stroke is a misfire when its detrended kinetic-energy drop
               (pipeline.crankspeed, rad^2/s^2) exceeds a threshold from an
               engine-speed table. The table is calibrated on healthy
               training runs: a high quantile of healthy drops per speed bin
               times a margin, like an ECU calibrator's no-false-detection map.
  type A       misfires are counted over consecutive 200-revolution blocks;
               a block whose misfire rate reaches the catalyst-damage rate
               sets a pending DTC (a flashing MIL on a real vehicle).
  type B       the same over 1000-revolution blocks at the lower emissions
               rate; a real vehicle confirms it on a second drive cycle
               before lighting the MIL (two-trip logic, not simulated here).
  DTC          P030N when one cylinder holds at least single_cylinder_share
               of the block's misfires, otherwise P0300 (random/multiple).

Blocks start when the monitor starts (t = 0 of the recording), as on a real
ECU after key-on; the monitor has no knock or valve capability. Rates and
block lengths come from configs/mvp.json obd_misfire. The catalyst-damage rate
there (5 % of strokes) is at the strict end of real calibrations, which
favours the ECU in the comparison. This module reads no evaluation truth.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CRANK_MODEL_VERSION = "crank-1.0.0"


@dataclass
class CrankCalibration:
    """Thresholds learned from healthy training runs (artifacts/models/crank_monitor.json)."""

    rpm_bin_edges: list[float]
    stroke_thresholds: list[float]  # rad^2/s^2 per rpm bin (OBD per-stroke decision)
    cylinder_threshold: float  # crank-path per-cylinder window roughness excess
    model_version: str = CRANK_MODEL_VERSION
    training: dict[str, Any] = field(default_factory=dict)

    def stroke_threshold(self, rpm: np.ndarray) -> np.ndarray:
        edges = np.asarray(self.rpm_bin_edges, dtype=np.float64)
        index = np.clip(np.searchsorted(edges, np.asarray(rpm, dtype=np.float64)) - 1,
                        0, len(self.stroke_thresholds) - 1)
        return np.asarray(self.stroke_thresholds)[index]

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CrankCalibration":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def calibrate(healthy_strokes: list[dict[str, np.ndarray]], cylinder_stats: np.ndarray,
              config: dict[str, Any], training: dict[str, Any]) -> CrankCalibration:
    """Thresholds from healthy strokes ({rpm, drop}) and per-cylinder window statistics.

    Each rpm bin uses quantile x margin of its healthy drops; a bin with too few
    strokes borrows the nearest populated bin, scaled by (rpm ratio)^2 because
    timing noise in omega^2 grows with speed squared.
    """
    obd, crank = config["obd_misfire"], config["crank_detector"]
    edges = np.asarray(obd["rpm_bin_edges"], dtype=np.float64)
    rpm = np.concatenate([s["rpm"] for s in healthy_strokes])
    drop = np.concatenate([s["drop"] for s in healthy_strokes])
    keep = np.isfinite(rpm) & np.isfinite(drop)
    rpm, drop = rpm[keep], drop[keep]
    centres = (edges[:-1] + edges[1:]) / 2
    thresholds = np.full(centres.size, np.nan)
    counts = np.zeros(centres.size, dtype=int)
    for k in range(centres.size):
        mine = drop[(rpm >= edges[k]) & (rpm < edges[k + 1])]
        counts[k] = mine.size
        if mine.size >= obd["min_strokes_per_bin"]:
            thresholds[k] = obd["threshold_margin"] * np.quantile(
                mine, obd["stroke_threshold_quantile"])
    populated = np.flatnonzero(np.isfinite(thresholds))
    if populated.size == 0:
        raise ValueError("no rpm bin has enough healthy strokes to calibrate the OBD monitor")
    for k in np.flatnonzero(~np.isfinite(thresholds)):
        near = populated[np.argmin(np.abs(populated - k))]
        thresholds[k] = thresholds[near] * (centres[k] / centres[near]) ** 2
    stats = np.asarray(cylinder_stats, dtype=np.float64)
    stats = stats[np.isfinite(stats)]
    cylinder_threshold = crank["threshold_margin"] * float(
        np.quantile(stats, crank["cylinder_threshold_quantile"]))
    training = {**training, "strokes_per_bin": counts.tolist(), "healthy_strokes": int(drop.size),
                "healthy_windows": int(stats.size)}
    return CrankCalibration(edges.tolist(), thresholds.round(3).tolist(),
                            round(cylinder_threshold, 4), training=training)


@dataclass
class ObdMisfireMonitor:
    """Streaming type A / type B misfire counter with pending DTCs."""

    calibration: CrankCalibration
    config: dict[str, Any]

    def __post_init__(self):
        obd = self.config["obd_misfire"]
        cylinders = self.config["engine"]["cylinders"]
        strokes_per_rev = cylinders / (self.config["engine"]["strokes"] / 2)
        self.cylinders = cylinders
        self.blocks = {
            "A": {"strokes": round(obd["type_a_revs"] * strokes_per_rev),
                  "rate": obd["type_a_rate_pct"] / 100, "revs": obd["type_a_revs"]},
            "B": {"strokes": round(obd["type_b_revs"] * strokes_per_rev),
                  "rate": obd["type_b_rate_pct"] / 100, "revs": obd["type_b_revs"]},
        }
        self.share = obd["single_cylinder_share"]
        self.reset()

    def reset(self) -> None:
        self.count = {kind: np.zeros(self.cylinders, dtype=int) for kind in self.blocks}
        self.seen = dict.fromkeys(self.blocks, 0)
        self.total = np.zeros(self.cylinders, dtype=int)
        self.strokes = 0
        self.dtcs: list[dict[str, Any]] = []

    def update(self, time_s: np.ndarray, cylinder: np.ndarray, rpm: np.ndarray,
               drop: np.ndarray) -> list[dict[str, Any]]:
        """Feed strokes in time order; returns DTCs set by blocks completed now.

        Strokes without a cylinder (no cam phase) or a finite drop still count
        toward the block length, as an ECU's revolution counter would, but can
        never be misfires.
        """
        new = []
        threshold = self.calibration.stroke_threshold(rpm)
        misfire = np.isfinite(drop) & (np.asarray(drop) > threshold) & (np.asarray(cylinder) > 0)
        for t, c, hit in zip(time_s, cylinder, misfire, strict=True):
            self.strokes += 1
            if hit:
                self.total[c - 1] += 1
            for kind, block in self.blocks.items():
                if hit:
                    self.count[kind][c - 1] += 1
                self.seen[kind] += 1
                if self.seen[kind] == block["strokes"]:
                    dtc = self._close_block(kind, float(t))
                    if dtc:
                        new.append(dtc)
                        self.dtcs.append(dtc)
                    self.count[kind][:] = 0
                    self.seen[kind] = 0
        return new

    def _close_block(self, kind: str, t: float) -> dict[str, Any] | None:
        block, count = self.blocks[kind], self.count[kind]
        rate = count.sum() / block["strokes"]
        if rate < block["rate"]:
            return None
        worst = int(np.argmax(count))
        single = count[worst] >= self.share * count.sum()
        return {"type": kind, "t_s": round(t, 3), "rate_pct": round(100 * rate, 2),
                "code": f"P030{worst + 1}" if single else "P0300",
                "cylinder": worst + 1 if single else None,
                "misfires": count.tolist()}

    def first_dtc(self, kind: str | None = None) -> dict[str, Any] | None:
        for dtc in self.dtcs:
            if kind is None or dtc["type"] == kind:
                return dtc
        return None

    def snapshot(self) -> dict[str, Any]:
        """Counters for display: current block progress and misfires so far."""
        return {
            "strokes": self.strokes,
            "misfires_total": self.total.tolist(),
            "blocks": {kind: {"progress": round(self.seen[kind] / b["strokes"], 3),
                              "revs": b["revs"], "rate_limit_pct": 100 * b["rate"],
                              "misfires": self.count[kind].tolist()}
                       for kind, b in self.blocks.items()},
            "dtcs": list(self.dtcs),
        }
