"""Independent crankshaft-speed misfire path and the simulated ECU monitor.

The acoustic/vibration detector is one way to see a misfire. The crank wheel
(data contract 1.2) is a second, physically separate one: a cylinder that does
not fire lets the crank slow through its expansion stroke. CrankPath runs at
every decision time on the same causal window as the acoustic features:

  detector   each cylinder's mean roughness score in the window minus the
             median over cylinders (its excess). A cylinder whose excess
             passes the calibrated threshold is a misfire candidate; the
             candidate is confirmed and cleared like any alert (AlertState).
             An excess between half the threshold and the threshold holds
             the current state (hysteresis), so an intermittent misfire near
             the threshold does not flicker on and off.
             The cylinder is named from crank angle alone, independently of
             the acoustic localizer.
  ECU        every stroke is passed once, in time order, to the simulated
             OBD-II monitor (models.obd_misfire), whose 200/1000-revolution
             blocks set P030N codes: the benchmark for detection time.

Strokes are final once the wheel has passed the next stroke and the running
median's neighbours: a stroke is processed at the first decision time that is
at least crank_detector.stroke_lag_s after it. Nothing here reads truth.
"""

from typing import Any

import numpy as np

from engine_diagnostics.models.obd_misfire import CrankCalibration, ObdMisfireMonitor
from engine_diagnostics.pipeline.crankspeed import (
    PHASE_CAM,
    StrokeSpeeds,
    crank_window_features,
    stroke_speeds,
)
from engine_diagnostics.pipeline.cylinders import window_pulses
from engine_diagnostics.runtime.alerts import AlertState

MISFIRE = "misfire"


def cylinder_excess(strokes: StrokeSpeeds, cylinders: int) -> np.ndarray:
    """Each cylinder's mean roughness score minus the median over cylinders (NaN if unknown)."""
    out = np.full(cylinders, np.nan)
    if strokes.phase != PHASE_CAM:
        return out
    finite = np.isfinite(strokes.score)
    for c in range(1, cylinders + 1):
        mine = strokes.score[finite & (strokes.cylinder == c)]
        if mine.size:
            out[c - 1] = mine.mean()
    if np.isfinite(out).sum() == cylinders:
        out = out - np.median(out)
    else:
        out[:] = np.nan
    return out


class CrankPath:
    """Crank-speed misfire detector plus OBD monitor for one replay."""

    def __init__(self, calibration: CrankCalibration, config: dict[str, Any]):
        self.calibration = calibration
        self.config = config
        crank = config["crank_detector"]
        self.lag_s = crank["stroke_lag_s"]
        self.cylinders = config["engine"]["cylinders"]
        self.alert = AlertState(confirm=crank["confirm_updates"], clear=crank["clear_updates"])
        self.obd = ObdMisfireMonitor(calibration, config)
        self.reset()

    def reset(self) -> None:
        self.alert.reset()
        self.obd.reset()
        self.done_until = -np.inf  # strokes before this time were already passed to the ECU
        self.cylinder: int | None = None

    def update(self, t: float, edges_s: np.ndarray | None,
               pulses_s: np.ndarray | None) -> dict[str, Any]:
        """Decision-time record for the window [t - window_s, t)."""
        window_s = self.config["analysis"]["window_s"]
        if edges_s is None:
            return {"status": "unavailable"}
        edges = window_pulses(edges_s, t - window_s, t)
        pulses = window_pulses(pulses_s, t - window_s, t)
        strokes = stroke_speeds(edges, pulses, self.config)
        excess = cylinder_excess(strokes, self.cylinders)
        quality = "ok" if np.isfinite(excess).all() else "no_phase"
        decision = "normal"
        candidate = None
        if quality == "ok":
            candidate = int(np.argmax(excess)) + 1
            threshold = self.calibration.cylinder_threshold
            if excess[candidate - 1] > threshold:
                decision = MISFIRE
            elif excess[candidate - 1] > threshold / 2:
                decision = "uncertain"  # holds the state: neither confirms nor clears
        active = self.alert.update(decision, quality)
        if active and decision == MISFIRE:
            self.cylinder = candidate
        elif not active:
            self.cylinder = None

        final = (strokes.time_s >= self.done_until) & (strokes.time_s < t - self.lag_s)
        new_dtcs = self.obd.update(strokes.time_s[final], strokes.cylinder[final],
                                   strokes.rpm[final], strokes.drop[final])
        if final.any():
            self.done_until = t - self.lag_s
        flagged = (np.isfinite(strokes.drop)
                   & (strokes.drop > self.calibration.stroke_threshold(strokes.rpm))
                   & (strokes.cylinder > 0))
        return {
            "status": "alert" if active else (
                "candidate" if decision == MISFIRE else
                "elevated" if decision == "uncertain" else "normal"),
            "quality": quality,
            "phase": strokes.phase,
            "alert": active == MISFIRE,
            "cylinder": self.cylinder if active else None,
            "candidate_cylinder": candidate if decision == MISFIRE else None,
            "excess": [None if not np.isfinite(v) else round(float(v), 3) for v in excess],
            "threshold": self.calibration.cylinder_threshold,
            "flagged_strokes": [int(np.sum(flagged & (strokes.cylinder == c)))
                                for c in range(1, self.cylinders + 1)],
            "rpm": float(np.median(strokes.rpm)) if strokes.rpm.size else None,
            "features": crank_window_features(strokes, self.cylinders),
            "speed_trace": strokes.speed_trace,
            "stroke_marks": stroke_marks(strokes, t, flagged),
            "obd": self.obd.snapshot(),
            "obd_new": new_dtcs,
        }


def stroke_marks(strokes: StrokeSpeeds, t: float, flagged: np.ndarray,
                 seconds: float = 0.5) -> dict[str, list]:
    """TDC times, cylinders and 90-degree segment speeds over the trace span, for display."""
    recent = strokes.time_s >= t - seconds
    return {"t_s": strokes.time_s[recent].round(5).tolist(),
            "cylinder": strokes.cylinder[recent].tolist(),
            "rpm": strokes.rpm[recent].round(1).tolist(),
            "flagged": flagged[recent].tolist()}
