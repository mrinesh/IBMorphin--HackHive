"""Crankshaft speed from the 60-2 trigger wheel (data contract 1.2).

A production ECU detects misfire from crankshaft speed: each power stroke
accelerates the crank, and a cylinder that does not fire lets it slow down
through its expansion stroke. This module turns tooth edge times into that
measurement, using only edges and cam pulses inside the decision window:

  decode      the long period across the missing teeth marks the tooth after
              the gap (first_tooth_angle_deg after cylinder 1 firing TDC, mod
              360); every other period spans one 6-degree pitch. The cam pulse
              (cam_sync_angle_deg mod 720) then picks which of the two
              revolutions of the cycle each edge belongs to.
  segments    the time to turn through a 90-degree segment centred on each
              firing TDC gives that stroke's speed omega_k. Centring on TDC
              makes the second-order reciprocating-inertia ripple identical
              for every segment, so it cancels between strokes.
  metric      drop_k = omega_k^2 - omega_{k+1}^2 (rad^2/s^2) is the kinetic
              energy lost across stroke k's expansion; a misfire's lost
              combustion energy gives about the same value at any speed. A
              running median over 5 neighbouring strokes is subtracted, so
              speed ramps and steady load changes cancel.
  score       the detrended drop divided by a robust spread (1.4826 x MAD over
              the window's strokes): a dimensionless roughness for the
              independent crank-speed misfire path.

The stroke attributed to cylinder firing_order[p] has its firing TDC at
p * 180 degrees (mod 720). Without a cam pulse the wheel alone cannot tell
cylinder 1 from 4 (or 2 from 3), and the phase is reported as wheel_only.
This module never reads evaluation truth; it is an inference-side feature
path like pipeline.features.
"""

from dataclasses import dataclass, field

import numpy as np

from engine_diagnostics.data.run_io import wheel_gaps

PHASE_CAM, PHASE_WHEEL, PHASE_NONE = "cam_sync", "wheel_only", "none"
SEGMENT_DEG = 90.0
DETREND_STROKES = 5
MAD_TO_SIGMA = 1.4826
MIN_SPREAD = 1.0  # rad^2/s^2; floors the robust spread of a perfectly smooth window
FLAG_SCORE = 6.0  # default per-stroke roughness flag (calibrated value lives in the model)
SPEED_TRACE_POINTS = 200


@dataclass
class WheelAngles:
    """Unwrapped crank angle (degrees after cylinder 1 firing TDC) at each edge.

    With phase wheel_only the angle is only known mod 360 (cylinder pairs share
    a revolution position); with cam_sync it is absolute mod 720.
    """

    edges_s: np.ndarray
    angle_deg: np.ndarray
    phase: str


@dataclass
class StrokeSpeeds:
    """Per-stroke crank speed in one window, ordered by time.

    time_s is each stroke's firing TDC time, position its firing-order
    position (0..cylinders-1, or position mod 2 with wheel_only phase),
    cylinder the cylinder number (-1 when unphased), rpm the 90-degree
    segment speed, drop the detrended kinetic-energy drop across the stroke's
    expansion (NaN when the next segment is outside the window), and score
    the drop over the window's robust spread.
    """

    time_s: np.ndarray
    position: np.ndarray
    cylinder: np.ndarray
    rpm: np.ndarray
    drop: np.ndarray
    score: np.ndarray
    phase: str
    spread: float = float("nan")
    speed_trace: dict = field(default_factory=dict)


def decode_teeth(edges_s: np.ndarray, wheel: dict, pulses_s: np.ndarray | None,
                 cam_angle_deg: float) -> WheelAngles | None:
    """Crank angle of every edge after the first gap; None when no gap is seen.

    Edges between two gaps are counted one pitch apart; each gap starts a new
    revolution at first_tooth_angle_deg, so a lost tooth only misplaces edges
    until the next gap. Edges before the first gap are counted backwards from it.
    """
    edges = np.asarray(edges_s, dtype=np.float64)
    gaps = wheel_gaps(edges)
    if gaps.size == 0:
        return None
    pitch = 360.0 / wheel["teeth_per_rev"]
    first = float(wheel["first_tooth_angle_deg"])
    starts = gaps + 1  # index of each revolution's first tooth
    angle = np.full(edges.size, np.nan)
    for r, start in enumerate(starts):
        stop = starts[r + 1] if r + 1 < starts.size else edges.size
        count = min(stop - start, wheel["teeth_per_rev"] - wheel["missing_teeth"])
        angle[start:start + count] = first + 360.0 * r + pitch * np.arange(count)
    before = np.arange(starts[0])
    angle[before] = first - pitch * (wheel["missing_teeth"] + starts[0] - before)
    known = np.isfinite(angle)
    edges, angle = edges[known], angle[known]

    phase = PHASE_WHEEL
    if pulses_s is not None and len(pulses_s):
        pulses = np.asarray(pulses_s, dtype=np.float64)
        inside = pulses[(pulses >= edges[0]) & (pulses <= edges[-1])]
        if inside.size:
            # The cam pulse sits at cam_angle (mod 720): shift by whole revolutions.
            at_pulse = float(np.interp(inside[0], edges, angle))
            angle = angle + 360.0 * np.round((cam_angle_deg - at_pulse) / 360.0)
            phase = PHASE_CAM
    return WheelAngles(edges, angle, phase)


def crank_angle_from_teeth(times_s: np.ndarray, wheel_angles: WheelAngles) -> np.ndarray:
    """Crank angle at arbitrary times inside the decoded edges (NaN outside)."""
    times = np.asarray(times_s, dtype=np.float64)
    edges, angle = wheel_angles.edges_s, wheel_angles.angle_deg
    out = np.interp(times, edges, angle)
    out[(times < edges[0]) | (times > edges[-1])] = np.nan
    return out


def stroke_speeds(edges_s: np.ndarray, pulses_s: np.ndarray | None, config: dict) -> StrokeSpeeds:
    """Segment speeds, kinetic-energy drops and roughness scores of a window's strokes."""
    engine, wheel = config["engine"], config["streams"]["crank_wheel"]
    cylinders = engine["cylinders"]
    spacing = 720.0 / cylinders
    empty = StrokeSpeeds(*(np.zeros(0) for _ in range(6)), phase=PHASE_NONE)
    edges = np.asarray(edges_s, dtype=np.float64)
    if edges.size < 2 * wheel["teeth_per_rev"]:
        return empty
    decoded = decode_teeth(edges, wheel, pulses_s, engine["cam_sync_angle_deg"])
    if decoded is None:
        return empty
    angle, times = decoded.angle_deg, decoded.edges_s
    half = SEGMENT_DEG / 2
    tdc = np.arange(np.ceil((angle[0] + half) / spacing) * spacing,
                    angle[-1] - half + 1e-9, spacing)
    if tdc.size < 2:
        return empty
    segment_s = np.interp(tdc + half, angle, times) - np.interp(tdc - half, angle, times)
    omega = np.deg2rad(SEGMENT_DEG) / segment_s
    raw = np.full(tdc.size, np.nan)
    raw[:-1] = omega[:-1] ** 2 - omega[1:] ** 2
    drop = raw - running_median(raw, DETREND_STROKES)
    finite = np.isfinite(drop)
    spread = MIN_SPREAD
    if finite.sum() >= DETREND_STROKES:
        mad = np.median(np.abs(drop[finite] - np.median(drop[finite])))
        spread = max(MAD_TO_SIGMA * mad, MIN_SPREAD)
    position = (np.round(np.mod(tdc, 720.0) / spacing).astype(int)) % cylinders
    cylinder = np.full(tdc.size, -1)
    if decoded.phase == PHASE_CAM:
        cylinder = np.asarray(engine["firing_order"])[position]
    else:
        position = position % (cylinders // 2)
    return StrokeSpeeds(
        time_s=np.interp(tdc, angle, times), position=position, cylinder=cylinder,
        rpm=SEGMENT_DEG / 6.0 / segment_s, drop=drop, score=drop / spread,
        phase=decoded.phase, spread=spread,
        speed_trace=speed_trace(times, 360.0 / wheel["teeth_per_rev"]))


def running_median(x: np.ndarray, width: int) -> np.ndarray:
    """Centred running median ignoring NaN (shorter at the ends)."""
    x = np.asarray(x, dtype=np.float64)
    half = width // 2
    out = np.full(x.size, np.nan)
    for k in range(x.size):
        part = x[max(0, k - half):k + half + 1]
        part = part[np.isfinite(part)]
        if part.size:
            out[k] = np.median(part)
    return out


def speed_trace(edges_s: np.ndarray, pitch_deg: float, seconds: float = 0.5,
                points: int = SPEED_TRACE_POINTS) -> dict:
    """Instantaneous speed (rpm per tooth period) over the last `seconds`, decimated.

    Gap periods are dropped. For display only (the dashboard's crank panel).
    """
    edges = np.asarray(edges_s, dtype=np.float64)
    if edges.size < 3:
        return {"t_s": [], "rpm": []}
    period = np.diff(edges)
    keep = np.ones(period.size, dtype=bool)
    keep[wheel_gaps(edges)] = False
    mid = (edges[:-1] + edges[1:]) / 2
    recent = keep & (mid >= edges[-1] - seconds)
    t, rpm = mid[recent], pitch_deg / 6.0 / period[recent]  # degrees per second / 6
    if t.size > points:
        index = np.linspace(0, t.size - 1, points).round().astype(int)
        t, rpm = t[index], rpm[index]
    return {"t_s": t.round(5).tolist(), "rpm": rpm.round(1).tolist()}


def crank_feature_names(cylinders: int = 4) -> list[str]:
    """Window crank features: per-cylinder roughness and flagged-stroke fraction."""
    per = [f"crank_cyl{c}_{name}" for c in range(1, cylinders + 1)
           for name in ("score_max", "score_mean", "flagged")]
    return per + ["crank_score_max", "crank_flagged", "crank_rpm", "crank_spread"]


def crank_window_features(strokes: StrokeSpeeds, cylinders: int = 4,
                          flag_score: float = FLAG_SCORE) -> dict[str, float]:
    """Per-cylinder roughness summary of one window (NaN without a cam-phased wheel)."""
    names = crank_feature_names(cylinders)
    out = dict.fromkeys(names, float("nan"))
    finite = np.isfinite(strokes.score)
    if strokes.phase != PHASE_CAM or finite.sum() < cylinders:
        return out
    score, cyl = strokes.score[finite], strokes.cylinder[finite]
    for c in range(1, cylinders + 1):
        mine = score[cyl == c]
        if mine.size:
            out[f"crank_cyl{c}_score_max"] = float(mine.max())
            out[f"crank_cyl{c}_score_mean"] = float(mine.mean())
            out[f"crank_cyl{c}_flagged"] = float(np.mean(mine > flag_score))
    out["crank_score_max"] = float(score.max())
    out["crank_flagged"] = float(np.mean(score > flag_score))
    out["crank_rpm"] = float(np.median(strokes.rpm))
    out["crank_spread"] = float(strokes.spread)
    return out
