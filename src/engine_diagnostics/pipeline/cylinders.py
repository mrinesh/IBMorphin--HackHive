"""Cylinder attribution of stroke-synchronous vibration signatures.

One accelerometer hears every cylinder through the same path, so cylinders
differ only in *when* they fire. Each 180-degree firing interval found by the
stroke-synchronous features is attributed to a position in the firing order:

  cam_sync   with the cam phase reference (contract 1.1), the crank angle of
             each stroke onset is interpolated between the cam pulses in the
             window (720 degrees per pulse, at cam_sync_angle_deg after
             cylinder 1's firing TDC). Position p fires cylinder
             firing_order[p], so the attribution is absolute.
  relative   without a reference stream (contract 1.0 data), strokes are
             numbered from the first stroke in the window. Positions are then
             a rotation of the true ones; runtime.localization aligns them
             across windows into run-stable slots, but cannot name cylinders.

For each position, the mean stroke energy per crank-angle sector is computed in
three bands: combustion (300-1000 Hz, weak or missing for misfire), mid
(2-5 kHz, valve impacts), and high (5-9 kHz, knock ringing). The signature is
each position's log ratio to the median over positions, so engine-wide level
changes (load, sensor gain) cancel and only one-cylinder differences remain.
A valve fault on cylinder c appears in later positions (its valves close while
other cylinders fire), which the localizer learns from the full rotated vector.

Like every window feature, the signature uses only waveform samples and cam
pulses inside [t - window_s, t). It is not part of the detector feature vector.
"""

from dataclasses import dataclass

import numpy as np

from engine_diagnostics.data.run_io import cam_cycle_steps

CYLINDER_FEATURE_VERSION = "1.0.0"
PHASE_CAM, PHASE_RELATIVE, PHASE_NONE = "cam_sync", "relative", "none"
COMBUSTION_HZ = (300, 1000)  # same band as the stroke reference (features.SYNC_REFERENCE_HZ)
SIGNATURE_BANDS_HZ = {"comb": COMBUSTION_HZ, "mid": (2000, 5000), "high": (5000, 9000)}
SIGNATURE_SECTORS_DEG = ((0, 15), (15, 30), (30, 45), (45, 90), (90, 150), (150, 180))
MIN_STROKES_PER_POSITION = 2
CAM_EXTRAPOLATE_CYCLES = 1.0  # strokes further than this outside the pulses are unphased
CONTRAST_FLOOR = 1e-3     # relative floor in the log ratio; bounds empty sectors


@dataclass
class CylinderSignature:
    """Per firing-order position signature of one decision window.

    contrast[p] is the log ratio of position p's sector energies to the median
    over positions, in signature_names() order. With phase cam_sync, position p
    is cylinder firing_order[p]; with relative, position 0 is the window's
    first stroke. stroke_times_s are absolute onset times of the attributed
    strokes and stroke_positions their positions (used to align relative
    positions across windows).
    """

    contrast: np.ndarray
    strokes: np.ndarray
    phase: str
    stroke_times_s: np.ndarray
    stroke_positions: np.ndarray


def signature_names() -> list[str]:
    """Names of one position's signature values (band-major, then sector)."""
    return [f"{band}_{a}_{b}" for band in SIGNATURE_BANDS_HZ for a, b in SIGNATURE_SECTORS_DEG]


def cylinder_columns(engine: dict) -> list[str]:
    """Feature-table columns: each cylinder's signature, by cylinder number."""
    return [f"cyl{c}_{name}" for c in range(1, engine["cylinders"] + 1)
            for name in signature_names()]


def window_pulses(cam_sync_s: np.ndarray | None, t_start_s: float,
                  t_end_s: float) -> np.ndarray | None:
    """Cam pulses inside [t_start, t_end); None when the run has no reference stream."""
    if cam_sync_s is None:
        return None
    pulses = np.asarray(cam_sync_s, dtype=np.float64)
    return pulses[(pulses >= t_start_s) & (pulses < t_end_s)]


def crank_angle(times_s: np.ndarray, pulses_s: np.ndarray, angle_deg: float,
                rpm: float) -> np.ndarray:
    """Crank angle (degrees after cylinder 1 firing TDC, unwrapped) at each time.

    Pulse k is assigned cycle n_k, counting a gap of several local cycle
    periods as missing pulses (data.run_io.cam_cycle_steps), and sits at
    720 n_k + angle_deg. Angles are interpolated
    linearly between pulses and extrapolated with the nearest cycle's speed
    (RPM sets the speed when there is a single pulse). Times further than
    CAM_EXTRAPOLATE_CYCLES outside the pulses get NaN.
    """
    times = np.asarray(times_s, dtype=np.float64)
    pulses = np.asarray(pulses_s, dtype=np.float64)
    if pulses.size == 0:
        return np.full(times.shape, np.nan)
    if pulses.size == 1:
        if not np.isfinite(rpm) or rpm <= 0:
            return np.full(times.shape, np.nan)
        period = 120.0 / rpm
        cycles = np.zeros(1)
    else:
        steps = cam_cycle_steps(np.diff(pulses))
        cycles = np.concatenate([[0.0], np.cumsum(steps)])
        period = float(np.median(np.diff(pulses) / steps))
    angles = 720.0 * cycles + angle_deg
    out = np.interp(times, pulses, angles)
    if pulses.size == 1:
        slope_first = slope_last = 720.0 / period
    else:
        slope_first = (angles[1] - angles[0]) / (pulses[1] - pulses[0])
        slope_last = (angles[-1] - angles[-2]) / (pulses[-1] - pulses[-2])
    before, after = times < pulses[0], times > pulses[-1]
    out[before] = angles[0] - slope_first * (pulses[0] - times[before])
    out[after] = angles[-1] + slope_last * (times[after] - pulses[-1])
    limit = CAM_EXTRAPOLATE_CYCLES * period
    out[(pulses[0] - times > limit) | (times - pulses[-1] > limit)] = np.nan
    return out


def firing_positions(times_s: np.ndarray, pulses_s: np.ndarray, angle_deg: float, rpm: float,
                     cylinders: int) -> np.ndarray:
    """Firing-order position (0..cylinders-1) of each stroke onset; -1 if unphased.

    Position p's TDC is at p * 720 / cylinders degrees. Onsets come shortly
    after TDC (combustion offset plus sensor delay), far less than half the
    spacing, so the nearest TDC identifies the position.
    """
    spacing = 720.0 / cylinders
    angle = crank_angle(times_s, pulses_s, angle_deg, rpm)
    positions = np.full(angle.shape, -1, dtype=int)
    known = np.isfinite(angle)
    positions[known] = np.round(np.mod(angle[known], 720.0) / spacing).astype(int) % cylinders
    return positions


def cylinder_signature(stroke_energy: dict[str, np.ndarray], onsets_s: np.ndarray,
                       t_start_s: float, pulses_s: np.ndarray | None, rpm: float,
                       engine: dict) -> CylinderSignature | None:
    """Signature from per-stroke energy matrices, or None if it cannot be formed.

    stroke_energy[band] is (strokes, bins) energy over each firing interval,
    resampled to equal crank-angle bins; onsets_s are the interval start times
    relative to the window start. pulses_s None means no reference stream
    (relative positions); an empty array means the stream lost every pulse in
    this window, so no signature is formed.
    """
    cylinders = engine["cylinders"]
    n = len(onsets_s)
    times = t_start_s + np.asarray(onsets_s, dtype=np.float64)
    if pulses_s is None:
        phase, positions = PHASE_RELATIVE, np.arange(n) % cylinders
    else:
        phase = PHASE_CAM
        positions = firing_positions(times, pulses_s, engine["cam_sync_angle_deg"], rpm,
                                     cylinders)
    if n == 0 or not (positions >= 0).any():
        return None
    first = next(iter(stroke_energy.values()))
    bins = first.shape[1]
    degrees = np.arange(bins) * 180 / bins
    sectors = []
    for band in SIGNATURE_BANDS_HZ:
        energy = stroke_energy[band]
        for a, b in SIGNATURE_SECTORS_DEG:
            sectors.append(energy[:, (degrees >= a) & (degrees < b)].sum(axis=1))
    per_stroke = np.column_stack(sectors)  # (strokes, bands * sectors)
    counts = np.array([np.sum(positions == p) for p in range(cylinders)])
    if counts.min() < MIN_STROKES_PER_POSITION:
        return None
    means = np.array([per_stroke[positions == p].mean(axis=0) for p in range(cylinders)])
    reference = np.median(means, axis=0)
    floor = CONTRAST_FLOOR * np.maximum(reference, 1e-30)
    contrast = np.log((means + floor) / (reference + floor))
    known = positions >= 0
    return CylinderSignature(contrast=contrast, strokes=counts, phase=phase,
                             stroke_times_s=times[known], stroke_positions=positions[known])


def by_cylinder(signature: CylinderSignature, engine: dict) -> dict[str, float]:
    """Feature-table values (cylinder_columns order) for a cam-phased signature."""
    names = signature_names()
    values = {}
    for p, cylinder in enumerate(engine["firing_order"]):
        values.update({f"cyl{cylinder}_{n}": float(v)
                       for n, v in zip(names, signature.contrast[p], strict=True)})
    return {c: values[c] for c in cylinder_columns(engine)}


def contrast_from_row(row, engine: dict) -> np.ndarray:
    """Rebuild the (positions, values) contrast of a feature-table row (cam phase)."""
    names = signature_names()
    return np.array([[row[f"cyl{c}_{n}"] for n in names] for c in engine["firing_order"]],
                    dtype=np.float64)
