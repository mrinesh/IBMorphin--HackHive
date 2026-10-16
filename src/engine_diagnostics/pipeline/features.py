"""Window features shared by offline training and replay inference.

Every feature uses only the trailing window [t - window_s, t) of waveform
samples and telemetry readings strictly before t. Spectral filtering is done
per window in the frequency domain, so no filter state carries between
windows and offline extraction equals replay extraction for the same window.
Labels, run/engine IDs, and simulation truth are never inputs.

Two named sets are produced for every window:
  feature_names()       base set, FEATURE_VERSION 0.1.0 (frozen; v1, temporary)
  all_feature_names()   base set plus stroke-synchronous vibration features,
                        SYNC_FEATURE_VERSION 0.3.0 (v2 multi-fault model)
Models record the names they use; replay selects those columns by name.

Each window also carries a per-cylinder signature (pipeline.cylinders,
CYLINDER_FEATURE_VERSION) for fault localization. It is kept outside the
detector vector so the frozen detector feature versions are unchanged; the
feature table stores it as cyl<N>_* columns plus cyl_phase.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert, welch
from scipy.stats import kurtosis

from engine_diagnostics.pipeline.cylinders import (
    COMBUSTION_HZ,
    PHASE_CAM,
    PHASE_NONE,
    SIGNATURE_BANDS_HZ,
    CylinderSignature,
    by_cylinder,
    cylinder_columns,
    cylinder_signature,
    window_pulses,
)

FEATURE_VERSION = "0.1.0"
SYNC_FEATURE_VERSION = "0.3.0"  # 0.2.0 was a rejected v1 candidate set
MODALITIES = ("audio", "vibration")
BAND_EDGES_HZ = (500, 1500, 3000, 5000, 7000)  # between each modality's low edge and high_hz
HIGH_ENVELOPE_HZ = (5000, 10000)
MID_ENVELOPE_HZ = (500, 3000)
PER_MODALITY = (
    ["log_rms", "crest", "kurtosis", "log_band_power"]
    + [f"band_frac_{k}" for k in range(len(BAND_EDGES_HZ) + 1)]
    + ["env_cv_high", "env_kurtosis_high", "env_cv_mid", "firing_peak_cv"]
)
CONTEXT = ("rpm", "load", "throttle")
CONTEXT_MAX_AGE_KEYS = {
    "rpm": "rpm_max_age_s", "load": "load_throttle_max_age_s",
    "throttle": "load_throttle_max_age_s",
}


# Stroke-synchronous vibration features. Each 180-degree firing interval is
# referenced to the rising edge of the 300-1000 Hz combustion envelope and
# resampled to SYNC_BINS crank-angle bins (RPM from past telemetry sets the
# expected interval; missing strokes are filled at that period). Strokes repeat
# every four for the same cylinder, so a one-cylinder fault shows up as excess
# energy in one of the four stroke groups; the angle sector and band of that
# excess separate knock (5-9 kHz just after combustion) from valve impacts.
SYNC_REFERENCE_HZ = COMBUSTION_HZ  # (300, 1000)
SYNC_BANDS_HZ = {band: SIGNATURE_BANDS_HZ[band] for band in ("mid", "high")}
SYNC_SECTORS_DEG = ((0, 15), (15, 30), (30, 45), (45, 90), (90, 150), (150, 180))
SYNC_BINS = 90  # 2 crank degrees per bin
SYNC_MIN_STROKES = 8


def feature_names() -> list[str]:
    """Base model input names; the order is part of FEATURE_VERSION (frozen)."""
    names = [f"{m}_{f}" for m in MODALITIES for f in PER_MODALITY]
    return names + [f"ctx_{c}" for c in CONTEXT]


def sync_feature_names() -> list[str]:
    return [f"sync_{band}_cyl_excess_{a}_{b}" for band in SYNC_BANDS_HZ
            for a, b in SYNC_SECTORS_DEG] + ["sync_ref_jitter"]


def all_feature_names() -> list[str]:
    """Every extracted feature, in window vector order (SYNC_FEATURE_VERSION)."""
    return feature_names() + sync_feature_names()


@dataclass
class WindowResult:
    t_end_s: float
    features: np.ndarray
    quality: str  # "ok", "clipped", "missing_waveform", or "stale_telemetry"
    cylinders: CylinderSignature | None = None  # per-cylinder signature for localization


def window_ends(duration_s: float, config: dict[str, Any]) -> np.ndarray:
    """Decision times t with a full trailing window inside [0, duration]."""
    window, step = config["analysis"]["window_s"], config["analysis"]["step_s"]
    count = int(np.floor((duration_s - window) / step + 1e-9)) + 1
    return window + step * np.arange(max(count, 0))


def window_slice(t_end_s: float, fs_hz: float, t0_s: float, config: dict) -> slice:
    """Sample indices covering [t_end - window_s, t_end) for a stream starting at t0."""
    window = config["analysis"]["window_s"]
    start = round((t_end_s - window - t0_s) * fs_hz)
    # A window starting before the stream yields a short slice (missing data), never a wrap.
    return slice(max(start, 0), max(start + round(window * fs_hz), 0))


def telemetry_context(telemetry: pd.DataFrame, t_end_s: float,
                      config: dict) -> tuple[dict[str, float], bool]:
    """Latest valid reading before t for each context field; flags stale values."""
    runtime = config["runtime"]
    values, fresh = {}, True
    for name in CONTEXT:
        rows = telemetry[(telemetry["signal"] == name) & (telemetry["valid"] == 1)
                         & (telemetry["t_s"] < t_end_s)]
        if rows.empty:
            values[name], fresh = float("nan"), False
            continue
        last = rows.iloc[-1]
        values[name] = float(last["value"])
        if t_end_s - last["t_s"] > runtime[CONTEXT_MAX_AGE_KEYS[name]]:
            fresh = False
    return values, fresh


def extract_window(waves: dict[str, np.ndarray], rates: dict[str, float],
                   context: dict[str, float], context_fresh: bool,
                   full_scales: dict[str, float], t_end_s: float, config: dict, *,
                   cam_sync_s: np.ndarray | None = None,
                   vibration_start_s: float | None = None) -> WindowResult:
    """Features for one decision window; `waves` hold exactly the window samples.

    cam_sync_s are the cam pulses inside the window (None: the run has no
    reference stream) and vibration_start_s the time of the first vibration
    sample (default t_end - window_s); both only affect the cylinder signature.
    """
    analysis, runtime = config["analysis"], config["runtime"]
    quality = "ok" if context_fresh else "stale_telemetry"
    parts = []
    sync = np.full(len(sync_feature_names()), np.nan)
    signature = None
    for name in MODALITIES:
        x = np.asarray(waves[name], dtype=np.float64)
        expected = round(analysis["window_s"] * rates[name])
        missing = 1 - np.count_nonzero(np.isfinite(x)) / expected if expected else 1
        if x.size != expected or missing > runtime["max_missing_fraction"]:
            quality = "missing_waveform"
            parts.append(np.full(len(PER_MODALITY), np.nan))
            continue
        x = np.nan_to_num(x)
        if np.mean(np.abs(x) >= full_scales[name]) > runtime["max_clipped_fraction"]:
            quality = "clipped" if quality == "ok" else quality
        low = analysis[f"{name}_low_hz"]
        firing_hz = _firing_rate_hz(context["rpm"], config["engine"])
        parts.append(_modality_features(x, rates[name], low, analysis, firing_hz))
        if name == "vibration":
            fs = rates[name]
            envelopes = {band: _band_envelope(x, fs, edges)
                         for band, edges in SIGNATURE_BANDS_HZ.items()}
            found = _stroke_references(x, fs, context["rpm"], envelopes["comb"])
            sync = _sync_from_strokes(envelopes, fs, found)
            if found is not None:
                refs = found[0]
                start = t_end_s - analysis["window_s"] if vibration_start_s is None \
                    else vibration_start_s
                energy = {band: _stroke_matrix(env, fs, refs) ** 2
                          for band, env in envelopes.items()}
                signature = cylinder_signature(energy, refs[:-1], start, cam_sync_s,
                                               context["rpm"], config["engine"])
    features = np.concatenate(parts + [np.array([context[c] for c in CONTEXT]), sync])
    return WindowResult(t_end_s=float(t_end_s), features=features, quality=quality,
                        cylinders=signature)


def extract_run(bundle, config: dict) -> pd.DataFrame:
    """Offline features for every decision window of a loaded RunBundle."""
    duration = bundle.metadata["duration_s"]
    rates = {m: getattr(bundle, f"{m}_fs_hz") for m in MODALITIES}
    starts = {m: getattr(bundle, f"{m}_t0_s") for m in MODALITIES}
    scales = {m: bundle.metadata["streams"][m]["full_scale"] for m in MODALITIES}
    window_s = config["analysis"]["window_s"]
    empty = dict.fromkeys(cylinder_columns(config["engine"]), np.nan)
    rows = []
    for t in window_ends(duration, config):
        slices = {m: window_slice(t, rates[m], starts[m], config) for m in MODALITIES}
        waves = {m: getattr(bundle, m)[slices[m]] for m in MODALITIES}
        context, fresh = telemetry_context(bundle.telemetry, t, config)
        result = extract_window(
            waves, rates, context, fresh, scales, t, config,
            cam_sync_s=window_pulses(bundle.cam_sync_s, t - window_s, t),
            vibration_start_s=starts["vibration"] + slices["vibration"].start
            / rates["vibration"])
        signature = result.cylinders
        cam = signature is not None and signature.phase == PHASE_CAM
        rows.append({"t_end_s": result.t_end_s, "quality": result.quality,
                     **dict(zip(all_feature_names(), result.features, strict=True)),
                     "cyl_phase": signature.phase if signature is not None else PHASE_NONE,
                     **(by_cylinder(signature, config["engine"]) if cam else empty)})
    return pd.DataFrame(rows)


def _firing_rate_hz(rpm: float, engine: dict) -> float:
    # Each cylinder fires once per (strokes / 2) revolutions.
    return rpm / 60 * engine["cylinders"] / (engine["strokes"] / 2)


def _modality_features(x: np.ndarray, fs: float, low_hz: float, analysis: dict,
                       firing_hz: float) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2))
    crest = np.max(np.abs(x)) / rms if rms > 0 else 0.0
    nperseg = analysis["welch_segment_samples"]
    freqs, psd = welch(x, fs=fs, window=analysis["fft_window"], nperseg=nperseg,
                       noverlap=round(nperseg * analysis["welch_overlap_fraction"]),
                       detrend=False, scaling="density")
    edges = (low_hz, *BAND_EDGES_HZ, analysis["high_hz"])
    df = freqs[1] - freqs[0]
    powers = np.array([psd[(freqs >= lo) & (freqs < hi)].sum() * df
                       for lo, hi in zip(edges[:-1], edges[1:], strict=True)])
    total = powers.sum()
    fractions = powers / total if total > 0 else np.zeros_like(powers)

    high_env = _band_envelope(x, fs, HIGH_ENVELOPE_HZ)
    mid_env = _band_envelope(x, fs, MID_ENVELOPE_HZ)
    return np.concatenate([
        [np.log10(rms + 1e-12), crest, _kurtosis(x), np.log10(total + 1e-20)],
        fractions,
        [_cv(high_env), _kurtosis(high_env), _cv(mid_env), _firing_peak_cv(mid_env, fs, firing_hz)],
    ])


def _band_envelope(x: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """Hilbert envelope of the window after an FFT-domain band-pass (window-local)."""
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1 / fs)
    spectrum[(freqs < band[0]) | (freqs >= min(band[1], fs / 2))] = 0
    return np.abs(hilbert(np.fft.irfft(spectrum, n=x.size)))


def _firing_peak_cv(envelope: np.ndarray, fs: float, firing_hz: float) -> float:
    """Variation of per-firing envelope peaks; observed RPM sets the firing period."""
    if not np.isfinite(firing_hz) or firing_hz <= 0:
        return float("nan")
    width = max(1, round(0.002 * fs))
    smooth = np.convolve(envelope, np.ones(width) / width, mode="same")
    period = 1 / firing_hz
    peaks, _ = find_peaks(smooth, distance=max(1, round(0.7 * period * fs)))
    return _cv(smooth[peaks]) if peaks.size >= 3 else float("nan")


def _stroke_references(x: np.ndarray, fs: float, rpm: float,
                       envelope: np.ndarray | None = None) -> tuple[np.ndarray, float] | None:
    """Combustion onset times: half-peak rising edge of the low-band envelope.

    `envelope` may pass the window's precomputed SYNC_REFERENCE_HZ envelope.
    """
    if not np.isfinite(rpm) or rpm <= 0:
        return None
    period = 30.0 / rpm  # four cylinders: one firing per 180 crank degrees
    if envelope is None:
        envelope = _band_envelope(x, fs, SYNC_REFERENCE_HZ)
    peaks, _ = find_peaks(envelope, distance=max(1, round(0.7 * period * fs)))
    if peaks.size < SYNC_MIN_STROKES:
        return None
    strong = peaks[envelope[peaks] >= 0.3 * np.median(envelope[peaks])]
    onsets = []
    for p in strong:
        lo = max(0, p - round(0.4 * period * fs))
        below = np.nonzero(envelope[lo:p] < 0.5 * envelope[p])[0]
        onsets.append((lo + below[-1]) / fs if below.size else p / fs)
    filled = [onsets[0]]  # weak (e.g. misfired) strokes: fill at the observed spacing
    for t in onsets[1:]:
        gap = t - filled[-1]
        n = max(1, round(gap / period))
        filled.extend(filled[-1] + gap * k / n for k in range(1, n))
        filled.append(t)
    return np.array(filled), period


def _stroke_matrix(envelope: np.ndarray, fs: float, refs: np.ndarray) -> np.ndarray:
    """Envelope over each firing interval [refs[i], refs[i+1]), at SYNC_BINS angle bins."""
    grid = np.arange(envelope.size)
    return np.array([np.interp(np.linspace(a * fs, b * fs, SYNC_BINS, endpoint=False),
                               grid, envelope)
                     for a, b in zip(refs[:-1], refs[1:], strict=True)])


def _stroke_sync_features(x: np.ndarray, fs: float, rpm: float) -> np.ndarray:
    """Per-cylinder excess energy by crank-angle sector, plus reference jitter.

    Excess is (largest - median) of the four stroke-group means, relative to
    the window's mean stroke energy per degree. Zero when strokes cannot be
    referenced (no detectable firing pattern), which is the no-excess value.
    """
    envelopes = {band: _band_envelope(x, fs, edges) for band, edges in SYNC_BANDS_HZ.items()}
    return _sync_from_strokes(envelopes, fs, _stroke_references(x, fs, rpm))


def _sync_from_strokes(envelopes: dict[str, np.ndarray], fs: float,
                       found: tuple[np.ndarray, float] | None) -> np.ndarray:
    """_stroke_sync_features from precomputed band envelopes and stroke references."""
    out = np.zeros(len(sync_feature_names()))
    if found is None:
        return out
    refs, period = found
    usable = (refs.size - 1) // 4 * 4
    if usable < SYNC_MIN_STROKES:
        return out
    refs = refs[:usable + 1]
    degrees = np.arange(SYNC_BINS) * 180 / SYNC_BINS
    k = 0
    for band in SYNC_BANDS_HZ:
        energy = _stroke_matrix(envelopes[band], fs, refs) ** 2
        total = energy.mean() + 1e-20
        for a, b in SYNC_SECTORS_DEG:
            sector = energy[:, (degrees >= a) & (degrees < b)].sum(axis=1)
            groups = np.array([sector[g::4].mean() for g in range(4)])
            out[k] = (groups.max() - np.median(groups)) / (total * (b - a))
            k += 1
    out[k] = np.std(np.diff(refs)) / period
    return out


def _kurtosis(values: np.ndarray) -> float:
    """Excess kurtosis; 0 for a constant window (for example, fully clipped)."""
    return float(kurtosis(values)) if np.ptp(values) > 0 else 0.0


def _cv(values: np.ndarray) -> float:
    mean = np.mean(values)
    return float(np.std(values) / mean) if mean > 0 else 0.0
