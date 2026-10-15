"""Window features shared by offline training and replay inference.

Every feature uses only the trailing window [t - window_s, t) of waveform
samples and telemetry readings strictly before t. Spectral filtering is done
per window in the frequency domain, so no filter state carries between
windows and offline extraction equals replay extraction for the same window.
Labels, run/engine IDs, and simulation truth are never inputs.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert, welch
from scipy.stats import kurtosis

FEATURE_VERSION = "0.1.0"
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


def feature_names() -> list[str]:
    """Ordered model input names; the order is part of FEATURE_VERSION."""
    names = [f"{m}_{f}" for m in MODALITIES for f in PER_MODALITY]
    return names + [f"ctx_{c}" for c in CONTEXT]


@dataclass
class WindowResult:
    t_end_s: float
    features: np.ndarray
    quality: str  # "ok", "clipped", "missing_waveform", or "stale_telemetry"


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
                   full_scales: dict[str, float], t_end_s: float, config: dict) -> WindowResult:
    """Features for one decision window; `waves` hold exactly the window samples."""
    analysis, runtime = config["analysis"], config["runtime"]
    quality = "ok" if context_fresh else "stale_telemetry"
    parts = []
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
    features = np.concatenate(parts + [np.array([context[c] for c in CONTEXT])])
    return WindowResult(t_end_s=float(t_end_s), features=features, quality=quality)


def extract_run(bundle, config: dict) -> pd.DataFrame:
    """Offline features for every decision window of a loaded RunBundle."""
    duration = bundle.metadata["duration_s"]
    rates = {m: getattr(bundle, f"{m}_fs_hz") for m in MODALITIES}
    starts = {m: getattr(bundle, f"{m}_t0_s") for m in MODALITIES}
    scales = {m: bundle.metadata["streams"][m]["full_scale"] for m in MODALITIES}
    rows = []
    for t in window_ends(duration, config):
        waves = {m: getattr(bundle, m)[window_slice(t, rates[m], starts[m], config)]
                 for m in MODALITIES}
        context, fresh = telemetry_context(bundle.telemetry, t, config)
        result = extract_window(waves, rates, context, fresh, scales, t, config)
        rows.append({"t_end_s": result.t_end_s, "quality": result.quality,
                     **dict(zip(feature_names(), result.features, strict=True))})
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


def _kurtosis(values: np.ndarray) -> float:
    """Excess kurtosis; 0 for a constant window (for example, fully clipped)."""
    return float(kurtosis(values)) if np.ptp(values) > 0 else 0.0


def _cv(values: np.ndarray) -> float:
    mean = np.mean(values)
    return float(np.std(values) / mean) if mean > 0 else 0.0
