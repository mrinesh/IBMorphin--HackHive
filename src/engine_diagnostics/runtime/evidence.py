"""Per-alert evidence: plain-language indicators for sound, vibration, and RPM.

For an active alert, each signal gets a claim that fits the alerted fault (for
example "irregular firing rhythm" for misfire), a strength measured as robust
z-scores of the model's own features against its normal reference, and the
time segments in the decision window that show the pattern, so a display can
highlight them. Indicators summarize what changed; they do not prove a cause.

Inputs are the decision window's waveforms and the RPM readings received before
the decision time; labels and simulation truth are never used.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from engine_diagnostics.pipeline.features import (
    HIGH_ENVELOPE_HZ,
    MID_ENVELOPE_HZ,
    _band_envelope,
)

SUPPORT_Z = 3.0  # robust z at or above this supports the claim
WEAK_Z = 2.0     # between WEAK_Z and SUPPORT_Z is weak support
PULSE_Z = 3.0    # a pulse this many robust deviations from the window's pulses is abnormal
WEAK_PULSE = 0.8     # ...and must also be below this fraction of the median to be weak
STRONG_PULSE = 1.25  # ...or above this multiple of the median to be strong
GAP_PERIODS = 1.5    # spacing above this many firing periods is a missing pulse
RPM_DIP_NOISE = 4.0  # RPM residual dip of this many noise units counts as fluctuation
RPM_NOISE_FLOOR = 1.0  # rpm; avoids treating quantized telemetry as noise-free
RPM_MATCH_S = 0.15   # an RPM dip within this time of a pulse segment accompanies it

# What each fault should look like per signal. Features are scored in the given
# direction ("up": increase supports; "both": any change supports).
FAULT_EVIDENCE = {
    "misfire": {
        "headline": "Misfire evidence",
        "sound": ("irregular firing rhythm", ["audio_firing_peak_cv"], "up", "pulses"),
        "vibration": ("repeated abnormal pulses",
                      ["vibration_firing_peak_cv", "vibration_env_cv_mid"], "up", "pulses"),
        "rpm": "accompanying fluctuation",
    },
    "knock_like": {
        "headline": "Knock evidence",
        "sound": ("sharp high-frequency bursts",
                  ["audio_env_kurtosis_high", "audio_env_cv_high"], "up", "bursts"),
        "vibration": ("high-frequency ringing after combustion",
                      ["vibration_env_kurtosis_high", "vibration_env_cv_high"], "up", "bursts"),
        "rpm": None,
    },
    "valve_clearance_like": {
        "headline": "Valve-clearance evidence",
        "sound": ("changed valve-impact pattern", ["audio_env_cv_mid", "audio_crest"],
                  "both", "pulses"),
        "vibration": ("altered impact timing and shape",
                      ["vibration_env_cv_mid", "vibration_env_kurtosis_high", "vibration_crest"],
                      "both", "pulses"),
        "rpm": None,
    },
}


@dataclass
class Indicator:
    signal: str
    claim: str
    status: str  # "supports", "weak", "not seen", or "not expected"
    detail: str
    segments: list[tuple[float, float]] = field(default_factory=list)
    trace_t: np.ndarray = field(default_factory=lambda: np.empty(0))
    trace_y: np.ndarray = field(default_factory=lambda: np.empty(0))
    trace_label: str = ""


@dataclass
class EvidencePanel:
    fault: str
    headline: str
    t_end_s: float
    indicators: list[Indicator]
    supporting: int
    expected: int
    assessment: str


def build_evidence(fault: str, waves: dict[str, np.ndarray], rates: dict[str, float],
                   t_end_s: float, window_s: float, rpm_t: np.ndarray, rpm_values: np.ndarray,
                   firing_hz: float, z_by_name: dict[str, float]) -> EvidencePanel:
    """Evidence for one alerted fault in the window [t_end - window_s, t_end)."""
    spec = FAULT_EVIDENCE[fault]
    t_start = t_end_s - window_s
    indicators = []
    for signal, modality in (("sound", "audio"), ("vibration", "vibration")):
        claim, names, direction, pattern = spec[signal]
        x = np.nan_to_num(np.asarray(waves[modality], dtype=np.float64))
        fs = rates[modality]
        if pattern == "bursts":
            env = _band_envelope(x, fs, HIGH_ENVELOPE_HZ)
            segments, count_text = _bursts(env, fs, t_start, firing_hz)
            label = f"{modality} 5-10 kHz envelope"
        else:
            env = _band_envelope(x, fs, MID_ENVELOPE_HZ)
            segments, count_text = _pulses(env, fs, t_start, firing_hz,
                                           strong=signal == "vibration" or direction == "both")
            label = f"{modality} 0.5-3 kHz envelope"
        z, worst = _strength(names, direction, z_by_name)
        status = _status(z)
        detail = (f"{_describe(worst)} {_z_text(z)} vs normal; {count_text}"
                  if worst else f"no reference feature available; {count_text}")
        step = max(1, round(fs / 2000))  # about 2 kHz display rate is enough for an envelope
        indicators.append(Indicator(
            signal, claim, status, detail, segments,
            t_start + np.arange(0, env.size, step) / fs, env[::step], label))

    for ind in indicators:
        ind.segments = _clip(ind.segments, t_start, t_end_s)
    pulse_segments = [s for ind in indicators for s in ind.segments]
    indicators.append(_rpm_indicator(spec["rpm"], rpm_t, rpm_values, pulse_segments))
    expected = [i for i in indicators if i.status != "not expected"]
    supporting = sum(i.status == "supports" for i in expected)
    return EvidencePanel(fault, spec["headline"], t_end_s, indicators, supporting,
                         len(expected), _assessment(supporting, len(expected), indicators))


def _strength(names: list[str], direction: str, z_by_name: dict[str, float]):
    scored = [(z_by_name[n] if direction == "up" else abs(z_by_name[n]), n)
              for n in names if n in z_by_name and np.isfinite(z_by_name[n])]
    if not scored:
        return float("nan"), None
    best = max(scored)
    return best[0], (best[1], z_by_name[best[1]])


def _status(z: float) -> str:
    if not np.isfinite(z):
        return "not seen"
    return "supports" if z >= SUPPORT_Z else "weak" if z >= WEAK_Z else "not seen"


def _describe(worst) -> str:
    name, _ = worst
    return name.replace("_", " ")


def _z_text(z: float) -> str:
    return f"robust z {z:+.1f}" if np.isfinite(z) else "robust z n/a"


def _firing_pulses(env: np.ndarray, fs: float, firing_hz: float, smooth_s: float):
    """One envelope peak per firing interval, with each peak's robust deviation."""
    period = 1 / firing_hz
    width = max(1, round(smooth_s * fs))
    smooth = np.convolve(env, np.ones(width) / width, mode="same")
    peaks, _ = find_peaks(smooth, distance=max(1, round(0.7 * period * fs)))
    # Pulses cut by the window edge, and window-local filter edge effects, look falsely weak.
    edge = round(period * fs)
    peaks = peaks[(peaks >= edge) & (peaks < env.size - edge)]
    heights = smooth[peaks]
    median = np.median(heights) if peaks.size else 0.0
    spread = 1.4826 * np.median(np.abs(heights - median)) + 1e-9 * (median + 1e-20)
    return peaks, heights, median, (heights - median) / spread, period


def _pulses(env: np.ndarray, fs: float, t_start: float, firing_hz: float,
            strong: bool) -> tuple[list[tuple[float, float]], str]:
    """Firing pulses that are weak, missing, or (optionally) abnormally strong."""
    if not np.isfinite(firing_hz) or firing_hz <= 0 or env.size == 0:
        return [], "firing period unknown (no valid RPM)"
    peaks, heights, median, z, period = _firing_pulses(env, fs, firing_hz, 0.002)
    if peaks.size < 3:
        return [], "too few pulses to compare"
    times = t_start + peaks / fs
    half = period / 2
    weak = [(t - half, t + half) for t, h, d in zip(times, heights, z, strict=True)
            if d <= -PULSE_Z and h < WEAK_PULSE * median]
    loud = ([(t - half, t + half) for t, h, d in zip(times, heights, z, strict=True)
             if d >= PULSE_Z and h > STRONG_PULSE * median] if strong else [])
    gaps = [(a + half, b - half) for a, b in zip(times[:-1], times[1:], strict=True)
            if b - a > GAP_PERIODS * period]
    parts = [f"{len(weak)} weak", f"{len(gaps)} missing"]
    if strong:
        parts.append(f"{len(loud)} strong")
    return _merge(weak + gaps + loud), f"{', '.join(parts)} of {peaks.size} pulses"


def _bursts(env: np.ndarray, fs: float, t_start: float,
            firing_hz: float) -> tuple[list[tuple[float, float]], str]:
    """Firings whose high-band burst is far stronger than the window's typical firing."""
    if not np.isfinite(firing_hz) or firing_hz <= 0 or env.size == 0:
        return [], "firing period unknown (no valid RPM)"
    peaks, heights, median, z, period = _firing_pulses(env, fs, firing_hz, 0.0005)
    if peaks.size < 3:
        return [], "too few firings to compare"
    times = t_start + peaks / fs
    pad = period / 4
    hits = [(t - pad, t + pad) for t, h, d in zip(times, heights, z, strict=True)
            if d >= PULSE_Z and h > STRONG_PULSE * median]
    return _merge(hits), f"{len(hits)} of {peaks.size} firings with strong bursts"


def _rpm_indicator(claim: str | None, rpm_t: np.ndarray, rpm_values: np.ndarray,
                   pulse_segments: list[tuple[float, float]]) -> Indicator:
    t, v = np.asarray(rpm_t, dtype=float), np.asarray(rpm_values, dtype=float)
    keep = np.isfinite(t) & np.isfinite(v)
    t, v = t[keep], v[keep]
    if claim is None:
        return Indicator("rpm", "no RPM change expected for this fault", "not expected",
                         "RPM is shown for context only", trace_t=t, trace_y=v, trace_label="RPM")
    if t.size < 5:
        return Indicator("rpm", claim, "not seen", "too few RPM readings in the window",
                         trace_t=t, trace_y=v, trace_label="RPM")
    trend = np.polyval(np.polyfit(t, v, 1), t)  # remove ramps; keep short dips
    residual = v - trend
    noise = max(1.4826 * np.median(np.abs(residual - np.median(residual))), RPM_NOISE_FLOOR)
    dip_idx = np.nonzero(residual < -RPM_DIP_NOISE * noise)[0]
    step = np.median(np.diff(t))
    segments = _merge([(t[k] - step, t[k] + step) for k in dip_idx])
    matched = [(a, b) for a, b in segments
               if any(a - RPM_MATCH_S <= q and p <= b + RPM_MATCH_S for p, q in pulse_segments)]
    deepest = -residual.min() / noise
    if not segments:
        status = "not seen"
    elif matched:
        status = "supports"
    else:
        status = "weak"
    detail = (f"{len(segments)} dips below trend (deepest {deepest:.1f}x noise of "
              f"{noise:.1f} rpm); {len(matched)} coincide with abnormal pulses")
    return Indicator("rpm", claim, status, detail, segments, t, v, "RPM")


def _assessment(supporting: int, expected: int, indicators: list[Indicator]) -> str:
    """Factual count of the signals behind the alert (the model's confidence is shown apart)."""
    which = [i.signal for i in indicators if i.status == "supports"]
    partial = [i.signal for i in indicators if i.status == "weak"]
    text = f"{supporting} of {expected} signals support this alert"
    if which:
        text += f" ({', '.join(which)})"
    if partial:
        text += f"; partial: {', '.join(partial)}"
    return text + "."


def _clip(segments: list[tuple[float, float]], lo: float, hi: float) -> list[tuple[float, float]]:
    return [(max(a, lo), min(b, hi)) for a, b in segments if b > lo and a < hi]


def _merge(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for a, b in sorted(segments):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]
