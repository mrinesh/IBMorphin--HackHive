"""Health timeline: interpretable features against the healthy baseline, with stages.

Tracked lines are expressed relative to their own developing threshold (1.0 =
threshold): two interpretable features as robust z against the healthy baseline
of the same operating range, and the model's overall anomaly score. Each
decision window gets a stage:
  healthy             every tracked line is below its threshold
  developing anomaly  a tracked line stays at or above its threshold, without a
                      confirmed alert
  persistent fault    the model's confirmed alert (its persistence rule) is active
  no data             the window failed quality checks; no assessment

"Stays" means the median of the last SMOOTH_WINDOWS good windows (causal, about
2 s), so single noisy windows do not change the stage. Thresholds were chosen
on core_v1 tuning engines for about 1% of truth-healthy windows each (see
docs/TESTING.md). Stages describe the current state relative to the healthy
baseline. They are not a forecast or a remaining-useful-life estimate: that
needs run-to-failure degradation data.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# (feature, label, developing threshold in robust z); an increase moves away from healthy.
TRACKED = (
    ("audio_firing_peak_cv", "Firing-rhythm irregularity (sound)", 4.0),
    ("vibration_env_kurtosis_high", "High-frequency impulsiveness (vibration)", 4.0),
)
OVERALL = "Overall deviation (model anomaly score)"  # threshold: the model's own
HEALTHY, DEVELOPING, FAULT, NO_DATA = ("healthy", "developing anomaly", "persistent fault",
                                       "no data")
STAGES = (HEALTHY, DEVELOPING, FAULT, NO_DATA)
SMOOTH_WINDOWS = 4


@dataclass
class HealthTimeline:
    t: np.ndarray
    level: pd.DataFrame  # smoothed tracked lines / their thresholds (columns = labels)
    stage: list[str]

    def transitions(self) -> list[tuple[float, str]]:
        """First time of each stage change, ignoring no-data windows."""
        out: list[tuple[float, str]] = []
        for t, s in zip(self.t, self.stage, strict=True):
            if s != NO_DATA and (not out or out[-1][1] != s):
                out.append((float(t), s))
        return out

    def summary(self) -> str:
        steps = self.transitions()
        if not steps:
            return "No assessed windows yet."
        parts = [steps[0][1]] + [f"{s} at {t:.1f} s" for t, s in steps[1:]]
        return " → ".join(parts)


def health_timeline(t, z_by_feature: pd.DataFrame, anomaly_score, anomaly_threshold: float,
                    alert_active, quality_ok) -> HealthTimeline:
    """Stages for decision windows in time order; inputs are aligned per window."""
    t = np.asarray(t, dtype=float)
    ok = np.asarray(quality_ok, dtype=bool)
    alert = np.asarray(alert_active, dtype=bool)
    columns = {}
    for name, label, threshold in TRACKED:
        raw = (z_by_feature[name].to_numpy(dtype=float) if name in z_by_feature
               else np.full(t.size, np.nan))
        columns[label] = _causal_median(raw, ok) / threshold
    score = _causal_median(np.asarray(anomaly_score, dtype=float), ok)
    columns[OVERALL] = score / anomaly_threshold
    level = pd.DataFrame(columns, index=t)
    peak = level.max(axis=1, skipna=True).to_numpy()
    stage = []
    for k in range(t.size):
        if not ok[k]:
            stage.append(NO_DATA)
        elif alert[k]:
            stage.append(FAULT)
        elif np.isfinite(peak[k]) and peak[k] >= 1.0:
            stage.append(DEVELOPING)
        else:
            stage.append(HEALTHY)
    return HealthTimeline(t=t, level=level, stage=stage)


def _causal_median(values: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Median of the last SMOOTH_WINDOWS good values up to each window (NaN if none)."""
    out = np.full(values.size, np.nan)
    history: list[float] = []
    for k, (v, good) in enumerate(zip(values, ok, strict=True)):
        if good and np.isfinite(v):
            history = (history + [float(v)])[-SMOOTH_WINDOWS:]
        if history:
            out[k] = np.median(history)
    return out
