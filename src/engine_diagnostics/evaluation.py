"""Alert-episode metrics against evaluation truth (evaluation side only).

Decisions are the per-window records produced by replay (or, equivalently,
offline features in time order). Truth is used here only to score them.
"""

from typing import Any

import numpy as np

from engine_diagnostics.runtime.alerts import AlertState

SUSTAINED_S = 3.0  # fault intervals at least this long count as sustained events
DETECTION_GRACE_S = 2.0  # an alert may start up to this long after an interval ends


def alert_sequence(accepted: list[str], quality: list[str], confirm: int, clear: int) -> list:
    state = AlertState(confirm=confirm, clear=clear)
    return [state.update(a, q) for a, q in zip(accepted, quality, strict=True)]


def episodes(times: list[float], alerts: list) -> list[tuple[float, float, str]]:
    """Contiguous runs of the same active alert: (start_s, end_s, class)."""
    found, start, current = [], None, None
    for t, alert in zip(times, alerts, strict=True):
        if alert != current:
            if current is not None:
                found.append((start, t, current))
            start, current = t, alert
    if current is not None:
        found.append((start, times[-1], current))
    return found


def _intervals(truth: dict[str, Any]) -> list[tuple[float, float]]:
    raw = truth.get("active_intervals") or []
    if raw and not isinstance(raw[0], (list, tuple)):
        raw = [raw]
    return [(float(a), float(b)) for a, b in raw]


def score_run(times: list[float], alerts: list, labels: list[str], truth: dict[str, Any],
              window_s: float, step_s: float) -> dict[str, Any]:
    condition = truth["condition"]
    intervals = _intervals(truth)
    found = episodes(times, alerts)
    false_alerts, wrong_class = 0, 0
    for start, _, alert in found:
        window = (start - window_s, start)
        overlaps = any(a < window[1] and b > window[0] for a, b in intervals)
        if not overlaps:
            false_alerts += 1
        elif alert != condition:
            wrong_class += 1
    sustained, brief = [], []
    for a, b in intervals:
        hit = [s for s, _, alert in found
               if alert == condition and a <= s <= b + DETECTION_GRACE_S]
        record = {"start_s": a, "end_s": b, "detected": bool(hit),
                  "latency_s": (min(hit) - a) if hit else None}
        (sustained if b - a >= SUSTAINED_S else brief).append(record)
    normal_minutes = sum(label == "normal" for label in labels) * step_s / 60
    return {"false_alert_episodes": false_alerts, "wrong_class_episodes": wrong_class,
            "normal_minutes": normal_minutes, "sustained": sustained, "brief": brief}


def summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    normal_minutes = sum(s["normal_minutes"] for s in scores)
    false_alerts = sum(s["false_alert_episodes"] for s in scores)
    sustained = [e for s in scores for e in s["sustained"]]
    brief = [e for s in scores for e in s["brief"]]
    latencies = [e["latency_s"] for e in sustained if e["detected"]]
    return {
        "runs": len(scores),
        "normal_minutes": normal_minutes,
        "false_alert_episodes": false_alerts,
        "false_alerts_per_normal_minute": false_alerts / normal_minutes if normal_minutes else None,
        "wrong_class_episodes": sum(s["wrong_class_episodes"] for s in scores),
        "sustained_events": len(sustained),
        "sustained_recall": (sum(e["detected"] for e in sustained) / len(sustained)
                             if sustained else None),
        "latency_p50_s": float(np.percentile(latencies, 50)) if latencies else None,
        "latency_p95_s": float(np.percentile(latencies, 95)) if latencies else None,
        "brief_events": len(brief),
        "brief_recall": sum(e["detected"] for e in brief) / len(brief) if brief else None,
    }


def severity_at(truth: dict[str, Any], t: float) -> float:
    """True simulated severity at time t (0 when inactive); evaluation only."""
    if not any(a <= t < b for a, b in _intervals(truth)):
        return 0.0
    if not truth.get("degradation"):
        return float(truth["severity"])
    onset = float(truth["onset_s"])
    end = max(b for _, b in _intervals(truth))
    progress = min(max((t - onset) / (end - onset), 0.0), 1.0)
    return float(truth["severity"] + (truth["severity_end"] - truth["severity"]) * progress)
