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


def localization_episodes(records: list[dict[str, Any]], fault: str, true_cylinder: int | None,
                          intervals: list[tuple[float, float]] | None = None
                          ) -> list[dict[str, Any]]:
    """Score cylinder localization over each alert episode of one fault.

    An episode is a run of consecutive decisions with `fault` among the active
    alerts. Answers are judged only on decisions made while the fault is truly
    active (t_end inside one of its `intervals`), as detection is scored
    against each fault's own intervals: `first` is the first cylinder reported
    then (what a technician would act on), `final` the last, and `changes` how
    often it changed. An episode that never overlaps the intervals is a false
    alert (`true_alert` False), as is every episode when true_cylinder is None.
    `first_any` is the first cylinder reported at any time in the episode,
    including before onset (the stricter, cause-agnostic view).
    intervals None treats the whole run as active.
    """
    def active(t: float) -> bool:
        return intervals is None or any(a < t <= b for a, b in intervals)

    found, current = [], None
    for r in records:
        alerts = r.get("alerts") if "alerts" in r else ([r["alert"]] if r.get("alert") else [])
        if fault not in (alerts or []):
            current = None
            continue
        if current is None:
            current = {"start_s": r["t_end_s"], "end_s": r["t_end_s"], "first": None,
                       "first_s": None, "final": None, "changes": 0, "first_any": None,
                       "overlaps": False, "overlap_start_s": None,
                       "true_cylinder": true_cylinder}
            found.append(current)
        current["end_s"] = r["t_end_s"]
        where = (r.get("localization") or {}).get(fault)
        cylinder = where.get("cylinder") if where else None
        if cylinder is not None and current["first_any"] is None:
            current["first_any"] = cylinder
        if not active(r["t_end_s"]):
            continue
        if not current["overlaps"]:
            current["overlaps"], current["overlap_start_s"] = True, r["t_end_s"]
        if cylinder is not None:
            if current["first"] is None:
                current["first"], current["first_s"] = cylinder, r["t_end_s"]
            elif cylinder != current["final"]:
                current["changes"] += 1
            current["final"] = cylinder
    for e in found:
        e["true_alert"] = true_cylinder is not None and e["overlaps"]
        e["delay_s"] = (None if e["first_s"] is None
                        else e["first_s"] - max(e["start_s"], e["overlap_start_s"]))
        judged = e["true_alert"]
        e["first_correct"] = None if not judged or e["first"] is None \
            else e["first"] == true_cylinder
        e["final_correct"] = None if not judged or e["final"] is None \
            else e["final"] == true_cylinder
        e["first_any_correct"] = None if not judged or e["first_any"] is None \
            else e["first_any"] == true_cylinder
    return found


def summarize_localization(found: list[dict[str, Any]]) -> dict[str, Any]:
    """Episode-level localization rates for true alerts; false alerts counted apart.

    Delays are from the later of alert start and fault onset to the first
    cylinder named while the fault is active.
    """
    true_alerts = [e for e in found if e["true_alert"]]
    located = [e for e in true_alerts if e["first"] is not None]
    any_located = [e for e in true_alerts if e["first_any"] is not None]
    delays = [e["delay_s"] for e in located]
    return {
        "true_alert_episodes": len(true_alerts),
        "localized_fraction": len(located) / len(true_alerts) if true_alerts else None,
        "first_correct": (sum(e["first_correct"] for e in located) / len(located)
                          if located else None),
        "final_correct": (sum(e["final_correct"] for e in located) / len(located)
                          if located else None),
        "first_any_correct": (sum(e["first_any_correct"] for e in any_located)
                              / len(any_located) if any_located else None),
        "wrong_cylinder_episodes": sum(not e["first_correct"] for e in located),
        "episodes_with_changes": sum(e["changes"] > 0 for e in located),
        "delay_p50_s": float(np.percentile(delays, 50)) if delays else None,
        "delay_p95_s": float(np.percentile(delays, 95)) if delays else None,
        "false_alert_episodes": sum(not e["true_alert"] for e in found),
        "false_alert_episodes_localized": sum(
            e["first_any"] is not None for e in found if not e["true_alert"]),
    }
