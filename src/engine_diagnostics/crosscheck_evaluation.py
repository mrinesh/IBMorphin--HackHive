"""Calibrate the independent sensor paths and benchmark them (data contract 1.2).

Evaluation side (reads truth). Calibration uses only training-split runs:
those without misfire set the OBD per-stroke threshold table and the crank
detector's per-cylinder threshold; fault-free runs set the thermal path's
healthy spreads and thresholds. No fault label tunes any threshold. The
benchmark replays held-out test runs through ReplayEngine exactly as the
dashboard does (acoustic model, optional localizer, crank path, OBD monitor
and thermal path in one pass).

OBD section, for every misfire interval:

  acoustic   the first misfire alert of the acoustic/vibration model
  crank      the first alert of the independent crank-speed detector
  ECU A/B    the first type A (200 rev) and type B (1000 rev) pending DTC

Lead = ECU time - acoustic time. An ECU that sets no code before the run ends
is reported as censored, not as a number. Knock and valve intervals show what
the ECU monitor cannot cover.

Thermal section: for every fault interval, whether the thermal pattern for that
fault alerted (and on which cylinder), its latency, and thermal false alerts
on fault-free runs. The confirmation section reports how many acoustic alert
episodes an independent path agreed with, and an experimental fused knock
alert (acoustic knock AND thermal head_hot) next to the plain acoustic one.
Synthetic data only.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from engine_diagnostics.data.run_io import find_runs, load_run, thermal_signals
from engine_diagnostics.models.obd_misfire import CrankCalibration, calibrate
from engine_diagnostics.pipeline.crankspeed import stroke_speeds
from engine_diagnostics.pipeline.cylinders import window_pulses
from engine_diagnostics.pipeline.features import window_ends
from engine_diagnostics.pipeline.labels import FAULTS, fault_cylinders, fault_intervals
from engine_diagnostics.runtime.crosscheck import cylinder_excess
from engine_diagnostics.runtime.replay import replay_run
from engine_diagnostics.runtime.thermal import (
    KINDS,
    PATTERN_FAULT,
    ThermalCalibration,
    latest_readings,
    relative,
)

MISFIRE = "misfire"
GRACE_S = 2.0  # an alert may start up to this long after an interval ends (as evaluation.py)


def load_truths(truth_dir: Path, run_ids: list[str]) -> dict[str, dict]:
    return {r: json.loads((Path(truth_dir) / f"{r}_events.json").read_text(encoding="utf-8"))
            for r in run_ids}


def run_strokes(bundle, config: dict) -> tuple[dict[str, np.ndarray], list[float]]:
    """Every stroke once (the replay partition) and the window maximum cylinder excess."""
    window_s, lag = config["analysis"]["window_s"], config["crank_detector"]["stroke_lag_s"]
    cylinders = config["engine"]["cylinders"]
    done_until = -np.inf
    rpm, drop, excess = [], [], []
    for t in window_ends(bundle.metadata["duration_s"], config):
        strokes = stroke_speeds(window_pulses(bundle.crank_tooth_s, t - window_s, t),
                                window_pulses(bundle.cam_sync_s, t - window_s, t), config)
        final = (strokes.time_s >= done_until) & (strokes.time_s < t - lag)
        if final.any():
            rpm.append(strokes.rpm[final])
            drop.append(strokes.drop[final])
            done_until = t - lag
        values = cylinder_excess(strokes, cylinders)
        if np.isfinite(values).all():
            excess.append(float(values.max()))
    def join(parts):
        return np.concatenate(parts) if parts else np.zeros(0)

    return {"rpm": join(rpm), "drop": join(drop)}, excess


def calibrate_crank(raw_dir: Path, truth_dir: Path, config: dict,
                    splits: tuple[str, ...] = ("train",)) -> CrankCalibration:
    """Thresholds from the given splits' runs that contain no misfire."""
    run_ids = [p.name for p in find_runs(raw_dir)]
    truths = load_truths(truth_dir, run_ids)
    chosen = [r for r in run_ids if truths[r].get("split") in splits
              and MISFIRE not in fault_intervals(truths[r])]
    strokes, excess = [], []
    for run_id in chosen:
        bundle = load_run(Path(raw_dir) / run_id)
        if bundle.crank_tooth_s is None:
            continue
        healthy, values = run_strokes(bundle, config)
        strokes.append(healthy)
        excess += values
    if not strokes:
        raise ValueError(f"no {'/'.join(splits)} runs with a crank wheel and no misfire "
                         f"under {raw_dir}")
    training = {"raw_dir": str(raw_dir), "splits": list(splits), "runs": chosen,
                "obd_misfire": config["obd_misfire"], "crank_detector": config["crank_detector"]}
    return calibrate(strokes, np.asarray(excess), config, training)


def calibrate_thermal(raw_dir: Path, truth_dir: Path, config: dict,
                      splits: tuple[str, ...] = ("train",)) -> ThermalCalibration:
    """Healthy spreads and z thresholds from fault-free runs of the given splits.

    Each run is processed causally as ThermalPath does: per-cylinder baseline
    from the first baseline_s, then the change of every reading after it.
    """
    defaults = ThermalCalibration(spread_c={}, thresholds={})
    tuning = config["thermal_detector"]
    run_ids = [p.name for p in find_runs(raw_dir)]
    truths = load_truths(truth_dir, run_ids)
    chosen = [r for r in run_ids if truths[r].get("split") in splits
              and not fault_intervals(truths[r])]
    cylinders = config["engine"]["cylinders"]
    changes = {kind: [] for kind in KINDS}
    used = []
    for run_id in chosen:
        bundle = load_run(Path(raw_dir) / run_id)
        if not set(thermal_signals(config)) <= set(bundle.telemetry["signal"]):
            continue
        used.append(run_id)
        samples, baseline = {k: [] for k in KINDS}, None
        for t in window_ends(bundle.metadata["duration_s"], config):
            readings = latest_readings(bundle.telemetry, t, cylinders, defaults.max_age_s)
            rel = {k: relative(readings[k]) for k in KINDS}
            if not all(np.isfinite(rel[k]).all() for k in KINDS):
                continue
            if baseline is None:
                for k in KINDS:
                    samples[k].append(rel[k])
                if t >= defaults.baseline_s:
                    baseline = {k: np.median(np.vstack(samples[k]), axis=0) for k in KINDS}
                continue
            for k in KINDS:
                changes[k].append(rel[k] - baseline[k])
    if not used:
        raise ValueError(f"no fault-free {'/'.join(splits)} runs with cylinder temperatures "
                         f"under {raw_dir}")
    spread = {}
    z = {}
    for k in KINDS:
        values = np.concatenate(changes[k])
        mad = np.median(np.abs(values - np.median(values)))
        spread[k] = round(float(max(1.4826 * mad, tuning["min_spread_c"][k])), 3)
        z[k] = values / spread[k]
    q, margin, floor = tuning["quantile"], tuning["threshold_margin"], tuning["min_z"]
    hot = max(floor, margin * max(np.quantile(z["egt"], q), np.quantile(z["cht"], q)))
    cold = max(floor, margin * float(np.quantile(-z["egt"], q)))
    thresholds = {"cold": round(float(cold), 2), "hot": round(float(hot), 2),
                  "not_hot": round(float(hot) / 2, 2)}
    training = {"raw_dir": str(raw_dir), "splits": list(splits), "runs": used,
                "readings": {k: int(np.concatenate(changes[k]).size) for k in KINDS},
                "thermal_detector": tuning}
    return ThermalCalibration(spread_c=spread, thresholds=thresholds,
                              confirm_updates=tuning["confirm_updates"],
                              clear_updates=tuning["clear_updates"], training=training)


def _first(times: list[float], flags: list[bool], start: float, end: float) -> float | None:
    """Start of the first alert episode beginning inside [start, end + grace]."""
    previous = False
    for t, flag in zip(times, flags, strict=True):
        if flag and not previous and start <= t <= end + GRACE_S:
            return t
        previous = flag
    return None


def _episodes_outside(times: list[float], flags: list[bool],
                      intervals: list[tuple[float, float]], window_s: float) -> int:
    """Alert episodes whose decision window does not overlap any fault interval."""
    count, previous = 0, False
    for t, flag in zip(times, flags, strict=True):
        if flag and not previous and not any(a < t and b > t - window_s for a, b in intervals):
            count += 1
        previous = flag
    return count


def benchmark_run(records: list[dict[str, Any]], truth: dict, config: dict) -> dict[str, Any]:
    window_s = config["analysis"]["window_s"]
    duration = float(records[-1]["t_end_s"]) if records else 0.0
    times = [r["t_end_s"] for r in records]
    intervals = fault_intervals(truth)
    cylinders = fault_cylinders(truth)
    crank = [r.get("crank") or {} for r in records]
    crank_flags = [bool(c.get("alert")) for c in crank]
    dtcs = (crank[-1].get("obd") or {}).get("dtcs", []) if crank else []
    misfire_intervals = intervals.get(MISFIRE, [])
    out: dict[str, Any] = {"run_id": truth["run_id"], "faults": sorted(intervals),
                           "duration_s": duration, "misfire": [], "other_faults": []}
    for a, b in misfire_intervals:
        acoustic = _first(times, [MISFIRE in r.get("alerts", []) for r in records], a, b)
        crank_t = _first(times, crank_flags, a, b)
        crank_cyl = next((c.get("cylinder") for t, c in zip(times, crank, strict=True)
                          if crank_t is not None and t >= crank_t and c.get("cylinder")), None)
        located = None
        if acoustic is not None:
            for t, r in zip(times, records, strict=True):
                where = (r.get("localization") or {}).get(MISFIRE) or {}
                if t >= acoustic and where.get("cylinder") is not None:
                    located = where["cylinder"]
                    break
        ecu = {kind: next((d for d in dtcs if d["type"] == kind and d["t_s"] >= a), None)
               for kind in ("A", "B")}
        entry = {"start_s": a, "end_s": b, "true_cylinder": cylinders.get(MISFIRE),
                 "acoustic_s": acoustic, "acoustic_cylinder": located,
                 "crank_s": crank_t, "crank_cylinder": crank_cyl}
        for kind, dtc in ecu.items():
            entry[f"ecu_{kind}_s"] = dtc["t_s"] if dtc else None
            entry[f"ecu_{kind}_code"] = dtc["code"] if dtc else None
            entry[f"ecu_{kind}_cylinder"] = dtc["cylinder"] if dtc else None
        out["misfire"].append(entry)
    for fault in FAULTS:
        if fault == MISFIRE:
            continue
        for a, b in intervals.get(fault, []):
            out["other_faults"].append({
                "fault": fault, "start_s": a, "end_s": b,
                "acoustic_s": _first(times, [fault in r.get("alerts", []) for r in records], a, b)})
    every = [iv for ivs in intervals.values() for iv in ivs]
    out["crank_false_alert_episodes"] = _episodes_outside(times, crank_flags,
                                                          misfire_intervals, window_s)
    out["ecu_false_dtcs"] = [d for d in dtcs
                             if not any(a <= d["t_s"] for a, _ in misfire_intervals)]
    out["any_fault_intervals"] = every
    return out


def thermal_run(records: list[dict[str, Any]], truth: dict, config: dict) -> dict[str, Any]:
    """Per fault interval: the thermal pattern for that fault, its time and cylinder."""
    window_s = config["analysis"]["window_s"]
    times = [r["t_end_s"] for r in records]
    thermal = [r.get("thermal") or {} for r in records]
    intervals, cylinders = fault_intervals(truth), fault_cylinders(truth)
    out = {"run_id": truth["run_id"], "faults": sorted(intervals),
           "duration_s": float(times[-1]) if times else 0.0, "intervals": [],
           "false_alert_episodes": {}}
    for pattern, fault in PATTERN_FAULT.items():
        flags = [bool(((th.get("patterns") or {}).get(pattern) or {}).get("alert"))
                 for th in thermal]
        for a, b in intervals.get(fault, []):
            hit = _first(times, flags, a, b)
            cylinder = None
            if hit is not None:
                cylinder = next((th["patterns"][pattern]["cylinder"]
                                 for t, th in zip(times, thermal, strict=True)
                                 if t >= hit and th["patterns"][pattern]["cylinder"]), None)
            out["intervals"].append({"fault": fault, "start_s": a, "end_s": b,
                                     "true_cylinder": cylinders.get(fault), "thermal_s": hit,
                                     "thermal_cylinder": cylinder})
        out["false_alert_episodes"][fault] = _episodes_outside(
            times, flags, intervals.get(fault, []), window_s)
    return out


def confirmation_run(records: list[dict[str, Any]], truth: dict, config: dict) -> dict[str, Any]:
    """Acoustic alert episodes, which independent path agreed, and a fused knock alert."""
    window_s = config["analysis"]["window_s"]
    times = [r["t_end_s"] for r in records]
    intervals = fault_intervals(truth)
    episodes = []
    for fault in FAULTS:
        flags = [fault in r.get("alerts", []) for r in records]
        start = None
        for k, flag in enumerate(flags + [False]):
            if flag and start is None:
                start = k
            elif not flag and start is not None:
                span = records[start:k]
                true = any(a < times[start] and b > times[start] - window_s
                           for a, b in intervals.get(fault, []))
                thermal = any(fault in ((r.get("thermal") or {}).get("alerts") or [])
                              for r in span)
                crank = fault == MISFIRE and any((r.get("crank") or {}).get("alert")
                                                 for r in span)
                episodes.append({"fault": fault, "start_s": times[start], "true": true,
                                 "thermal_confirmed": thermal, "crank_confirmed": crank})
                start = None
    knock = "knock_like"
    fused = [knock in r.get("alerts", []) and knock in ((r.get("thermal") or {}).get("alerts")
                                                      or []) for r in records]
    plain = [knock in r.get("alerts", []) for r in records]
    variants = {}
    for name, flags in (("acoustic", plain), ("acoustic_and_thermal", fused)):
        variants[name] = {
            "detected": [_first(times, flags, a, b) for a, b in intervals.get(knock, [])],
            "false_alert_episodes": _episodes_outside(times, flags, intervals.get(knock, []),
                                                      window_s)}
    knock_free = sum(1 for t in times if not any(a < t and b > t - window_s
                                                for a, b in intervals.get(knock, [])))
    return {"episodes": episodes, "knock_variants": variants,
            "knock_free_minutes": knock_free * config["analysis"]["step_s"] / 60,
            "knock_intervals": intervals.get(knock, [])}


def summarize_thermal(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    fault_free = [r for r in runs if not r["faults"]]
    minutes = sum(r["duration_s"] for r in fault_free) / 60
    for fault in PATTERN_FAULT.values():
        events = [e for r in runs for e in r["intervals"] if e["fault"] == fault]
        caught = [e for e in events if e["thermal_s"] is not None]
        named = [e for e in caught if e["thermal_cylinder"] is not None]
        without = [r for r in runs if fault not in r["faults"]]
        out[fault] = {
            "intervals": len(events),
            "detected": round(len(caught) / len(events), 3) if events else None,
            "latency_s": _stats([e["thermal_s"] - e["start_s"] for e in caught]),
            "cylinder_correct": (round(sum(e["thermal_cylinder"] == e["true_cylinder"]
                                           for e in named) / len(named), 3) if named else None),
            "false_alert_episodes_on_runs_without_it": sum(
                r["false_alert_episodes"][fault] for r in without),
            "false_alert_episodes_on_fault_free_runs": sum(
                r["false_alert_episodes"][fault] for r in fault_free),
        }
    out["fault_free_minutes"] = round(minutes, 2)
    return out


def summarize_confirmation(runs: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = [e for r in runs for e in r["episodes"]]
    out = {}
    for fault in FAULTS:
        for truth_flag, name in ((True, "true_alerts"), (False, "false_alerts")):
            group = [e for e in episodes if e["fault"] == fault and e["true"] == truth_flag]
            out.setdefault(fault, {})[name] = {
                "episodes": len(group),
                "thermal_confirmed": sum(e["thermal_confirmed"] for e in group),
                "crank_confirmed": sum(e["crank_confirmed"] for e in group)}
    minutes = sum(r["knock_free_minutes"] for r in runs)
    fused = {}
    for name in ("acoustic", "acoustic_and_thermal"):
        detected = [(t, iv) for r in runs
                    for t, iv in zip(r["knock_variants"][name]["detected"], r["knock_intervals"],
                                     strict=True)]
        sustained = [(t, iv) for t, iv in detected if iv[1] - iv[0] >= 3.0]
        false = sum(r["knock_variants"][name]["false_alert_episodes"] for r in runs)
        fused[name] = {
            "sustained_intervals": len(sustained),
            "sustained_recall": (round(sum(t is not None for t, _ in sustained)
                                       / len(sustained), 3) if sustained else None),
            "latency_s": _stats([t - iv[0] for t, iv in sustained if t is not None]),
            "false_alert_episodes": false,
            "false_alerts_per_knock_free_minute": round(false / minutes, 3) if minutes else None}
    out["knock_fusion_experiment"] = fused
    return out


def _stats(values: list[float]) -> dict[str, float | None]:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "p50": None, "p95": None, "mean": None}
    return {"n": len(values), "p50": round(float(np.percentile(values, 50)), 3),
            "p95": round(float(np.percentile(values, 95)), 3),
            "mean": round(float(np.mean(values)), 3)}


def summarize_benchmark(runs: list[dict[str, Any]]) -> dict[str, Any]:
    events = [e for r in runs for e in r["misfire"]]
    n = len(events)

    def rate(key):
        return round(sum(e[key] is not None for e in events) / n, 3) if n else None

    def correct(key):
        named = [e for e in events if e[key] is not None]
        return (round(sum(e[key] == e["true_cylinder"] for e in named) / len(named), 3)
                if named else None)

    leads_a = [e["ecu_A_s"] - e["acoustic_s"] for e in events
               if e["ecu_A_s"] is not None and e["acoustic_s"] is not None]
    censored = [e for e in events if e["acoustic_s"] is not None and e["ecu_A_s"] is None]
    both_independent = [e for e in events if e["acoustic_cylinder"] is not None
                        and e["crank_cylinder"] is not None]
    others = [o for r in runs for o in r["other_faults"]]
    no_misfire = [r for r in runs if not r["misfire"]]
    minutes = sum(r["duration_s"] for r in no_misfire) / 60
    return {
        "misfire_intervals": n,
        "detected": {"acoustic": rate("acoustic_s"), "crank": rate("crank_s"),
                     "ecu_type_a": rate("ecu_A_s"), "ecu_type_b": rate("ecu_B_s")},
        "latency_s": {
            "acoustic": _stats([e["acoustic_s"] - e["start_s"] for e in events
                                if e["acoustic_s"] is not None]),
            "crank": _stats([e["crank_s"] - e["start_s"] for e in events
                             if e["crank_s"] is not None]),
            "ecu_type_a": _stats([e["ecu_A_s"] - e["start_s"] for e in events
                                  if e["ecu_A_s"] is not None]),
            "ecu_type_b": _stats([e["ecu_B_s"] - e["start_s"] for e in events
                                  if e["ecu_B_s"] is not None])},
        "acoustic_lead_over_ecu_type_a_s": _stats(leads_a),
        "acoustic_alerted_ecu_type_a_silent": len(censored),
        # Time from the acoustic alert to the end of the run, during which the ECU stayed silent.
        "ecu_type_a_silent_for_s_after_acoustic_alert": _stats(
            [r["duration_s"] - e["acoustic_s"] for r in runs for e in r["misfire"]
             if e["acoustic_s"] is not None and e["ecu_A_s"] is None]),
        "cylinder_correct": {"acoustic_localizer": correct("acoustic_cylinder"),
                             "crank": correct("crank_cylinder"),
                             "ecu_type_a": correct("ecu_A_cylinder")},
        "independent_paths_agree": (round(sum(e["acoustic_cylinder"] == e["crank_cylinder"]
                                              for e in both_independent)
                                          / len(both_independent), 3)
                                    if both_independent else None),
        "other_faults": {f: {"intervals": sum(o["fault"] == f for o in others),
                             "acoustic_detected": sum(o["fault"] == f and o["acoustic_s"]
                                                      is not None for o in others),
                             "ecu_capability": "none (crank-speed misfire monitor only)"}
                         for f in FAULTS if f != MISFIRE},
        "false_alarms_on_runs_without_misfire": {
            "runs": len(no_misfire), "minutes": round(minutes, 2),
            "crank_alert_episodes": sum(r["crank_false_alert_episodes"] for r in no_misfire),
            "ecu_dtcs": sum(len(r["ecu_false_dtcs"]) for r in no_misfire)},
    }


def crosscheck_benchmark(model, raw_dir: Path, truth_dir: Path, config: dict,
                         crank: CrankCalibration, thermal: ThermalCalibration,
                         localizer=None, splits: tuple[str, ...] = ("test",)) -> dict[str, Any]:
    """One replay per held-out run; OBD, thermal and confirmation sections per subset."""
    run_ids = [p.name for p in find_runs(raw_dir)]
    truths = load_truths(truth_dir, run_ids)
    report: dict[str, Any] = {
        "status": "Independent sensor paths and simulated OBD monitor on held-out engines; "
                  "synthetic data only",
        "acoustic_model_version": model.model_version,
        "localizer_version": getattr(localizer, "model_version", None),
        "crank_model_version": crank.model_version,
        "thermal_model_version": thermal.model_version,
        "splits": list(splits), "obd_misfire": config["obd_misfire"],
        "crank_thresholds": {"rpm_bin_edges": crank.rpm_bin_edges,
                             "stroke_thresholds": crank.stroke_thresholds,
                             "cylinder_threshold": crank.cylinder_threshold},
        "thermal_calibration": {"spread_c": thermal.spread_c, "thresholds": thermal.thresholds,
                                "baseline_s": thermal.baseline_s},
    }
    for subset, challenge in (("core", False), ("challenge", True)):
        chosen = [r for r in run_ids if truths[r].get("split") in splits
                  and bool(truths[r].get("challenge")) == challenge]
        obd, heat, confirm = [], [], []
        for run_id in chosen:
            records = replay_run(load_run(Path(raw_dir) / run_id), model, config,
                                 localizer=localizer, crank=crank, thermal=thermal)
            obd.append(benchmark_run(records, truths[run_id], config))
            heat.append(thermal_run(records, truths[run_id], config))
            confirm.append(confirmation_run(records, truths[run_id], config))
        if obd:
            report[subset] = {
                "runs": len(obd),
                "obd": {**summarize_benchmark(obd), "per_run": obd},
                "thermal": {**summarize_thermal(heat), "per_run": heat},
                "confirmation": summarize_confirmation(confirm),
            }
    return report
