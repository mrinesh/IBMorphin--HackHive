"""One-time final evaluation of a frozen model on the held-out test engines.

Window metrics use the offline feature table; alert metrics replay every test
run through ReplayEngine exactly as the dashboard does. Truth is read only
after decisions are made, to score them.
"""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from engine_diagnostics.data.run_io import load_run
from engine_diagnostics.evaluation import (
    localization_episodes,
    score_run,
    severity_at,
    summarize,
    summarize_localization,
)
from engine_diagnostics.models.diagnostic import thin
from engine_diagnostics.models.temporary import (
    ABLATIONS,
    UNCERTAIN,
    classification_metrics,
    fit_svm,
)
from engine_diagnostics.pipeline.labels import window_labels
from engine_diagnostics.runtime.replay import replay_run

DEGRADATION_WINDOW = 10  # decisions (5 s) in the rolling valve-probability indicator
VALVE = "valve_clearance_like"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def degradation_indicator(records: list[dict[str, Any]]) -> np.ndarray:
    """Rolling mean of the valve-class probability: a simulated wear trend, not RUL."""
    values = np.array([r.get("probabilities", {}).get(VALVE, np.nan) for r in records])
    series = pd.Series(values).rolling(DEGRADATION_WINDOW, min_periods=1).mean()
    return series.to_numpy()


def window_metrics(model, rows: pd.DataFrame) -> dict[str, Any]:
    clean = rows[(rows["label"] != "transition") & (rows["quality"] == "ok")]
    results = model.predict(clean[model.feature_names].to_numpy())
    truth = clean["label"].to_numpy()
    predicted = np.array([r["predicted"] for r in results])
    accepted = np.array([r["accepted"] for r in results])
    resolved = accepted != UNCERTAIN
    failures = sorted(
        ({"run_id": run, "t_end_s": float(t), "truth": tr, "predicted": r["predicted"],
          "probability": r["probability"]}
         for run, t, tr, r in zip(clean["run_id"], clean["t_end_s"], truth, results, strict=True)
         if r["predicted"] != tr),
        key=lambda f: -f["probability"])[:5]
    return {
        "top_class": classification_metrics(truth, predicted, model.classes),
        "accepted": {"coverage": float(resolved.mean()) if len(resolved) else None,
                     **classification_metrics(truth[resolved], accepted[resolved], model.classes)},
        "excluded_windows": {"transition": int((rows["label"] == "transition").sum()),
                             "non_ok_quality": int((rows["quality"] != "ok").sum())},
        "most_confident_errors": failures,
    }


def replay_metrics(model, run_ids: list[str], raw_dir: Path, truths: dict[str, dict],
                   config: dict) -> dict[str, Any]:
    window_s, step_s = config["analysis"]["window_s"], config["analysis"]["step_s"]
    scores, latencies, degradation = [], [], []
    for run_id in run_ids:
        records = replay_run(load_run(raw_dir / run_id), model, config)
        truth = truths[run_id]
        times = [r["t_end_s"] for r in records]
        scores.append(score_run(times, [r["alert"] for r in records],
                                window_labels(truth, np.array(times), window_s),
                                truth, window_s, step_s))
        latencies += [r["processing_ms"] for r in records]
        if truth.get("degradation"):
            indicator = degradation_indicator(records)
            if np.all(np.isnan(indicator)):
                degradation.append({"run_id": run_id, "spearman": None,
                                    "note": "no ok-quality windows (data-quality state only)"})
                continue
            severity = [severity_at(truth, t) for t in times]
            valid = ~np.isnan(indicator)
            rho = spearmanr(indicator[valid], np.array(severity)[valid]).statistic
            degradation.append({"run_id": run_id, "spearman": float(rho),
                                "indicator_first_5s": float(np.nanmean(indicator[:10])),
                                "indicator_last_5s": float(np.nanmean(indicator[-10:]))})
    rhos = [d["spearman"] for d in degradation if d["spearman"] is not None]
    return {**summarize(scores),
            "processing_ms_p50": float(np.percentile(latencies, 50)),
            "processing_ms_p95": float(np.percentile(latencies, 95)),
            "degradation_runs": degradation,
            "degradation_spearman_median": float(np.median(rhos)) if rhos else None}


def evaluate_final(model, table: pd.DataFrame, truths: dict[str, dict], raw_dir: Path,
                   config: dict, model_path: Path) -> dict[str, Any]:
    test = table[table["split"] == "test"]
    report = {
        "status": "FINAL evaluation on held-out test engines; synthetic data only",
        "model_version": model.model_version, "feature_version": model.feature_version,
        "model_sha256": sha256(model_path),
        "dataset_manifest_sha256": (sha256(raw_dir / "manifest.json")
                                    if (raw_dir / "manifest.json").is_file() else None),
        "test_engines": sorted(test["engine_id"].unique()),
        "settings": {"probability_min": model.probability_min,
                     "anomaly_threshold": model.anomaly_threshold,
                     "gate_fault_with_anomaly": model.gate_fault_with_anomaly,
                     "confirm_updates": model.confirm_updates,
                     "clear_updates": model.clear_updates,
                     "svm": model.notes.get("svm_selected")},
    }
    for subset in ("core", "challenge"):
        rows = test[test["subset"] == subset]
        if rows.empty:
            continue
        report[subset] = {
            "runs": int(rows["run_id"].nunique()),
            "windows": window_metrics(model, rows),
            "replay": replay_metrics(model, sorted(rows["run_id"].unique()), raw_dir,
                                     truths, config),
        }
    # Modality ablations: same partitions and hyperparameters, top-class macro-F1.
    train = table[(table["split"] == "train") & (table["subset"] == "core")
                  & (table["label"] != "transition") & (table["quality"] == "ok")]
    thinned = thin(train, model.notes.get("train_windows_used", 8000))
    core = test[(test["subset"] == "core") & (test["label"] != "transition")
                & (test["quality"] == "ok")]
    selected = model.notes.get("svm_selected", {"C": 10.0, "gamma": "scale"})
    report["ablations"] = {}
    for name, keep in ABLATIONS.items():
        names = [n for n in model.feature_names if keep(n)]
        svm = fit_svm(thinned, names, C=selected["C"], gamma=selected["gamma"])
        metrics = classification_metrics(core["label"].to_numpy(),
                                         svm.predict(core[names].to_numpy()), model.classes)
        report["ablations"][name] = {"macro_f1": metrics["macro_f1"], "recall": metrics["recall"]}
    return report


def multi_window_metrics(model, rows: pd.DataFrame) -> dict[str, Any]:
    """Per-fault and set-level window metrics for a multi-fault model."""
    from engine_diagnostics.models.multifault import multilabel_metrics
    from engine_diagnostics.pipeline.labels import FAULTS

    clean = rows[rows["quality"] == "ok"]
    results = model.predict(clean[model.feature_names].to_numpy())
    y_true = clean[[f"y_{f}" for f in FAULTS]].to_numpy(dtype=float)
    y_pred = np.array([[f in r["detected"] for f in FAULTS] for r in results])
    return {**multilabel_metrics(y_true, y_pred, list(FAULTS)),
            "excluded_windows": {"non_ok_quality": int((rows["quality"] != "ok").sum())}}


def single_label_baseline(model, rows: pd.DataFrame) -> dict[str, Any]:
    """A one-class model (frozen v1) on the same windows: its accepted class as the set."""
    from engine_diagnostics.models.multifault import multilabel_metrics
    from engine_diagnostics.pipeline.labels import FAULTS

    clean = rows[rows["quality"] == "ok"]
    results = model.predict(clean[model.feature_names].to_numpy())
    y_true = clean[[f"y_{f}" for f in FAULTS]].to_numpy(dtype=float)
    y_pred = np.array([[r["accepted"] == f for f in FAULTS] for r in results])
    return {"model_version": model.model_version,
            **multilabel_metrics(y_true, y_pred, list(FAULTS))}


def multi_replay_metrics(model, run_ids: list[str], raw_dir: Path, truths: dict[str, dict],
                         config: dict) -> dict[str, Any]:
    """Replay every run; score each fault's alert episodes against its own intervals."""
    from engine_diagnostics.pipeline.labels import FAULTS, fault_intervals, fault_window_labels

    window_s, step_s = config["analysis"]["window_s"], config["analysis"]["step_s"]
    per_fault = {f: [] for f in FAULTS}
    runs, latencies = [], []
    for run_id in run_ids:
        records = replay_run(load_run(raw_dir / run_id), model, config)
        truth = truths[run_id]
        times = [r["t_end_s"] for r in records]
        labels = fault_window_labels(truth, np.array(times), window_s)
        intervals = fault_intervals(truth)
        caught = {}
        for fault in FAULTS:
            alerts = [fault if fault in r.get("alerts", []) else None for r in records]
            own = {"condition": fault, "active_intervals": intervals.get(fault, [])}
            score = score_run(times, alerts, list(np.where(labels[fault] == 0, "normal", "other")),
                              own, window_s, step_s)
            per_fault[fault].append(score)
            if fault in intervals and score["sustained"]:
                caught[fault] = all(e["detected"] for e in score["sustained"])
        runs.append({"run_id": run_id, "faults": sorted(intervals),
                     "sustained_caught": caught,
                     "false_alert_episodes": {f: per_fault[f][-1]["false_alert_episodes"]
                                              for f in FAULTS}})
        latencies += [r["processing_ms"] for r in records]
    by_count = {}
    for n in range(len(FAULTS) + 1):
        group = [r for r in runs if len(r["faults"]) == n]
        judged = [r for r in group if r["sustained_caught"]]
        if group:
            by_count[str(n)] = {
                "runs": len(group),
                "runs_all_sustained_faults_alerted": (
                    sum(all(r["sustained_caught"].values()) for r in judged) / len(judged)
                    if judged else None),
                "runs_with_any_false_alert": sum(
                    any(r["false_alert_episodes"].values()) for r in group) / len(group),
            }
    return {"per_fault": {f: summarize(s) for f, s in per_fault.items()},
            "by_active_faults": by_count,
            "processing_ms_p50": float(np.percentile(latencies, 50)),
            "processing_ms_p95": float(np.percentile(latencies, 95)),
            "runs": runs}


def evaluate_final_multi(model, table: pd.DataFrame, truths: dict[str, dict], raw_dir: Path,
                         config: dict, model_path: Path, baseline=None) -> dict[str, Any]:
    test = table[table["split"] == "test"]
    report = {
        "status": "FINAL multi-fault evaluation on held-out test engines; synthetic data only",
        "model_version": model.model_version, "feature_version": model.feature_version,
        "model_sha256": sha256(model_path),
        "dataset_manifest_sha256": (sha256(raw_dir / "manifest.json")
                                    if (raw_dir / "manifest.json").is_file() else None),
        "test_engines": sorted(test["engine_id"].unique()),
        "settings": {"thresholds": model.thresholds, "bands": model.bands,
                     "confirm_updates": model.confirm_updates,
                     "clear_updates": model.clear_updates,
                     "anomaly_threshold": model.anomaly_threshold,
                     "svm": model.notes.get("svm_selected")},
    }
    for subset in ("core", "challenge"):
        rows = test[test["subset"] == subset]
        if rows.empty:
            continue
        report[subset] = {
            "runs": int(rows["run_id"].nunique()),
            "windows": multi_window_metrics(model, rows),
            "replay": multi_replay_metrics(model, sorted(rows["run_id"].unique()), raw_dir,
                                           truths, config),
        }
        if baseline is not None:
            report[subset]["single_label_baseline"] = single_label_baseline(baseline, rows)
    return report


def localization_replay_metrics(model, localizer, run_ids: list[str], raw_dir: Path,
                                truths: dict[str, dict], config: dict) -> dict[str, Any]:
    """Replay runs with detector + localizer; score each fault's alert episodes."""
    from engine_diagnostics.pipeline.labels import fault_cylinders, fault_intervals

    episodes = {f: [] for f in localizer.faults}
    latencies = []
    for run_id in run_ids:
        records = replay_run(load_run(raw_dir / run_id), model, config, localizer=localizer)
        cylinders = fault_cylinders(truths[run_id])
        intervals = fault_intervals(truths[run_id])
        for fault in localizer.faults:
            for e in localization_episodes(records, fault, cylinders.get(fault),
                                           intervals.get(fault, [])):
                episodes[fault].append({"run_id": run_id, **e})
        latencies += [r["processing_ms"] for r in records]
    return {"per_fault": {f: summarize_localization(e) for f, e in episodes.items()},
            "processing_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
            "episodes": episodes}


def evaluate_final_localization(model, localizer, table: pd.DataFrame,
                                truths: dict[str, dict], raw_dir: Path, config: dict,
                                model_path: Path, localizer_path: Path) -> dict[str, Any]:
    """Window-level (given the fault) and replay-level localization on test engines."""
    from engine_diagnostics.models.localizer import evaluate_windows

    test = table[table["split"] == "test"]
    report = {
        "status": "FINAL cylinder-localization evaluation on held-out test engines; "
                  "synthetic data only",
        "localizer_version": localizer.model_version,
        "cylinder_feature_version": localizer.feature_version,
        "localizer_sha256": sha256(localizer_path),
        "detector_version": model.model_version, "detector_sha256": sha256(model_path),
        "dataset_manifest_sha256": (sha256(raw_dir / "manifest.json")
                                    if (raw_dir / "manifest.json").is_file() else None),
        "test_engines": sorted(test["engine_id"].unique()),
        "settings": {"min_confidence": localizer.min_confidence,
                     "temperatures": localizer.temperatures},
        "windows_given_fault": evaluate_windows(localizer, test, truths, config, "test"),
    }
    for subset in ("core", "challenge"):
        rows = test[test["subset"] == subset]
        if not rows.empty:
            report[subset] = localization_replay_metrics(
                model, localizer, sorted(rows["run_id"].unique()), raw_dir, truths, config)
    return report
