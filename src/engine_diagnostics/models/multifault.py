"""Version 2 multi-fault model: one calibrated detector per fault.

Each fault (misfire, knock_like, valve_clearance_like) has its own RBF SVM,
fault present vs absent, so any combination can be reported in one window.
Partitions are used strictly by role, as in v1:
  train        SVM fits and the operating-bin normal baseline
  tuning       per-fault SVM hyperparameters, thresholds, alert persistence
  calibration  sigmoid probability calibration of each frozen SVM
  test         final evaluation only (see engine_diagnostics.cli evaluate-multi)
A window's label for one fault is NaN near that fault's changes and is then
ignored for that fault only. The normal baseline (anomaly score) only decides
whether a window with no detected fault is shown as normal or uncertain; it
never gates or clears a fault.
"""

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import f1_score

from engine_diagnostics.evaluation import alert_sequence, score_run, summarize
from engine_diagnostics.models.diagnostic import (
    FALSE_ALERT_LIMIT,
    SVM_GRID,
    DiagnosticModel,
    _baseline,
    run_sequences,
    thin,
)
from engine_diagnostics.models.temporary import NORMAL, UNCERTAIN, fit_svm
from engine_diagnostics.pipeline.features import FEATURE_VERSION, feature_names
from engine_diagnostics.pipeline.labels import FAULTS, fault_intervals

MODEL_VERSION = "v2.0.0"
ABSENT = "absent"
FAULT, CLEAR = "fault", "normal"  # per-fault decisions fed to MultiAlertState
ANOMALY_QUANTILE = 0.99
DECISION_GRID = {
    "threshold": (0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    "band": (0.0, 0.1, 0.2),  # below threshold by less than this: uncertain, not clear
    "confirm_updates": (2, 3),
}


@dataclass
class MultiFaultModel:
    feature_names: list[str]
    fault_classes: list[str]
    detectors: dict[str, Any]
    svms: dict[str, Any]
    baseline_names: list[str]
    bin_stats: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]
    global_stats: tuple[np.ndarray, np.ndarray]
    normal_scores: np.ndarray
    anomaly_threshold: float = 0.0
    thresholds: dict[str, float] = field(default_factory=dict)
    bands: dict[str, float] = field(default_factory=dict)
    confirm_updates: dict[str, int] = field(default_factory=dict)
    clear_updates: int = 3
    trained_on: list[str] = field(default_factory=list)
    model_version: str = MODEL_VERSION
    feature_version: str = FEATURE_VERSION
    notes: dict[str, Any] = field(default_factory=dict)

    # Same operating-bin normal baseline as v1.
    operating_bin = staticmethod(DiagnosticModel.operating_bin)
    anomaly_score = DiagnosticModel.anomaly_score

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        """Calibrated P(fault present), one column per fault in fault_classes."""
        x = np.atleast_2d(x)
        columns = []
        for fault in self.fault_classes:
            detector = self.detectors[fault]
            columns.append(detector.predict_proba(x)[:, list(detector.classes_).index(fault)])
        return np.column_stack(columns)

    def fault_decisions(self, proba_row: np.ndarray) -> dict[str, str]:
        decisions = {}
        for fault, p in zip(self.fault_classes, proba_row, strict=True):
            threshold = self.thresholds[fault]
            decisions[fault] = (FAULT if p >= threshold
                                else UNCERTAIN if p >= threshold - self.bands[fault] else CLEAR)
        return decisions

    def predict(self, x: np.ndarray) -> list[dict[str, Any]]:
        """Per-fault probabilities and decisions; `accepted` names the detected set.

        accepted is the sorted detected faults joined with '+', else uncertain
        when any fault is borderline or the window deviates from normal, else
        normal. Alerts use fault_decisions, one independent state per fault.
        """
        x = np.atleast_2d(x)
        proba = self.probabilities(x)
        anomaly = self.anomaly_score(x)
        results = []
        for k in range(len(x)):
            decisions = self.fault_decisions(proba[k])
            detected = sorted(f for f, d in decisions.items() if d == FAULT)
            borderline = sorted(f for f, d in decisions.items() if d == UNCERTAIN)
            anomalous = bool(anomaly[k] > self.anomaly_threshold)
            if detected:
                accepted = "+".join(detected)
            else:
                accepted = UNCERTAIN if borderline or anomalous else NORMAL
            distance = [abs(p - self.thresholds[f])
                        for f, p in zip(self.fault_classes, proba[k], strict=True)]
            results.append({
                "predicted": "+".join(detected) or NORMAL,
                "probability": float(proba[k].max()),
                "margin": float(min(distance)),
                "probabilities": dict(zip(self.fault_classes, map(float, proba[k]), strict=True)),
                "fault_decisions": decisions, "detected": detected,
                "uncertain_faults": borderline,
                "anomaly_score": float(anomaly[k]), "anomalous": anomalous,
                "accepted": accepted,
            })
        return results


def fault_rows(table: pd.DataFrame, split: str, fault: str) -> pd.DataFrame:
    """Clean core windows of a partition where this fault's label is known."""
    rows = table[(table["split"] == split) & (table["quality"] == "ok")
                 & table[f"y_{fault}"].notna()]
    if "subset" in rows:
        rows = rows[rows["subset"] == "core"]
    return rows.assign(label=np.where(rows[f"y_{fault}"] == 1, fault, ABSENT)).reset_index(
        drop=True)


def normal_rows(table: pd.DataFrame, split: str) -> pd.DataFrame:
    """Windows known to have no fault at all."""
    rows = table[(table["split"] == split) & (table["quality"] == "ok")]
    if "subset" in rows:
        rows = rows[rows["subset"] == "core"]
    return rows[np.all(rows[[f"y_{f}" for f in FAULTS]].to_numpy() == 0, axis=1)]


def fault_episode_summary(model: MultiFaultModel, fault: str, cache: list[dict[str, Any]],
                          truths: dict[str, dict], config: dict, threshold: float, band: float,
                          confirm: int) -> dict[str, Any]:
    """Replay-equivalent alert episodes for one fault against its own intervals."""
    window_s, step_s = config["analysis"]["window_s"], config["analysis"]["step_s"]
    column = model.fault_classes.index(fault)
    scores = []
    for item in cache:
        run, ok = item["run"], item["ok"]
        decisions = np.full(len(run), UNCERTAIN, dtype=object)
        p = item["proba"][:, column]
        decisions[ok] = np.where(p >= threshold, fault,
                                 np.where(p >= threshold - band, UNCERTAIN, NORMAL))
        alerts = alert_sequence(list(decisions), list(run["quality"]), confirm,
                                model.clear_updates)
        labels = np.where(run[f"y_{fault}"].to_numpy() == 0, NORMAL, "other")
        truth = truths[run["run_id"].iloc[0]]
        own = {"condition": fault, "active_intervals": fault_intervals(truth).get(fault, [])}
        scores.append(score_run(list(run["t_end_s"]), alerts, list(labels), own,
                                window_s, step_s))
    return summarize(scores)


def fit_multifault_model(table: pd.DataFrame, truths: dict[str, dict], config: dict,
                         max_train_windows: int = 6000) -> MultiFaultModel:
    names = feature_names()
    detectors, svms, search, svm_selected = {}, {}, {}, {}
    counts = {}
    for fault in FAULTS:
        train = fault_rows(table, "train", fault)
        tuning = fault_rows(table, "tuning", fault)
        calibration = fault_rows(table, "calibration", fault)
        thinned = thin(train, max_train_windows)
        counts[fault] = {"train": len(train), "train_used": len(thinned),
                         "train_present": int((train["label"] == fault).sum()),
                         "tuning": len(tuning), "calibration": len(calibration)}

        # 1. SVM hyperparameters by this fault's F1 on tuning engines.
        trials = []
        for c, gamma in itertools.product(SVM_GRID["C"], SVM_GRID["gamma"]):
            svm = fit_svm(thinned, names, C=c, gamma=gamma)
            score = f1_score(tuning["label"], svm.predict(tuning[names].to_numpy()),
                             pos_label=fault, average="binary", zero_division=0)
            trials.append({"C": c, "gamma": gamma, "tuning_f1": float(score), "model": svm})
        best = max(trials, key=lambda t: t["tuning_f1"])
        search[fault] = [{k: v for k, v in t.items() if k != "model"} for t in trials]
        svm_selected[fault] = {"C": best["C"], "gamma": best["gamma"]}
        svms[fault] = best["model"]

        # 2. Sigmoid calibration of the frozen SVM on calibration engines.
        detectors[fault] = CalibratedClassifierCV(FrozenEstimator(best["model"]),
                                                  method="sigmoid").fit(
            calibration[names].to_numpy(), calibration["label"].to_numpy())

    # 3. Normal baseline per operating bin from training windows with no fault.
    baseline_names = [n for n in names if not n.startswith("ctx_")]
    normal = normal_rows(table, "train")
    bins, global_stats = _baseline(normal, baseline_names)
    model = MultiFaultModel(
        feature_names=names, fault_classes=list(FAULTS), detectors=detectors, svms=svms,
        baseline_names=baseline_names, bin_stats=bins, global_stats=global_stats,
        normal_scores=np.empty(0), clear_updates=config["runtime"]["clear_fault_updates"],
        trained_on=sorted(table.loc[(table["split"] == "train"), "run_id"].unique()),
    )
    model.normal_scores = model.anomaly_score(normal[names].to_numpy())
    model.anomaly_threshold = float(np.quantile(model.normal_scores, ANOMALY_QUANTILE))

    # 4. Per-fault threshold, band, and persistence by alert episodes on tuning runs.
    cache = []
    for run in run_sequences(table, "tuning"):
        ok = run["quality"].to_numpy() == "ok"
        x = run.loc[ok, names].to_numpy()
        cache.append({"run": run, "ok": ok,
                      "proba": model.probabilities(x) if ok.any()
                      else np.empty((0, len(FAULTS)))})
    decision_trials, decision_selected, limit_met = {}, {}, {}
    for fault in FAULTS:
        trials = []
        for threshold, band, confirm in itertools.product(*DECISION_GRID.values()):
            summary = fault_episode_summary(model, fault, cache, truths, config,
                                            threshold, band, confirm)
            trials.append({"threshold": threshold, "band": band, "confirm_updates": confirm,
                           **summary})
        feasible = [t for t in trials if t["false_alerts_per_normal_minute"] is not None
                    and t["false_alerts_per_normal_minute"] <= FALSE_ALERT_LIMIT]
        chosen = max(feasible or trials,
                     key=lambda t: (t["sustained_recall"] or 0,
                                    -(t["false_alerts_per_normal_minute"] or 0),
                                    -(t["latency_p50_s"] or 99)))
        model.thresholds[fault] = chosen["threshold"]
        model.bands[fault] = chosen["band"]
        model.confirm_updates[fault] = chosen["confirm_updates"]
        decision_trials[fault], decision_selected[fault] = trials, chosen
        limit_met[fault] = bool(feasible)
    model.notes = {
        "windows": counts, "normal_train_windows": len(normal),
        "svm_search": search, "svm_selected": svm_selected,
        "anomaly_quantile": ANOMALY_QUANTILE,
        "decision_trials": decision_trials, "decision_selected": decision_selected,
        "false_alert_limit_met_on_tuning": limit_met,
    }
    return model


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       faults: list[str]) -> dict[str, Any]:
    """Per-fault and set-level window metrics. NaN truth is ignored per fault.

    Set-level metrics use windows whose labels are known for every fault, and
    are broken down by how many faults are truly active (0-3).
    """
    y_pred = y_pred.astype(bool)
    per_fault = {}
    for k, fault in enumerate(faults):
        known = ~np.isnan(y_true[:, k])
        t, p = y_true[known, k] == 1, y_pred[known, k]
        tp, fp, fn = int(np.sum(t & p)), int(np.sum(~t & p)), int(np.sum(t & ~p))
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else None
        per_fault[fault] = {"windows": int(known.sum()), "present": int(t.sum()),
                            "precision": precision, "recall": recall, "f1": f1,
                            "false_positive_rate": fp / int((~t).sum()) if (~t).any() else None}
    known = ~np.isnan(y_true).any(axis=1)
    truth, pred = y_true[known] == 1, y_pred[known]
    active = truth.sum(axis=1)
    by_count = {}
    for n in range(len(faults) + 1):
        rows = active == n
        if not rows.any():
            continue
        t, p = truth[rows], pred[rows]
        by_count[str(n)] = {
            "windows": int(rows.sum()),
            "exact_set": float(np.mean(np.all(t == p, axis=1))),
            "all_true_detected": float(np.mean(np.all(p | ~t, axis=1))),
            "no_false_fault": float(np.mean(np.all(~p | t, axis=1))),
            "recall": {f: (float(p[t[:, k], k].mean()) if t[:, k].any() else None)
                       for k, f in enumerate(faults)},
        }
    f1s = [m["f1"] for m in per_fault.values() if m["f1"] is not None]
    return {
        "per_fault": per_fault,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "set_windows": int(known.sum()),
        "exact_set": float(np.mean(np.all(truth == pred, axis=1))) if known.any() else None,
        "hamming_loss": float(np.mean(truth != pred)) if known.any() else None,
        "by_active_faults": by_count,
    }
