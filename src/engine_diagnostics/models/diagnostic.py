"""Version 1 diagnostic model: operating-bin normal baseline, calibrated RBF SVM.

Partitions are used strictly by role:
  train        SVM fit and normal-baseline statistics
  tuning       SVM hyperparameters, decision thresholds, alert persistence
  calibration  sigmoid probability calibration of the frozen SVM
  test         final evaluation only (see engine_diagnostics.cli evaluate)
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
from engine_diagnostics.models.temporary import NORMAL, UNCERTAIN, fit_svm, trainable
from engine_diagnostics.pipeline.features import FEATURE_VERSION, feature_names

MODEL_VERSION = "v1.0.0"
RPM_EDGES = (1600.0, 2400.0)  # operating bins: <1600, 1600-2400, >=2400 rpm
LOAD_EDGE = 50.0  # percent
MIN_BIN_WINDOWS = 50
SVM_GRID = {"C": (1.0, 10.0, 30.0), "gamma": ("scale", 0.03)}
DECISION_GRID = {
    "probability_min": (0.4, 0.5, 0.6, 0.7),
    "anomaly_quantile": (0.95, 0.99, 0.995),
    "gate_fault_with_anomaly": (True, False),
    "confirm_updates": (2, 3),
}
FALSE_ALERT_LIMIT = 0.2  # episodes per normal minute (checklist §7 goal)


@dataclass
class DiagnosticModel:
    feature_names: list[str]
    classes: list[str]
    svm: Any
    calibrated: Any
    baseline_names: list[str]
    bin_stats: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]
    global_stats: tuple[np.ndarray, np.ndarray]
    normal_scores: np.ndarray
    anomaly_threshold: float = 0.0
    probability_min: float = 0.5
    gate_fault_with_anomaly: bool = True
    confirm_updates: int = 2
    clear_updates: int = 3
    trained_on: list[str] = field(default_factory=list)
    model_version: str = MODEL_VERSION
    feature_version: str = FEATURE_VERSION
    notes: dict[str, Any] = field(default_factory=dict)

    # Normal baseline -------------------------------------------------------
    @staticmethod
    def operating_bin(rpm: float, load: float) -> tuple[int, int] | None:
        if not (np.isfinite(rpm) and np.isfinite(load)):
            return None
        return int(np.searchsorted(RPM_EDGES, rpm, side="right")), int(load >= LOAD_EDGE)

    def anomaly_score(self, x: np.ndarray) -> np.ndarray:
        """RMS robust z-score against normal windows of the same RPM/load bin."""
        x = np.atleast_2d(x)
        idx = [self.feature_names.index(n) for n in self.baseline_names]
        rpm = x[:, self.feature_names.index("ctx_rpm")]
        load = x[:, self.feature_names.index("ctx_load")]
        scores = np.empty(len(x))
        for k in range(len(x)):
            median, scale = self.bin_stats.get(self.operating_bin(rpm[k], load[k]),
                                               self.global_stats)
            z = (x[k, idx] - median) / scale
            scores[k] = np.sqrt(np.mean(np.clip(z, -20, 20) ** 2))
        return scores

    # Decisions -------------------------------------------------------------
    def probabilities(self, x: np.ndarray) -> np.ndarray:
        return self.calibrated.predict_proba(np.atleast_2d(x))

    def predict(self, x: np.ndarray) -> list[dict[str, Any]]:
        x = np.atleast_2d(x)
        proba = self.probabilities(x)
        anomaly = self.anomaly_score(x)
        results = []
        for k in range(len(x)):
            order = np.argsort(proba[k])
            predicted = self.classes[order[-1]]
            top = float(proba[k, order[-1]])
            anomalous = bool(anomaly[k] > self.anomaly_threshold)
            accepted = self._accept(predicted, top, anomalous)
            results.append({
                "predicted": predicted, "probability": top,
                "margin": float(top - proba[k, order[-2]]),
                "probabilities": dict(zip(self.classes, map(float, proba[k]), strict=True)),
                "anomaly_score": float(anomaly[k]), "anomalous": anomalous,
                "accepted": accepted,
            })
        return results

    def _accept(self, predicted: str, probability: float, anomalous: bool) -> str:
        if probability < self.probability_min:
            return UNCERTAIN
        if predicted == NORMAL:
            return NORMAL if not anomalous else UNCERTAIN
        if self.gate_fault_with_anomaly and not anomalous:
            return UNCERTAIN
        return predicted


def thin(train: pd.DataFrame, max_windows: int, seed: int = 0) -> pd.DataFrame:
    """Class-balanced subsample capped at max_windows, spread across runs."""
    per_class = max_windows // train["label"].nunique()
    parts = [rows.sample(min(len(rows), per_class), random_state=seed)
             for _, rows in train.groupby("label")]
    return pd.concat(parts).sort_index().reset_index(drop=True)


def _baseline(normal: pd.DataFrame, names: list[str]):
    def stats(rows):
        values = rows[names].to_numpy()
        median = np.median(values, axis=0)
        scale = 1.4826 * np.median(np.abs(values - median), axis=0)
        return median, np.where(scale > 0, scale, 1.0)

    bins = {}
    keys = [DiagnosticModel.operating_bin(r, ld)
            for r, ld in zip(normal["ctx_rpm"], normal["ctx_load"], strict=True)]
    normal = normal.assign(_bin=keys)
    for key, rows in normal.groupby("_bin"):
        if key is not None and len(rows) >= MIN_BIN_WINDOWS:
            bins[key] = stats(rows)
    return bins, stats(normal)


def run_sequences(table: pd.DataFrame, split: str) -> list[pd.DataFrame]:
    """Every window of each core run in a partition, in time order (replay order)."""
    rows = table[(table["split"] == split) & (table["subset"] == "core")]
    return [run.sort_values("t_end_s") for _, run in rows.groupby("run_id")]


def score_cache(model: DiagnosticModel, runs: list[pd.DataFrame]) -> list[dict[str, Any]]:
    """Threshold-independent outputs per run, so decision settings can be swept cheaply."""
    cache = []
    for run in runs:
        ok = run["quality"].to_numpy() == "ok"
        x = run.loc[ok, model.feature_names].to_numpy()
        proba = model.probabilities(x) if ok.any() else np.empty((0, len(model.classes)))
        cache.append({"run": run, "ok": ok, "proba": proba,
                      "anomaly": model.anomaly_score(x) if ok.any() else np.empty(0)})
    return cache


def episode_summary(model: DiagnosticModel, cache: list[dict[str, Any]],
                    truths: dict[str, dict], config: dict) -> dict[str, Any]:
    window_s, step_s = config["analysis"]["window_s"], config["analysis"]["step_s"]
    scores = []
    for item in cache:
        run, ok = item["run"], item["ok"]
        accepted = np.full(len(run), UNCERTAIN, dtype=object)
        top = item["proba"].argmax(axis=1)
        accepted[ok] = [
            model._accept(model.classes[k], float(item["proba"][i, k]),
                          bool(item["anomaly"][i] > model.anomaly_threshold))
            for i, k in enumerate(top)]
        alerts = alert_sequence(list(accepted), list(run["quality"]),
                                model.confirm_updates, model.clear_updates)
        scores.append(score_run(list(run["t_end_s"]), alerts, list(run["label"]),
                                truths[run["run_id"].iloc[0]], window_s, step_s))
    return summarize(scores)


def fit_diagnostic_model(table: pd.DataFrame, truths: dict[str, dict], config: dict,
                         max_train_windows: int = 8000) -> DiagnosticModel:
    names = feature_names()
    train = trainable(table, "train")
    tuning = trainable(table, "tuning")
    calibration = trainable(table, "calibration")
    thinned = thin(train, max_train_windows)

    # 1. SVM hyperparameters by macro-F1 on tuning engines.
    search = []
    for c, gamma in itertools.product(SVM_GRID["C"], SVM_GRID["gamma"]):
        svm = fit_svm(thinned, names, C=c, gamma=gamma)
        score = f1_score(tuning["label"], svm.predict(tuning[names].to_numpy()), average="macro")
        search.append({"C": c, "gamma": gamma, "tuning_macro_f1": float(score), "model": svm})
    best = max(search, key=lambda s: s["tuning_macro_f1"])

    # 2. Normal baseline per operating bin from all clean training normal windows.
    baseline_names = [n for n in names if not n.startswith("ctx_")]
    bins, global_stats = _baseline(train[train["label"] == NORMAL], baseline_names)

    # 3. Sigmoid calibration of the frozen SVM on calibration engines.
    calibrated = CalibratedClassifierCV(FrozenEstimator(best["model"]), method="sigmoid")
    calibrated.fit(calibration[names].to_numpy(), calibration["label"].to_numpy())

    model = DiagnosticModel(
        feature_names=names, classes=list(calibrated.classes_), svm=best["model"],
        calibrated=calibrated, baseline_names=baseline_names, bin_stats=bins,
        global_stats=global_stats, normal_scores=np.empty(0),
        clear_updates=config["runtime"]["clear_fault_updates"],
        trained_on=sorted(train["run_id"].unique()),
    )
    model.normal_scores = model.anomaly_score(
        train.loc[train["label"] == NORMAL, names].to_numpy())

    # 4. Decision and alert settings by replay-equivalent episodes on tuning runs.
    tuning_cache = score_cache(model, run_sequences(table, "tuning"))
    trials = []
    for p_min, quantile, gate, confirm in itertools.product(*DECISION_GRID.values()):
        model.probability_min, model.gate_fault_with_anomaly = p_min, gate
        model.anomaly_threshold = float(np.quantile(model.normal_scores, quantile))
        model.confirm_updates = confirm
        summary = episode_summary(model, tuning_cache, truths, config)
        trials.append({"probability_min": p_min, "anomaly_quantile": quantile,
                       "gate_fault_with_anomaly": gate, "confirm_updates": confirm, **summary})
    feasible = [t for t in trials
                if t["false_alerts_per_normal_minute"] is not None
                and t["false_alerts_per_normal_minute"] <= FALSE_ALERT_LIMIT]
    pool = feasible or trials
    chosen = max(pool, key=lambda t: (t["sustained_recall"] or 0,
                                      -(t["wrong_class_episodes"]),
                                      -(t["latency_p50_s"] or 99)))
    model.probability_min = chosen["probability_min"]
    model.gate_fault_with_anomaly = chosen["gate_fault_with_anomaly"]
    model.confirm_updates = chosen["confirm_updates"]
    model.anomaly_threshold = float(np.quantile(model.normal_scores, chosen["anomaly_quantile"]))
    model.notes = {
        "train_windows": len(train), "train_windows_used": len(thinned),
        "tuning_windows": len(tuning), "calibration_windows": len(calibration),
        "svm_search": [{k: v for k, v in s.items() if k != "model"} for s in search],
        "svm_selected": {"C": best["C"], "gamma": best["gamma"]},
        "baseline_bins": {f"{k}": int(v) for k, v in _bin_counts(train, baseline_names).items()},
        "decision_trials": trials,
        "decision_selected": chosen,
        "false_alert_limit_met_on_tuning": bool(feasible),
    }
    return model


def _bin_counts(train: pd.DataFrame, names: list[str]) -> dict:
    normal = train[train["label"] == NORMAL]
    keys = [DiagnosticModel.operating_bin(r, ld)
            for r, ld in zip(normal["ctx_rpm"], normal["ctx_load"], strict=True)]
    return pd.Series(keys).value_counts().to_dict()
