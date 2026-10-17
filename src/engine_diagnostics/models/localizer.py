"""Cylinder localizer: which cylinder carries each detected fault.

Detection (v1/v2 models) says *what* is wrong; this model says *where*. It
scores each firing-order position of a window's cylinder signature
(pipeline.cylinders) as the candidate for one fault, with one model per fault:

  candidate vector  the window's contrast rows rotated so the candidate comes
                    first, x_p = [c[p], c[p+1], c[p+2], c[p+3]] (positions mod
                    4), plus their absolute values (valve faults may raise or
                    lower impact energy). The rotation lets the model learn
                    where a fault on position p shows up in *other* positions,
                    such as valve impacts that land while later cylinders fire.
  scorer            standardized L2 logistic regression on candidate rows
                    (candidate is the affected cylinder or not). The same
                    weights score every position, so the model is rotation-
                    equivariant and also applies to relative (cam-less) slots.
  probabilities     softmax of the four candidate scores over a temperature
                    fitted by log loss (a conditional-logit calibration).

Partitions are used strictly by role, as in v1/v2:
  train        logistic regression fits
  tuning       regularization C, then the per-fault min_confidence
  calibration  softmax temperature
  test         final evaluation only (engine-diagnostics evaluate-localization)

Training uses windows with the cam phase reference in which the fault is
present (its per-fault label is 1). Labels come from evaluation truth; truth
never reaches the signature or a prediction.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from engine_diagnostics.pipeline.cylinders import (
    CYLINDER_FEATURE_VERSION,
    PHASE_CAM,
    signature_names,
)
from engine_diagnostics.pipeline.labels import FAULTS, fault_cylinders

MODEL_VERSION = "localizer-1.0.0"
C_GRID = (0.01, 0.1, 1.0, 10.0)
CONFIDENCE_GRID = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
TARGET_ACCURACY = 0.95  # min_confidence: accepted windows must be at least this accurate
MAX_TRAIN_WINDOWS = 8000


@dataclass
class CylinderLocalizer:
    faults: list[str]
    firing_order: list[int]
    signature_names: list[str]
    scorers: dict[str, Pipeline]
    temperatures: dict[str, float] = field(default_factory=dict)
    min_confidence: dict[str, float] = field(default_factory=dict)
    model_version: str = MODEL_VERSION
    feature_version: str = CYLINDER_FEATURE_VERSION
    trained_on: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def scores(self, fault: str, contrast: np.ndarray) -> np.ndarray:
        """Raw candidate scores, one per firing-order position."""
        return self.scorers[fault].decision_function(candidate_matrix(contrast))

    def position_probabilities(self, fault: str, contrast: np.ndarray) -> np.ndarray:
        """Calibrated P(position p carries the fault), summing to 1 over positions."""
        return softmax(self.scores(fault, contrast) / self.temperatures.get(fault, 1.0))


def candidate_matrix(contrast: np.ndarray) -> np.ndarray:
    """(positions, 2 * positions * values) rotated candidate vectors of one window."""
    contrast = np.asarray(contrast, dtype=np.float64)
    n = contrast.shape[0]
    rows = np.array([np.concatenate([contrast[(p + k) % n] for k in range(n)])
                     for p in range(n)])
    return np.hstack([rows, np.abs(rows)])


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def localization_rows(table: pd.DataFrame, truths: dict[str, dict], split: str, fault: str,
                      engine: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Contrasts (windows, positions, values), true positions, and the rows used.

    Clean, cam-phased core windows of a partition where this fault is present.
    """
    rows = table[(table["split"] == split) & (table["quality"] == "ok")
                 & (table["cyl_phase"] == PHASE_CAM) & (table[f"y_{fault}"] == 1)]
    if "subset" in rows:
        rows = rows[rows["subset"] == "core"]
    cylinder = rows["run_id"].map(lambda r: fault_cylinders(truths[r]).get(fault))
    rows = rows[cylinder.notna()].reset_index(drop=True)
    cylinder = cylinder[cylinder.notna()].astype(int).to_numpy()
    order = list(engine["firing_order"])
    names = signature_names()
    columns = [f"cyl{c}_{n}" for c in order for n in names]  # contrast_from_row layout
    contrasts = rows[columns].to_numpy(dtype=np.float64).reshape(len(rows), len(order),
                                                                 len(names))
    positions = np.array([order.index(c) for c in cylinder], dtype=int)
    return contrasts, positions, rows


def _fit_scorer(contrasts: np.ndarray, positions: np.ndarray, c: float) -> Pipeline:
    x = np.vstack([candidate_matrix(w) for w in contrasts])
    n = contrasts.shape[1]
    y = np.concatenate([np.arange(n) == p for p in positions]).astype(int)
    scorer = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
    return scorer.fit(x, y)


def _all_scores(scorer: Pipeline, contrasts: np.ndarray) -> np.ndarray:
    if not len(contrasts):
        return np.empty((0, 0))
    n = contrasts.shape[1]
    flat = scorer.decision_function(np.vstack([candidate_matrix(w) for w in contrasts]))
    return flat.reshape(len(contrasts), n)


def _fit_temperature(scores: np.ndarray, positions: np.ndarray) -> float:
    """Softmax temperature minimizing log loss of the true positions."""
    if not len(scores):
        return 1.0

    def nll(log_t):
        p = softmax(scores / np.exp(log_t))
        return -np.mean(np.log(p[np.arange(len(p)), positions] + 1e-12))

    result = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(result.x))


def window_localization_metrics(proba: np.ndarray, positions: np.ndarray,
                                min_confidence: float) -> dict[str, Any]:
    """Top-1 accuracy, accuracy and coverage at a confidence, and calibration."""
    if not len(proba):
        return {"windows": 0, "top1_accuracy": None, "coverage": None,
                "accepted_accuracy": None, "log_loss": None}
    top = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    accepted = confidence >= min_confidence
    correct = top == positions
    return {
        "windows": int(len(proba)),
        "top1_accuracy": float(correct.mean()),
        "coverage": float(accepted.mean()),
        "accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
        "log_loss": float(-np.mean(np.log(proba[np.arange(len(proba)), positions] + 1e-12))),
        "mean_confidence": float(confidence.mean()),
    }


def confusion(proba: np.ndarray, positions: np.ndarray, firing_order: list[int]) -> dict:
    """Counts of predicted cylinder per true cylinder (top-1)."""
    out = {str(c): dict.fromkeys(map(str, firing_order), 0) for c in firing_order}
    for p, t in zip(proba.argmax(axis=1) if len(proba) else [], positions, strict=True):
        out[str(firing_order[t])][str(firing_order[p])] += 1
    return out


def fit_localizer(table: pd.DataFrame, truths: dict[str, dict], config: dict,
                  max_train_windows: int = MAX_TRAIN_WINDOWS) -> CylinderLocalizer:
    engine = config["engine"]
    if "cyl_phase" not in table:
        raise ValueError("feature table has no cylinder signature; rebuild it with "
                         "build-features on contract 1.1 data (cam_sync)")
    scorers, temperatures, min_conf, notes = {}, {}, {}, {}
    for fault in FAULTS:
        data = {split: localization_rows(table, truths, split, fault, engine)
                for split in ("train", "tuning", "calibration")}
        train_c, train_p, train_rows = data["train"]
        if not len(train_c):
            raise ValueError(f"no cam-phased training windows with {fault} present")
        rng = np.random.default_rng(0)
        if len(train_c) > max_train_windows:
            keep = np.sort(rng.choice(len(train_c), max_train_windows, replace=False))
            train_c, train_p = train_c[keep], train_p[keep]
        tune_c, tune_p, _ = data["tuning"]
        cal_c, cal_p, _ = data["calibration"]

        # 1. Regularization by top-1 accuracy, then log loss, on tuning engines.
        trials = []
        for c in C_GRID:
            scorer = _fit_scorer(train_c, train_p, c)
            proba = softmax(_all_scores(scorer, tune_c)) if len(tune_c) else np.empty((0, 4))
            metrics = window_localization_metrics(proba, tune_p, 0.0)
            trials.append({"C": c, **metrics, "model": scorer})
        best = max(trials, key=lambda t: ((t["top1_accuracy"] or 0), -(t["log_loss"] or 0)))
        scorers[fault] = best["model"]

        # 2. Softmax temperature on calibration engines.
        temperatures[fault] = _fit_temperature(_all_scores(best["model"], cal_c), cal_p)

        # 3. Smallest confidence whose accepted tuning windows meet TARGET_ACCURACY.
        tune_proba = softmax(_all_scores(best["model"], tune_c) / temperatures[fault]) \
            if len(tune_c) else np.empty((0, 4))
        sweep = [{"min_confidence": m, **window_localization_metrics(tune_proba, tune_p, m)}
                 for m in CONFIDENCE_GRID]
        feasible = [s for s in sweep if (s["accepted_accuracy"] or 0) >= TARGET_ACCURACY]
        chosen = feasible[0] if feasible else max(
            sweep, key=lambda s: (s["accepted_accuracy"] or 0, s["coverage"] or 0))
        min_conf[fault] = chosen["min_confidence"]
        cal_proba = softmax(_all_scores(best["model"], cal_c) / temperatures[fault]) \
            if len(cal_c) else np.empty((0, 4))
        notes[fault] = {
            "windows": {"train": int(len(data["train"][0])), "train_used": int(len(train_c)),
                        "tuning": int(len(tune_c)), "calibration": int(len(cal_c)),
                        "train_runs": int(train_rows["run_id"].nunique())},
            "c_search": [{k: v for k, v in t.items() if k != "model"} for t in trials],
            "C": best["C"], "temperature": temperatures[fault],
            "confidence_sweep": sweep, "min_confidence": min_conf[fault],
            "target_accuracy_met_on_tuning": bool(feasible),
            "tuning": window_localization_metrics(tune_proba, tune_p, min_conf[fault]),
            "calibration": window_localization_metrics(cal_proba, cal_p, min_conf[fault]),
            "tuning_confusion": confusion(tune_proba, tune_p, list(engine["firing_order"])),
        }
    return CylinderLocalizer(
        faults=list(FAULTS), firing_order=list(engine["firing_order"]),
        signature_names=signature_names(), scorers=scorers, temperatures=temperatures,
        min_confidence=min_conf,
        trained_on=sorted(table.loc[table["split"] == "train", "run_id"].unique()),
        notes={"per_fault": notes, "target_accuracy": TARGET_ACCURACY})


def evaluate_windows(localizer: CylinderLocalizer, table: pd.DataFrame,
                     truths: dict[str, dict], config: dict, split: str) -> dict[str, Any]:
    """Window-level localization on one partition, given that the fault is present."""
    out = {}
    for fault in localizer.faults:
        contrasts, positions, _ = localization_rows(table, truths, split, fault,
                                                    config["engine"])
        proba = np.array([localizer.position_probabilities(fault, c) for c in contrasts]) \
            if len(contrasts) else np.empty((0, 4))
        out[fault] = {**window_localization_metrics(proba, positions,
                                                    localizer.min_confidence[fault]),
                      "confusion": confusion(proba, positions, localizer.firing_order)}
    return out
