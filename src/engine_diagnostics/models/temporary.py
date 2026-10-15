"""Temporary normal-baseline detector and RBF SVM for the integration batch.

This proves the training-to-replay path. It is fitted on three virtual
engines and checked on one held-out engine, so its scores are pipeline
checks, not diagnostic performance.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from engine_diagnostics.pipeline.features import FEATURE_VERSION, feature_names

MODEL_VERSION = "temporary-0.2.0"
NORMAL = "normal"
UNCERTAIN = "uncertain"


@dataclass
class TemporaryModel:
    feature_names: list[str]
    classes: list[str]
    classifier: Any
    baseline_names: list[str]
    baseline_median: np.ndarray
    baseline_scale: np.ndarray
    anomaly_threshold: float
    margin_threshold: float
    trained_on: list[str]
    feature_version: str = FEATURE_VERSION
    model_version: str = MODEL_VERSION
    notes: dict[str, Any] = field(default_factory=dict)

    def anomaly_score(self, x: np.ndarray) -> np.ndarray:
        """RMS robust z-score of waveform features against normal training windows."""
        idx = [self.feature_names.index(n) for n in self.baseline_names]
        z = (np.atleast_2d(x)[:, idx] - self.baseline_median) / self.baseline_scale
        return np.sqrt(np.mean(np.clip(z, -20, 20) ** 2, axis=1))

    def predict(self, x: np.ndarray) -> list[dict[str, Any]]:
        """Class, decision margin, anomaly score, and accepted condition per window.

        A fault is accepted only when the SVM names a fault, the margin over
        the runner-up class is sufficient, and the window is anomalous
        relative to normal. Disagreements are reported as uncertain.
        """
        x = np.atleast_2d(x)
        scores = self.classifier.decision_function(x)
        order = np.argsort(scores, axis=1)
        margin = scores[np.arange(len(x)), order[:, -1]] - scores[np.arange(len(x)), order[:, -2]]
        anomaly = self.anomaly_score(x)
        results = []
        for k in range(len(x)):
            predicted = self.classes[order[k, -1]]
            anomalous = bool(anomaly[k] > self.anomaly_threshold)
            confident = bool(margin[k] >= self.margin_threshold)
            if predicted == NORMAL:
                accepted = NORMAL if confident and not anomalous else UNCERTAIN
            else:
                accepted = predicted if confident and anomalous else UNCERTAIN
            results.append({"predicted": predicted, "margin": float(margin[k]),
                            "anomaly_score": float(anomaly[k]), "anomalous": anomalous,
                            "accepted": accepted})
        return results


def trainable(table: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = table[(table["split"] == split) & (table["label"] != "transition")
                 & (table["quality"] == "ok")]
    return rows.reset_index(drop=True)


def fit_svm(train: pd.DataFrame, names: list[str]):
    # Give each source engine equal total influence so engines with more windows
    # do not dominate the shared decision boundary.
    domain_sizes = train.groupby("engine_id")["engine_id"].transform("size").to_numpy()
    sample_weight = len(train) / (train["engine_id"].nunique() * domain_sizes)
    model = make_pipeline(StandardScaler(),
                          SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
                              decision_function_shape="ovr"))
    return model.fit(train[names].to_numpy(), train["label"].to_numpy(),
                     standardscaler__sample_weight=sample_weight,
                     svc__sample_weight=sample_weight)


def fit_temporary_model(table: pd.DataFrame, anomaly_quantile: float = 0.99,
                        margin_threshold: float = 0.3) -> TemporaryModel:
    train = trainable(table, "train")
    names = feature_names()
    classifier = fit_svm(train, names)
    baseline_names = [n for n in names if not n.startswith("ctx_")]
    normal = train.loc[train["label"] == NORMAL, baseline_names].to_numpy()
    median = np.median(normal, axis=0)
    scale = 1.4826 * np.median(np.abs(normal - median), axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    model = TemporaryModel(
        feature_names=names, classes=list(classifier.classes_), classifier=classifier,
        baseline_names=baseline_names, baseline_median=median, baseline_scale=scale,
        anomaly_threshold=0.0, margin_threshold=margin_threshold,
        trained_on=sorted(train["run_id"].unique()),
    )
    normal_scores = model.anomaly_score(train.loc[train["label"] == NORMAL, names].to_numpy())
    model.anomaly_threshold = float(np.quantile(normal_scores, anomaly_quantile))
    model.notes = {"anomaly_quantile": anomaly_quantile, "train_windows": len(train),
                   "threshold_source": "training normal windows (no tuning partition yet)",
                   "domain_strategy": "equal total SVM sample weight per training engine",
                   "domain_limit": "anomaly baseline uses pooled source-engine data; "
                                   "no target-domain adaptation"}
    return model


def classification_metrics(truth: np.ndarray, predicted: np.ndarray,
                           classes: list[str]) -> dict[str, Any]:
    return {
        "windows": int(len(truth)),
        "macro_f1": float(f1_score(truth, predicted, labels=classes, average="macro",
                                   zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "recall": dict(zip(classes, map(float, recall_score(
            truth, predicted, labels=classes, average=None, zero_division=0)), strict=True)),
        "support": {c: int(np.sum(truth == c)) for c in classes},
        "confusion_labels": classes,
        "confusion": confusion_matrix(truth, predicted, labels=classes).tolist(),
    }


ABLATIONS = {
    "telemetry_only": lambda n: n.startswith("ctx_"),
    "audio_plus_context": lambda n: n.startswith(("audio_", "ctx_")),
    "vibration_plus_context": lambda n: n.startswith(("vibration_", "ctx_")),
    "fused_without_context": lambda n: not n.startswith("ctx_"),
    "fused": lambda n: True,
}


def evaluate(model: TemporaryModel, table: pd.DataFrame) -> dict[str, Any]:
    """Held-out-engine check plus leave-one-engine-out source-domain evaluation."""
    train, test = trainable(table, "train"), trainable(table, "test")
    truth = test["label"].to_numpy()
    results = model.predict(test[model.feature_names].to_numpy())
    predicted = np.array([r["predicted"] for r in results])
    accepted = np.array([r["accepted"] for r in results])
    resolved = accepted != UNCERTAIN
    report = {
        "model_version": model.model_version,
        "feature_version": model.feature_version,
        "status": "TEMPORARY pipeline check on one held-out engine; not a performance claim",
        "trained_on": model.trained_on,
        "tested_on": sorted(test["run_id"].unique()),
        "svm": classification_metrics(truth, predicted, model.classes),
        "accepted": {
            "coverage": float(resolved.mean()),
            **classification_metrics(truth[resolved], accepted[resolved], model.classes),
        },
        "anomaly_threshold": model.anomaly_threshold,
        "normal_windows_flagged_anomalous": float(np.mean(
            [r["anomalous"] for r, t in zip(results, truth, strict=True) if t == NORMAL])),
        "ablations": {},
        "cross_domain": evaluate_leave_one_engine_out(table, model.classes),
    }
    for name, keep in ABLATIONS.items():
        names = [n for n in model.feature_names if keep(n)]
        svm = fit_svm(train, names)
        report["ablations"][name] = classification_metrics(
            truth, svm.predict(test[names].to_numpy()), model.classes)
    return report


def evaluate_leave_one_engine_out(table: pd.DataFrame, classes: list[str]) -> dict[str, Any]:
    """Evaluate unseen synthetic engines by holding each training engine out.

    Reports both the deployed fused feature set and a waveform-only variant to
    expose reliance on engine-specific telemetry context.
    """
    source = trainable(table, "train")
    feature_sets = {
        "fused": feature_names(),
        "waveform_only": [n for n in feature_names() if not n.startswith("ctx_")],
    }
    results = {}
    for protocol, names in feature_sets.items():
        predictions, truths, per_engine = [], [], {}
        for engine_id in sorted(source["engine_id"].unique()):
            fit = source[source["engine_id"] != engine_id]
            holdout = source[source["engine_id"] == engine_id]
            if fit.empty or holdout.empty:
                continue
            classifier = fit_svm(fit, names)
            truth = holdout["label"].to_numpy()
            predicted = classifier.predict(holdout[names].to_numpy())
            predictions.extend(predicted)
            truths.extend(truth)
            per_engine[engine_id] = classification_metrics(truth, predicted, classes)
        pooled = classification_metrics(np.asarray(truths), np.asarray(predictions), classes)
        per_engine_f1 = [part["macro_f1"] for part in per_engine.values()]
        results[protocol] = {
            "macro_f1_mean_by_engine": float(np.mean(per_engine_f1)) if per_engine_f1 else 0.0,
            "macro_f1_std_by_engine": float(np.std(per_engine_f1)) if per_engine_f1 else 0.0,
            "pooled": pooled,
            "engines": per_engine,
        }
    return {
        "status": "synthetic leave-one-engine-out domain generalization check; not a real-world claim",
        "source_engines": sorted(source["engine_id"].unique()),
        "protocol": "train on all but one training engine, evaluate on the unseen engine",
        "results": results,
    }
