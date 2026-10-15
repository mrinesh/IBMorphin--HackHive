"""Engine-calibrated RBF SVM for the synthetic integration batch.

Each source engine is normalized against known-healthy windows. A new target
engine requires a separate known-healthy calibration run. Metrics are synthetic
transfer checks, not real-engine diagnostic performance.
"""

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from engine_diagnostics.pipeline.features import FEATURE_VERSION, feature_names

MODEL_VERSION = "temporary-0.3.0"
NORMAL = "normal"
UNCERTAIN = "uncertain"
DOMAIN_CALIBRATION_WINDOWS = 8


@dataclass
class TemporaryModel:
    feature_names: list[str]
    classifier_names: list[str]
    classes: list[str]
    classifier: Any
    baseline_names: list[str]
    baseline_median: np.ndarray
    baseline_scale: np.ndarray
    anomaly_threshold: float
    margin_threshold: float
    trained_on: list[str]
    calibration_windows: int = DOMAIN_CALIBRATION_WINDOWS
    feature_version: str = FEATURE_VERSION
    model_version: str = MODEL_VERSION
    notes: dict[str, Any] = field(default_factory=dict)
    target_median: np.ndarray | None = None
    target_scale: np.ndarray | None = None
    source_anomaly_threshold: float | None = None

    @property
    def calibrated(self) -> bool:
        return self.target_median is not None

    def reset_calibration(self) -> None:
        self.target_median = self.target_scale = None
        if self.source_anomaly_threshold is not None:
            self.anomaly_threshold = self.source_anomaly_threshold

    def calibrate(self, normal_windows: np.ndarray) -> None:
        """Adapt feature scale to known-healthy windows from the target engine."""
        x = np.atleast_2d(normal_windows)
        if x.shape[0] < self.calibration_windows:
            raise ValueError(f"need {self.calibration_windows} healthy calibration windows")
        indices = [self.feature_names.index(n) for n in self.classifier_names]
        values = x[:, indices]
        self.target_median = np.median(values, axis=0)
        scale = 1.4826 * np.median(np.abs(values - self.target_median), axis=0)
        self.target_scale = np.where(scale > 1e-8, scale, 1.0)
        adapted = (values - self.target_median) / self.target_scale
        self.anomaly_threshold = float(np.quantile(self.anomaly_score(adapted), 0.99))

    def anomaly_score(self, x: np.ndarray) -> np.ndarray:
        """RMS robust z-score of waveform features against normal training windows."""
        idx = [self.classifier_names.index(n) for n in self.baseline_names]
        z = (np.atleast_2d(x)[:, idx] - self.baseline_median) / self.baseline_scale
        return np.sqrt(np.mean(np.clip(z, -20, 20) ** 2, axis=1))

    def predict(self, x: np.ndarray) -> list[dict[str, Any]]:
        """Class, decision margin, anomaly score, and accepted condition per window.

        A fault is accepted only when the SVM names a fault, the margin over
        the runner-up class is sufficient, and the window is anomalous
        relative to normal. Disagreements are reported as uncertain.
        """
        x = np.atleast_2d(x)
        if not self.calibrated:
            raise RuntimeError("model needs known-healthy target-engine calibration before prediction")
        indices = [self.feature_names.index(n) for n in self.classifier_names]
        adapted = (x[:, indices] - self.target_median) / self.target_scale
        scores = self.classifier.decision_function(adapted)
        order = np.argsort(scores, axis=1)
        margin = scores[np.arange(len(x)), order[:, -1]] - scores[np.arange(len(x)), order[:, -2]]
        anomaly = self.anomaly_score(adapted)
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


def fit_svm(train: pd.DataFrame, names: list[str], scale: bool = True):
    # Give each source engine equal total influence so engines with more windows
    # do not dominate the shared decision boundary.
    domain_sizes = train.groupby("engine_id")["engine_id"].transform("size").to_numpy()
    sample_weight = len(train) / (train["engine_id"].nunique() * domain_sizes)
    classifier = SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
                     decision_function_shape="ovr")
    if scale:
        model = make_pipeline(StandardScaler(), classifier)
        return model.fit(train[names].to_numpy(), train["label"].to_numpy(),
                         standardscaler__sample_weight=sample_weight,
                         svc__sample_weight=sample_weight)
    return classifier.fit(train[names].to_numpy(), train["label"].to_numpy(),
                          sample_weight=sample_weight)


def _normal_reference(rows: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = rows[names].to_numpy()
    median = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - median), axis=0)
    return median, np.where(scale > 1e-8, scale, 1.0)


def _normalize_engine_domains(table: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Express each engine's features relative to its known-healthy baseline."""
    adapted = table.copy()
    for _, engine_rows in adapted.groupby("engine_id", sort=False):
        normal = engine_rows[engine_rows["label"] == NORMAL]
        if normal.empty:
            raise ValueError("domain normalization requires healthy reference windows per engine")
        median, scale = _normal_reference(normal, names)
        adapted.loc[engine_rows.index, names] = (
            engine_rows[names].to_numpy() - median
        ) / scale
    return adapted


def fit_temporary_model(table: pd.DataFrame, anomaly_quantile: float = 0.99,
                        margin_threshold: float = 0.3) -> TemporaryModel:
    train = trainable(table, "train")
    names = feature_names()
    classifier_names = [n for n in names if not n.startswith("ctx_")]
    adapted_train = _normalize_engine_domains(train, classifier_names)
    classifier = fit_svm(adapted_train, classifier_names, scale=False)
    baseline_names = classifier_names
    normal = adapted_train.loc[adapted_train["label"] == NORMAL, baseline_names].to_numpy()
    median = np.median(normal, axis=0)
    scale = 1.4826 * np.median(np.abs(normal - median), axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    model = TemporaryModel(
        feature_names=names, classifier_names=classifier_names,
        classes=list(classifier.classes_), classifier=classifier,
        baseline_names=baseline_names, baseline_median=median, baseline_scale=scale,
        anomaly_threshold=0.0, margin_threshold=margin_threshold,
        trained_on=sorted(train["run_id"].unique()),
    )
    normal_scores = model.anomaly_score(
        adapted_train.loc[adapted_train["label"] == NORMAL, classifier_names].to_numpy())
    model.anomaly_threshold = float(np.quantile(normal_scores, anomaly_quantile))
    model.source_anomaly_threshold = model.anomaly_threshold
    model.notes = {"anomaly_quantile": anomaly_quantile, "train_windows": len(train),
                   "threshold_source": "training normal windows (no tuning partition yet)",
                   "domain_strategy": "per-engine healthy-reference robust normalization plus "
                                     "equal total SVM sample weight per source engine",
                   "target_adaptation": f"separate known-healthy target run with at least "
                                        f"{model.calibration_windows} valid windows",
                   "domain_limit": "calibration requires a known-healthy reference run; "
                                   "validate on real engines before deployment"}
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
    """Held-out-engine check using labeled normal windows as an oracle reference."""
    train, test = trainable(table, "train"), trainable(table, "test")
    calibration = (test[test["label"] == NORMAL]
                   .sort_values(["run_id", "t_end_s"]))
    evaluation_model = copy.deepcopy(model)
    evaluation_model.calibrate(calibration[model.feature_names].to_numpy())
    truth = test["label"].to_numpy()
    results = evaluation_model.predict(test[model.feature_names].to_numpy())
    predicted = np.array([r["predicted"] for r in results])
    accepted = np.array([r["accepted"] for r in results])
    resolved = accepted != UNCERTAIN
    report = {
        "model_version": model.model_version,
        "feature_version": model.feature_version,
        "status": "synthetic held-out-engine check with known-healthy target calibration; "
                  "not a real-world performance claim",
        "target_calibration_windows": len(calibration),
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

    Reports raw-source baselines and a waveform-only model adapted from a short,
    known-healthy target-engine calibration run.
    """
    source = trainable(table, "train")
    feature_sets = {
        "fused": feature_names(),
        "waveform_only": [n for n in feature_names() if not n.startswith("ctx_")],
    }
    results = {name: {"predictions": [], "truths": [], "engines": {}}
               for name in feature_sets}
    results["healthy_calibrated_waveform"] = {
        "predictions": [], "truths": [], "engines": {},
    }
    calibrated_names = feature_sets["waveform_only"]
    for protocol, names in feature_sets.items():
        for engine_id in sorted(source["engine_id"].unique()):
            fit = source[source["engine_id"] != engine_id]
            holdout = source[source["engine_id"] == engine_id]
            if fit.empty or holdout.empty:
                continue
            classifier = fit_svm(fit, names)
            truth = holdout["label"].to_numpy()
            predicted = classifier.predict(holdout[names].to_numpy())
            results[protocol]["predictions"].extend(predicted)
            results[protocol]["truths"].extend(truth)
            results[protocol]["engines"][engine_id] = classification_metrics(
                truth, predicted, classes)

    for engine_id in sorted(source["engine_id"].unique()):
        fit = source[source["engine_id"] != engine_id]
        holdout = source[source["engine_id"] == engine_id]
        normal = (holdout[holdout["label"] == NORMAL]
                  .sort_values(["run_id", "t_end_s"]))
        if fit.empty or normal.empty or len(normal) < DOMAIN_CALIBRATION_WINDOWS:
            continue
        adapted_fit = _normalize_engine_domains(fit, calibrated_names)
        median, scale = _normal_reference(normal, calibrated_names)
        adapted_holdout = holdout[calibrated_names].to_numpy()
        adapted_holdout = (adapted_holdout - median) / scale
        classifier = fit_svm(adapted_fit, calibrated_names, scale=False)
        truth = holdout["label"].to_numpy()
        predicted = classifier.predict(adapted_holdout)
        results["healthy_calibrated_waveform"]["predictions"].extend(predicted)
        results["healthy_calibrated_waveform"]["truths"].extend(truth)
        results["healthy_calibrated_waveform"]["engines"][engine_id] = {
            **classification_metrics(truth, predicted, classes),
            "calibration_windows": len(normal),
        }

    for result in results.values():
        truths, predictions = np.asarray(result.pop("truths")), np.asarray(result.pop("predictions"))
        result["pooled"] = classification_metrics(truths, predictions, classes)
        engine_f1 = [part["macro_f1"] for part in result["engines"].values()]
        result["macro_f1_mean_by_engine"] = float(np.mean(engine_f1)) if engine_f1 else 0.0
        result["macro_f1_std_by_engine"] = float(np.std(engine_f1)) if engine_f1 else 0.0
    return {
        "status": "synthetic leave-one-engine-out check; calibrated protocol assumes a "
                  "separate known-healthy target run; not a real-world claim",
        "source_engines": sorted(source["engine_id"].unique()),
        "protocol": "train on all but one training engine, evaluate on the unseen engine",
        "calibration_windows": DOMAIN_CALIBRATION_WINDOWS,
        "results": results,
    }
