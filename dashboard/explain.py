"""Turn one ReplayEngine decision record into the dashboard's diagnosis panel.

Everything shown is derived from the record (model outputs and features for that
window) and the model's own thresholds and normal reference. Nothing here reads
evaluation truth, and nothing is invented: when the model gives no probability
(temporary model) the panel shows the decision margin instead, and the evidence
strength is a checklist of the model's own acceptance criteria, not a physical
severity.
"""

from typing import Any

import numpy as np

from dashboard.theme import CLASS_META, TOKENS, class_meta

QUALITY_TEXT = {
    "missing_waveform": "a waveform stream has a gap",
    "stale_telemetry": "RPM/load/throttle telemetry is missing or stale",
    "clipped": "a waveform is clipped",
}

MODEL_NOTE = ("Live model output on a recorded synthetic run. Evaluation truth is never "
              "used for this decision.")

# Plain-language names for the 14 per-modality features (pipeline/features.py).
# Band edges: BAND_EDGES_HZ between each modality's low edge and high_hz (10 kHz).
_BANDS = ("low-500 Hz", "0.5-1.5 kHz", "1.5-3 kHz", "3-5 kHz", "5-7 kHz", "7-10 kHz")
FEATURE_TEXT = {
    "log_rms": "overall level (RMS)",
    "crest": "peakiness (crest factor)",
    "kurtosis": "impulsiveness (kurtosis)",
    "log_band_power": "total band power",
    **{f"band_frac_{k}": f"share of power in {band}" for k, band in enumerate(_BANDS)},
    "env_cv_high": "burstiness of the 5-10 kHz envelope",
    "env_kurtosis_high": "spikiness of the 5-10 kHz envelope",
    "env_cv_mid": "variability of the 0.5-3 kHz envelope",
    "firing_peak_cv": "cycle-to-cycle variation of firing pulses",
}

# What each class means and what a technician would usually check on a real engine.
# General background, shown separately from model output.
CLASS_TEXT = {
    "normal": ("Sound and vibration signatures match the normal reference.", None),
    "misfire": ("A cylinder fails to burn its charge, so its firing pulse is weak or missing "
                "and the firing rhythm becomes irregular.",
                "spark plugs, ignition coils, injectors, compression"),
    "knock_like": ("Uncontrolled auto-ignition after firing causes sharp, short "
                   "high-frequency pressure bursts.",
                   "fuel octane, ignition timing, carbon deposits, cooling"),
    "valve_clearance_like": ("Excess valve-train clearance makes valves strike their seats "
                             "and rockers, adding sharp periodic impacts at camshaft rate.",
                             "valve lash adjustment, worn lifters, rockers or cam lobes"),
}


def _quality_text(quality: str) -> str:
    return QUALITY_TEXT.get(quality, quality.replace("_", " "))


def describe_feature(name: str) -> str:
    modality, _, feature = name.partition("_")
    return f"{modality.capitalize()} {FEATURE_TEXT.get(feature, feature.replace('_', ' '))}"


def reference_stats(record: dict[str, Any], model):
    """Normal reference for this window: operating bin (v1) or global (temporary)."""
    if hasattr(model, "bin_stats"):
        ctx = record["context"]
        key = model.operating_bin(ctx["rpm"], ctx["load"])
        return model.bin_stats.get(key, model.global_stats), key in model.bin_stats
    return (model.baseline_median, model.baseline_scale), False


def evidence(record: dict[str, Any], model, top: int = 3) -> tuple[list[dict], bool]:
    """Largest robust deviations of this window's features from the normal reference."""
    names = model.baseline_names
    values = np.array([record["features"][model.feature_names.index(n)] for n in names])
    (median, scale), matched = reference_stats(record, model)
    z = (values - median) / scale
    order = [k for k in np.argsort(-np.abs(z)) if np.isfinite(z[k])][:top]
    return [{"feature": names[k], "value": float(values[k]), "median": float(median[k]),
             "z": float(z[k])} for k in order], matched


def evidence_sentence(row: dict) -> str:
    """'Audio burstiness of the 5-10 kHz envelope is far above normal (z = +9.0)'."""
    size = abs(row["z"])
    degree = "far" if size >= 6 else "well" if size >= 3 else "slightly"
    direction = "above" if row["z"] > 0 else "below"
    return (f"{describe_feature(row['feature'])} is {degree} {direction} normal "
            f"(z = {row['z']:+.1f})")


def names(classes: str | None) -> str:
    """Readable name for a class or a v2 '+'-joined fault set (misfire+knock_like)."""
    if not classes:
        return "-"
    return " + ".join(CLASS_META.get(c, {"label": c})["label"] for c in classes.split("+"))


FAULTS = ("misfire", "knock_like", "valve_clearance_like")


def reported_faults(record: dict[str, Any] | None) -> list[dict[str, str]]:
    """Faults the dashboard reports now: active alerts plus this window's detected faults.

    Alerts are debounced: confirmed over several updates and held through brief
    dropouts. So a fault can be alerted while missing from one window, or detected
    before its alert is confirmed. Each entry's status says which: 'confirmed'
    (alert active and detected in this window), 'held' (alert active, not detected
    in this window) or 'unconfirmed' (detected in this window, alert not confirmed
    yet). Sorted by name, the order of the model's '+'-joined sets and the banners.
    """
    if record is None:
        return []
    alerts = record.get("alerts")
    if alerts is None:  # v1 / temporary model: one alert class (or None)
        alerts = (record.get("alert") or "").split("+")
    alerted = {a for a in alerts if a in FAULTS}
    detected = ({c for c in (record.get("accepted") or "").split("+") if c in FAULTS}
                if record.get("quality") == "ok" else set())
    return [{"fault": f, "status": "confirmed" if f in alerted and f in detected
             else "held" if f in alerted else "unconfirmed"}
            for f in sorted(alerted | detected)]


def reported_names(reported: list[dict[str, str]]) -> str:
    """'Knock + Misfire' for a reported_faults list."""
    return names("+".join(r["fault"] for r in reported))


def evidence_strength(record: dict[str, Any], model) -> dict[str, Any] | None:
    """Checklist of the model's own criteria for the classifier's class (0-3 met).

    This is how strongly the model's outputs support its answer, not how severe a
    fault is: the model does not estimate severity.
    """
    if record["quality"] != "ok" or not record.get("predicted"):
        return None
    predicted = record["predicted"]
    name = names(predicted)
    if hasattr(model, "fault_classes"):  # v2: every detected fault must clear its own threshold
        faults = record["detected"] or list(model.fault_classes)
        probs = record["probabilities"]
        if record["detected"]:
            confident = all(probs[f] >= model.thresholds[f] for f in faults)
            first = "Each detected fault's probability ≥ its threshold"
        else:
            confident = all(probs[f] < model.thresholds[f] for f in faults)
            first = "Every fault probability below its threshold"
    elif "probability" in record:
        confident = record["probability"] >= model.probability_min
        first = (f"Confidence {record['probability']:.2f} "
                 f"{'≥' if confident else '<'} {model.probability_min:.2f}")
    else:
        confident = record["margin"] >= model.margin_threshold
        first = (f"Margin {record['margin']:.2f} "
                 f"{'≥' if confident else '<'} {model.margin_threshold:.2f}")
    anomalous = bool(record["anomalous"])
    if predicted == "normal":
        criteria = [(first, confident),
                    ("Anomaly score within the normal range", not anomalous),
                    ("No active alert", not record.get("alert"))]
    else:
        criteria = [(first, confident),
                    ("Anomaly score above its threshold", anomalous),
                    (f"Alert confirmed for {name}",
                     set(predicted.split("+"))
                     <= set(record.get("alerts") or [record.get("alert")]))]
    level = sum(ok for _, ok in criteria)
    return {"for": name, "level": level, "criteria": criteria}


def diagnose(record: dict[str, Any] | None, model, confirm_updates: int) -> dict[str, Any]:
    """Diagnosis panel content for the latest decision (or the warm-up state)."""
    version = getattr(model, "model_version", "?")
    if record is None:
        return {"meta": CLASS_META["uncertain"] | {"label": "Warming up"},
                "subtitle": "Waiting for the first full analysis window.",
                "metrics": [], "alert": None, "model_version": version, "note": MODEL_NOTE,
                "why": [], "about": None}

    quality = record["quality"]
    accepted = record["accepted"]
    meta = class_meta(accepted, quality)
    # Title: every fault reported now (alerts and this window's detections), in the
    # first fault's style, so it names the same faults as the alert banners.
    reported = reported_faults(record)
    if reported:
        meta = class_meta(reported[0]["fault"]) | {"label": reported_names(reported)}
    if quality != "ok":
        subtitle = f"No prediction: {_quality_text(quality)}. Alert state is held."
    elif accepted == "uncertain":
        subtitle = "No fault above its detection threshold"
    else:
        subtitle = f"Classifier: {names(record['predicted'])}"

    metrics = []
    if hasattr(model, "fault_classes") and quality == "ok":
        for fault, p in record["probabilities"].items():
            threshold = model.thresholds[fault]
            metrics.append({"label": f"P({names(fault)})", "value": f"{p:.2f}",
                            "fraction": p, "marker": threshold,
                            "color": TOKENS["cyan"] if p >= threshold else TOKENS["slate"]})
    elif "probability" in record and quality == "ok":
        p_min = model.probability_min
        metrics.append({"label": "Calibrated confidence", "value": f"{record['probability']:.2f}",
                        "fraction": record["probability"], "marker": p_min,
                        "color": TOKENS["cyan"] if record["probability"] >= p_min
                        else TOKENS["slate"]})
    elif quality == "ok":
        metrics.append({"label": "Decision margin", "value": f"{record['margin']:.2f}",
                        "fraction": None})
    if quality == "ok":
        threshold = model.anomaly_threshold
        score = record["anomaly_score"]
        # Bar spans 0..2x threshold; the marker sits at the threshold.
        metrics.append({"label": "Anomaly score", "value": f"{score:.2f} / {threshold:.2f}",
                        "fraction": score / (2 * threshold), "marker": 0.5,
                        "color": TOKENS["amber"] if score > threshold else TOKENS["cyan"]})
    metrics.append({"label": "Window ends", "value": f"{record['t_end_s']:.1f} s",
                    "fraction": None})
    metrics.append({"label": "Data quality", "value": quality.replace("_", " "),
                    "fraction": None})

    why, about = [], None
    if quality == "ok" and "features" in record:
        rows, matched = evidence(record, model)
        why = [evidence_sentence(row) for row in rows]
        why_ref = ("vs normal windows in the same RPM/load bin" if matched
                   else "vs all training normal windows")
    else:
        why_ref = None
    topic = record.get("predicted") if accepted == "uncertain" else accepted
    if quality == "ok" and topic in CLASS_TEXT:
        about = {"class": CLASS_META[topic]["label"], "means": CLASS_TEXT[topic][0],
                 "checks": CLASS_TEXT[topic][1]}

    alert = record.get("alert")
    if isinstance(confirm_updates, dict):  # v2: per-fault confirmation counts
        confirm_updates = max(confirm_updates[f] for f in alert.split("+")) if alert else 0
    return {"meta": meta, "subtitle": subtitle, "metrics": metrics,
            "alert": (f"{names(alert)} "
                      f"(confirmed by {confirm_updates} consistent updates)") if alert else None,
            "model_version": version, "note": MODEL_NOTE, "why": why, "why_ref": why_ref,
            "about": about}
