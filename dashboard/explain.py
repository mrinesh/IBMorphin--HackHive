"""Turn one ReplayEngine decision record into the dashboard's diagnosis panel.

Everything shown is derived from the record (model outputs for that window) and the
model's own thresholds. Nothing here reads evaluation truth, and nothing is invented:
when the model gives no probability (temporary model) the panel shows the decision
margin instead.
"""

from typing import Any

from dashboard.theme import CLASS_META, TOKENS, class_meta

QUALITY_TEXT = {
    "missing_waveform": "a waveform stream has a gap",
    "stale_telemetry": "RPM/load/throttle telemetry is missing or stale",
    "clipped": "a waveform is clipped",
}

MODEL_NOTE = ("Live model output on a recorded synthetic run. Evaluation truth is never "
              "used for this decision.")


def _quality_text(quality: str) -> str:
    return QUALITY_TEXT.get(quality, quality.replace("_", " "))


def uncertain_reason(record: dict[str, Any], model) -> str:
    """Why the accepted decision is 'uncertain', from the same rules the model applies."""
    predicted = record.get("predicted")
    anomalous = bool(record.get("anomalous"))
    if "probability" in record:
        if record["probability"] < model.probability_min:
            return (f"classifier confidence {record['probability']:.2f} is below the "
                    f"{model.probability_min:.2f} acceptance level")
        if predicted == "normal" and anomalous:
            return "classifier says normal, but the anomaly score is above its threshold"
        if getattr(model, "gate_fault_with_anomaly", False) and not anomalous:
            return f"classifier says {predicted}, but the anomaly score is not elevated"
        return "decision not accepted"
    if record.get("margin", 0.0) < getattr(model, "margin_threshold", 0.0):
        return "decision margin is below the acceptance threshold"
    if predicted == "normal" and anomalous:
        return "classifier says normal, but the anomaly score is above its threshold"
    if predicted != "normal" and not anomalous:
        return f"classifier says {predicted}, but the anomaly score is not elevated"
    return "decision not accepted"


def diagnose(record: dict[str, Any] | None, model, confirm_updates: int) -> dict[str, Any]:
    """Diagnosis panel content for the latest decision (or the warm-up state)."""
    version = getattr(model, "model_version", "?")
    if record is None:
        return {"meta": CLASS_META["uncertain"] | {"label": "Warming up"},
                "subtitle": "Waiting for the first full analysis window.",
                "metrics": [], "alert": None, "model_version": version, "note": MODEL_NOTE}

    quality = record["quality"]
    accepted = record["accepted"]
    meta = class_meta(accepted, quality)
    if quality != "ok":
        subtitle = f"No prediction: {_quality_text(quality)}. Alert state is held."
    elif accepted == "uncertain":
        subtitle = f"Not accepted: {uncertain_reason(record, model)}."
    else:
        subtitle = f"Classifier: {CLASS_META.get(record['predicted'], meta)['label']}"

    metrics = []
    if "probability" in record and quality == "ok":
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

    alert = record.get("alert")
    return {"meta": meta, "subtitle": subtitle, "metrics": metrics,
            "alert": (f"{CLASS_META.get(alert, {'label': alert})['label']} "
                      f"(confirmed by {confirm_updates} consistent updates)") if alert else None,
            "model_version": version, "note": MODEL_NOTE}
