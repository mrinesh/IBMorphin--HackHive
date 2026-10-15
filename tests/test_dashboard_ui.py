"""HUD theme and diagnosis-panel helpers (pure functions; no runs or model needed)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("plotly")
from dashboard import explain, theme  # noqa: E402

V1 = SimpleNamespace(model_version="v-test", probability_min=0.7, anomaly_threshold=2.5,
                     gate_fault_with_anomaly=False)
TEMPORARY = SimpleNamespace(model_version="t-test", margin_threshold=0.4, anomaly_threshold=2.0)


def record(**overrides):
    base = {"t_end_s": 12.0, "quality": "ok", "predicted": "knock_like", "probability": 0.82,
            "margin": 0.6, "anomaly_score": 3.1, "anomalous": True, "accepted": "knock_like",
            "alert": None}
    return base | overrides


def test_valve_class_is_named_as_valve_train_impacts_not_generic_vibration():
    assert theme.CLASS_META["valve_clearance_like"]["label"] == "Valve-train impacts"
    assert "vibration" not in theme.CLASS_META["valve_clearance_like"]["label"].lower()


def test_bad_quality_shows_no_decision_whatever_the_class():
    assert theme.class_meta("knock_like", "missing_waveform")["label"] == "No decision"
    assert theme.class_meta("misfire", "ok")["label"] == "Misfire"
    assert theme.class_meta(None)["label"] == "Uncertain"


def test_diagnose_warm_up_and_accepted_fault():
    assert explain.diagnose(None, V1, 2)["meta"]["label"] == "Warming up"
    result = explain.diagnose(record(alert="knock_like"), V1, 2)
    assert result["meta"]["label"] == "Knock-like"
    assert result["alert"] == "Knock-like (confirmed by 2 consistent updates)"
    labels = [m["label"] for m in result["metrics"]]
    assert labels[:2] == ["Calibrated confidence", "Anomaly score"]
    assert result["metrics"][0]["value"] == "0.82"  # the model's number, not a derived one


def test_uncertain_reason_follows_the_model_acceptance_rules():
    low = record(probability=0.55, accepted="uncertain")
    assert "below the 0.70 acceptance level" in explain.diagnose(low, V1, 2)["subtitle"]
    normal_but_anomalous = record(predicted="normal", accepted="uncertain")
    assert "anomaly score is above" in explain.uncertain_reason(normal_but_anomalous, V1)
    temp = {k: v for k, v in record(margin=0.1, accepted="uncertain").items()
            if k != "probability"}
    assert "margin is below" in explain.uncertain_reason(temp, TEMPORARY)


def test_temporary_model_shows_margin_not_an_invented_confidence():
    temp = {k: v for k, v in record().items() if k != "probability"}
    labels = [m["label"] for m in explain.diagnose(temp, TEMPORARY, 2)["metrics"]]
    assert "Decision margin" in labels and "Calibrated confidence" not in labels


def test_quality_failure_has_no_scores():
    result = explain.diagnose(record(quality="missing_waveform", accepted="uncertain"), V1, 2)
    assert result["meta"]["label"] == "No decision"
    assert "waveform stream has a gap" in result["subtitle"]
    assert {m["label"] for m in result["metrics"]} == {"Window ends", "Data quality"}


def test_cards_escape_text_and_format_values():
    card = theme.status_card_html(explain.diagnose(record(), V1, 2) | {"subtitle": "<b>x</b>"})
    assert "<b>x</b>" not in card and "&lt;b&gt;x&lt;/b&gt;" in card
    tele = theme.telemetry_cards_html([
        {"label": "RPM", "fmt": ",.0f", "unit": "rpm", "status": "fresh", "age_s": 0.02,
         "value": 2940.4, "history": [2900.0, 2950.0, 2940.0]},
        {"label": "Load", "fmt": ".1f", "unit": "%", "status": "no data", "age_s": None,
         "value": None, "history": []},
    ])
    assert "2,940" in tele and ">Live<" in tele and "Last reading 0.0 s ago" in tele
    assert "No data" in tele and "<svg" not in tele


def test_evidence_strength_is_a_checklist_of_model_criteria():
    confirmed = explain.evidence_strength(record(alert="knock_like"), V1)
    assert (confirmed["level"], confirmed["label"]) == (3, "Confirmed")
    assert confirmed["criteria"][0] == ("Confidence 0.82 ≥ 0.70", True)
    unconfirmed = explain.evidence_strength(record(anomalous=False), V1)
    assert (unconfirmed["level"], unconfirmed["label"]) == (1, "Weak")
    normal = explain.evidence_strength(record(predicted="normal", accepted="normal",
                                              anomalous=False), V1)
    assert normal["for"] == "Normal" and normal["level"] == 3
    temp = {k: v for k, v in record(margin=0.1).items() if k != "probability"}
    assert explain.evidence_strength(temp, TEMPORARY)["criteria"][0] == ("Margin 0.10 < 0.40",
                                                                         False)
    assert explain.evidence_strength(record(quality="clipped"), V1) is None


def test_evidence_ranks_finite_robust_deviations_from_the_normal_reference():
    import numpy as np

    model = SimpleNamespace(
        baseline_names=["audio_firing_peak_cv", "vibration_env_cv_high", "audio_crest",
                        "audio_log_rms"],
        feature_names=["audio_firing_peak_cv", "vibration_env_cv_high", "audio_crest",
                       "audio_log_rms", "ctx_rpm"],
        baseline_median=np.array([0.1, 1.0, 3.0, 0.0]),
        baseline_scale=np.array([0.01, 0.5, 1.0, 1.0]))
    window = record(features=[0.25, 0.0, np.nan, 0.5, 2000.0])
    rows, matched = explain.evidence(window, model)
    assert not matched
    assert [r["feature"] for r in rows] == ["audio_firing_peak_cv", "vibration_env_cv_high",
                                            "audio_log_rms"]  # NaN feature skipped
    assert rows[0]["z"] == pytest.approx(15.0)
    assert explain.evidence_sentence(rows[0]) == (
        "Audio cycle-to-cycle variation of firing pulses is far above normal (z = +15.0)")
    assert explain.evidence_sentence(rows[1]).startswith(
        "Vibration burstiness of the 5-10 kHz envelope is slightly below normal")


def test_background_text_is_marked_as_not_model_output():
    card = theme.status_card_html(explain.diagnose(record(), V1, 2))
    assert "general background, not model output" in card
    assert "Typical checks on a real engine" in card
    assert "Evidence strength for Knock-like" in card
