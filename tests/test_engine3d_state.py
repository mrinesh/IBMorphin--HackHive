"""State sent to the 3D engine view: model output and sensor data only, small payload."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("streamlit")
from dashboard import engine3d  # noqa: E402

NAMES = ["vibration_log_rms", "audio_log_rms"]
MODEL = SimpleNamespace(model_version="v", probability_min=0.7, anomaly_threshold=2.5,
                        gate_fault_with_anomaly=False, baseline_names=NAMES,
                        feature_names=[*NAMES, "ctx_rpm"],
                        baseline_median=np.array([1.0, 0.0]), baseline_scale=np.array([0.1, 1.0]))
TRUTH = {"condition": "knock_like", "affected_cylinder": 3, "active_intervals": [[12.5, 30]]}


def record(**overrides):
    base = {"t_end_s": 16.0, "quality": "ok", "predicted": "knock_like", "probability": 0.82,
            "margin": 0.6, "anomaly_score": 3.1, "anomalous": True, "accepted": "knock_like",
            "alert": "knock_like", "features": [1.30103, 0.0, 2000.0]}
    return base | overrides


def state(rec, **kwargs):
    return engine3d.scene_state(rec, MODEL, t_now=kwargs.pop("t_now", 16.0), playing=True,
                                rpm=1520.0, firing_order=[1, 3, 4, 2], **kwargs)


def test_effect_and_intensity_come_from_the_model():
    s = state(record())
    assert (s["effect"], s["intensity"], s["alert"]) == ("knock_like", 0.82, "knock_like")
    assert s["vibration_ratio"] == pytest.approx(2.0, rel=1e-3)  # 10 ** (1.30103 - 1.0)
    assert "cylinder not identified (no localizer loaded)" in s["status_text"]
    assert s["model_cylinder"] is None
    held = state(record(accepted="uncertain", probability=0.5))
    assert (held["effect"], held["intensity"]) == ("knock_like", 0.45)
    assert "alert held: Knock-like" in held["status_text"]
    normal = state(record(accepted="normal", predicted="normal", alert=None))
    assert normal["effect"] is None and normal["intensity"] == 0.0


def test_no_cylinder_without_the_truth_overlay():
    assert state(record())["truth_cylinder"] is None
    assert state(record(), truth=TRUTH)["truth_cylinder"] == 3
    assert state(record(), truth=TRUTH, t_now=5.0)["truth_cylinder"] is None  # before onset
    assert state(record(), truth=TRUTH | {"condition": "normal"})["truth_cylinder"] is None


def test_model_cylinder_comes_from_the_localizer():
    located = {"knock_like": {"status": "localized", "cylinder": 4, "slot": None,
                              "confidence": 0.93}}
    s = state(record(localization=located), truth=TRUTH)
    assert s["model_cylinder"] == 4 and s["truth_cylinder"] == 3  # shown side by side
    assert "cylinder 4 (confidence 0.93)" in s["status_text"]
    locating = state(record(localization={}))
    assert locating["model_cylinder"] is None and "locating cylinder" in locating["status_text"]
    slot = {"knock_like": {"status": "slot_only", "cylinder": None, "slot": "B",
                           "confidence": 0.9}}
    relative = state(record(localization=slot))
    assert relative["model_cylinder"] is None
    assert "firing slot B (no cam reference" in relative["status_text"]


def test_quality_failure_and_warm_up():
    bad = state(record(quality="missing_waveform", accepted="uncertain", alert=None))
    assert bad["effect"] is None and bad["vibration_ratio"] == 1.0
    assert bad["status_text"].startswith("Model: no decision")
    assert state(None)["status_text"] == "Model: warming up"


def test_payload_stays_small():
    assert len(json.dumps(state(record(), truth=TRUTH))) < 600


def test_view_controls_are_labelled_buttons():
    html = engine3d._HTML
    for label in ("Car", "Engine", "Reset view"):
        assert f">{label}</button>" in html
    assert html.count('type="button"') == 3 and 'aria-pressed="true"' in html
    assert 'role="img"' in html and "arrow keys" in html  # keyboard alternative to dragging
