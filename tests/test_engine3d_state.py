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


def state(rec, model=MODEL, **kwargs):
    return engine3d.scene_state(rec, model, t_now=kwargs.pop("t_now", 16.0), playing=True,
                                rpm=1520.0, firing_order=[1, 3, 4, 2], **kwargs)


def pick(entry, *keys):
    return tuple(entry[k] for k in keys)


def test_effect_and_intensity_come_from_the_model():
    s = state(record())
    assert (s["effect"], s["alert"]) == ("knock_like", "knock_like")
    assert s["faults"] == [{"fault": "knock_like", "status": "confirmed", "cylinder": None,
                            "intensity": 0.82, "color": s["edge_color"]}]
    assert s["vibration_ratio"] == pytest.approx(2.0, rel=1e-3)  # 10 ** (1.30103 - 1.0)
    assert "cylinder not identified (no localizer loaded)" in s["status_text"]
    held = state(record(accepted="uncertain", probability=0.5))
    assert held["effect"] == "knock_like"
    assert pick(held["faults"][0], "status", "intensity") == ("held", 0.45)
    assert "Knock-like, alert held" in held["status_text"]
    assert "confidence" not in held["status_text"]  # nothing detected in this window
    normal = state(record(accepted="normal", predicted="normal", alert=None))
    assert normal["effect"] is None and normal["faults"] == []
    assert normal["status_text"] == "Model: Normal"


def test_no_cylinder_without_the_truth_overlay():
    assert state(record())["truth_cylinder"] is None
    assert state(record(), truth=TRUTH)["truth_cylinder"] == 3
    assert state(record(), truth=TRUTH, t_now=5.0)["truth_cylinder"] is None  # before onset
    assert state(record(), truth=TRUTH | {"condition": "normal"})["truth_cylinder"] is None


def test_model_cylinder_comes_from_the_localizer():
    located = {"knock_like": {"status": "localized", "cylinder": 4, "slot": None,
                              "confidence": 0.93}}
    s = state(record(localization=located), truth=TRUTH)
    assert s["faults"][0]["cylinder"] == 4 and s["truth_cylinder"] == 3  # side by side
    assert "Knock-like, confidence 0.82, cylinder 4 (location confidence 0.93)" in s["status_text"]
    locating = state(record(localization={}))
    assert locating["faults"][0]["cylinder"] is None
    assert "locating cylinder" in locating["status_text"]
    slot = {"knock_like": {"status": "slot_only", "cylinder": None, "slot": "B",
                           "confidence": 0.9}}
    relative = state(record(localization=slot))
    assert relative["faults"][0]["cylinder"] is None
    assert "firing slot B (no cam reference" in relative["status_text"]


V2 = SimpleNamespace(**vars(MODEL), fault_classes=("misfire", "knock_like",
                                                   "valve_clearance_like"))


def v2_record(**overrides):
    """Knock (cylinder 1) held by its alert, misfire (cylinder 4) detected this window."""
    return record(predicted="misfire", accepted="misfire", detected=["misfire"],
                  probabilities={"misfire": 0.97, "knock_like": 0.4, "valve_clearance_like": 0.1},
                  alerts=["knock_like", "misfire"], alert="knock_like+misfire",
                  localization={
                      "knock_like": {"status": "localized", "cylinder": 1, "slot": None,
                                     "confidence": 1.0},
                      "misfire": {"status": "localized", "cylinder": 4, "slot": None,
                                  "confidence": 0.98}}) | overrides


def test_every_reported_fault_is_shown_on_its_own_cylinder():
    s = state(v2_record(), model=V2)
    by_fault = {f["fault"]: f for f in s["faults"]}
    assert list(by_fault) == ["knock_like", "misfire"]  # sorted, like the alert banners
    assert pick(by_fault["knock_like"], "cylinder", "status", "intensity") == (1, "held", 0.45)
    assert pick(by_fault["misfire"], "cylinder", "status", "intensity") == (4, "confirmed", 0.97)
    assert s["effect"] == "knock_like" and s["alert"] == "knock_like+misfire"
    # Each fault is paired with its own cylinder (misfire was once labelled with knock's).
    assert s["status_text"] == (
        "Model: Knock-like, alert held, cylinder 1 (location confidence 1.00); "
        "Misfire, confidence 0.97, cylinder 4 (location confidence 0.98)")
    both = state(v2_record(accepted="knock_like+misfire", detected=["knock_like", "misfire"]),
                 model=V2)
    assert [f["intensity"] for f in both["faults"]] == [0.4, 0.97]  # each its own probability
    assert all(f["status"] == "confirmed" for f in both["faults"])


def test_engine_keeps_running_after_the_replay_ends_but_not_when_paused():
    playing = state(record())
    assert (playing["animate"], playing["finished"]) == (True, False)
    paused = engine3d.scene_state(record(), MODEL, t_now=12.0, playing=False, rpm=1520.0,
                                  firing_order=[1, 3, 4, 2])
    assert (paused["animate"], paused["finished"]) == (False, False)
    ended = engine3d.scene_state(record(), MODEL, t_now=30.0, playing=False, rpm=1520.0,
                                 firing_order=[1, 3, 4, 2], finished=True)
    assert (ended["animate"], ended["finished"], ended["playing"]) == (True, True, False)
    assert ended["faults"] and ended["effect"] == "knock_like"  # final decision kept


def test_quality_failure_and_warm_up():
    bad = state(record(quality="missing_waveform", accepted="uncertain", alert=None))
    assert bad["effect"] is None and bad["vibration_ratio"] == 1.0
    assert bad["status_text"].startswith("Model: no decision")
    assert state(None)["status_text"] == "Model: warming up"


SENSORS = {
    "thermal": {"egt_c": [712.4, 705.0, 452.3, 698.9], "cht_c": [180.0, 179.0, 150.0, 181.0],
                "patterns": {"cold": {"alert": True, "cylinder": 3},
                             "head_hot": {"alert": False, "cylinder": None}}},
    "crank": {"alert": True, "cylinder": 3, "speed_trace": {"t_s": [0.1] * 200,
                                                           "rpm": [1500.0] * 200}},
}


def test_exhaust_temperatures_and_independent_cylinders():
    s = state(record(**SENSORS))
    assert s["egt_c"] == [712, 705, 452, 699]
    assert s["egt"][2] < 0.4 < 0.7 < s["egt"][0] <= 1.0
    assert (s["crank_cylinder"], s["thermal_cylinder"]) == (3, 3)
    plain = state(record())
    assert plain["egt"] is None and plain["crank_cylinder"] is None
    stale = state(record(thermal={"egt_c": [700.0, None, 690.0, 680.0]}))
    assert stale["egt"] is None


def test_payload_stays_small():
    assert len(json.dumps(state(record(), truth=TRUTH))) < 700
    # Three faults at once is the most the models report.
    three = v2_record(alerts=["knock_like", "misfire", "valve_clearance_like"])
    assert len(json.dumps(state(three, model=V2, truth=TRUTH))) < 1000
    # Per-tooth speed traces stay out of the payload; only four temperatures go in.
    assert len(json.dumps(state(record(**SENSORS), truth=TRUTH))) < 700


def test_view_controls_are_labelled_buttons():
    html = engine3d._HTML  # engine view only: the car view was removed
    assert ">Reset view</button>" in html and html.count('type="button"') == 1
    assert ">Car</button>" not in html
    assert 'role="img"' in html and "arrow keys" in html  # keyboard alternative to dragging
