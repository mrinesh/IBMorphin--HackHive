"""Dashboard smoke test with Streamlit's AppTest (needs runs and a trained model)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard/app.py"

needs_integration = pytest.mark.skipif(
    not (ROOT / "artifacts/models/temporary.joblib").is_file()
    or not any((ROOT / "data/raw").glob("run_*")),
    reason="needs the integration runs and train-temporary output",
)


def widget(widgets, label):
    return next(w for w in widgets if w.label == label)


@needs_integration
def test_dashboard_renders_banner_and_seeks_with_live_predictions():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert any("SYNTHETIC DATA" in m.value for m in app.markdown)
    widget(app.sidebar.selectbox, "Dataset").set_value("integration_v1").run()
    widget(app.sidebar.selectbox, "Run").set_value("run_00012").run()
    widget(app.sidebar.slider, "Seek (s)").set_value(12.0).run()
    widget(app.sidebar.button, "Seek").click().run()
    assert not app.exception
    state = app.session_state
    assert state["stream_t"] == pytest.approx(12.0)
    assert [r["t_end_s"] for r in state["records"]][-1] == 12.0
    assert state["records"][-1]["model_version"] == "temporary-0.2.0"
    assert any("ALERT" in e.value for e in app.error)


@needs_integration
def test_play_advances_stream_time_in_real_time():
    import time

    app = AppTest.from_file(str(APP), default_timeout=60).run()
    widget(app.sidebar.selectbox, "Dataset").set_value("integration_v1").run()
    widget(app.sidebar.button, "Play").click().run()
    assert app.session_state["playing"]
    time.sleep(1.5)
    app.run()
    assert not app.exception
    assert 1.0 <= app.session_state["stream_t"] <= 2.5  # 1x pacing, 40 ms packets
    if app.session_state["stream_t"] < 2.0:  # no decision before a full 2 s window
        assert app.session_state["records"] == []


V1_MODEL = ROOT / "artifacts/models/diagnostic_v1.joblib"


@pytest.mark.skipif(not V1_MODEL.is_file() or not (ROOT / "data/raw/core_v1").is_dir()
                    or not (ROOT / "artifacts/models/multifault_v2.joblib").is_file(),
                    reason="needs core_v1/core_v2 runs and the v1 and v2 models")
def test_every_demo_preset_renders_with_its_model_panels():
    import json

    presets = json.loads((ROOT / "configs/demo_presets.json").read_text(encoding="utf-8"))
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    for preset in presets:
        widget(app.sidebar.selectbox, "Demo preset").set_value(preset["name"]).run()
        assert not app.exception, preset["name"]
        assert app.session_state["run_id"] == preset["run_id"]
        assert app.session_state["stream_t"] == pytest.approx(preset["start_s"], abs=0.05)
        widget(app.sidebar.slider, "Seek (s)").set_value(24.0).run()
        widget(app.sidebar.button, "Seek").click().run()
        assert not app.exception, preset["name"]
        records = app.session_state["records"]
        assert records and "probability" in records[-1]
        expected = "P(misfire)" if preset["dataset"] == "core_v2" else "Calibrated confidence"
        assert any(m.label == expected for m in app.metric), preset["name"]


@pytest.mark.skipif(not V1_MODEL.is_file() or not (ROOT / "data/raw/core_v1/run_00228").is_dir(),
                    reason="needs core_v1 runs and the v1 model")
def test_active_alert_shows_linked_evidence_panel():
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    widget(app.sidebar.selectbox, "Dataset").set_value("core_v1").run()
    widget(app.sidebar.selectbox, "Run").set_value("run_00228").run()  # misfire preset
    widget(app.sidebar.slider, "Seek (s)").set_value(19.0).run()
    widget(app.sidebar.button, "Seek").click().run()
    assert not app.exception
    assert app.session_state["records"][-1]["alert"] == "misfire"
    assert any(s.value == "Evidence for active alerts" for s in app.subheader)
    text = " ".join(m.value for m in app.markdown)
    assert "Possible misfire" in text
    for claim in ("irregular firing rhythm", "repeated abnormal pulses",
                  "accompanying fluctuation", "Assessment:"):
        assert claim in text


@pytest.mark.skipif(not V1_MODEL.is_file() or not (ROOT / "data/raw/core_v1").is_dir()
                    or not (ROOT / "evaluation_truth/core_v1").is_dir(),
                    reason="needs core_v1 runs, truth, and the v1 model")
def test_fault_injection_lab_labels_injected_signals_and_shows_response():
    app = AppTest.from_file(str(APP), default_timeout=180).run()
    widget(app.sidebar.radio, "Mode").set_value("Fault-injection lab").run()
    assert not app.exception
    widget(app.sidebar.button, "Apply and replay").click().run()  # default: misfire 0.5 at 8 s
    widget(app.sidebar.slider, "Seek (s)").set_value(12.0).run()
    widget(app.sidebar.button, "Seek").click().run()
    assert not app.exception
    state = app.session_state
    assert state["lab"].fault == "misfire" and state["bundle"].run_id.startswith("lab:")
    text = " ".join(m.value for m in app.markdown)
    assert "FAULT-INJECTION LAB" in text and "INJECTED misfire" in text
    assert "Lab response" in text and "Unmodified recording" in text
    assert state["records"][-1]["alert"] == "misfire"
    assert not any("evaluation truth" in c.label for c in app.sidebar.checkbox)

    widget(app.sidebar.radio, "Mode").set_value("Recorded replay").run()
    assert not app.exception
    assert app.session_state["lab"] is None
    assert "FAULT-INJECTION LAB" not in " ".join(m.value for m in app.markdown)


@pytest.mark.skipif(not V1_MODEL.is_file() or not (ROOT / "data/raw/core_v1/run_00216").is_dir(),
                    reason="needs core_v1 runs and the v1 model")
def test_health_timeline_tracks_stages_and_states_no_rul_claim():
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    widget(app.sidebar.selectbox, "Dataset").set_value("core_v1").run()
    widget(app.sidebar.selectbox, "Run").set_value("run_00216").run()  # valve degradation
    widget(app.sidebar.slider, "Seek (s)").set_value(24.0).run()
    widget(app.sidebar.button, "Seek").click().run()
    assert not app.exception
    summary = next(m.value for m in app.markdown if m.value.startswith("**Health stages so far:**"))
    assert summary.split("**")[2].strip().startswith("healthy")
    assert "persistent fault at" in summary
    caption = " ".join(c.value for c in app.caption)
    assert "deterioration demonstration" in caption
    assert "no forecast or remaining-useful-life claim" in caption


needs_holdout = pytest.mark.skipif(
    not (ROOT / "data/processed/integration_v1_features.parquet").is_file()
    or not (ROOT / "data/raw/run_00014").is_dir(),
    reason="needs the 16-run integration batch and build-features output",
)


def integration_app(timeout=60):
    app = AppTest.from_file(str(APP), default_timeout=timeout).run()
    widget(app.sidebar.selectbox, "Dataset").set_value("integration_v1").run()
    return app


@needs_integration
@needs_holdout
def test_hold_out_replays_a_training_engine_with_a_model_that_never_saw_it():
    app = integration_app(timeout=120)
    widget(app.sidebar.selectbox, "Run").set_value("run_00014").run()  # engine_001, valve
    widget(app.sidebar.checkbox, "Hold out this run's engine").check().run()
    widget(app.sidebar.slider, "Seek (s)").set_value(12.0).run()
    widget(app.sidebar.button, "Seek").click().run()
    assert not app.exception
    state = app.session_state
    assert state["held_out"] == "engine_001"
    assert state["records"][-1]["model_version"] == "temporary-0.2.0-without-engine_001"
    assert any("Unseen-engine replay" in w.value for w in app.warning)


@needs_integration
@needs_holdout
def test_test_engine_runs_cannot_be_held_out():
    app = integration_app()
    widget(app.sidebar.selectbox, "Run").set_value("run_00012").run()  # engine_004: test
    assert widget(app.sidebar.checkbox, "Hold out this run's engine").disabled
    assert app.session_state["held_out"] is None


@needs_integration
def test_cross_engine_panel_shows_leave_one_engine_out_report():
    app = integration_app()
    assert not app.exception
    assert any("Cross-engine check" in s.value for s in app.subheader)
    table = next(d.value for d in app.dataframe if "features" in d.value.columns)
    assert set(table["features"]) == {"fused", "waveform_only"}
    assert "pooled" in set(table["held-out engine"])


@pytest.mark.skipif(not V1_MODEL.is_file() or not (ROOT / "data/raw/core_v1").is_dir(),
                    reason="needs core_v1 runs and the v1 model")
def test_scenario_cards_load_the_tagged_preset_and_play_label_follows_state():
    import json

    presets = json.loads((ROOT / "configs/demo_presets.json").read_text(encoding="utf-8"))
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    for tag in ("normal", "knock_like", "misfire", "valve_clearance_like"):
        expected = next(p for p in presets if p["scenario"] == tag)
        widget(app.sidebar.segmented_control, "Scenario").set_value(tag).run()
        assert not app.exception, tag
        assert app.session_state["preset_select"] == expected["name"]
        assert app.session_state["run_id"] == expected["run_id"]
    widget(app.sidebar.selectbox, "Demo preset").set_value(presets[5]["name"]).run()
    assert app.session_state["scenario"] is None  # data-quality preset has no card
    widget(app.sidebar.button, "Play").click().run()
    assert app.session_state["playing"]
    assert any(b.label == "Pause" for b in app.sidebar.button)


V2_MODEL = ROOT / "artifacts/models/multifault_v2.joblib"


@pytest.mark.skipif(not V2_MODEL.is_file() or not (ROOT / "data/raw/core_v2").is_dir(),
                    reason="needs core_v2 runs and the v2 model")
def test_core_v2_shows_one_probability_and_alert_row_per_fault():
    import json

    truths = ROOT / "evaluation_truth/core_v2"
    run_id = next(json.loads(p.read_text(encoding="utf-8"))["run_id"]
                  for p in sorted(truths.glob("run_*_events.json"))
                  if len(json.loads(p.read_text(encoding="utf-8"))["conditions"]) == 3)
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    widget(app.sidebar.selectbox, "Dataset").set_value("core_v2").run()
    widget(app.sidebar.selectbox, "Run").set_value(run_id).run()
    widget(app.sidebar.slider, "Seek (s)").set_value(29.0).run()
    widget(app.sidebar.button, "Seek").click().run()
    assert not app.exception
    record = app.session_state["records"][-1]
    assert record["model_version"] == "v2.0.0"
    assert set(record["probabilities"]) == {"misfire", "knock_like", "valve_clearance_like"}
    assert isinstance(record["alerts"], list)
    assert {m.label for m in app.metric} >= {"P(misfire)", "P(knock_like)",
                                             "P(valve_clearance_like)", "Faults detected"}
    assert len([e for e in app.error if "ALERT" in e.value]) == len(record["alerts"])
