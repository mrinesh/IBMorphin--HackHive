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


@pytest.mark.skipif(not V1_MODEL.is_file() or not (ROOT / "data/raw/core_v1").is_dir(),
                    reason="needs core_v1 runs and the v1 model")
def test_every_demo_preset_renders_with_v1_panels():
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
        assert any(m.label == "Calibrated confidence" for m in app.metric)


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
