"""Dashboard smoke test with Streamlit's AppTest (needs runs and a trained model)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard/app.py"

pytestmark = pytest.mark.skipif(
    not (ROOT / "artifacts/models/temporary.joblib").is_file()
    or not any((ROOT / "data/raw").glob("run_*")),
    reason="needs generated runs and train-temporary output",
)


def widget(widgets, label):
    return next(w for w in widgets if w.label == label)


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
    assert state["records"][-1]["model_version"] == "temporary-0.1.0"
    assert any("ALERT" in e.value for e in app.error)


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
