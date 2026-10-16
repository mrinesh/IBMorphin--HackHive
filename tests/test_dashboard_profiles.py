"""Dashboard engine profiles with AppTest: optional, saving per engine, specs and warnings."""

from pathlib import Path

import joblib
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

from engine_diagnostics.config import load_config  # noqa: E402
from engine_diagnostics.data.run_io import load_run  # noqa: E402
from engine_diagnostics.history import HistoryStore  # noqa: E402
from engine_diagnostics.runtime.replay import replay_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard/app.py"
MODEL = ROOT / "artifacts/models/temporary.joblib"
RUN = "run_00012"

needs_integration = pytest.mark.skipif(
    not MODEL.is_file() or not (ROOT / "data/raw" / RUN).is_dir(),
    reason="needs the integration runs and train-temporary output",
)


def widget(widgets, label):
    return next(w for w in widgets if w.label == label)


@pytest.fixture
def history_dir(tmp_path, monkeypatch):
    path = tmp_path / "history"
    monkeypatch.setenv("ENGINE_HISTORY_DIR", str(path))
    return path


def open_run(app):
    widget(app.sidebar.selectbox, "Dataset").set_value("integration_v1").run()
    widget(app.sidebar.selectbox, "Run").set_value(RUN).run()


def seek(app, t):
    widget(app.sidebar.slider, "Seek (s)").set_value(t).run()
    widget(app.sidebar.button, "Seek").click().run()
    assert not app.exception


def seek_to_end(app):
    seek(app, float(app.session_state["bundle"].metadata["duration_s"]))


def create_profile(app, serial="SN-001", make=None, model=None):
    widget(app.sidebar.selectbox, "Engine").set_value("new").run()
    if make:
        widget(app.sidebar.selectbox, "Make").set_value(make).run()
    if model:
        widget(app.sidebar.selectbox, "Model").set_value(model).run()
    widget(app.sidebar.text_input, "Serial / ID").set_value(serial).run()
    widget(app.sidebar.button, "Create profile").click().run()
    assert not app.exception


@needs_integration
def test_without_a_profile_nothing_is_saved_and_decisions_match_offline_replay(history_dir):
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    open_run(app)
    assert widget(app.sidebar.selectbox, "Engine").value is None  # default: history off
    seek_to_end(app)
    assert not history_dir.exists()
    assert not any("Engine history" in s.value for s in app.subheader)
    config = load_config(ROOT / "configs/mvp.json", ROOT / "schemas/project-config.schema.json")
    offline = replay_run(load_run(ROOT / "data/raw" / RUN), joblib.load(MODEL), config)
    keys = ("t_end_s", "accepted", "alert")
    assert ([{k: r[k] for k in keys} for r in app.session_state["records"]]
            == [{k: r[k] for k in keys} for r in offline])


@needs_integration
def test_catalog_profile_saves_the_finished_replay_with_results_raw_data_and_notes(history_dir):
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    open_run(app)
    create_profile(app)
    store = HistoryStore(history_dir)
    (profile,) = store.list_profiles()
    assert profile.make == "DRIVEX Sim" and profile.spec_source == "catalog"
    assert widget(app.sidebar.selectbox, "Engine").value == profile.id

    seek(app, 6.0)
    widget(app.button, "Save run so far").click().run()
    seek_to_end(app)  # seeking restarts the replay; the finished replay is saved once
    runs = store.list_runs(profile.id)
    assert [r.complete for r in runs] == [True, False]
    run = runs[0]
    assert app.session_state["saved_history_run"] == run.id
    assert run.source["recording_run_id"] == RUN and run.source["mode"] == "recorded"
    assert run.source["model_version"] == "temporary-0.2.0"
    assert set(run.source) == {"mode", "dataset", "recording_run_id", "held_out_engine",
                               "config", "model_version", "feature_version", "lab_injection"}
    records = app.session_state["records"]
    assert len(store.decisions(run.id)) == len(records)
    alerted = {r["alert"] for r in records if r["alert"]}
    assert run.summary["predicted_fault"] == ("+".join(sorted(alerted)) or "normal")
    raw = store.load_raw(run.id)
    assert raw.audio.size == app.session_state["bundle"].audio.size
    assert any("Engine history" in s.value for s in app.subheader)
    assert app.main.dataframe  # run history table

    widget(app.text_area, "New note").set_value("replaced injector 3")
    widget(app.button, "Add note").click().run()
    assert not app.exception
    (note,) = store.list_notes(profile.id)
    assert note.text == "replaced injector 3" and note.run_id == run.id


@needs_integration
def test_non_four_cylinder_engine_saves_with_a_reliability_warning(history_dir):
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    open_run(app)
    create_profile(app, make="Ford", model="EcoBoost 1.0 I3")
    assert any("may not be reliable" in w.value for w in app.sidebar.warning)
    assert any("may not be reliable" in w.value for w in app.main.warning)
    (profile,) = HistoryStore(history_dir).list_profiles()
    assert profile.specs["cylinders"] == 3
    seek_to_end(app)
    assert len(HistoryStore(history_dir).list_runs(profile.id)) == 1


@needs_integration
def test_manual_specs_are_validated_before_a_profile_is_created(history_dir):
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    widget(app.sidebar.selectbox, "Engine").set_value("new").run()
    widget(app.sidebar.selectbox, "Make").set_value("Other (not in catalog)…").run()
    widget(app.sidebar.text_input, "Make").set_value("Acme").run()
    widget(app.sidebar.text_input, "Model").set_value("V2 test").run()
    widget(app.sidebar.text_input, "Serial / ID").set_value("A-1").run()
    assert any("Not in the catalog" in c.value for c in app.sidebar.caption)
    widget(app.sidebar.number_input, "RPM min").set_value(9000.0).run()
    assert any("rpm_min must be below rpm_max" in e.value for e in app.sidebar.error)
    assert widget(app.sidebar.button, "Create profile").disabled
    widget(app.sidebar.number_input, "RPM min").set_value(900.0).run()
    widget(app.sidebar.text_input, "Firing order").set_value("1-3-4").run()
    assert any("each cylinder 1-4" in e.value for e in app.sidebar.error)
    widget(app.sidebar.text_input, "Firing order").set_value("1-2-4-3").run()
    widget(app.sidebar.button, "Create profile").click().run()
    (profile,) = HistoryStore(history_dir).list_profiles()
    assert profile.spec_source == "manual" and profile.specs["firing_order"] == [1, 2, 4, 3]
    assert profile.specs["rpm_min"] == 900.0
