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
    assert not any("Diagnosis and report" in s.value for s in app.subheader)
    assert not any(b.label in ("Diagnose", "Export PDF") for b in app.button)
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


@pytest.fixture
def gemini(monkeypatch, tmp_path):
    """Mocked Gemini: no .env key, and every client call raises the error set by the test."""
    import google.genai
    from google.genai import errors

    from engine_diagnostics import diagnosis

    monkeypatch.setattr(diagnosis, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    calls = []

    class RateLimited:
        def __init__(self, **kwargs):
            self.models = self

        def generate_content(self, **kwargs):
            calls.append(kwargs)
            raise errors.ClientError(429, {"error": {"code": 429, "message": "quota",
                                                     "status": "RESOURCE_EXHAUSTED"}})

    monkeypatch.setattr(google.genai, "Client", RateLimited)
    return calls


@needs_integration
def test_diagnose_falls_back_without_a_key_and_on_429_and_saves_each_diagnosis(
        history_dir, gemini, monkeypatch):
    app = AppTest.from_file(str(APP), default_timeout=180).run()
    open_run(app)
    create_profile(app)
    seek_to_end(app)
    assert any("Diagnosis and report" in s.value for s in app.subheader)

    widget(app.button, "Diagnose").click().run()  # no key
    assert not app.exception
    assert any("No Gemini API key" in i.value for i in app.info)
    assert any("Possible causes" in m.value for m in app.markdown)
    assert any("Decision support only" in c.value for c in app.caption)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    widget(app.button, "Diagnose").click().run()  # simulated 429
    assert not app.exception and len(gemini) == 1
    assert any("rate limit" in i.value for i in app.info)

    (profile,) = HistoryStore(history_dir).list_profiles()
    saved = HistoryStore(history_dir).list_diagnoses(profile.id)
    assert len(saved) == 2 and all(d.fallback for d in saved)
    assert widget(app.selectbox, "Saved diagnoses").value == saved[0].id
    widget(app.selectbox, "Saved diagnoses").set_value(saved[1].id).run()
    assert any("No Gemini API key" in i.value for i in app.info)


@needs_integration
def test_report_preview_and_pdf_export(history_dir, gemini):
    app = AppTest.from_file(str(APP), default_timeout=180).run()
    open_run(app)
    create_profile(app)
    seek_to_end(app)
    widget(app.toggle, "Show report").set_value(True).run()
    assert not app.exception and not gemini  # building a report never calls Gemini
    headings = [m.value for m in app.markdown]
    for section in ("**1. Engine profile and specs**", "**2. Latest run**",
                    "**3. Latest diagnosis**", "**4. Run history and trends**"):
        assert section in headings
    assert any("No diagnosis saved yet" in i.value for i in app.info)
    _, pdf, name = app.session_state["report_pdf"]
    assert pdf.startswith(b"%PDF-") and name.endswith(".pdf")


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
