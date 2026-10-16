"""Diagnose and the engine report: context, Gemini validation, fallback, saving, PDF.

Gemini is always mocked: an autouse fixture makes constructing a real client fail,
so no test can reach the network.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
genai_errors = pytest.importorskip("google.genai.errors")

from engine_diagnostics import diagnosis as D  # noqa: E402
from engine_diagnostics import report as R  # noqa: E402
from engine_diagnostics.config import load_config  # noqa: E402
from engine_diagnostics.diagnosis_context import (  # noqa: E402
    build_context,
    context_ids,
    crank_summary,
)
from engine_diagnostics.diagnosis_fallback import rule_based_diagnosis  # noqa: E402
from engine_diagnostics.history import HistoryStore, RawData, summarize_run  # noqa: E402
from tests.test_crankspeed import ROOT, synthetic_wheel  # noqa: E402

I4 = {"cylinders": 4, "strokes": 4, "displacement_l": 1.6, "firing_order": [1, 3, 4, 2],
      "rpm_min": 800, "rpm_max": 3500, "load_min": 0.2, "load_max": 0.8}
V6 = {**I4, "cylinders": 6, "firing_order": [1, 2, 3, 4, 5, 6]}
SETTINGS = D.DiagnosisSettings(model="gemini-test-flash", timeout_s=1.0)


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/mvp.json", ROOT / "schemas/project-config.schema.json")


@pytest.fixture(autouse=True)
def no_network(monkeypatch, tmp_path):
    """No real Gemini client and no key unless a test sets one."""
    import google.genai

    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to create a real Gemini client")

    monkeypatch.setattr(google.genai, "Client", refuse)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(D, "ENV_FILE", tmp_path / "missing.env")


def record(t, accepted="normal", alert=None, p_misfire=0.05, score=1.0):
    return {"t_end_s": t, "quality": "ok", "predicted": accepted, "accepted": accepted,
            "alert": alert, "margin": 0.5, "anomaly_score": score,
            "probabilities": {"normal": 1 - p_misfire, "misfire": p_misfire,
                              "knock_like": 0.02, "valve_clearance_like": 0.02},
            "model_version": "multifault-2.0.0", "feature_version": "0.2.0",
            "features": np.array([0.1, 0.2]), "context": {"rpm": 1500.0, "load": 40.0}}


HEALTHY = [record(2.0 + 0.5 * k) for k in range(8)]
MISFIRE = ([record(2.0 + 0.5 * k) for k in range(4)]
           + [record(4.0 + 0.5 * k, "misfire", "misfire", 0.9, 4.0) for k in range(4)])


def telemetry(seconds=12.0, cold_cylinder=3, ltft=12.0):
    rows = []
    for t in np.arange(0, seconds, 0.1):
        for c in range(1, 5):
            egt = 650.0 - (90.0 if c == cold_cylinder and t > 1.0 else 0.0)
            rows.append((t, f"egt_cyl{c}", egt, "degC", 1))
            rows.append((t, f"cht_cyl{c}", 180.0, "degC", 1))
        rows.append((t, "short_term_fuel_trim", 1.0 + t, "percent", 1))
        rows.append((t, "long_term_fuel_trim", ltft, "percent", 1))
    return pd.DataFrame(rows, columns=["t_s", "signal", "value", "unit", "valid"])


def raw_with_misfire(cylinder=3) -> RawData:
    edges, cam, _, _ = synthetic_wheel(seconds=4.0, misfire_cylinder=cylinder, misfire_every=3,
                                     jitter_s=1e-6)
    rng = np.random.default_rng(0)
    return RawData(audio=rng.normal(0, 1, 48_000).astype(np.float32),
                   vibration=rng.normal(0, 1, 12_000).astype(np.float32),
                   audio_fs_hz=24_000.0, vibration_fs_hz=6_000.0, telemetry=telemetry(),
                   cam_sync_s=cam, crank_tooth_s=edges)


@pytest.fixture
def store(tmp_path):
    return HistoryStore(tmp_path / "history", raw_keep=5)


def save(store, profile, records, raw=None, mode="recorded"):
    stages = ["healthy"] * len(records)
    if any(r["alert"] for r in records):
        stages[-1] = "persistent fault"
    source = {"mode": mode, "dataset": "core_v3", "recording_run_id": "run_00042",
              "lab_injection": {"fault": "misfire", "cylinder": 3} if mode == "lab" else None}
    return store.save_run(profile.id, records, source=source,
                          summary={**summarize_run(records, stages), "anomaly_threshold": 2.0},
                          feature_names=["a", "b"], duration_s=6.0, complete=True, raw=raw,
                          stages=stages)


@pytest.fixture
def engine(store):
    """Two healthy runs, a plug repair note, then a misfire run with crank/thermal data."""
    profile = store.create_profile("DRIVEX Sim", "Synthetic 1.6 I4", "SN-1", I4, "catalog")
    first = save(store, profile, HEALTHY)
    save(store, profile, HEALTHY)
    store.add_note(profile.id, "Replaced spark plugs on all cylinders", first)
    save(store, profile, MISFIRE, raw=raw_with_misfire(), mode="lab")
    return profile


@pytest.fixture
def context(store, engine, config):
    return build_context(store, engine, config=config)


class FakeModels:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return type("Response", (), {"text": self.reply})()


class FakeClient:
    def __init__(self, reply=None, error=None):
        self.models = FakeModels(reply, error)


def answer(context, **overrides):
    run = context["current_run"]["id"]
    data = {
        "summary": "Cylinder 3 is misfiring. The plugs were replaced, so check the coil next.",
        "causes": [{"title": "Misfire on cylinder 3", "kind": "detected", "confidence": "high",
                    "reasoning": "Alert and crank roughness on cylinder 3.",
                    "confidence_notes": "A coil swap test would confirm it.",
                    "evidence_ids": [run, f"S{run[1:]}-crank"]},
                   {"title": "Failing ignition coil", "kind": "suggested",
                    "confidence": "medium", "reasoning": "Plugs are new (note).",
                    "evidence_ids": ["N1", "R999"]}],
        "actions": [{"step": "Swap coil 3 with coil 2 and re-test.", "urgency": "soon",
                     "addresses": ["Failing ignition coil"]}],
        "data_gaps": ["Only three runs."],
    }
    data.update(overrides)
    return json.dumps(data)


# -- context ------------------------------------------------------------------------------
def test_context_has_stable_citable_ids_and_no_waveforms_or_lab_settings(context):
    ids = context_ids(context)
    assert {"SPEC", "R1", "R2", "R3", "N1", "S3-crank", "S3-thermal", "S3-fuel",
            "S3-spectrum", "T-misfire", "T-anomaly", "T-health"} <= ids
    assert context["current_run"]["id"] == "R3"  # newest first
    assert [r["id"] for r in context["past_runs"]] == ["R2", "R1"]
    assert context["notes"][0]["run"] == "R1"
    assert context["engine"]["spec_source"] == "catalog"
    text = json.dumps(context)
    assert len(text) < 20_000  # summaries only, no raw samples
    for leaked in ("lab_injection", "run_00042", "truth"):
        assert leaked not in text
    assert "fault-injection lab" in context["current_run"]["mode"]


def test_signal_summaries_find_the_misfiring_cylinder(context):
    sig = {s["id"]: s for s in context["signal_summaries"]}
    crank = sig["S3-crank"]
    assert crank["phase"] == "cam_sync" and crank["roughest"] == "cyl3"
    assert crank["per_cylinder"]["cyl3"]["flagged_stroke_pct"] > 10
    assert sig["S3-thermal"]["egt"]["cyl3"]["vs_median_c"] < -50
    assert sig["S3-fuel"]["long_term_fuel_trim"]["mean"] == pytest.approx(12.0)
    assert sig["S3-fuel"]["short_term_fuel_trim"]["drift"] > 2
    misfire = next(t for t in context["trends"] if t["id"] == "T-misfire")
    assert misfire["runs_alerted"] == 1 and misfire["latest_alert_streak"] == 1


def test_crank_summary_is_none_without_a_wheel(config):
    raw = raw_with_misfire()
    raw.crank_tooth_s = None
    assert crank_summary(raw, config) is None


def test_pruned_runs_have_no_signal_summaries(store, config):
    profile = store.create_profile("A", "B", "C", I4, "manual")
    run = save(store, profile, HEALTHY, raw=raw_with_misfire())
    store.raw_keep = 0
    store.prune_raw(profile.id)
    context = build_context(store, profile, config=config)
    assert context["current_run"]["id"] == f"R{run}"
    assert context["current_run"]["raw_data"] == "pruned"
    assert context["signal_summaries"] == []


# -- validation ---------------------------------------------------------------------------
def test_causes_citing_nonexistent_evidence_are_dropped_or_flagged(context):
    causes = json.loads(answer(context))["causes"] + [
        {"title": "Cracked piston", "kind": "suggested", "confidence": "low",
         "reasoning": "made up", "evidence_ids": ["R77", "N55"]}]
    actions = [{"step": "Pull the head.", "urgency": "soon", "addresses": ["Cracked piston"]},
               {"step": "Swap coil 3.", "urgency": "soon",
                "addresses": ["Failing ignition coil"]}]
    out = D.validate_output(answer(context, causes=causes, actions=actions), context)
    assert [c["title"] for c in out["causes"]] == ["Misfire on cylinder 3",
                                                   "Failing ignition coil"]
    coil = out["causes"][1]
    assert coil["evidence_ids"] == ["N1"] and coil["unverified_ids"] == ["R999"]
    assert out["dropped_causes"] == [{"title": "Cracked piston", "evidence_ids": ["R77", "N55"],
                                      "reason": "no evidence ID exists in the data"}]
    assert [a["step"] for a in out["actions"]] == ["Swap coil 3."]


@pytest.mark.parametrize("bad", [
    "not json {", "[]", json.dumps({"summary": "x"}),
    json.dumps({"summary": "", "causes": [], "actions": [], "data_gaps": []}),
    json.dumps({"summary": "x", "causes": "none", "actions": [], "data_gaps": []}),
])
def test_malformed_output_is_a_failure(context, bad):
    with pytest.raises(D.DiagnosisError) as error:
        D.validate_output(bad, context)
    assert error.value.reason == "invalid"


def test_malformed_causes_are_dropped(context):
    out = D.validate_output(answer(context, causes=[
        {"title": "No kind", "confidence": "high", "reasoning": "x", "evidence_ids": ["R3"]},
        {"title": "Good", "kind": "detected", "confidence": "low", "reasoning": "x",
         "evidence_ids": ["R3"]}]), context)
    assert [c["title"] for c in out["causes"]] == ["Good"]
    assert out["dropped_causes"][0]["reason"] == "malformed"


# -- Gemini and fallback ------------------------------------------------------------------
def test_valid_gemini_answer_is_used_with_prompt_schema_and_compact_context(context):
    client = FakeClient(reply=answer(context))
    outcome = D.diagnose(context, SETTINGS, client)
    assert not outcome.fallback and outcome.model == "gemini-test-flash"
    assert outcome.reason is None
    assert outcome.output["causes"][0]["title"] == "Misfire on cylinder 3"
    call = client.models.calls[0]
    assert call["model"] == "gemini-test-flash"
    assert call["config"].system_instruction == D.SYSTEM_PROMPT
    assert call["config"].response_json_schema == D.RESPONSE_SCHEMA
    assert call["config"].response_mime_type == "application/json"
    sent = json.loads(call["contents"].split("\n", 1)[1])
    assert sent == json.loads(json.dumps(context))


def test_without_a_key_the_basic_diagnosis_is_used(context):
    outcome = D.diagnose(context, SETTINGS)  # no client, no key: never builds a client
    assert outcome.fallback and outcome.model == D.FALLBACK_MODEL
    assert "No Gemini API key" in outcome.reason
    assert outcome.output["causes"]


@pytest.mark.parametrize(("error", "words"), [
    (genai_errors.ClientError(429, {"error": {"code": 429, "message": "Resource exhausted",
                                               "status": "RESOURCE_EXHAUSTED"}}), "rate limit"),
    (genai_errors.ClientError(403, {"error": {"message": "denied"}}), "rejected the API key"),
    (genai_errors.ServerError(503, {"error": {"message": "unavailable"}}), "could not be"),
    (TimeoutError("slow"), "did not answer in time"),
    (OSError("no route to host"), "could not be"),
])
def test_api_failures_fall_back_with_a_short_reason(context, error, words):
    outcome = D.diagnose(context, SETTINGS, FakeClient(error=error))
    assert outcome.fallback and words in outcome.reason
    assert "Traceback" not in outcome.reason


@pytest.mark.parametrize("reply", ["{not json", json.dumps({"summary": "only"}), None])
def test_invalid_gemini_output_falls_back(context, reply):
    outcome = D.diagnose(context, SETTINGS, FakeClient(reply=reply))
    assert outcome.fallback and "expected format" in outcome.reason


def test_an_answer_with_only_made_up_evidence_falls_back(context):
    reply = answer(context, causes=[{"title": "Ghost", "kind": "suggested", "confidence": "high",
                                     "reasoning": "x", "evidence_ids": ["R404"]}])
    assert D.diagnose(context, SETTINGS, FakeClient(reply=reply)).fallback


def test_rule_based_diagnosis_cites_only_real_evidence(context):
    out = rule_based_diagnosis(context)
    checked = D.validate_output(json.dumps(out), context)
    assert checked["dropped_causes"] == [] and len(checked["causes"]) == len(out["causes"])
    assert not any(c.get("unverified_ids") for c in checked["causes"])
    titles = [c["title"] for c in out["causes"]]
    assert "Misfire" in titles
    assert any("cylinder 3" in t.lower() for t in titles)  # crank and EGT summaries
    assert any("mixture" in t for t in titles)     # long-term trim 12 %
    misfire = next(c for c in out["causes"] if c["title"] == "Misfire")
    assert "N1" in misfire["evidence_ids"]          # the plug repair did not fix it
    order = ["stop_now", "soon", "monitor"]
    urgencies = [a["urgency"] for a in out["actions"]]
    assert urgencies == sorted(urgencies, key=order.index)


def test_rule_based_lowers_confidence_for_non_four_cylinder_engines(store, config):
    profile = store.create_profile("V", "6", "X", V6, "manual")
    save(store, profile, MISFIRE)
    save(store, profile, MISFIRE)
    out = rule_based_diagnosis(build_context(store, profile, config=config))
    misfire = next(c for c in out["causes"] if c["title"] == "Misfire")
    assert misfire["confidence"] == "medium"  # high, lowered one step
    assert "SPEC" in misfire["evidence_ids"]
    assert any("6 cylinders" in g for g in out["data_gaps"])


def test_rule_based_with_no_runs(store, config):
    profile = store.create_profile("A", "B", "C", I4, "manual")
    out = rule_based_diagnosis(build_context(store, profile, config=config))
    assert out["causes"] == [] and "no saved runs" in out["summary"]


def test_api_key_comes_from_the_environment_or_dotenv(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nOTHER=1\nGEMINI_API_KEY='from-file'\n", encoding="utf-8")
    assert D.api_key(env) == "from-file"
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert D.api_key(env) == "from-env"


def test_settings_name_a_free_tier_flash_model():
    settings = D.load_settings()
    assert "flash" in settings.model and settings.timeout_s > 0


# -- saving -------------------------------------------------------------------------------
def test_diagnoses_are_saved_to_history_and_listed_newest_first(store, engine, config):
    first = D.run_diagnosis(store, engine, config, SETTINGS)
    client = FakeClient(reply=answer(build_context(store, engine, config=config)))
    second = D.run_diagnosis(store, engine, config, SETTINGS, client)
    assert first.fallback and first.fallback_reason and first.model == D.FALLBACK_MODEL
    assert not second.fallback and second.model == "gemini-test-flash"
    assert second.latest_run_id == 3 and len(second.inputs_digest) == 64
    assert first.inputs_digest == second.inputs_digest  # same history, same inputs
    listed = store.list_diagnoses(engine.id)
    assert [d.id for d in listed] == [second.id, first.id]
    assert listed[0].output == second.output
    assert store.latest_diagnosis(engine.id).id == second.id
    assert HistoryStore(store.root).latest_diagnosis(engine.id).output["summary"]


def test_reading_diagnoses_of_an_empty_history_writes_nothing(tmp_path):
    store = HistoryStore(tmp_path / "none")
    assert store.list_diagnoses(1) == [] and store.latest_diagnosis(1) is None
    assert not store.root.exists()


# -- report -------------------------------------------------------------------------------
def test_report_builds_without_calling_gemini(store, engine, monkeypatch):
    monkeypatch.setattr(D, "call_gemini", lambda *a, **k: pytest.fail("report called Gemini"))
    rep = R.build_report(store, engine)
    assert rep.diagnosis is None and rep.current_run["id"] == 3
    assert rep.timeline["probabilities"]["misfire"][-1] == pytest.approx(0.9)
    assert rep.likelihoods["kind"] == "likelihood"
    assert [r["run"] for r in rep.history] == ["#3", "#2", "#1"]
    assert rep.trends["runs"] == ["#1", "#2", "#3"]
    assert "lab_injection" not in json.dumps(rep.__dict__, default=str)


def test_report_localization_is_text():
    loc = {"misfire": {"cylinder": 3, "confidence": 0.98},
           "knock_like": {"cylinder": None, "slot": "C", "confidence": 1.0}}
    assert R.localization_text(loc) == ("Misfire: cylinder 3 (confidence 0.98); Knock-like: "
                                        "firing slot C, cylinder not resolved (confidence 1.00)")
    assert R.localization_text(None) == "-"
    assert R.localization_text("cylinder 2") == "cylinder 2"


def pdf_pages(data: bytes) -> int:
    return len(re.findall(rb"/Type /Page\b", data))


def test_pdf_has_all_sections_and_charts(store, engine, config):
    D.run_diagnosis(store, engine, config, SETTINGS)
    rep = R.build_report(store, engine, now="2026-10-17T09:30:00+00:00")
    assert rep.diagnosis is not None and rep.diagnosis["fallback"]
    data = R.render_pdf(rep)
    assert data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF")
    assert pdf_pages(data) >= 2
    assert len(re.findall(rb"/Subtype /Image", data)) == 3  # timeline, likelihoods, trend
    assert rep.file_name() == "engine_report_DRIVEX_Sim_Synthetic_1_6_I4_SN_1_2026-10-17.pdf"


def test_pdf_for_an_engine_without_runs_or_diagnosis(store):
    profile = store.create_profile("V", "6", "X", V6, "manual", "Odd one")
    rep = R.build_report(store, profile)
    assert rep.current_run is None and rep.diagnosis is None and rep.engine["warning"]
    data = R.render_pdf(rep)
    assert data.startswith(b"%PDF-") and pdf_pages(data) == 1
    assert b"/Subtype /Image" not in data
