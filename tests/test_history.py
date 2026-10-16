"""Per-engine history store: profiles, runs, results, notes, raw data, and pruning."""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from engine_diagnostics.history import (
    HistoryStore,
    RawData,
    raw_from_bundle,
    summarize_run,
)

I4 = {"cylinders": 4, "strokes": 4, "displacement_l": 1.6, "firing_order": [1, 3, 4, 2],
      "rpm_min": 800, "rpm_max": 3500, "load_min": 0.2, "load_max": 0.8}
FEATURES = ["audio_rms", "vibration_rms"]


def record(t, accepted="normal", alert=None, quality="ok", p_misfire=0.1, score=1.0):
    return {"t_end_s": t, "quality": quality, "predicted": accepted, "accepted": accepted,
            "alert": alert, "margin": 0.5, "anomaly_score": score if quality == "ok"
            else float("nan"), "probabilities": {"normal": 1 - p_misfire, "misfire": p_misfire},
            "model_version": "diagnostic-1.0.0", "feature_version": "0.1.0",
            "features": np.array([0.1, float("nan")]), "context": {"rpm": 1500.0, "load": 40.0}}


RECORDS = [record(2.0), record(2.5, quality="stale_telemetry"),
           record(3.0, "misfire", p_misfire=0.8, score=3.0),
           record(3.5, "misfire", "misfire", p_misfire=0.9, score=4.0)]
STAGES = ["healthy", "no data", "developing anomaly", "persistent fault"]


def raw(seconds=1.0):
    tel = pd.DataFrame({"t_s": [0.0, 0.02], "signal": ["rpm", "rpm"], "value": [1500.0, np.nan],
                        "unit": ["rpm", "rpm"], "valid": [1, 0]})
    return RawData(audio=np.arange(round(24 * seconds), dtype=np.float32),
                   vibration=np.ones(round(25.6 * seconds), dtype=np.float32),
                   audio_fs_hz=24.0, vibration_fs_hz=25.6, telemetry=tel,
                   cam_sync_s=np.array([0.1, 0.4]))


@pytest.fixture
def store(tmp_path):
    return HistoryStore(tmp_path / "history", raw_keep=2)


@pytest.fixture
def profile(store):
    return store.create_profile("Toyota", "2ZR-FE 1.8 I4", "SN-001", I4, "catalog", "Blue car")


def save(store, profile, raw_data=None, records=RECORDS):
    return store.save_run(profile.id, records, source={"mode": "recorded", "run_id": "run_00001"},
                          summary=summarize_run(records, STAGES[:len(records)]),
                          feature_names=FEATURES, duration_s=30.0, complete=True, raw=raw_data,
                          stages=STAGES[:len(records)])


def test_reading_an_empty_history_writes_nothing(tmp_path):
    store = HistoryStore(tmp_path / "history")
    assert store.list_profiles() == [] and store.list_runs(1) == [] and store.list_notes(1) == []
    assert store.get_profile(1) is None and store.load_raw(1) is None
    assert not (tmp_path / "history").exists()


def test_profile_round_trip_and_unique_identity(store, profile):
    assert store.get_profile(profile.id) == profile
    assert profile.specs == I4 and profile.spec_source == "catalog"
    assert profile.label == "Blue car (Toyota 2ZR-FE 1.8 I4 · SN-001)"
    assert store.find_profile(" toyota", "2zr-fe 1.8 i4", "sn-001") == profile
    with pytest.raises(ValueError, match="already exists"):
        store.create_profile("TOYOTA", "2ZR-FE  1.8 I4", "sn-001", I4, "manual")
    other = store.create_profile("Toyota", "2ZR-FE 1.8 I4", "SN-002", I4, "manual")
    assert [p.id for p in store.list_profiles()] == [profile.id, other.id]


@pytest.mark.parametrize(
    ("args", "error"),
    [(("", "M", "S", I4, "manual"), "required"),
     (("A", "M", "  ", I4, "manual"), "required"),
     (("A", "M", "S", {**I4, "rpm_min": 9000}, "manual"), "rpm_min must be below"),
     (("A", "M", "S", I4, "guessed"), "spec_source")],
)
def test_invalid_profiles_are_rejected(store, args, error):
    with pytest.raises(ValueError, match=error):
        store.create_profile(*args)
    assert store.list_profiles() == []


def test_non_four_cylinder_profiles_save_with_history(store):
    three = {**I4, "cylinders": 3, "firing_order": [1, 2, 3]}
    p = store.create_profile("Ford", "EcoBoost 1.0 I3", "X1", three, "catalog")
    run_id = save(store, p, raw())
    assert store.get_run(run_id).specs["cylinders"] == 3


def test_saved_run_keeps_metadata_results_and_decisions(store, profile):
    run_id = save(store, profile, raw())
    run = store.get_run(run_id)
    assert run.profile_id == profile.id and run.complete and run.raw_available
    assert run.source == {"mode": "recorded", "run_id": "run_00001"}
    assert run.specs == I4 and run.analyzed_s == 3.5 and run.duration_s == 30.0
    assert run.created_utc.endswith("+00:00")
    summary = run.summary
    assert summary["predicted_fault"] == "misfire"
    assert summary["alerts"] == [{"fault": "misfire", "onset_s": 3.5}]
    assert summary["max_probabilities"]["misfire"] == pytest.approx(0.9)
    assert summary["peak_anomaly_score"] == 4.0 and summary["ok_windows"] == 3
    assert summary["final_health_stage"] == "persistent fault"
    assert summary["decision_counts"] == {"misfire": 2, "normal": 1, "stale_telemetry": 1}
    decisions = store.decisions(run_id)
    assert list(decisions["t_end_s"]) == [2.0, 2.5, 3.0, 3.5]
    assert list(decisions["health_stage"]) == STAGES
    assert decisions["alert"].iloc[-1] == "misfire"
    assert np.isnan(decisions["anomaly_score"].iloc[1])
    assert decisions["features"].iloc[0] == [0.1, None]  # NaN stored as null
    assert decisions["probabilities"].iloc[3]["misfire"] == pytest.approx(0.9)


def test_saved_run_keeps_the_specs_it_was_analysed_with(store, profile):
    first = save(store, profile)
    store.update_specs(profile.id, {**I4, "rpm_max": 6400}, "catalog_edited")
    second = save(store, profile)
    assert store.get_run(first).specs["rpm_max"] == 3500
    assert store.get_run(second).specs["rpm_max"] == 6400
    assert store.get_profile(profile.id).spec_source == "catalog_edited"
    with pytest.raises(ValueError, match="load_max"):
        store.update_specs(profile.id, {**I4, "load_max": 2}, "manual")


def test_raw_data_round_trip(store, profile):
    original = raw()
    loaded = store.load_raw(save(store, profile, original))
    np.testing.assert_array_equal(loaded.audio, original.audio)
    np.testing.assert_array_equal(loaded.vibration, original.vibration)
    np.testing.assert_array_equal(loaded.cam_sync_s, original.cam_sync_s)
    assert loaded.audio.dtype == np.float32 and loaded.audio_fs_hz == 24.0
    pd.testing.assert_frame_equal(loaded.telemetry, original.telemetry)


def test_runs_list_newest_first_per_engine(store, profile):
    other = store.create_profile("Honda", "K20 2.0 I4", "H1", I4, "catalog")
    ids = [save(store, profile) for _ in range(3)]
    save(store, other)
    assert [r.id for r in store.list_runs(profile.id)] == ids[::-1]
    assert len(store.list_runs(other.id)) == 1


def test_pruning_keeps_results_and_notes_but_removes_old_raw_data(store, profile):
    ids = [save(store, profile, raw()) for _ in range(4)]
    note = store.add_note(profile.id, "replaced injector 3", run_id=ids[0])
    runs = {r.id: r for r in store.list_runs(profile.id)}
    assert [runs[i].raw_available for i in ids] == [False, False, True, True]
    for run_id in ids[:2]:
        assert store.load_raw(run_id) is None
        assert not (store.root / "raw" / f"profile_{profile.id}" / f"run_{run_id}").exists()
        assert runs[run_id].summary["predicted_fault"] == "misfire"  # results kept
        assert len(store.decisions(run_id)) == len(RECORDS)
    assert store.load_raw(ids[-1]) is not None
    assert store.list_notes(profile.id) == [note]
    with sqlite3.connect(store.db_path) as con:
        pruned = con.execute("SELECT raw_pruned_utc FROM runs WHERE id = ?", (ids[0],)).fetchone()
    assert pruned[0] is not None


def test_pruning_is_per_engine_and_skips_runs_saved_without_raw(store, profile):
    other = store.create_profile("Honda", "K20 2.0 I4", "H1", I4, "catalog")
    first_other = save(store, other, raw())
    for _ in range(3):
        save(store, profile, raw())
    save(store, profile)  # no raw data: does not take a raw slot
    assert store.load_raw(first_other) is not None
    assert sum(r.raw_available for r in store.list_runs(profile.id)) == 2


def test_retention_is_configurable(tmp_path, profile, monkeypatch):
    monkeypatch.setenv("ENGINE_HISTORY_RAW_KEEP", "1")
    store = HistoryStore(tmp_path / "env")
    assert store.raw_keep == 1
    assert HistoryStore(tmp_path / "d", raw_keep=None).raw_keep == 1
    monkeypatch.delenv("ENGINE_HISTORY_RAW_KEEP")
    assert HistoryStore(tmp_path / "d").raw_keep == 10
    with pytest.raises(ValueError):
        HistoryStore(tmp_path / "d", raw_keep=-1)


def test_history_dir_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGINE_HISTORY_DIR", str(tmp_path / "elsewhere"))
    assert HistoryStore().root == tmp_path / "elsewhere"


def test_notes_attach_to_engine_or_one_of_its_runs(store, profile):
    other = store.create_profile("Honda", "K20 2.0 I4", "H1", I4, "catalog")
    run_id = save(store, profile)
    general = store.add_note(profile.id, "  bought used, 80k km  ")
    on_run = store.add_note(profile.id, "replaced injector 3", run_id=run_id)
    assert general.text == "bought used, 80k km" and general.run_id is None
    assert store.list_notes(profile.id) == [on_run, general]
    with pytest.raises(ValueError, match="does not belong"):
        store.add_note(other.id, "wrong engine", run_id=run_id)
    with pytest.raises(ValueError, match="text"):
        store.add_note(profile.id, "   ")
    with pytest.raises(ValueError, match="no engine profile"):
        store.add_note(999, "missing")


def test_summary_handles_runs_without_alerts_or_good_windows():
    assert summarize_run([record(2.0)])["predicted_fault"] == "normal"
    bad = summarize_run([record(2.0, quality="missing_waveform")])
    assert bad["predicted_fault"] == "no data" and bad["peak_anomaly_score"] is None
    assert summarize_run([])["predicted_fault"] == "no data"


def test_summary_tracks_multi_fault_alert_onsets():
    rows = [dict(record(2.0), alerts=["misfire"], alert="misfire"),
            dict(record(2.5), alerts=["knock_like", "misfire"], alert="knock_like+misfire"),
            dict(record(3.0), alerts=[], alert=None),
            dict(record(3.5), alerts=["misfire"], alert="misfire")]
    summary = summarize_run(rows)
    assert summary["predicted_fault"] == "knock_like+misfire"
    assert summary["alerts"] == [{"fault": "misfire", "onset_s": 2.0},
                                 {"fault": "knock_like", "onset_s": 2.5},
                                 {"fault": "misfire", "onset_s": 3.5}]
    json.dumps(summary, allow_nan=False)


def test_raw_from_bundle_keeps_only_received_data():
    class Bundle:
        audio = np.arange(240, dtype=np.float32)
        vibration = np.arange(256, dtype=np.float32)
        audio_fs_hz, vibration_fs_hz = 24.0, 25.6
        telemetry = pd.DataFrame({"t_s": [0.0, 4.0, 6.0], "signal": ["rpm"] * 3,
                                  "value": [1.0, 2.0, 3.0], "unit": ["rpm"] * 3, "valid": [1] * 3,
                                  "extra": [0, 0, 0]})
        cam_sync_s = np.array([1.0, 5.5])

    data = raw_from_bundle(Bundle, 5.0)
    assert data.audio.size == 120 and data.vibration.size == 128
    assert list(data.telemetry["t_s"]) == [0.0, 4.0]
    assert list(data.telemetry.columns) == ["t_s", "signal", "value", "unit", "valid"]
    assert list(data.cam_sync_s) == [1.0]


def test_failed_save_leaves_no_partial_run(store, profile):
    with pytest.raises(ValueError, match="align"):
        store.save_run(profile.id, RECORDS, source={}, summary={}, feature_names=FEATURES,
                       duration_s=30, complete=True, raw=raw(), stages=["healthy"])
    assert store.list_runs(profile.id) == []
    assert not (store.root / "raw").exists()


def test_stored_history_contains_no_truth_fields(store, profile):
    run_id = save(store, profile, raw())
    folder = store.root / "raw" / f"profile_{profile.id}" / f"run_{run_id}"
    assert sorted(p.name for p in folder.iterdir()) == ["signals.npz", "telemetry.csv"]
    with sqlite3.connect(store.db_path) as con:
        text = " ".join(str(v) for row in con.execute("SELECT * FROM runs") for v in row)
    for word in ("onset", "severity", "seed", "label_timeline", "affected_cylinder"):
        assert word not in text.replace("onset_s", "")  # alert onsets are predictions
