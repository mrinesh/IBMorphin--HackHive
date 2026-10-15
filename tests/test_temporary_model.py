"""Alert persistence and the temporary model's training/decision rules."""

import pytest

from engine_diagnostics.runtime.alerts import AlertState

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from engine_diagnostics.models import temporary as T  # noqa: E402
from engine_diagnostics.pipeline.features import feature_names  # noqa: E402

CLASSES = ["knock_like", "misfire", "normal", "valve_clearance_like"]


def test_alert_needs_two_consistent_faults_and_three_normals_to_clear():
    state = AlertState(confirm=2, clear=3)
    assert state.update("misfire") is None
    assert state.update("knock_like") is None  # inconsistent: candidate restarts
    assert state.update("knock_like") == "knock_like"
    assert state.update("normal") == "knock_like"
    assert state.update("normal") == "knock_like"
    assert state.update("normal") is None


def test_uncertain_and_bad_quality_hold_state():
    state = AlertState()
    state.update("misfire")
    assert state.update("uncertain") is None
    assert state.update("misfire") == "misfire"  # uncertain did not reset the count
    assert state.update("normal", quality="clipped") == "misfire"
    state.reset()
    assert state.active is None


def synthetic_table(seed=0):
    """Separable classes: each fault shifts a different block of features."""
    rng = np.random.default_rng(seed)
    names = feature_names()
    rows = []
    for split, engines in (("train", ["engine_001", "engine_002"]), ("test", ["engine_003"])):
        for engine in engines:
            for c, label in enumerate(CLASSES + ["transition"]):
                x = rng.normal(0, 1, (30, len(names)))
                if label not in ("normal", "transition"):
                    x[:, 6 * c:6 * c + 6] += 8
                for k in range(30):
                    rows.append({"run_id": f"{engine}_{label}", "engine_id": engine,
                                 "split": split, "t_end_s": 2 + 0.5 * k, "label": label,
                                 "quality": "ok", **dict(zip(names, x[k], strict=True))})
    return pd.DataFrame(rows)


def test_training_uses_only_clean_training_windows():
    table = synthetic_table()
    train = T.trainable(table, "train")
    assert set(train["split"]) == {"train"} and "transition" not in set(train["label"])
    model = T.fit_temporary_model(table)
    assert model.classes == CLASSES
    assert all(run.startswith(("engine_001", "engine_002")) for run in model.trained_on)
    assert model.anomaly_threshold > 0


def test_accepted_decision_requires_agreement():
    table = synthetic_table()
    model = T.fit_temporary_model(table)
    test = T.trainable(table, "test")
    results = model.predict(test[model.feature_names].to_numpy())
    assert len(results) == len(test)
    for result in results:
        if result["accepted"] != T.UNCERTAIN:
            assert result["accepted"] == result["predicted"]
        if result["predicted"] != T.NORMAL and not result["anomalous"]:
            assert result["accepted"] == T.UNCERTAIN
    report = T.evaluate(model, table)
    assert report["svm"]["macro_f1"] > 0.9
    assert set(report["ablations"]) == set(T.ABLATIONS)
    assert "not a performance claim" in report["status"]
    assert "cross_domain" in report


def test_engine_weights_give_each_engine_equal_total():
    table = T.trainable(synthetic_table(), "train")
    table = pd.concat([table, table[table["engine_id"] == "engine_001"]], ignore_index=True)
    weights = T.engine_balanced_weights(table)
    totals = pd.Series(weights).groupby(table["engine_id"]).sum()
    assert totals["engine_001"] == pytest.approx(totals["engine_002"])
    assert weights.mean() == pytest.approx(1.0)


def test_leave_one_engine_out_holds_out_only_training_engines():
    report = T.evaluate_leave_one_engine_out(synthetic_table(), CLASSES)
    assert report["source_engines"] == ["engine_001", "engine_002"]
    assert set(report["results"]) == {"fused", "waveform_only"}
    for part in report["results"].values():
        assert sorted(part["engines"]) == report["source_engines"]
        assert part["pooled"]["windows"] == len(T.trainable(synthetic_table(), "train"))


def test_holdout_model_never_trains_on_the_held_out_engine():
    model = T.fit_holdout_model(synthetic_table(), "engine_001")
    assert model.trained_on and not any(r.startswith("engine_001") for r in model.trained_on)
    assert model.model_version == f"{T.MODEL_VERSION}-without-engine_001"
    assert model.notes["held_out_engine"] == "engine_001"
