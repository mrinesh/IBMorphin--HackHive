"""Episode scoring and the v1 model's use of partitions and decision rules."""

import pytest

from engine_diagnostics import evaluation as E

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from engine_diagnostics.models import diagnostic as D  # noqa: E402
from engine_diagnostics.pipeline.features import feature_names  # noqa: E402

CLASSES = ["knock_like", "misfire", "normal", "valve_clearance_like"]


def test_episode_scoring_counts_false_wrong_and_detected():
    times = [2.0 + 0.5 * k for k in range(20)]  # 2.0 .. 11.5
    alerts = [None] * 2 + ["misfire"] * 2 + [None] * 8 + ["knock_like"] * 3 + ["misfire"] * 5
    truth = {"condition": "misfire", "active_intervals": [[7.0, 12.0]]}
    labels = ["normal"] * 10 + ["misfire"] * 10
    score = E.score_run(times, alerts, labels, truth, window_s=2.0, step_s=0.5)
    assert score["false_alert_episodes"] == 1  # misfire alert at 3.0 s, before any fault
    assert score["wrong_class_episodes"] == 1  # knock alert during the misfire interval
    assert score["sustained"][0]["detected"]
    assert score["sustained"][0]["latency_s"] == pytest.approx(2.5)  # misfire alert from 9.5 s
    summary = E.summarize([score])
    assert summary["false_alerts_per_normal_minute"] == pytest.approx(1 / (10 * 0.5 / 60))


def test_brief_intervals_are_reported_separately_and_severity_ramps():
    truth = {"condition": "valve_clearance_like", "active_intervals": [[5.0, 6.0], [10.0, 30.0]],
             "degradation": True, "onset_s": 10.0, "severity": 0.1, "severity_end": 0.9}
    score = E.score_run([2.0, 2.5], [None, None], ["normal", "normal"], truth, 2.0, 0.5)
    assert len(score["brief"]) == 1 and len(score["sustained"]) == 1
    assert E.severity_at(truth, 0.0) == 0.0
    assert E.severity_at(truth, 20.0) == pytest.approx(0.5)


def synthetic_table(seed=0):
    rng = np.random.default_rng(seed)
    names = feature_names()
    rows, truths = [], {}
    splits = {"train": 4, "tuning": 2, "calibration": 2, "test": 1}
    run = 0
    for split, engines in splits.items():
        for e in range(engines):
            for c, condition in enumerate(CLASSES):
                run += 1
                run_id = f"run_{run:05d}"
                onset = None if condition == "normal" else 8.0
                truths[run_id] = {"condition": condition,
                                  "active_intervals": [] if onset is None else [[onset, 20.0]]}
                for k in range(37):
                    t = 2.0 + 0.5 * k
                    label = "normal" if onset is None or t - 2 < onset - 0.5 else (
                        condition if t - 2 >= onset + 0.5 else "transition")
                    x = rng.normal(0, 1, len(names))
                    x[names.index("ctx_rpm")] = 1500 + 400 * e
                    x[names.index("ctx_load")] = 40
                    if label not in ("normal", "transition"):
                        x[6 * c:6 * c + 6] += 6
                    rows.append({"run_id": run_id, "engine_id": f"{split}_{e}", "split": split,
                                 "subset": "core", "t_end_s": t, "label": label, "quality": "ok",
                                 **dict(zip(names, x, strict=True))})
    return pd.DataFrame(rows), truths


@pytest.fixture(scope="module")
def fitted():
    table, truths = synthetic_table()
    config = {"analysis": {"window_s": 2.0, "step_s": 0.5},
              "runtime": {"clear_fault_updates": 3}}
    return D.fit_diagnostic_model(table, truths, config, max_train_windows=2000), table


def test_model_uses_partitions_by_role(fitted):
    model, table = fitted
    train_runs = set(table.loc[table["split"] == "train", "run_id"])
    assert set(model.trained_on) <= train_runs
    assert model.notes["train_windows_used"] <= 2000
    assert {"C", "gamma"} == set(model.notes["svm_selected"])
    assert len(model.notes["decision_trials"]) == np.prod(
        [len(v) for v in D.DECISION_GRID.values()])
    assert model.notes["false_alert_limit_met_on_tuning"]


def test_predictions_are_calibrated_and_gated(fitted):
    model, table = fitted
    test = table[(table["split"] == "test") & (table["label"] != "transition")]
    results = model.predict(test[model.feature_names].to_numpy())
    for result in results:
        assert sum(result["probabilities"].values()) == pytest.approx(1.0)
        if result["probability"] < model.probability_min:
            assert result["accepted"] == "uncertain"
        if result["predicted"] == "normal" and result["anomalous"]:
            assert result["accepted"] == "uncertain"
    accuracy = np.mean([r["predicted"] == t for r, t in zip(results, test["label"], strict=True)])
    assert accuracy > 0.9


def test_operating_bins():
    assert D.DiagnosticModel.operating_bin(1000, 30) == (0, 0)
    assert D.DiagnosticModel.operating_bin(2000, 60) == (1, 1)
    assert D.DiagnosticModel.operating_bin(3000, 50) == (2, 1)
    assert D.DiagnosticModel.operating_bin(float("nan"), 50) is None
