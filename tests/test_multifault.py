"""Per-fault labels, independent per-fault alerts, and the v2 multi-fault model."""

from pathlib import Path

import pytest

from engine_diagnostics.runtime.alerts import MultiAlertState

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from engine_diagnostics.config import load_config  # noqa: E402
from engine_diagnostics.models import multifault as M  # noqa: E402
from engine_diagnostics.pipeline.features import feature_names  # noqa: E402
from engine_diagnostics.pipeline.labels import (  # noqa: E402
    FAULTS,
    fault_window_labels,
    window_labels,
)

ROOT = Path(__file__).resolve().parents[1]
MIS, KNOCK, VALVE = FAULTS


def timeline(*segments):
    return [{"start_s": a, "end_s": b, "label": label} for a, b, label in segments]


def test_staggered_faults_get_their_own_labels_and_transitions():
    truth = {"condition": "knock_like+misfire",
             "faults": [{"condition": MIS, "active_intervals": [[8, 30]]},
                        {"condition": KNOCK, "active_intervals": [[12, 15], [20, 30]]}],
             "label_timeline": timeline((0, 30, "unused"))}
    t = np.array([7.5, 9.0, 11.0, 13.0, 14.5, 18.0, 20.0, 23.0, 30.0])
    y = fault_window_labels(truth, t, 2.0)
    assert np.array_equal(y[MIS], [0, np.nan, 1, 1, 1, 1, 1, 1, 1], equal_nan=True)
    assert np.array_equal(y[KNOCK], [0, 0, 0, np.nan, 1, 0, np.nan, 1, 1], equal_nan=True)
    assert np.all(y[VALVE] == 0)


def test_single_fault_labels_agree_with_the_label_timeline():
    truth = {"condition": MIS, "active_intervals": [[8.0, 12.0], [15.0, 20.0]],
             "label_timeline": timeline((0, 7.5, "normal"), (7.5, 8.5, "transition"),
                                        (8.5, 11.5, MIS), (11.5, 12.5, "transition"),
                                        (12.5, 14.5, "normal"), (14.5, 15.5, "transition"),
                                        (15.5, 20, MIS))}
    t = np.arange(2.0, 20.01, 0.5)
    y = fault_window_labels(truth, t, 2.0)[MIS]
    for label, value in zip(window_labels(truth, t, 2.0), y, strict=True):
        expected = {"normal": 0.0, MIS: 1.0}.get(label, np.nan)
        assert value == expected or (np.isnan(value) and np.isnan(expected))


def test_faults_raise_and_clear_independently():
    state = MultiAlertState(FAULTS, confirm={f: 2 for f in FAULTS}, clear=3)
    both = {MIS: "fault", KNOCK: "fault", VALVE: "normal"}
    assert state.update(both) == []
    assert state.update(both) == [KNOCK, MIS]
    only_knock = {MIS: "normal", KNOCK: "fault", VALVE: "normal"}
    for _ in range(2):
        assert state.update(only_knock) == [KNOCK, MIS]
    assert state.update(only_knock) == [KNOCK]  # misfire cleared, knock stays
    assert state.update({MIS: "uncertain", KNOCK: "normal"}, quality="gap") == [KNOCK]


SETS = [(), (MIS,), (KNOCK,), (VALVE,), (MIS, KNOCK), (MIS, VALVE), (KNOCK, VALVE),
        (MIS, KNOCK, VALVE)]
SPLITS = {"train": 4, "tuning": 2, "calibration": 2, "test": 2}
ONSET = 8.0


def synthetic(seed=0):
    """Each fault shifts its own block of six features while active (fixture only)."""
    rng = np.random.default_rng(seed)
    names = feature_names()
    rows, truths, engine = [], {}, 0
    t_end = np.arange(2.0, 30.01, 0.5)
    for split, n_engines in SPLITS.items():
        for _ in range(n_engines):
            engine += 1
            for faults in SETS:
                run_id = f"run_{engine:03d}_{'_'.join(faults) or 'normal'}"
                truth = {"condition": "+".join(sorted(faults)) or "normal",
                         "faults": [{"condition": f, "active_intervals": [[ONSET, 30.0]]}
                                    for f in faults],
                         "active_intervals": [[ONSET, 30.0]] if faults else [],
                         "label_timeline": timeline((0, 30, "unused"))}
                truths[run_id] = truth
                y = fault_window_labels(truth, t_end, 2.0)
                x = rng.normal(0, 1, (len(t_end), len(names)))
                for k, fault in enumerate(FAULTS):
                    x[:, 6 * k:6 * k + 6] += 6 * np.nan_to_num(y[fault], nan=0.5)[:, None]
                for i, t in enumerate(t_end):
                    rows.append({"run_id": run_id, "engine_id": f"engine_{engine:03d}",
                                 "split": split, "subset": "core", "t_end_s": t,
                                 "label": "unused", "quality": "ok",
                                 **{f"y_{f}": y[f][i] for f in FAULTS},
                                 **dict(zip(names, x[i], strict=True))})
    return pd.DataFrame(rows), truths


@pytest.fixture(scope="module")
def fitted():
    config = load_config(ROOT / "configs/mvp.json", ROOT / "schemas/project-config.schema.json")
    table, truths = synthetic()
    return M.fit_multifault_model(table, truths, config), table


def test_model_reports_every_active_fault_and_none_on_normal(fitted):
    model, table = fitted
    test = table[(table["split"] == "test") & (table["t_end_s"] >= ONSET + 3)]
    results = model.predict(test[model.feature_names].to_numpy())
    held = 0
    for (_, row), result in zip(test.iterrows(), results, strict=True):
        truth = sorted(f for f in FAULTS if row[f"y_{f}"] == 1)
        assert result["detected"] == truth
        if truth:
            assert result["accepted"] == "+".join(truth)
        else:  # a fault-free window beyond the normal baseline is held as uncertain
            assert result["accepted"] in ("normal", "uncertain")
            held += result["accepted"] == "uncertain"
    assert held <= 0.05 * len(results)
    assert set(model.thresholds) == set(FAULTS) == set(model.confirm_updates)


def test_multilabel_metrics_break_down_by_number_of_active_faults(fitted):
    model, table = fitted
    test = table[table["split"] == "test"]
    y_true = test[[f"y_{f}" for f in FAULTS]].to_numpy(dtype=float)
    perfect = M.multilabel_metrics(y_true, np.nan_to_num(y_true) == 1, list(FAULTS))
    assert perfect["exact_set"] == 1.0 and perfect["hamming_loss"] == 0.0
    assert set(perfect["by_active_faults"]) == {"0", "1", "2", "3"}
    first_only = np.zeros_like(y_true, dtype=bool)
    first_only[:, 0] = np.nan_to_num(y_true[:, 0]) == 1  # a one-fault-at-a-time guess
    partial = M.multilabel_metrics(y_true, first_only, list(FAULTS))
    assert partial["by_active_faults"]["2"]["all_true_detected"] < 0.5
    assert partial["by_active_faults"]["0"]["no_false_fault"] == 1.0
