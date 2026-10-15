"""Check MATLAB-generated multi-fault truth in evaluation_truth/core_v2 when it exists."""

import json
from collections import Counter
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from engine_diagnostics.pipeline.labels import (  # noqa: E402
    FAULTS,
    fault_intervals,
    fault_window_labels,
)

ROOT = Path(__file__).resolve().parents[1]
TRUTH = ROOT / "evaluation_truth/core_v2"
CONFIG = json.loads((ROOT / "configs/core_v2.json").read_text(encoding="utf-8"))
FILES = sorted(TRUTH.glob("run_*_events.json")) if TRUTH.is_dir() else []

pytestmark = pytest.mark.skipif(
    not FILES, reason="no core_v2 truth; run generate_dataset(Config=\"configs/core_v2.json\")")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def truths():
    return [load(p) for p in FILES]


def test_every_fault_set_has_its_configured_runs(truths):
    engines = sum(len(v) for v in CONFIG["engine_splits"].values())
    expected = Counter()
    for entry in CONFIG["fault_sets"]:
        name = "+".join(sorted(entry["faults"])) or "normal"
        expected[name] += engines * entry["runs_per_engine"]
    for faults in CONFIG["challenge"]["fault_sets"]:
        expected["+".join(sorted(faults))] += len(CONFIG["engine_splits"]["test"])
    assert Counter(t["condition"] for t in truths) == expected


def test_each_fault_starts_at_its_own_onset_and_only_listed_faults_appear(truths):
    for truth in truths:
        listed = {f["condition"]: f for f in truth["faults"]}
        assert sorted(listed) == sorted(truth["conditions"])
        tagged = Counter(e["fault"] for e in truth["events"] if e["faulted"])
        assert set(tagged) == set(listed), truth["run_id"]
        for event in truth["events"]:
            if event["faulted"]:
                assert event["time_s"] >= listed[event["fault"]]["onset_s"] - 1e-9


def test_misfired_strokes_never_knock(truths):
    for truth in truths:
        if not {"misfire", "knock_like"} <= set(truth["conditions"]):
            continue
        bursts = [(e["cylinder"], e["time_s"]) for e in truth["events"]
                  if e["kind"] == "knock_burst"]
        for e in truth["events"]:
            if e["fault"] == "misfire" and e["kind"] == "combustion":
                assert not any(c == e["cylinder"] and e["time_s"] < t <= e["time_s"] + 0.005
                               for c, t in bursts), truth["run_id"]


def test_multi_fault_runs_have_windows_with_every_fault_active_together(truths):
    t_end = np.arange(2.0, CONFIG["duration_s"] + 1e-9, 0.5)
    for truth in truths:
        if len(truth["conditions"]) < 2:
            continue
        y = fault_window_labels(truth, t_end, 2.0)
        together = np.all([y[f] == 1 for f in truth["conditions"]], axis=0)
        assert together.any(), truth["run_id"]
        for fault in FAULTS:
            if fault not in truth["conditions"]:
                assert np.all(y[fault] == 0)
        assert sorted(fault_intervals(truth)) == sorted(truth["conditions"])
