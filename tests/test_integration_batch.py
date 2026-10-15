"""Validate MATLAB-exported runs in data/raw when they exist (generated, not tracked)."""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("scipy")
pytest.importorskip("pandas")

from engine_diagnostics.config import load_config  # noqa: E402
from engine_diagnostics.data import run_io  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
RUNS = run_io.find_runs(RAW) if RAW.is_dir() else []

pytestmark = pytest.mark.skipif(
    not RUNS, reason="no exported runs; run generate_integration_batch() in MATLAB")


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/mvp.json", ROOT / "schemas/project-config.schema.json")


def test_batch_size_matches_configuration(config):
    assert len(RUNS) >= config["dataset"]["integration_runs"]


@pytest.mark.parametrize("run_dir", RUNS, ids=lambda p: p.name)
def test_matlab_export_round_trips(run_dir, config):
    bundle = run_io.load_run(run_dir)
    assert run_io.validate_run(bundle, config) == []
    assert bundle.metadata["duration_s"] == config["dataset"]["integration_duration_s"]


@pytest.mark.parametrize("run_dir", RUNS[:1], ids=lambda p: p.name)
def test_run_loads_without_evaluation_truth(run_dir, config, tmp_path):
    # A copy with no evaluation_truth anywhere nearby must load and validate.
    copy = shutil.copytree(run_dir, tmp_path / "raw" / run_dir.name)
    assert run_io.validate_run(run_io.load_run(copy), config) == []
