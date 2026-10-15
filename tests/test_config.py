"""Check safeguards for aliasing, broken splits, and hidden-truth configuration."""

import copy
import json
from pathlib import Path

import pytest

from engine_diagnostics.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas/project-config.schema.json"


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/mvp.json").read_text(encoding="utf-8"))


def test_shared_mvp_configuration_is_valid():
    config = load_config(ROOT / "configs/mvp.json", SCHEMA)
    assert config["dataset"]["engine_count"] == 30


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        ("analysis", "high_hz", 12000, "Nyquist"),
        ("analysis", "step_s", 3, "step_s"),
        ("analysis", "step_s", 0.333333, "integer sample"),
        ("analysis", "stft_hop_samples", 4096, "STFT hop"),
        ("analysis", "welch_segment_samples", 100000, "spectral segments"),
        ("dataset", "engine_count", 31, "split counts"),
        ("dataset", "runs_per_engine", 7, "condition classes"),
    ],
)
def test_unsafe_numeric_contracts_are_rejected(config, tmp_path, section, field, value, error):
    config[section][field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_config(path, SCHEMA)


def test_hidden_truth_is_not_an_allowed_context_field(config, tmp_path):
    modified = copy.deepcopy(config)
    modified["inference_context_fields"].append("true_severity")
    path = tmp_path / "leakage.json"
    path.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(ValueError, match="inference_context_fields"):
        load_config(path, SCHEMA)
