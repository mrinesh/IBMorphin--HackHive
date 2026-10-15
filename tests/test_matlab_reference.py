"""Python features agree with MATLAB reference values (matlab/validate_generator.m)."""

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from engine_diagnostics.config import load_config  # noqa: E402
from engine_diagnostics.data.run_io import load_run  # noqa: E402
from engine_diagnostics.pipeline import features as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reports/generated/generator_validation/matlab_features.json"
RAW = ROOT / "data/raw/core_v1"

pytestmark = pytest.mark.skipif(not REFERENCE.is_file(),
                                reason="run validate_generator() in MATLAB first")


def test_python_features_match_matlab_reference():
    config = load_config(ROOT / "configs/mvp.json", ROOT / "schemas/project-config.schema.json")
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if isinstance(reference, dict):
        reference = [reference]
    assert reference, "reference file is empty"
    compared = 0
    for entry in reference:
        run_dir = RAW / entry["run_id"]
        if not run_dir.is_dir():
            continue
        bundle = load_run(run_dir)
        modality = entry["name"].split("_", 1)[0]
        fs = getattr(bundle, f"{modality}_fs_hz")
        x = getattr(bundle, modality)[F.window_slice(entry["window_end_s"], fs, 0.0, config)]
        values = F._modality_features(
            np.nan_to_num(x.astype(np.float64)), fs, config["analysis"][f"{modality}_low_hz"],
            config["analysis"], float("nan"))
        python = dict(zip((f"{modality}_{n}" for n in F.PER_MODALITY), values, strict=True))
        assert python[entry["name"]] == pytest.approx(entry["value"], rel=1e-6, abs=1e-9), entry
        compared += 1
    assert compared == len(reference)
