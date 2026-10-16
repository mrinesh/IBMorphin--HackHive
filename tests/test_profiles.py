"""Engine catalog lookup, spec validation (schema rules), and the reliability warning."""

import json
from pathlib import Path

import pytest

from engine_diagnostics.profiles import (
    CATALOG_PATH,
    load_catalog,
    lookup,
    reliability_warning,
    spec_source,
    validate_specs,
)

ROOT = Path(__file__).resolve().parents[1]
I4 = {"cylinders": 4, "strokes": 4, "displacement_l": 1.6, "firing_order": [1, 3, 4, 2],
      "rpm_min": 800, "rpm_max": 3500, "load_min": 0.2, "load_max": 0.8}


def test_catalog_entries_are_valid_and_mostly_inline_four_four_stroke():
    catalog = load_catalog()
    assert 3 <= len(catalog) <= 5
    assert all(validate_specs(e.engine) == [] for e in catalog)
    supported = [e for e in catalog if reliability_warning(e.engine) is None]
    assert len(supported) > len(catalog) / 2
    assert len(supported) < len(catalog)  # at least one engine outside 4-cylinder 4-stroke


def test_simulated_engine_matches_the_project_configuration():
    mvp = json.loads((ROOT / "configs/mvp.json").read_text(encoding="utf-8"))["engine"]
    entry = lookup(load_catalog(), "DRIVEX Sim", "Synthetic 1.6 I4")
    assert {k: mvp[k] for k in entry.engine} == entry.engine


def test_lookup_ignores_case_and_spacing_and_misses_unknown_engines():
    catalog = load_catalog()
    entry = lookup(catalog, "  toyota ", "2zr-fe   1.8 i4")
    assert entry is not None and entry.make == "Toyota"
    assert lookup(catalog, "Toyota", "Unknown model") is None
    assert lookup(catalog, "Nobody", "2ZR-FE 1.8 I4") is None


def test_lookup_reads_a_custom_catalog(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"engines": [{"make": "A", "model": "B", "engine": I4}]}),
                    encoding="utf-8")
    assert lookup(load_catalog(path), "a", "b").engine == I4
    assert CATALOG_PATH.is_file()


def test_spec_source_distinguishes_catalog_edited_and_manual():
    entry = lookup(load_catalog(), "DRIVEX Sim", "Synthetic 1.6 I4")
    assert spec_source(entry, dict(entry.engine)) == "catalog"
    assert spec_source(entry, {**entry.engine, "rpm_max": 3500.0}) == "catalog"  # same value
    assert spec_source(entry, {**entry.engine, "rpm_max": 4000}) == "catalog_edited"
    assert spec_source(None, I4) == "manual"


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"displacement_l": 0}, "displacement_l"),
        ({"rpm_min": -5}, "rpm_min"),
        ({"rpm_min": 4000}, "rpm_min must be below rpm_max"),
        ({"rpm_min": 3500}, "rpm_min must be below rpm_max"),
        ({"load_max": 1.2}, "load_max"),
        ({"load_min": -0.1}, "load_min"),
        ({"load_min": 0.8}, "load_min must be below load_max"),
        ({"firing_order": [1, 3, 3, 2]}, "non-unique"),
        ({"firing_order": [1, 3, 4]}, "each cylinder 1-4"),
        ({"firing_order": [1, 3, 4, 5]}, "each cylinder 1-4"),
        ({"firing_order": [0, 1, 2, 3]}, "firing_order"),
        ({"cylinders": 0}, "cylinders"),
        ({"strokes": 3}, "strokes"),
        ({"displacement_l": "1.6"}, "displacement_l"),
    ],
)
def test_invalid_specs_are_rejected(change, error):
    problems = validate_specs({**I4, **change})
    assert any(error in p for p in problems), problems


def test_missing_and_unknown_fields_are_rejected():
    assert any("rpm_max" in p for p in validate_specs({k: v for k, v in I4.items()
                                                       if k != "rpm_max"}))
    assert validate_specs({**I4, "turbo": True})


def test_non_four_cylinder_engines_validate_but_carry_a_warning():
    three = {**I4, "cylinders": 3, "firing_order": [1, 2, 3]}
    six = {**I4, "cylinders": 6, "firing_order": [1, 5, 3, 6, 2, 4]}
    two_stroke = {**I4, "strokes": 2}
    for specs in (three, six, two_stroke):
        assert validate_specs(specs) == []
        assert "may not be reliable" in reliability_warning(specs)
    assert reliability_warning(I4) is None


def test_schema_is_not_widened():
    schema = json.loads((ROOT / "schemas/project-config.schema.json").read_text(encoding="utf-8"))
    engine = schema["properties"]["engine"]["properties"]
    assert engine["cylinders"] == {"const": 4} and engine["strokes"] == {"const": 4}
