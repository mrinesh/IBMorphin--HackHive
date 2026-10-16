"""Engine profiles: identify one physical engine and find its specifications.

A profile is make + model + serial/ID (plus an optional nickname) and the engine
specs in the shape of the configuration schema's `engine` section. Specs come
from configs/engine_catalog.json when make and model are listed there, or are
entered by hand; `spec_source` records which ("catalog", "catalog_edited",
"manual").

Specs are validated with the schema's own engine rules, except that `cylinders`
and `strokes` are not pinned to 4: such a profile may be saved, but the
diagnostics are tuned for inline 4-cylinder, 4-stroke engines, so
`reliability_warning` returns a warning for it. `cam_sync_angle_deg` is a
generator setting, not an identity spec, and is not part of a profile.

Profile specs are descriptive metadata. They are never passed to feature
extraction or the model (the analysis keeps using configs/mvp.json).
"""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "configs/engine_catalog.json"
SCHEMA_PATH = ROOT / "schemas/project-config.schema.json"
SPEC_FIELDS = ("cylinders", "strokes", "displacement_l", "firing_order",
               "rpm_min", "rpm_max", "load_min", "load_max")
SPEC_SOURCES = ("catalog", "catalog_edited", "manual")
SUPPORTED = {"cylinders": 4, "strokes": 4}


@dataclass(frozen=True)
class CatalogEntry:
    make: str
    model: str
    engine: dict[str, Any]


def load_catalog(path: Path = CATALOG_PATH) -> list[CatalogEntry]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CatalogEntry(e["make"], e["model"], dict(e["engine"])) for e in data["engines"]]


def _key(text: str) -> str:
    return " ".join(text.split()).casefold()


def lookup(catalog: list[CatalogEntry], make: str, model: str) -> CatalogEntry | None:
    """Catalog entry for make + model, ignoring case and repeated spaces."""
    return next((e for e in catalog if _key(e.make) == _key(make)
                 and _key(e.model) == _key(model)), None)


def spec_source(entry: CatalogEntry | None, specs: dict[str, Any]) -> str:
    """'catalog' if specs equal the catalog entry, 'catalog_edited' if changed, else 'manual'."""
    if entry is None:
        return "manual"
    same = all(_normalized(entry.engine[f]) == _normalized(specs.get(f)) for f in SPEC_FIELDS)
    return "catalog" if same else "catalog_edited"


def _normalized(value):
    if isinstance(value, list | tuple):
        return [int(v) for v in value]
    return float(value) if isinstance(value, int | float) else value


def _spec_schema(schema_path: Path = SCHEMA_PATH) -> dict[str, Any]:
    """The schema's engine section with the 4-cylinder / 4-stroke pins relaxed."""
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    engine = copy.deepcopy(schema["properties"]["engine"])
    engine["properties"] = {f: engine["properties"][f] for f in SPEC_FIELDS}
    engine["required"] = list(SPEC_FIELDS)
    engine["properties"]["cylinders"] = {"type": "integer", "minimum": 1}
    engine["properties"]["strokes"] = {"enum": [2, 4]}
    order = dict(engine["properties"]["firing_order"])
    for bound in ("minItems", "maxItems"):
        order.pop(bound, None)
    order["items"] = {k: v for k, v in order["items"].items() if k != "maximum"}
    engine["properties"]["firing_order"] = order
    return engine


def validate_specs(specs: dict[str, Any], schema_path: Path = SCHEMA_PATH) -> list[str]:
    """Problems with the specs (empty when valid): schema rules plus cross-field checks."""
    validator = Draft202012Validator(_spec_schema(schema_path))
    errors = [f"{'.'.join(map(str, e.absolute_path)) or 'engine'}: {e.message}"
              for e in sorted(validator.iter_errors(specs), key=lambda e: list(e.absolute_path))]
    if errors:
        return errors
    if specs["rpm_min"] >= specs["rpm_max"]:
        errors.append("rpm_min must be below rpm_max")
    if specs["load_min"] >= specs["load_max"]:
        errors.append("load_min must be below load_max")
    order, cylinders = specs["firing_order"], specs["cylinders"]
    if sorted(order) != list(range(1, cylinders + 1)):
        errors.append(f"firing_order must list each cylinder 1-{cylinders} exactly once")
    return errors


def reliability_warning(specs: dict[str, Any]) -> str | None:
    """Warning for engines outside the configuration the diagnostics are tuned for."""
    if all(specs.get(k) == v for k, v in SUPPORTED.items()):
        return None
    return (f"This engine has {specs.get('cylinders')} cylinders and {specs.get('strokes')} "
            "strokes. The diagnostics are tuned for inline 4-cylinder, 4-stroke engines, so "
            "results may not be reliable for it. History is still saved.")
