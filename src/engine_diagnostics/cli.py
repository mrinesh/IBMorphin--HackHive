"""Workspace diagnostics, configuration validation, and run-bundle validation."""

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

from engine_diagnostics import __version__
from engine_diagnostics.config import load_config

DEPENDENCIES = ("jsonschema",)
OPTIONAL_DEPENDENCIES = (
    "numpy", "scipy", "pandas", "pyarrow", "scikit-learn", "streamlit", "plotly"
)
WORKSPACE_DIRECTORIES = (
    "matlab", "configs", "schemas", "data/raw", "data/processed", "evaluation_truth",
    "artifacts/models", "artifacts/logs", "reports/generated", "dashboard", "docs", "tests",
    "src/engine_diagnostics/data", "src/engine_diagnostics/pipeline",
    "src/engine_diagnostics/models", "src/engine_diagnostics/runtime",
)


def _doctor(root: Path) -> int:
    print(f"Engine diagnostics {__version__} | Python {sys.version.split()[0]}")
    print(f"Workspace: {root}")
    healthy = True
    for name in DEPENDENCIES:
        try:
            print(f"[OK] {name} {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"[MISSING] {name}")
            healthy = False
    for name in OPTIONAL_DEPENDENCIES:
        try:
            print(f"[OK] Optional {name} {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"[INFO] Optional {name} not installed; enable the analysis/dashboard extras")
    for directory in WORKSPACE_DIRECTORIES:
        if not (root / directory).is_dir():
            print(f"[MISSING] Directory: {directory}")
            healthy = False
    try:
        load_config(root / "configs/mvp.json", root / "schemas/project-config.schema.json")
        print("[OK] Shared configuration and numeric contracts")
    except (OSError, ValueError) as error:
        print(f"[INVALID] {error}")
        healthy = False
    matlab = shutil.which("matlab")
    if matlab:
        print(f"[INFO] MATLAB executable found: {matlab}; product licenses are not verified")
    else:
        print("[INFO] MATLAB not on PATH; run matlab/initialize_project.m in MATLAB")
    print("Project stage: integration batch. Features, models, and dashboard remain planned.")
    return 0 if healthy else 1


def _validate_runs(raw_dir: Path, config: dict) -> int:
    try:
        from engine_diagnostics.data import run_io
    except ImportError as error:
        print(f"validate-runs needs the analysis extra ({error.name} missing); run "
              "`uv sync --locked --all-extras`", file=sys.stderr)
        return 1
    run_dirs = run_io.find_runs(raw_dir) if raw_dir.is_dir() else []
    if not run_dirs:
        print(f"No run_NNNNN directories in {raw_dir}", file=sys.stderr)
        return 1
    failures = 0
    for run_dir in run_dirs:
        try:
            issues = run_io.validate_run(run_io.load_run(run_dir), config)
        except (OSError, ValueError) as error:
            issues = [str(error)]
        failures += bool(issues)
        print(f"[OK] {run_dir.name}" if not issues else f"[INVALID] {run_dir.name}")
        for issue in issues:
            print(f"  {issue}")
    if failures:
        print(f"{failures} of {len(run_dirs)} runs invalid; manifest not written")
        return 1
    manifest = run_io.write_manifest(raw_dir, run_dirs)
    print(f"{len(run_dirs)} runs valid; manifest written to {manifest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Engine diagnostics project initialization tools")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check the local workspace and dependencies")
    doctor.add_argument("--root", type=Path, default=Path.cwd())
    validate = commands.add_parser("validate-config", help="Validate the shared scenario settings")
    validate.add_argument("config", type=Path, nargs="?", default=Path("configs/mvp.json"))
    validate.add_argument("--schema", type=Path, default=Path("schemas/project-config.schema.json"))
    runs = commands.add_parser("validate-runs", help="Validate exported run bundles")
    runs.add_argument("raw_dir", type=Path, nargs="?", default=Path("data/raw"))
    runs.add_argument("--config", type=Path, default=Path("configs/mvp.json"))
    runs.add_argument("--schema", type=Path, default=Path("schemas/project-config.schema.json"))
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.root.resolve())
    try:
        config = load_config(args.config, args.schema)
    except (OSError, ValueError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1
    if args.command == "validate-runs":
        return _validate_runs(args.raw_dir, config)
    print(json.dumps({
        "status": "valid",
        "schema_version": config["schema_version"],
        "conditions": config["conditions"],
        "window_s": config["analysis"]["window_s"],
        "step_s": config["analysis"]["step_s"],
        "core_runs": config["dataset"]["engine_count"] * config["dataset"]["runs_per_engine"],
    }, indent=2))
    return 0
