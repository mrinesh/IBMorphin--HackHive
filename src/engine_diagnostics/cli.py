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
    warned = 0
    for run_dir in run_dirs:
        warnings = []
        try:
            bundle = run_io.load_run(run_dir)
            issues = run_io.validate_run(bundle, config)
            warnings = run_io.run_warnings(bundle, config) if not issues else []
        except (OSError, ValueError) as error:
            issues = [str(error)]
        failures += bool(issues)
        warned += bool(warnings)
        status = "[INVALID]" if issues else "[WARN]" if warnings else "[OK]"
        print(f"{status} {run_dir.name}")
        for message in issues + warnings:
            print(f"  {message}")
    if warned:
        print(f"{warned} runs carry declared data-quality problems (allowed)")
    if failures:
        print(f"{failures} of {len(run_dirs)} runs invalid; manifest not written")
        return 1
    manifest = run_io.write_manifest(raw_dir, run_dirs)
    print(f"{len(run_dirs)} runs valid; manifest written to {manifest}")
    return 0


def _build_features(args, config: dict) -> int:
    try:
        from engine_diagnostics.pipeline.dataset import build_feature_table
        from engine_diagnostics.pipeline.features import FEATURE_VERSION
    except ImportError as error:
        print(f"build-features needs the analysis extra ({error.name} missing)", file=sys.stderr)
        return 1
    try:
        table = build_feature_table(args.raw_dir, args.truth_dir, args.batch_config, config)
    except (OSError, ValueError) as error:
        print(f"Feature build failed: {error}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)
    counts = table.groupby(["split", "label"]).size()
    print(f"{len(table)} windows, feature version {FEATURE_VERSION} -> {args.output}")
    print(counts.to_string())
    return 0


def _train_temporary(args) -> int:
    try:
        import joblib
        import pandas as pd

        from engine_diagnostics.models.temporary import evaluate, fit_temporary_model
    except ImportError as error:
        print(f"train-temporary needs the analysis extra ({error.name} missing)", file=sys.stderr)
        return 1
    if not args.features.is_file():
        print(f"{args.features} not found; run build-features first", file=sys.stderr)
        return 1
    table = pd.read_parquet(args.features)
    model = fit_temporary_model(table)
    report = evaluate(model, table)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["status"])
    for name in ("svm", "accepted"):
        part = report[name]
        print(f"{name:9s} macro-F1 {part['macro_f1']:.3f}  windows {part['windows']}  "
              f"recall {json.dumps({k: round(v, 2) for k, v in part['recall'].items()})}")
    print(f"accepted coverage {report['accepted']['coverage']:.3f}; normal windows flagged "
          f"anomalous {report['normal_windows_flagged_anomalous']:.3f}")
    for name, part in report["ablations"].items():
        print(f"ablation {name:24s} macro-F1 {part['macro_f1']:.3f}")
    print(f"model -> {args.model}\nreport -> {args.report}")
    return 0


def _replay(args, config: dict) -> int:
    try:
        import joblib
        import numpy as np

        from engine_diagnostics.data.run_io import load_run, validate_run
        from engine_diagnostics.runtime.replay import prediction_log, replay_run
    except ImportError as error:
        print(f"replay needs the analysis extra ({error.name} missing)", file=sys.stderr)
        return 1
    run_dir = args.raw_dir / args.run_id
    if not args.model.is_file():
        print(f"{args.model} not found; run train-temporary first", file=sys.stderr)
        return 1
    bundle = load_run(run_dir)
    issues = validate_run(bundle, config)
    if issues:
        print("\n".join(issues), file=sys.stderr)
        return 1
    model = joblib.load(args.model)
    print(f"SYNTHETIC DATA - RECORDED REPLAY | {args.run_id} | {model.model_version} | "
          f"speed {'fast' if not args.speed else f'{args.speed:g}x'}")

    def show(d):
        alert = d["alert"] or "-"
        print(f"t={d['t_end_s']:5.1f}s  {d['quality']:8s}  svm={d['predicted'] or '-':21s} "
              f"accepted={d['accepted']:21s} anomaly={d['anomaly_score']:5.2f}  "
              f"ALERT={alert:21s} {d['processing_ms']:5.1f} ms")

    records = replay_run(bundle, model, config, speed=args.speed, on_decision=show)
    log = prediction_log(records)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{args.run_id}_predictions.csv"
    log.to_csv(log_path, index=False)
    ms = log["processing_ms"].to_numpy()
    print(f"{len(log)} decisions; processing p50 {np.percentile(ms, 50):.1f} ms, "
          f"p95 {np.percentile(ms, 95):.1f} ms; log -> {log_path}")

    truth_path = args.truth_dir / f"{args.run_id}_events.json"
    if args.with_truth and truth_path.is_file():
        # Evaluation only: compared after replay, never shown to the engine.
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        alerts = log[log["alert"].notna()]
        print(f"[evaluation] truth condition {truth['condition']}, onset {truth['onset_s']}")
        if alerts.empty:
            print("[evaluation] no alert raised")
        else:
            first = alerts.iloc[0]
            delay = ("" if truth["onset_s"] is None
                     else f", {first['t_end_s'] - truth['onset_s']:.1f} s after onset")
            print(f"[evaluation] first alert {first['alert']} at {first['t_end_s']:.1f} s{delay}")
    return 0


def _load_truths(table, truth_dir: Path) -> dict:
    from engine_diagnostics.pipeline.labels import load_truth

    return {run_id: load_truth(truth_dir, run_id) for run_id in table["run_id"].unique()}


def _train(args, config: dict) -> int:
    try:
        import joblib
        import pandas as pd

        from engine_diagnostics.models.diagnostic import fit_diagnostic_model
    except ImportError as error:
        print(f"train needs the analysis extra ({error.name} missing)", file=sys.stderr)
        return 1
    if not args.features.is_file():
        print(f"{args.features} not found; run build-features first", file=sys.stderr)
        return 1
    table = pd.read_parquet(args.features)
    model = fit_diagnostic_model(table, _load_truths(table, args.truth_dir), config)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    notes = model.notes
    selected = notes["decision_selected"]
    summary = {k: v for k, v in notes.items() if k != "decision_trials"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"model_version": model.model_version, **summary,
                                       "decision_trials": notes["decision_trials"]},
                                      indent=2, default=str) + "\n", encoding="utf-8")
    print(f"{model.model_version}: train {notes['train_windows_used']} of "
          f"{notes['train_windows']} windows; SVM {notes['svm_selected']}")
    for trial in notes["svm_search"]:
        print(f"  tuning macro-F1 {trial['tuning_macro_f1']:.3f}  C={trial['C']} "
              f"gamma={trial['gamma']}")
    print(f"selected on tuning runs: p_min {selected['probability_min']}, anomaly quantile "
          f"{selected['anomaly_quantile']}, gate {selected['gate_fault_with_anomaly']}, "
          f"confirm {selected['confirm_updates']}")
    fa = selected["false_alerts_per_normal_minute"]
    print(f"  tuning: false alerts/normal min {fa:.3f}, sustained recall "
          f"{selected['sustained_recall']:.3f}, latency p50 {selected['latency_p50_s']} s, "
          f"wrong-class episodes {selected['wrong_class_episodes']}")
    if not notes["false_alert_limit_met_on_tuning"]:
        print("  WARNING: no setting met the false-alert limit on tuning runs")
    print(f"model -> {args.model}\nreport -> {args.report}")
    return 0


def _evaluate(args, config: dict) -> int:
    try:
        import joblib
        import pandas as pd

        from engine_diagnostics.final_evaluation import evaluate_final
    except ImportError as error:
        print(f"evaluate needs the analysis extra ({error.name} missing)", file=sys.stderr)
        return 1
    if args.report.is_file() and not args.force:
        print(f"{args.report} exists: the test set has already been scored. Use --force "
              "only to reproduce the same frozen model's result.", file=sys.stderr)
        return 1
    table = pd.read_parquet(args.features)
    model = joblib.load(args.model)
    report = evaluate_final(model, table, _load_truths(table, args.truth_dir), args.raw_dir,
                            config, args.model)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(report["status"])
    for subset in ("core", "challenge"):
        if subset not in report:
            continue
        part = report[subset]
        top, acc, rep = part["windows"]["top_class"], part["windows"]["accepted"], part["replay"]
        print(f"[{subset}] {part['runs']} runs | window macro-F1 {top['macro_f1']:.3f} "
              f"(accepted {acc['macro_f1']:.3f} at coverage {acc['coverage']:.3f})")
        print(f"  recall {json.dumps({k: round(v, 2) for k, v in top['recall'].items()})}")
        print(f"  replay: false alerts/normal min {rep['false_alerts_per_normal_minute']:.3f}, "
              f"sustained recall {rep['sustained_recall']}, latency p50/p95 "
              f"{rep['latency_p50_s']}/{rep['latency_p95_s']} s, brief recall "
              f"{rep['brief_recall']}, processing p95 {rep['processing_ms_p95']:.1f} ms")
    for name, part in report["ablations"].items():
        print(f"ablation {name:24s} macro-F1 {part['macro_f1']:.3f}")
    print(f"report -> {args.report}")
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
    build = commands.add_parser("build-features", help="Labeled window features for all runs")
    build.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    build.add_argument("--truth-dir", type=Path, default=Path("evaluation_truth"))
    build.add_argument("--batch-config", type=Path, default=Path("configs/integration_batch.json"))
    build.add_argument("--output", type=Path,
                       default=Path("data/processed/integration_v1_features.parquet"))
    build.add_argument("--config", type=Path, default=Path("configs/mvp.json"))
    build.add_argument("--schema", type=Path, default=Path("schemas/project-config.schema.json"))
    train = commands.add_parser("train-temporary",
                                help="Fit the temporary baseline/SVM and check held-out engines")
    train.add_argument("--features", type=Path,
                       default=Path("data/processed/integration_v1_features.parquet"))
    train.add_argument("--model", type=Path, default=Path("artifacts/models/temporary.joblib"))
    train.add_argument("--report", type=Path,
                       default=Path("reports/generated/temporary_model_report.json"))
    replay = commands.add_parser("replay", help="Replay a recorded run through the model")
    replay.add_argument("run_id")
    replay.add_argument("--speed", type=float, default=None,
                        help="real-time factor, e.g. 1 for 1x; default runs as fast as possible")
    replay.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    replay.add_argument("--model", type=Path, default=Path("artifacts/models/temporary.joblib"))
    replay.add_argument("--log-dir", type=Path, default=Path("artifacts/logs"))
    replay.add_argument("--truth-dir", type=Path, default=Path("evaluation_truth"))
    replay.add_argument("--with-truth", action="store_true",
                        help="after replay, compare alerts with evaluation truth")
    replay.add_argument("--config", type=Path, default=Path("configs/mvp.json"))
    replay.add_argument("--schema", type=Path, default=Path("schemas/project-config.schema.json"))
    fit = commands.add_parser("train", help="Fit the v1 model using train/tuning/calibration")
    fit.add_argument("--features", type=Path,
                     default=Path("data/processed/core_v1_features.parquet"))
    fit.add_argument("--truth-dir", type=Path, default=Path("evaluation_truth/core_v1"))
    fit.add_argument("--model", type=Path, default=Path("artifacts/models/diagnostic_v1.joblib"))
    fit.add_argument("--report", type=Path, default=Path("reports/generated/training_v1.json"))
    final = commands.add_parser("evaluate", help="Score the frozen model once on test engines")
    final.add_argument("--features", type=Path,
                       default=Path("data/processed/core_v1_features.parquet"))
    final.add_argument("--truth-dir", type=Path, default=Path("evaluation_truth/core_v1"))
    final.add_argument("--raw-dir", type=Path, default=Path("data/raw/core_v1"))
    final.add_argument("--model", type=Path, default=Path("artifacts/models/diagnostic_v1.joblib"))
    final.add_argument("--report", type=Path,
                       default=Path("reports/generated/final_evaluation_v1.json"))
    final.add_argument("--force", action="store_true")
    for sub in (fit, final):
        sub.add_argument("--config", type=Path, default=Path("configs/mvp.json"))
        sub.add_argument("--schema", type=Path,
                         default=Path("schemas/project-config.schema.json"))
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.root.resolve())
    if args.command == "train-temporary":
        return _train_temporary(args)
    try:
        config = load_config(args.config, args.schema)
    except (OSError, ValueError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1
    if args.command == "validate-runs":
        return _validate_runs(args.raw_dir, config)
    if args.command == "build-features":
        return _build_features(args, config)
    if args.command == "replay":
        return _replay(args, config)
    if args.command == "train":
        return _train(args, config)
    if args.command == "evaluate":
        return _evaluate(args, config)
    print(json.dumps({
        "status": "valid",
        "schema_version": config["schema_version"],
        "conditions": config["conditions"],
        "window_s": config["analysis"]["window_s"],
        "step_s": config["analysis"]["step_s"],
        "core_runs": config["dataset"]["engine_count"] * config["dataset"]["runs_per_engine"],
    }, indent=2))
    return 0
