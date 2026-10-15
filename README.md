# Multisensor Engine Diagnostics

MATLAB/Python project for the Honeywell hackathon problem on engine signature
extraction and pattern recognition.

A MATLAB simulator generates synchronized microphone, accelerometer, and
ECU-style telemetry for a four-cylinder engine under normal operation and three
modeled faults (misfire, knock-like bursts, valve-clearance-like impacts). A
Python pipeline validates the data, extracts 31 window features, and trains a
normal-baseline detector plus a calibrated SVM. Recorded runs are replayed in
40 ms packets through the same code into a local Streamlit evidence dashboard.

On four held-out virtual engines: macro-F1 0.835, no false alerts during normal
replay, and 13 ms processing per decision; mild faults and the reserved
challenge conditions are largely missed. **All data is synthetic**; these results
do not establish real-engine accuracy, physical wear, or remaining useful life.
A leave-one-engine-out check on the 16-run integration batch shows poor transfer
between synthetic engines (mean macro-F1 0.389); the dashboard can replay a run
with its engine held out to show this live.

## Quick start

Results and design: [PROJECT_REPORT.md](PROJECT_REPORT.md). Demo walkthrough:
[docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md).

Run from the repository root (macOS/Linux shell or PowerShell):

```powershell
uv sync --locked --all-extras --group quality
uv run --all-extras engine-diagnostics doctor
uv run --all-extras --group quality pytest
```

**Dashboard from the sample package (no MATLAB needed).** Unzip
`submission_sample.zip` (seven held-out demo runs, their evaluation truth, the
frozen v1 model, and the final reports) at the repository root, then:

```powershell
uv run --all-extras streamlit run dashboard/app.py
```

Choose a demo preset in the sidebar and press Play.

**Full reproduction.** Generate the core dataset in MATLAB (about 40 minutes in
one process, or split `Runs` across several processes), then run the Python
pipeline:

```matlab
addpath('matlab'); generate_dataset(); validate_generator();
```

```powershell
scripts/run_pipeline.sh            # validate, features, train
scripts/run_pipeline.sh --final    # also score the test engines once
```

The pipeline reproduces the feature table and model bit for bit.

## MATLAB setup

Open MATLAB with the repository root as its current folder (requires Signal
Processing Toolbox; Statistics and Machine Learning Toolbox for `kurtosis`):

```matlab
addpath('matlab');
project = initialize_project();          % settings and installed products
summary = generate_dataset();            % core_v1: data/raw/core_v1, evaluation_truth/core_v1
report = validate_generator();           % generator checks and MATLAB feature references
runs = generate_integration_batch();     % optional 16-run integration batch
```

On macOS MATLAB is at `/Applications/MATLAB_R2026b.app` and not on PATH. For
command-line batch generation from a path containing an apostrophe, launch
`matlab -batch` with `-sd` pointing to a symlink without one. See
[matlab/README.md](matlab/README.md).

## Command reference

| Command | Purpose |
|---|---|
| `engine-diagnostics validate-runs <dir>` | Contract validation, warnings, SHA-256 manifest |
| `engine-diagnostics build-features ...` | Labeled window feature table (Parquet) |
| `engine-diagnostics train` | v1 model: train/tuning/calibration partitions |
| `engine-diagnostics evaluate` | One-time test-engine scoring |
| `engine-diagnostics replay <run> --raw-dir data/raw/core_v1 --model artifacts/models/diagnostic_v1.joblib --truth-dir evaluation_truth/core_v1 --speed 1 --with-truth` | Terminal replay with prediction log |
| `engine-diagnostics train-temporary` | Temporary model for the integration batch |

Prefix each with `uv run --all-extras`.

## Structure

```text
configs/                   mvp.json (shared settings), core_v1.json, integration_batch.json, demo_presets.json
schemas/                   Configuration schema
matlab/                    Generator entry points, export, validation, review plots
matlab/sim/                Scenario sampling, crank/event model, faults, sensor and telemetry rendering
src/engine_diagnostics/    CLI, data contract, features, labels, models, evaluation, replay
dashboard/                 Streamlit replay dashboard
scripts/                   One-shot Python pipeline
data/raw/                  Exported run bundles (generated, not tracked)
data/processed/            Feature tables (generated)
evaluation_truth/          Evaluation-only labels and source events (generated)
artifacts/                 Models, prediction logs, submission sample (generated)
reports/generated/         Validation and evaluation reports (generated)
tests/                     Contract, DSP, leakage, replay, model, and dashboard tests
docs/                      Data contract, decisions, testing record, demo script
```

Generated datasets, virtual environments, private truth, and experiment artifacts
are ignored by Git. Local commits and verification reporting follow
[AGENTS.md](AGENTS.md). Publishing or pushing requires a separate instruction.

## Status

The submission prototype is complete; see the acceptance record in
[PROJECT_REPORT.md](PROJECT_REPORT.md) §11.6. Remaining items are listed in
[docs/TESTING.md](docs/TESTING.md).

## Project documents

- [Complete project report](PROJECT_REPORT.md)
- [Implementation checklist](PROJECT_CHECKLIST.md)
- [Data contract](docs/DATA_CONTRACT.md)
- [Decision log](docs/DECISIONS.md)
- [Verification status and pending checks](docs/TESTING.md)
