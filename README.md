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

**Simultaneous faults.** The `core_v2` MATLAB dataset puts 0-3 faults in the same
run with independent onsets. The v2 model has one calibrated detector and one
alert per fault plus crank-synchronous vibration features. On four new held-out
engines it scores window macro-F1 0.884 (v1: 0.621 on the same windows) and
alerts every sustained fault in 10 of 12 two-fault and 4 of 4 three-fault runs;
valve recall (0.62) and knock false alerts (0.30 per fault-free minute) remain
weak. See PROJECT_REPORT §11.7.

**Crank wheel, ECU benchmark, cylinder temperatures.** The `core_v3` dataset
(data contract 1.2) adds two signal families to the same recordings: a 60-2
crank trigger wheel and per-cylinder exhaust gas and head temperatures. They
feed independent paths that confirm the acoustic alerts and name the cylinder:
crank-speed misfire detection and an EGT/CHT cross-check. A simulated OBD-II
misfire monitor serves as the benchmark. On held-out engines the acoustic
misfire alert came a median 4.5 s before the ECU's type A code, and in 8 of 25
misfire intervals the ECU had set no code by the end of the run. The ECU
monitor cannot see knock or valve faults. Requiring the head-temperature
confirmation cut knock false alerts from 0.30 to 0 per minute, at some cost in
recall. See PROJECT_REPORT §11.9.

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

Choose a demo preset in the sidebar and press Play. Optionally create an
**Engine profile** in the sidebar (make, model, serial; specs from
`configs/engine_catalog.json` or by hand) to save each analysis and notes to that
engine's history in `data/engine_history/`; see
[dashboard/README.md](dashboard/README.md#engine-profiles-and-history).

With a profile selected, **Diagnose** ranks possible causes and next steps from
that engine's history, and **Report** shows a report you can export as PDF
([details](dashboard/README.md#possible-causes-diagnose-and-the-engine-report)).
Diagnose uses Google Gemini's free tier when a key is set and a basic rule-based
version otherwise. To set up a free key:

1. Sign in at <https://aistudio.google.com/apikey> and choose **Create API key**
   (no billing needed).
2. Copy `.env.example` to `.env` at the repository root (git-ignored) and set
   `GEMINI_API_KEY=<your key>`, or set `GEMINI_API_KEY` in the environment.
   Never commit the key.

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
| `engine-diagnostics train-localizer` | Cylinder localizer (core_v2 features with the cam reference) |
| `engine-diagnostics evaluate-localization` | One-time test-engine scoring of cylinder localization |
| `engine-diagnostics replay <run> ... --localizer artifacts/models/cylinder_localizer.joblib` | Replay that also names the affected cylinder of each alert |
| `engine-diagnostics calibrate-crosscheck` | Crank-speed, OBD monitor and EGT/CHT thresholds from core_v3 training runs |
| `engine-diagnostics evaluate-crosscheck` | Acoustic vs crank vs simulated ECU misfire timing, thermal detection, fused knock experiment on test engines |

Prefix each with `uv run --all-extras`.

## Structure

```text
configs/                   mvp.json (shared settings), core_v1/v2/v3.json, integration_batch.json, demo_presets.json, engine_catalog.json
schemas/                   Configuration schema
matlab/                    Generator entry points, export, validation, review plots
matlab/sim/                Scenario sampling, crank/event model, faults, sensor and telemetry rendering
src/engine_diagnostics/    CLI, data contract, features, labels, models, evaluation, replay
dashboard/                 Streamlit replay dashboard
scripts/                   One-shot Python pipeline
data/raw/                  Exported run bundles (generated, not tracked)
data/processed/            Feature tables (generated)
data/engine_history/       Optional per-engine history saved by the dashboard (not tracked)
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
