# Multisensor Engine Diagnostics

MATLAB/Python project for the Honeywell hackathon problem on engine signature
extraction and pattern recognition.

**Current stage: first end-to-end milestone (checklist §3).** A config-driven
MATLAB generator produces synchronized audio, vibration, and telemetry runs
(16 × 20 s on 4 virtual engines: normal-only, or normal then misfire,
knock-like, or valve-clearance-like). Python validates them against the data
contract, extracts window features, trains a temporary baseline/SVM model on
three engines, and replays runs in 40 ms packets with persistent alerts, in the
terminal or a local Streamlit dashboard. Cross-engine replay uses a separate
known-healthy calibration recording from the same engine. Synthetic leave-one-
engine-out scores improve with calibration but still miss many knock events;
real cross-domain validity is unproven.

## Quick start

Run these commands from the repository root in PowerShell:

```powershell
uv sync --locked
uv run engine-diagnostics doctor
uv run engine-diagnostics validate-config
uv run pytest
```

Select `.venv\Scripts\python.exe` as the IDE interpreter. The local interpreter is
Python 3.13; supported package range is 3.11-3.13. Exact dependency versions are
recorded in `uv.lock`; pytest is included by default. Initial setup installs the
configuration tools and tests. Install the scientific/dashboard stack before
starting the integration milestone:

```powershell
uv sync --locked --all-extras --group quality
uv run --all-extras --group quality ruff check .
```

The full environment includes NumPy, SciPy, pandas, PyArrow, scikit-learn,
Streamlit, Plotly, and Ruff. `doctor` distinguishes optional packages from missing
setup requirements. Use `uv run --all-extras` for future analysis/dashboard
commands, so dependency synchronization retains those extras.

With Python instead of `uv`, editable setup is possible using
`python -m pip install -e .` (or `-e ".[analysis,dashboard]"` for the full stack),
but use the lockfile workflow for the team's shared
environment. See the [uv locking/syncing documentation](https://docs.astral.sh/uv/concepts/projects/sync/).

## MATLAB setup

Open MATLAB with the repository root as its current folder:

```matlab
addpath('matlab');
project = initialize_project();
```

This loads the same JSON settings and prints the installed products. Generate
and check the integration batch (requires Signal Processing Toolbox):

```matlab
runs = generate_integration_batch();   % data/raw and evaluation_truth
```

```powershell
uv run --all-extras engine-diagnostics validate-runs
uv run --all-extras engine-diagnostics build-features
uv run --all-extras engine-diagnostics train-temporary
uv run --all-extras engine-diagnostics replay run_00012 --calibration-run run_00010 --speed 1 --with-truth
uv run --all-extras streamlit run dashboard/app.py
uv run --all-extras pytest
```

On macOS MATLAB is at `/Applications/MATLAB_R2026b.app` and not on PATH. See
[MATLAB setup](matlab/README.md).

## Structure

```text
configs/                   Shared JSON settings
schemas/                   Configuration schema
matlab/                    MATLAB initialization and future generator
src/engine_diagnostics/    Python CLI and future analysis modules
dashboard/                 Dashboard workstream notes
data/raw/                  Future exported run bundles
data/processed/            Future validated telemetry/features
evaluation_truth/          Evaluation-only labels and latent truth
artifacts/models/          Future model bundles
artifacts/logs/            Future prediction logs
reports/generated/         Future validation and evaluation reports
tests/                     Configuration safety checks
docs/                      Data contract and decisions
```

Generated datasets, virtual environments, private truth, and experiment artifacts
are ignored by Git. Local commits and verification reporting follow
[AGENTS.md](AGENTS.md). Publishing or pushing requires a separate instruction.

## Agreed MVP settings

- Four classes: normal, misfire, knock-like bursts, valve-clearance-like impacts.
- Audio at 24 kHz, one vibration axis at 25.6 kHz, and slower telemetry.
- Two-second analysis windows updated every half-second.
- MATLAB event-based synthesis and file export; Python owns inference features.
- Normal-baseline detector plus a feature-based SVM, followed by persistent alerts.
- Local replay dashboard with signatures, evidence, quality states, and a simulated
  degradation trend. Physical wear and remaining useful life remain unvalidated.

These are configured intentions, not implemented diagnostic capabilities. The
validator checks schema, sampling limits, window relationships, and split counts.

## Next milestone

Improve generator fidelity and decorrelate scenario RPM/load from the fault
class (a telemetry-only model currently finds a shortcut), add robustness
settings, then generate and freeze the engine-split dataset (checklist §4-§5).
The handoff is specified in [DATA_CONTRACT.md](docs/DATA_CONTRACT.md).

## Project documents

- [Complete project report](PROJECT_REPORT.md)
- [Implementation checklist](PROJECT_CHECKLIST.md)
- [Data contract](docs/DATA_CONTRACT.md)
- [Decision log](docs/DECISIONS.md)
- [Verification status and pending checks](docs/TESTING.md)
