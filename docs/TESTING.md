# Verification status and checks still required

Updated: 15 October 2026. This record covers project initialization, not a working
engine-diagnostics pipeline.

## Completed verification

| Check | Recorded result | Scope |
|---|---|---|
| Configuration tests | 9 passed on Python 3.13.7 | Valid defaults; rejected Nyquist, timing, segment, split, class-balance, and hidden-context violations |
| Shared configuration | Passed | JSON schema and cross-field relationships |
| Environment doctor | Passed setup checks | Reports missing optional packages and MATLAB discovery separately |
| Dependency lock | Offline consistency check passed | 50 packages resolved; only setup/test dependencies installed |
| Source compilation | Passed | Python syntax under `src/` and `tests/` |
| macOS full Python stack (2026-10-15) | Passed | `uv sync --locked --all-extras --group quality` on Python 3.13.2 / uv 0.9.7; NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, pandas, PyArrow, Streamlit, Plotly import |
| macOS pytest and Ruff (2026-10-15) | Passed | 9 tests passed; `ruff check .` reported no issues |
| MATLAB initializer (2026-10-15) | Passed | `initialize_project()` ran in R2026b (26.2) via `/Applications/MATLAB_R2026b.app/bin/matlab -batch`; config loaded |
| MATLAB product inventory (2026-10-15) | Recorded | `ver` lists only MATLAB and MATLAB Copilot. `license('test')` is true for Signal, Simulink, Statistics, and Powertrain Blockset, but these are licensed, not installed: `butter`, `pwelch`, `hann`, `envelope`, `fitcsvm` are undefined |
| MATLAB toolboxes installed (2026-10-15) | Passed | `ver` now lists Signal Processing Toolbox and Statistics and Machine Learning Toolbox; `butter`, `pwelch`, `hann`, `envelope`, `fitcsvm` (RBF), and `kurtosis` executed |

The first pytest attempt failed because the agent's Windows sandbox denied access
to pytest temporary/cache directories. All nine tests passed when run with normal
filesystem access. This was an environment restriction, not a passing sandbox run.

## Needs testing before implementation depends on it

| Component | Required check | Current reason/status |
|---|---|---|
| Simulink / Powertrain Blockset | Install only if the optional Simulink comparison is pursued | Licensed but not installed; not required for the MVP |
| MATLAB discovery in `doctor` | Add `/Applications/MATLAB_*.app/bin` to PATH or teach `doctor` to search it | `matlab` is not on PATH on macOS, so `doctor` reports it missing |
| Full Python stack on Windows | Repeat the extras install and imports on the original Windows machine | Verified on macOS only |
| Other Python versions | Exercise initialization on each version claimed as supported | Only Python 3.13.7 has been tested; package declares 3.11-3.13 |

## Required checks at later milestones

These components do not exist yet; they must not be described as tested features.

| Milestone | Verification needed |
|---|---|
| MATLAB integration batch | Repeatable generation; shapes, units, event timing, bandwidth, and cross-modal consistency |
| MATLAB/Python handoff | MAT Version 7 import; vector orientation, float32 values, sample rates, offsets, and telemetry parsing |
| Signal processing | Analytical fixtures, PSD scaling, causal filtering, no future samples, and offline/replay equivalence |
| Data partitions | Engine/run isolation before overlapping windows; preprocessing fitted on training only |
| Modeling | Held-out results, modality ablations, shortcut probes, uncertainty handling, and failure analysis |
| Dashboard/replay | Actual inference, quality states, seek/reset behavior, alerts, and measured latency |
| Demonstration | Three complete offline rehearsals and a clearly labeled fallback recording |

## Commands available now

```powershell
uv run engine-diagnostics doctor
uv run engine-diagnostics validate-config
uv run pytest
uv lock --check --offline
```

Update this file when verification status changes. Put relevant passed, failed,
and unexecuted checks in the associated commit body and task summary as well.
