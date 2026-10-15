# Verification status and checks still required

Updated: 15 October 2026. This record covers project initialization, not a working
engine-diagnostics pipeline.

## Completed verification

| Check | Recorded result | Scope |
|---|---|---|
| Configuration tests | 9 passed on Python 3.13.7 | Valid defaults; rejected Nyquist, timing, segment, split, class-balance, and hidden-context violations |
| Shared configuration | Passed | JSON schema and cross-field relationships |
| Environment doctor | Passed setup checks; finds MATLAB R2026b on PATH | Reports missing optional packages and MATLAB discovery separately |
| Dependency lock | Offline consistency check passed | 50 packages resolved; only setup/test dependencies installed |
| Source compilation | Passed | Python syntax under `src/` and `tests/` |
| MATLAB initializer | Passed in MATLAB R2026b (26.2.0.3386108) via `matlab -batch` | Loaded `configs/mvp.json`; Nyquist assertions passed |
| MATLAB products/licenses | Signal Processing (`butter`, `pwelch`) and Statistics and ML (`fitcknn`) executed | Installed: MATLAB, MATLAB Copilot, Signal Processing, Statistics and ML. Simulink, Powertrain Blockset, AUTOSAR Blockset not installed |

The first pytest attempt failed because the agent's Windows sandbox denied access
to pytest temporary/cache directories. All nine tests passed when run with normal
filesystem access. This was an environment restriction, not a passing sandbox run.

## Needs testing before implementation depends on it

| Component | Required check | Current reason/status |
|---|---|---|
| Simulink-based products | Install and license-check before any dependency on Simulink, Powertrain, or AUTOSAR Blockset | Not installed on this machine |
| Full Python stack | Run `uv sync --locked --all-extras --group quality`, then import scientific/dashboard packages | Full install stopped during slow downloads; extras are locked but not installed |
| Ruff | Run `uv run --all-extras --group quality ruff check .` | Quality tool not installed or executed |
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
