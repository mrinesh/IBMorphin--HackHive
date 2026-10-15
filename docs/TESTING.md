# Verification status and checks still required

Updated: 15 October 2026. This record covers initialization and the first integration
batch, not a working engine-diagnostics pipeline.

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
| MATLAB MCP connection (2026-10-15) | Passed | `evaluate_matlab_code` and `detect_matlab_toolboxes` attach to R2026b after `shareMATLABSession` |
| MATLAB generator static analysis (2026-10-15) | Passed | `checkcode` reports no messages on all 16 `.m` files |
| Integration batch generation (2026-10-15) | Passed | 3 runs x 20 s (misfire, knock-like, valve-clearance-like, onset 8 s); ~6.5 s per run; `waveforms.mat` ~3.6 MB, telemetry 40 KB, truth 0.3-0.75 MB per run |
| MATLAB run validation (2026-10-15) | Passed | `validate_run_matlab` on all 3: dtype/length, no clipping (peaks 13-17% of full scale), combustion count within 0.1% of RPM-derived expectation, faulted events only after onset, knock 5-9 kHz band power x2.13 after onset |
| Generator reproducibility (2026-10-15) | Passed | Regenerating run 1 gave identical arrays, telemetry CSV, and truth JSON (`isequal`) |
| Aliasing (2026-10-15) | Passed with note | Noise-free render: energy above 0.9 x Nyquist is -42 to -57 dB of total; full alias-free proof not attempted |
| Integration batch extended (2026-10-15) | Passed | 13 runs on 4 engines (train engine_001-003, test engine_004): 4 normal-only, 3 each of misfire/knock-like/valve-clearance-like; 80.6 s total; 48 MB public + 6.8 MB truth. Runs 1-3 regenerated bit-identical. `validate_run_matlab` 13/13; Python `validate-runs` 13/13 |
| Knock signature check (2026-10-15) | Passed | Event-locked 5-9 kHz energy after knocked vs unknocked combustion: x7.07, x3.67, x4.22. The earlier whole-band before/after check failed two valid runs (x1.33, x1.25) because normal valve modes and RPM ramps share that band, so it was replaced |
| Feature path (2026-10-15) | Passed | `build-features`: 481 windows from 13 runs in 5.7 s (~12 ms/window), all quality `ok`, no NaN. Unit tests: sine RMS/crest/kurtosis and band placement, impulsive and missing-firing fixtures, quality states, causal slicing, past-only telemetry context, label boundaries, features module imports no label/truth code |
| Temporary model pipeline check (2026-10-15) | Ran; not an evaluation | Trained on engine_001-003 (348 clean windows), checked on held-out engine_004 (133). SVM macro-F1 0.984; accepted decisions macro-F1 0.986 at 0.857 coverage; 5.5% of normal windows exceed the anomaly threshold. Ablations: telemetry-only 0.609, audio+context 0.984, vibration+context 0.745 (knock recall 0.05), fused without context 0.991. One held-out engine and clean synthetic faults make these scores optimistic |
| Alert persistence and model rules (2026-10-15) | Passed | Unit tests: confirm after 2 consistent faults, clear after 3 normals, uncertain/bad quality hold state; training uses only clean training windows; faults accepted only when SVM and anomaly detector agree |
| Replay (2026-10-15) | Passed | Replay features equal offline features exactly (max difference 0) on all 481 windows of 13 runs. Held-out engine_004: normal run no alert; misfire alert 1.5 s, knock 1.0 s, valve 2.0 s after onset (valve 13/37 updates uncertain). Processing p50 12.0 ms, p95 13.2 ms, max 40.4 ms (fast); 1x replay of run_00011 took 21.2 s wall for 20 s, p50 24.3 / p95 34.0 ms. Tests: offline equality, decisions unchanged by samples after t, alert sequence, reset |
| Truth null encoding (2026-10-15) | Fixed | Normal-run truth wrote `onset_s: []`; now `null` (MATLAB `jsonencode` writes NaN as null). Regenerated 13 runs; waveforms unchanged; 13/13 valid |
| Review figures (2026-10-15) | Inspected | `plot_run_overview` spectrograms/zooms show the intended knock band, weak misfire strokes, and shifted valve impacts |
| MATLAB/Python handoff (2026-10-15) | Passed | `validate-runs`: all 3 runs valid in Python (column orientation, float32 class, sample counts/rates, t0, summary sums/first/last vs MATLAB, telemetry units/rates/order); manifest written |
| Run I/O tests (2026-10-15) | Passed | 31 tests with extras (17 contract-violation/loader tests, 5 on real exported runs); 9 passed, 2 skipped in a base-only isolated env; Ruff clean |

The first pytest attempt failed because the agent's Windows sandbox denied access
to pytest temporary/cache directories. All nine tests passed when run with normal
filesystem access. This was an environment restriction, not a passing sandbox run.

## Needs testing before implementation depends on it

| Component | Required check | Current reason/status |
|---|---|---|
| Simulink / Powertrain Blockset | Install only if the optional Simulink comparison is pursued | Licensed but not installed; not required for the MVP |
| MATLAB/Python feature agreement | Compute the same features in MATLAB and compare with tolerances (checklist §6) | Not implemented; Python is the only feature implementation so far |
| Context shortcut | Decorrelate scenario RPM/load from condition, then repeat the telemetry-only ablation | Confirmed: telemetry-only recall 0.94 misfire / 0.95 valve on the held-out engine because scenarios use class-distinctive RPM/load ranges. Waveform-only features score 0.991, so the model does not need context, but scenarios must be decorrelated before any evaluation |
| Anomaly threshold and margin | Tune on a separate tuning partition | Threshold is the 99th percentile of training normal scores; margin 0.3 is untuned |
| Replay with nonzero stream offsets | Support `t0_s != 0` and clock drift in replay | Replay rejects nonzero `t0_s`; all current runs use 0 |
| Robustness settings | Generate and validate clock offsets/drift, missing telemetry, and clipping end to end | The format supports them; the generator does not produce them yet |
| MATLAB discovery in `doctor` | Add `/Applications/MATLAB_*.app/bin` to PATH or teach `doctor` to search it | `matlab` is not on PATH on macOS, so `doctor` reports it missing |
| Full Python stack on Windows | Repeat the extras install and imports on the original Windows machine | Verified on macOS only |
| Other Python versions | Exercise initialization on each version claimed as supported | Only Python 3.13.7 has been tested; package declares 3.11-3.13 |

## Required checks at later milestones

These components do not exist yet; they must not be described as tested features.

| Milestone | Verification needed |
|---|---|
| Signal processing | Analytical fixtures, PSD scaling, causal filtering, no future samples, and offline/replay equivalence |
| Data partitions | Engine/run isolation before overlapping windows; preprocessing fitted on training only |
| Modeling | Held-out results, modality ablations, shortcut probes, uncertainty handling, and failure analysis |
| Dashboard/replay | Actual inference, quality states, seek/reset behavior, alerts, and measured latency |
| Demonstration | Three complete offline rehearsals and a clearly labeled fallback recording |

## Commands available now

```powershell
uv run engine-diagnostics doctor
uv run engine-diagnostics validate-config
uv run --all-extras engine-diagnostics validate-runs
uv run --all-extras pytest
uv lock --check --offline
```

Update this file when verification status changes. Put relevant passed, failed,
and unexecuted checks in the associated commit body and task summary as well.
