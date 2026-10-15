# Decision log

## 2026-10-15: project initialization

- MATLAB generates the synthetic scenarios; Python owns the shared offline/replay
  feature implementation and the SVM/dashboard workflow.
- JSON is the common configuration format. Use MAT Version 7 numeric arrays,
  CSV telemetry, and JSON metadata for the future handoff.
- Python 3.13 is installed on this machine. The package supports 3.11-3.13;
  `.python-version` selects 3.13 locally. Resolve exact dependencies with `uv.lock`.
- MATLAB access is assumed by the user, but MATLAB is not discoverable on this
  machine's PATH during initialization. Execution and toolbox licenses remain
  unverified. Use `matlab/initialize_project.m` in the available MATLAB session.
- No dataset, diagnostic model, or dashboard is produced by initialization.
- No cloud or external service is required for the local MVP.
- Initialize a lightweight configuration/test environment. Declare the scientific
  and dashboard packages as locked optional extras and Ruff as the quality group;
  enable them with `uv sync --locked --all-extras --group quality`. The first full
  install was stopped because large downloads were slow, not because of a
  dependency compatibility error. Partial cache downloads are retained for reuse.

## 2026-10-15: macOS environment inventory

- Development continues on macOS (Python 3.13.2, uv 0.9.7) as well as Windows.
- MATLAB R2026b is installed at `/Applications/MATLAB_R2026b.app` (not on PATH);
  the MATLAB VS Code extension is also installed.
- Only base MATLAB is installed. Signal Processing, Statistics and Machine
  Learning, Simulink, and Powertrain Blockset are licensed but not installed.
- The full Python analysis/dashboard stack is installed and imports on macOS.
- Signal Processing Toolbox and Statistics and Machine Learning Toolbox were
  then installed and verified; Simulink and Powertrain Blockset remain uninstalled.

## 2026-10-15: first integration batch

- Organizers explicitly allow team-generated synthetic data.
- The first batch is 3 runs × 20 s instead of 12; each starts normal and
  switches to one fault at 8 s, so all four conditions appear. More runs later.
- MATLAB does not hash files (no portable SHA-256 without Java in this
  install); Python `validate-runs` writes `data/raw/manifest.json`.
- MATLAB may store integer-valued doubles compactly in MAT files; Python reads
  with `scipy.io.loadmat(..., mat_dtype=True)` so dtype checks see MATLAB classes.
- Events are keyed by crank angle, so a misfire's RPM dip shifts later events in
  every stream consistently.

## 2026-10-15: integration batch extended to 13 runs

- 13 runs × 20 s on 4 virtual engines. engine_001-003 are training engines
  (3 normal-only runs, 2 runs per fault); engine_004 is held out (one
  normal-only run and one run per fault). The split is in
  `configs/integration_batch.json` (`engine_splits`) and is engine-level.
- Normal runs record `onset_s` and `affected_cylinder` as null in the truth.
- The temporary model trained on this batch is a pipeline check, not an
  evaluation; four engines cannot support performance claims.

## 2026-10-15: final 30-hour plan and core dataset

- About 30 hours remain. Submission: code repository, demo video (recorded by
  the user), and written report. The demo runs on the macOS development machine.
- Core dataset `core_v1`: 30 engines × 8 runs × 30 s (2 per class), split
  18 train / 4 tuning / 4 calibration / 4 test, plus 16 challenge runs on the
  test engines (reserved RPM bands 3,000-3,500 and 800-900, 2× noise, shifted
  resonance, mild severity 0.15-0.3). Chosen by the user instead of 60 × 45 s.
- Scenarios are sampled (`sample_scenarios.m`) from distributions shared by all
  classes. A classifier on the observable scenario parameters scored 0.233
  balanced accuracy across engine folds (chance 0.25).
- Generator 0.2.0 adds recovery intervals, accelerated valve degradation
  (simulated time scale), unrelated impacts, combustion timing jitter, clipping,
  waveform gaps, and telemetry dropouts. Clock offsets/drift remain deferred.
- Clipping and declared gaps are data-quality warnings, not contract violations.

## Open decisions

- Assign team owners and record the actual hackathon deadline.
- Confirm organizer expectations for recorded-data replay in the demo.
- Revisit dataset scale only after timing the small integration batch.

## Initialization verification

- Local environment: Python 3.13.7; uv 0.9.4; 50 packages resolved in the lockfile.
- Lightweight package and setup/test dependencies installed in `.venv`.
- Shared JSON configuration and cross-field validation: passed.
- Configuration safety tests: 9 passed, including Nyquist, window, split-count,
  class-balance, and forbidden inference-context checks.
- `engine-diagnostics doctor`: passed setup checks, reporting optional stack and
  MATLAB discovery separately.
- Lockfile consistency check: passed offline.
- Python source compilation: passed.
- Tests required normal filesystem access because this agent's Windows sandbox
  denied pytest access to its temporary directories. The tests themselves passed.
- MATLAB execution, scientific-stack imports, and Ruff checks are not yet
  verified. No generator, sensor dataset, model, or dashboard was created.

## 2026-10-15: commit and verification convention

- Commit completed repository changes locally with relevant descriptive messages.
- Record performed verification and explicitly flag unexecuted checks in commit
  bodies, task summaries, and `docs/TESTING.md`.
- Persist the convention in `AGENTS.md` for subsequent work in this repository.
- Continue excluding environments, caches, secrets, and generated artifacts.
- Do not push or rewrite existing history without a separate instruction.
