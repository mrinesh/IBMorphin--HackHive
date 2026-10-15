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

## Open decisions

- Assign team owners and record the actual hackathon deadline.
- Confirm organizer expectations for team-generated data and replay.
- Install the licensed Signal Processing Toolbox, or keep the generator
  base-MATLAB-only (Python/SciPy already covers reference DSP).
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
