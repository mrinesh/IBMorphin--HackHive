# Track checklist

Step-by-step checklist for the split in [WORK_PLAN.md](WORK_PLAN.md). Tick items
in your own track only. Each item lists how to do it and what "done" means.
Record every check in [TESTING.md](TESTING.md) (append rows; do not edit the
other track's rows). Commands run from the repository root.

Owners: Track A = ______ (Windows), Track B = ______ (macOS). Deadline: ______.

## 0. Both: before splitting

- [ ] Push current `main` so both tracks start from the merged code.
  Done when: `git status` shows `main...origin/main` with no ahead/behind.
- [ ] Each person pulls and creates a branch: `git switch -c track-a/demo` or
  `git switch -c track-b/kaggle`.
- [ ] Fill in owners, deadline, and the B -> A handoff date here and in WORK_PLAN.md.
- [ ] Agree which machine's frozen v1 model is presented (see A1.4). Done when the
  choice and its SHA-256 are written in TESTING.md.
- [ ] Daily: `git fetch; git merge origin/main` into your branch; resolve conflicts
  the same day.

## Track A: demo and submission (Windows)

### A1. Environment and reproduction

- [x] Full Python stack installed: `uv sync --locked --all-extras --group quality`.
- [x] `uv run --all-extras engine-diagnostics doctor` reports all packages and MATLAB.
- [x] `uv run --all-extras --group quality pytest`: 68 passed, 0 skipped.
- [x] core_v1 generated (256 runs) and `train` completed on Windows.
- [ ] A1.4 Cross-platform difference: copy macOS `data/raw/core_v1/manifest.json`
  and compare with Windows; if run files differ, compare one run's
  `waveforms.mat` arrays in Python (`np.max(np.abs(a-b))`). Done when the first
  differing stage (MATLAB output, feature table, or model) is recorded.
- [ ] If presenting the Windows model: re-check the preset timings quoted in
  `docs/DEMO_SCRIPT.md` (e.g. knock alert ~15 s) and update any that moved.
- [ ] Do **not** run `engine-diagnostics evaluate` unless the team decides to
  re-score the final test; it is a one-time evaluation.

### A2. Dashboard review (all 7 presets)

Start: `uv run --all-extras streamlit run dashboard/app.py`, open
http://localhost:8501, speed 1x, truth overlay off.

For each preset confirm: plays without errors; banner visible; telemetry,
waveforms, spectrum/spectrogram, envelope render; decisions update every 0.5 s;
alert behavior matches the expectation below; seek and restart reset cleanly.

- [ ] 1. Normal operation with a load step (`run_00226`): load step ~9 s; no alert.
- [ ] 2. Knock-like bursts, then a load change (`run_00237`): bursts 12-15 s;
  knock alert ~15 s; evidence table shows high-band envelope statistics.
- [ ] 3. Intermittent misfire (`run_00235`): intermittent misfire decisions;
  alert behavior consistent with persistence rules.
- [ ] 4. Misfire with a recovery interval (`run_00228`): alert raises, clears
  during recovery, returns.
- [ ] 5. Valve clearance: accelerated degradation trend (`run_00216`), 2x speed:
  wear indicator rises.
- [ ] 6. Data-quality state: waveform gap (`run_00225`): gap ~12.5 s marked
  missing waveform; no prediction during gap; alert state held.
- [ ] 7. Limitation: mild knock at 3,300 rpm (`run_00247`): model misses it.
- [ ] Truth overlay toggles on/off and is off by default.
- [ ] No console errors in the Streamlit terminal during a full pass.
- [ ] Fix display-only issues on `track-a/*`; re-run `pytest tests/test_dashboard.py`.

### A3. Offline independence

- [ ] Close every MATLAB process (Task Manager shows none).
- [ ] Disconnect from the network, start the dashboard, play presets 1 and 2.
  Done when both play fully with MATLAB closed and offline.

### A4. Rehearsal

- [ ] Rehearsal 1 following `docs/DEMO_SCRIPT.md`, timed (target ~5 min).
- [ ] Rehearsal 2, consecutive, no restarts.
- [ ] Rehearsal 3, consecutive, no restarts. Record times and any glitches in TESTING.md.
- [ ] Every spoken number appears in `PROJECT_REPORT.md` §11; remove any that do not.

### A5. Recording and figures

- [ ] Screenshots: banner + normal decisions, knock alert with evidence table,
  truth overlay, degradation trend, data-quality gap, challenge miss.
- [ ] Full fallback video following the script; title card says
  "Synthetic data - recorded replay".
- [ ] Watch the video once end to end; check audio and readable text.

### A6. Submission package

- [ ] Build `artifacts/submission_sample.zip`: the 7 preset runs with their
  truth, the frozen v1 model, final and training reports (matching README).
- [ ] Test it: in a fresh clone, unzip at the repo root, `uv sync --locked
  --all-extras`, start the dashboard, play presets 1 and 7.
- [ ] Check organizer rules: naming, file-size limits, required items, deadline/time zone.
- [ ] Keep the zip out of Git (`artifacts/*.zip` is ignored); record its SHA-256.

### A7. Slides (synthetic results)

- [ ] Problem and approach; architecture (MATLAB -> exported files -> Python -> dashboard).
- [ ] Results with denominators: 4 held-out engines, 32 core runs, 1,568 windows;
  macro-F1 0.835; per-class recall; false alerts; latency; processing time.
- [ ] Weaknesses: valve/knock recall below 0.75 goal; challenge macro-F1 0.384,
  knock recall 0; mild faults missed.
- [ ] Ablations: telemetry-only near chance -> model uses sensor signatures.
- [ ] Space for Track B's real-data slide (from the handoff).
- [ ] "Responsible claim" slide: no real-engine accuracy, wear, or remaining-life claims.

### A8. Stretch (only after A1-A7)

- [ ] Generator robustness settings (clock offset/drift, missing telemetry,
  clipping) as a **new batch ID**; core_v1 stays unchanged.
- [ ] Replay support for nonzero `t0_s`, with tests.

## Track B: real data (Kaggle, macOS)

### B1. Dataset selection

- [ ] Shortlist candidates; for each record modality (audio/vibration/telemetry),
  sample rate, recording length, labels, number of distinct machines, size.
- [ ] Check the license allows hackathon/demo use and redistribution limits
  (do not include the data in the submission zip if redistribution is not allowed).
- [ ] Choose one; write `docs/REAL_DATA.md` with name, URL, version/date, license,
  download command, and SHA-256 of the downloaded archive.
- [ ] Store data under `data/raw/kaggle_<name>/` (ignored by Git). Never commit
  data or Kaggle API credentials (`kaggle.json`).

### B2. Label and format mapping

- [ ] Map dataset labels to normal / misfire / knock_like / valve_clearance_like;
  list unmapped labels and missing classes in REAL_DATA.md.
- [ ] Compare sample rates to the contract (audio 24 kHz, vibration 25.6 kHz);
  decide resample vs. native rate and document it.
- [ ] Decide how missing modalities are handled (e.g. audio-only features with
  telemetry absent) and document which features become unavailable.
- [ ] Keep real labels in a separate file from model inputs, like simulation truth.

### B3. Loader and pipeline

- [ ] `src/engine_diagnostics/data/kaggle.py`: converts recordings to the run-bundle
  contract or directly to feature windows using `pipeline/features.py` unchanged.
- [ ] `configs/kaggle_<name>.json` for paths, rates, and splits.
- [ ] New CLI subcommand(s) only (e.g. `kaggle-features`, `kaggle-evaluate`); no
  reorganization of existing commands in `cli.py`.
- [ ] If a dependency is needed (e.g. `kagglehub`), add it as an optional extra,
  run `uv lock`, commit `pyproject.toml` and `uv.lock` together.
- [ ] Tests in `tests/test_kaggle.py`: loader on a tiny fixture, label mapping,
  no label columns in feature inputs; skip cleanly when data is absent.
- [ ] Ruff clean; full `pytest` passes on macOS.

### B4. Evaluation

- [ ] Split by machine/recording (never random windows); record the split.
- [ ] Experiment 1: frozen synthetic v1 model applied to real data, unchanged.
- [ ] Experiment 2 (optional): model trained on real training split, scored on
  real test split. Outputs to `artifacts/models/kaggle_<name>*`.
- [ ] Report macro-F1, per-class precision/recall, support, confusion matrix,
  and coverage; write to `reports/generated/kaggle_<name>*.json`.
- [ ] Sanity check for shortcuts (e.g. recording device, file length, loudness
  differing by class).
- [ ] Record results in TESTING.md and a separate real-data section in
  `PROJECT_REPORT.md`; never merge with synthetic metrics.

### B5. Presentation handoff

- [ ] Optional dashboard page `dashboard/pages/real_data.py` (do not edit `app.py`).
- [ ] One slide: dataset, mapping, split, results with denominators, limitations.
- [ ] One figure (confusion matrix or example spectrogram).
- [ ] Hand slide + figure + numbers to Track A by the handoff date: ______.
- [ ] Open a pull request from `track-b/kaggle` into `main`.

## Final: both

- [ ] Both branches merged to `main`; `pytest` and Ruff pass on both machines.
- [ ] README instructions work for someone who did not build the project.
- [ ] Every claim in slides, video, and report traces to a row in TESTING.md
  or a section of PROJECT_REPORT.md.
- [ ] Submit; record the commit hash, model SHA-256, and zip SHA-256 submitted.
