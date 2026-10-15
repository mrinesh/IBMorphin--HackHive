# Work split: demo track and real-data track

Two people work in parallel. The synthetic demo (core_v1 + v1 model) is frozen
for presentation; the Kaggle work is a separate, additive track that must not
change demo inputs or outputs.

| | Track A: demo & submission | Track B: real-data (Kaggle) |
|---|---|---|
| Owner | Windows machine (MATLAB R2026b) | macOS machine |
| Branch | `track-a/<topic>` | `track-b/<topic>` |
| Goal | Rehearsed, reproducible synthetic demo and submission package | Honest evaluation of the pipeline on a real public dataset |

Fill in names and the deadline: Track A = ______, Track B = ______, deadline ______.
Step-by-step tasks with completion criteria: [TRACK_CHECKLIST.md](TRACK_CHECKLIST.md).

## Track A: demo and submission

Reproduce and harden what exists; do not retrain or regenerate the demo dataset.

1. Windows reproduction: confirm `scripts/run_pipeline.sh` steps reproduce the
   feature table (SHA-256 `b638614240a2…`) and v1 model (`805e7559bc6a…`); record
   results in `docs/TESTING.md`.
2. Dashboard visual review in a browser for all 7 presets; fix display-only issues.
3. Rehearse normal -> RPM/load change -> fault onset -> signature change -> alert ->
   evidence using `docs/DEMO_SCRIPT.md`; three consecutive offline rehearsals.
4. Verify replay works with MATLAB closed (exported files only).
5. Record screenshots and a clearly labeled fallback video.
6. Build the submission package (`artifacts/submission_sample.zip`) and check
   organizer rules, naming, and size limits.
7. Slides for synthetic results with denominators, held-out engines, and the
   challenge-set weaknesses (knock recall 0 on challenge runs).
8. Stretch, only after 1-7: generator robustness settings (clock offset/drift,
   missing telemetry, clipping) and nonzero `t0_s` replay, generated as a new
   batch ID so core_v1 stays unchanged.

## Track B: real-data (Kaggle)

1. Pick the dataset; record its name, version, URL, license, and checksum in
   `docs/REAL_DATA.md`. Confirm the license permits hackathon use.
2. Map its labels to the four MVP conditions, or document which classes it lacks.
   Record sample rates and modalities; note any missing audio/vibration/telemetry.
3. Loader in a new module (`src/engine_diagnostics/data/kaggle.py`) that converts
   recordings to the run-bundle contract or to feature windows directly.
4. Separate config, features, model, and report paths:
   `configs/kaggle_<name>.json`, `data/raw/kaggle_<name>/`,
   `data/processed/kaggle_<name>_features.parquet`,
   `artifacts/models/kaggle_<name>*.joblib`, `reports/generated/kaggle_<name>*`.
5. Evaluate with recording/machine-level splits (never random window splits).
   Report both the frozen synthetic-trained v1 model applied to real data and
   any model trained on real data, labeled separately.
6. Optional dashboard view as a separate Streamlit page
   (`dashboard/pages/real_data.py`) so `dashboard/app.py` is not edited.
7. One slide and a `PROJECT_REPORT.md` section on real-data results, kept
   apart from synthetic metrics. Synthetic scores never stand in for real ones.

## File ownership

| Files | Owner | Rule |
|---|---|---|
| `data/raw/core_v1/`, `configs/core_v1.json`, `configs/demo_presets.json`, v1 model | A | Frozen; nobody regenerates or retrains |
| `dashboard/app.py`, `docs/DEMO_SCRIPT.md` | A | B adds pages instead of editing |
| `matlab/` | A | B does not need MATLAB |
| `*kaggle*` files, `docs/REAL_DATA.md`, `dashboard/pages/` | B | A does not edit |
| `src/engine_diagnostics/cli.py` | Shared | B adds new subcommands only; no reorganization |
| `pipeline/features.py` | Shared | Changes alter the frozen model; agree first and bump the feature version |
| `pyproject.toml`, `uv.lock` | B (if adding deps) | On conflict keep one `pyproject.toml`, rerun `uv lock` |
| `docs/TESTING.md` | Shared | Append rows to the end of a table; never edit the other track's rows |
| `README.md`, `PROJECT_REPORT.md`, `configs/mvp.json` | Shared | Small edits in your own section; announce before editing |

## Working rules

- Branch from `main`, merge `main` into your branch daily, and merge back with a
  pull request. Do not commit directly to `main`.
- Data, models, and reports stay out of Git (see `.gitignore`); share them by
  regeneration commands and checksums.
- Real-data labels stay separate from model inputs, as with simulation truth.
- Handoff: B gives A final real-data numbers and one figure by ______ for slides.
