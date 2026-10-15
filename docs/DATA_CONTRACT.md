# Data contract, version 1.0

## Shared settings

`configs/mvp.json` is the authoritative configuration for both MATLAB and Python.
Its structural schema is `schemas/project-config.schema.json`. Validate using
`uv run engine-diagnostics validate-config` before generation. Numeric and timing
relationships are checked in Python as well as JSON structure.

## Run bundle

```text
data/raw/run_00001/
  waveforms.mat
  telemetry.csv
  metadata.json
data/raw/manifest.json              written by Python validate-runs
evaluation_truth/run_00001_events.json
```

Only explicit waveform fields and observed telemetry may reach inference. IDs are
bookkeeping for grouping/logging, not classifier inputs. The public files
(`waveforms.mat`, `telemetry.csv`, `metadata.json`) never contain the condition
label, fault schedule, severity, or seeds.

| File | Fields and rules |
|---|---|
| `waveforms.mat` | Version 7 (`-v7`); numeric `single` column vectors `audio` and `vibration`; `double` scalars `audio_fs_hz`, `vibration_fs_hz`, `audio_t0_s`, `vibration_t0_s`. MATLAB may store integer-valued doubles compactly; read with `loadmat(..., mat_dtype=True)` |
| `telemetry.csv` | Long-form columns `t_s,signal,value,unit,valid` (see Telemetry) |
| `metadata.json` | Public run description (see Metadata) |
| `manifest.json` | SHA-256 and byte size of every public file, per run |
| Private event JSON | Labels, fault schedule, source events, seeds (see Evaluation truth) |

## Time and indexing

- `t = 0` is the start of the generator's master clock for that run. All
  timestamps are elapsed seconds on that clock.
- Each waveform stream starts at its own `t0_s`, the stream's offset from the
  master clock. Sample `i` in zero-based Python indexing occurs at
  `t0_s + i / fs_hz`; MATLAB index `i+1` denotes the same sample.
- Sample counts are `round(duration_s * fs_hz)`. A 20-second run has 480,000
  audio and 512,000 vibration samples.
- `metadata.clock` records `offset_s` and `drift_ppm` for `audio`, `vibration`,
  and `telemetry`. A corrected time is `t_reported * (1 - drift_ppm * 1e-6) - offset_s`.
  The integration batch uses zero offsets and drift; nonzero values are later
  robustness settings.
- Two-second decision windows use `[t-2, t)`. Retain native waveform rates;
  combine per-modality features and fresh telemetry.

## Telemetry

| `signal` | `unit` | Rate | Freshness limit | Initial inference use |
|---|---|---:|---|---|
| `rpm` | `rpm` | 50 Hz | `runtime.rpm_max_age_s` | Operating context and approximate order features |
| `load` | `percent` | 10 Hz | `runtime.load_throttle_max_age_s` | Operating context |
| `throttle` | `percent` | 10 Hz | `runtime.load_throttle_max_age_s` | Operating context |
| `coolant_temp` | `degC` | 1 Hz | 2 s | Display |
| `short_term_fuel_trim` | `percent` | 2 Hz | 1 s | Display |
| `long_term_fuel_trim` | `percent` | 2 Hz | 1 s | Display |

- Sample `k` of a signal is nominally at `k / rate_hz`; timestamps lie in
  `[0, duration_s)`.
- Rows are sorted by `t_s`, then `signal`.
- `valid` is `1` or `0`. A missing reading keeps its row with an empty `value`
  and `valid = 0`. Readers must not interpolate across invalid rows silently.
- Display-only signals go stale after twice their nominal period.
- The engine configuration's load bounds are fractions (0-1); load and throttle
  telemetry are percent (0-100). The renderer converts explicitly.
- Telemetry rates are simulation choices, not guaranteed vehicle polling rates.

## Identifiers and versions

| Field | Format | Example |
|---|---|---|
| `run_id` | `run_` + 5 digits | `run_00001` |
| `engine_id` | `engine_` + 3 digits | `engine_001` |
| `batch_id` | lowercase name + version | `integration_v1` |
| `generator_version` | semantic version of the MATLAB generator | `0.1.0` |
| `config_schema_version` | `schema_version` from `configs/mvp.json` | `1.0` |
| `schema_version` | this data contract's version | `1.0` |

## Metadata

`metadata.json` contains:

- `schema_version`, `run_id`, `engine_id`, `batch_id`, `generator_version`,
  `config_schema_version`, `created_utc`, `synthetic` (always `true`), `duration_s`
- `streams.audio` and `streams.vibration`: `fs_hz`, `t0_s`, `unit`, `n_samples`
- `telemetry_signals`: for each signal, `rate_hz` and `unit`
- `clock`: `offset_s` and `drift_ppm` for `audio`, `vibration`, `telemetry`
- `waveform_summary.audio` and `waveform_summary.vibration`: `n_samples`,
  `n_missing`, `sum`, `sum_sq` (both excluding NaN), `first`, `last` (null if
  NaN), computed in MATLAB in double precision from the exported `single` arrays.
  Missing waveform samples are NaN; up to 5% per run is allowed. Python compares these against the loaded arrays
  to confirm values survived the export (round-trip check).

## Checksums

MATLAB does not write hashes; this R2026b installation has no portable built-in
for SHA-256 without Java. `engine-diagnostics validate-runs` computes SHA-256 and
byte size for each public file and writes `data/raw/manifest.json`. A locked
dataset is identified by its manifest.

## Evaluation truth

`evaluation_truth/<run_id>_events.json` is never read by feature extraction or
inference. It contains:

- `schema_version`, `run_id`, `engine_id`, `generator_version`
- `condition`: `normal`, one canonical fault class, or, for a run with several
  simultaneous faults, their sorted names joined with `+`
  (e.g. `knock_like+misfire`)
- `conditions`: the list of faults in the run (empty for normal)
- `faults`: one entry per fault with its own `condition`, `onset_s`,
  `affected_cylinder`, `severity`, `severity_end`, `degradation`, `recovery_s`,
  and `active_intervals`. Always a JSON list. Each fault has an independent
  lifecycle, so onsets are staggered and recoveries differ
- `onset_s`, `affected_cylinder` (`null` for normal runs), `severity` (0-1,
  generator-specific meaning; 0 for normal runs). For multi-fault runs these
  summarize the list: earliest onset, the cylinder if all faults share one
  (else `null`), and the largest severity; per-fault values are in `faults`
- `active_intervals`: `[start_s, end_s]` pairs where at least one fault is applied
- `label_timeline`: `start_s`, `end_s`, `label` segments. A label is `normal`,
  the active fault set (sorted names joined with `+`), or `transition` for the
  second around each change in that set; changes closer than 1 s share one
  transition
- `events`: every source event with `time_s`, `cylinder`, `kind`
  (`combustion`, `valve_impact`, `knock_burst`), `intensity`, `faulted`, and
  `fault` (the condition that changed or added it, or `""`). Truth written
  before multi-fault support has no `fault` field or `faults` list and is read
  as a single fault
- `seed`: `master_seed` and the random-stream indices used for the run
- Sampled datasets also record `split`, `challenge`, `profile_kind`,
  `degradation`, `severity_end`, `recovery_s`, and `disturbances` (impact rate,
  clip gain, noise scale, resonance shift, waveform gap, telemetry dropout).
  Event kinds include `unrelated_impact`. `scenarios.json` lists every run.

### Multi-fault generation

MATLAB applies a run's faults in the fixed order misfire → knock-like →
valve-clearance-like. Knock and valve events are timed after the misfire RPM
dips, and a misfired (weakened) power stroke on the same cylinder cannot also
knock. A batch configuration with `fault_sets` (core_v2) lists each set of
faults (empty = normal) and `runs_per_engine`; `challenge.fault_sets` lists
one run per set on each test engine. A batch may set its own `master_seed`.
Single-fault scenarios draw exactly the same random numbers as before, so
core_v1 and the integration batch regenerate bit-identically.

## Units and context

| Signal | Unit | Nominal rate | Initial inference use |
|---|---|---:|---|
| Audio | normalized sensor units, not SPL | 24,000 Hz | Future waveform features |
| Vibration | nominal simulated m/s^2, not physically calibrated | 25,600 Hz | Future waveform features |

Telemetry units and rates are listed under Telemetry.

## Label and partition rules

- Canonical classes: `normal`, `misfire`, `knock_like`, `valve_clearance_like`.
- Assign virtual engines to train/tuning/calibration/test before window creation.
- All variants of an engine remain in one partition.
- Keep clean classification windows separate from transition/ambiguous windows.
- Do not infer normal labels from the absence of a single intermittent event.
- Never supply seed, run/engine ID, severity, fault schedule, or latent crank
  truth as a classifier feature.

## Features

Ordered model input names are `engine_diagnostics.pipeline.features.feature_names()`
(`FEATURE_VERSION` 0.1.0): 14 per modality (`audio_*`, `vibration_*`) plus
`ctx_rpm`, `ctx_load`, `ctx_throttle`. Each decision time `t` uses waveform
samples in `[t-2, t)` and the latest valid telemetry strictly before `t`. A
window's quality is `ok`, `clipped`, `missing_waveform`, or `stale_telemetry`.
The labeled table (`build-features`) adds bookkeeping columns `run_id`,
`engine_id`, `split`, `subset`, `t_end_s`, `label`, `quality`, and per-fault
labels `y_misfire`, `y_knock_like`, `y_valve_clearance_like`; these are never
model inputs. A window is labeled only when it lies inside one truth segment;
otherwise it is `transition` and excluded from training. A per-fault label is
1 when the window lies inside that fault's active interval (more than 0.5 s
after a change), 0 when it is more than 0.5 s clear of every interval, and NaN
otherwise; NaN windows are ignored for that fault only.

## Temporary model

`engine-diagnostics train-temporary` writes `artifacts/models/temporary.joblib`
(model version `temporary-0.2.0`, feature version, ordered feature names,
training runs) and `reports/generated/temporary_model_report.json`. Each window
decision has `predicted` (SVM class), `margin`, `anomaly_score`, `anomalous`,
and `accepted` (the class, or `uncertain` when the SVM and the normal baseline
disagree or the margin is small).

## Multi-fault model (v2)

`engine-diagnostics train-multi` writes `artifacts/models/multifault_v2.joblib`
(model version `v2.0.0`) and `reports/generated/training_v2.json`. Each window
decision has `probabilities` (calibrated P(present) per fault),
`fault_decisions` (`fault`, `uncertain`, or `normal` per fault), `detected`,
`uncertain_faults`, `anomaly_score`, `anomalous`, and `accepted` (the detected
set joined with `+`, else `uncertain` if a fault is borderline or the window
deviates from the normal baseline, else `normal`). Replay records add `alerts`,
the list of active per-fault alerts, and `alert`, that list joined with `+`.
`engine-diagnostics evaluate-multi` scores the frozen model once on the core_v2
test engines and writes `reports/generated/final_evaluation_v2.json`.

## Prediction records

`engine-diagnostics replay` writes `artifacts/logs/<run_id>_predictions.csv`
with one row per decision: `t_end_s`, `quality`, `predicted`, `margin`,
`anomaly_score`, `anomalous`, `accepted`, `alert`, `processing_ms`,
`model_version`, `feature_version`. Replay ingests `runtime.packet_s` packets
and decides at `window_ends()`; it currently requires `t0_s = 0` for all streams.
Truth comparison (`--with-truth`) happens after replay and is printed under
`[evaluation]`.

## Integration batch checks

Verify MATLAB/Python array orientation, values, dtype, sampling rates, timestamps,
duration, units, and event consistency on the small integration batch before
creating the full collection.
