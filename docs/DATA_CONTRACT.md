# Data contract, version 1.0

## Shared settings

`configs/mvp.json` is the authoritative configuration for both MATLAB and Python.
Its structural schema is `schemas/project-config.schema.json`. Validate using
`uv run engine-diagnostics validate-config` before generation. Numeric and timing
relationships are checked in Python as well as JSON structure.

## Planned run bundle

```text
data/raw/run_00001/
  waveforms.mat
  telemetry.csv
  metadata.json
evaluation_truth/run_00001_events.json
```

These paths describe future outputs, not generated examples. Only explicit
waveform fields and observed telemetry may reach inference. IDs are bookkeeping
for grouping/logging, not classifier inputs.

| File | Fields and rules |
|---|---|
| `waveforms.mat` | Version 7 (`-v7`); numeric `single` column vectors `audio` and `vibration`; scalar `audio_fs_hz`, `vibration_fs_hz`, `audio_t0_s`, `vibration_t0_s` |
| `telemetry.csv` | Long-form columns `t_s,signal,value,unit,valid`; timestamps are elapsed seconds; missing values are explicit and invalid |
| `metadata.json` | `schema_version`, `run_id`, `engine_id`, `duration_s`, `synthetic`, `generator_version`, sensor units and clock corrections |
| Private event JSON | Fault class, active intervals, actual event timestamps, affected cylinder, severity schedule, seed/configuration references |

Sample `i` in zero-based Python indexing occurs at `t0_s + i / fs_hz`. MATLAB
index `i+1` denotes that same sample. Two-second decision windows use `[t-2, t)`.
Retain native waveform rates; combine per-modality features and fresh telemetry.

## Units and context

| Signal | Unit | Nominal rate | Initial inference use |
|---|---|---:|---|
| Audio | normalized sensor units, not SPL | 24,000 Hz | Future waveform features |
| Vibration | nominal simulated m/s^2, not physically calibrated | 25,600 Hz | Future waveform features |
| RPM | rpm | 50 Hz | Operating context and approximate order features |
| Estimated load | % | 10 Hz | Operating context |
| Throttle | % | 10 Hz | Operating context |
| Coolant temperature | degrees C | 1 Hz | Display |
| Short-term fuel trim | % correction | 2 Hz | Display |
| Long-term fuel trim | % correction | 2 Hz | Display |

The engine configuration's load bounds are fractions (0-1); observed load
telemetry is percent (0-100). Make the conversion explicit in the future renderer.
Telemetry rates are simulation choices, not guaranteed vehicle polling rates.

## Label and partition rules

- Canonical classes: `normal`, `misfire`, `knock_like`, `valve_clearance_like`.
- Assign virtual engines to train/tuning/calibration/test before window creation.
- All variants of an engine remain in one partition.
- Keep clean classification windows separate from transition/ambiguous windows.
- Do not infer normal labels from the absence of a single intermittent event.
- Never supply seed, run/engine ID, severity, fault schedule, or latent crank
  truth as a classifier feature.

## Next milestone checks

Verify MATLAB/Python array orientation, values, dtype, sampling rates, timestamps,
duration, units, and event consistency on the small integration batch before
creating the full collection. The initializer validates configuration only;
run-file validation will be implemented with that milestone.
