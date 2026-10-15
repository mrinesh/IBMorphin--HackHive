# Replay dashboard

```powershell
uv run --all-extras streamlit run dashboard/app.py
```

`.streamlit/config.toml` keeps the server on `localhost` and disables usage
statistics. The app lists every dataset whose runs and model exist:

| Dataset | Runs | Model |
|---|---|---|
| `core_v1` (final) | `data/raw/core_v1` | `artifacts/models/diagnostic_v1.joblib` (`engine-diagnostics train`) |
| `core_v2` (multi-fault) | `data/raw/core_v2` | `artifacts/models/multifault_v2.joblib` (`engine-diagnostics train-multi`) |
| `integration_v1` | `data/raw` | `artifacts/models/temporary.joblib` (`engine-diagnostics train-temporary`) |

## Demo presets

`configs/demo_presets.json` holds seven presets on held-out test engines. Each
selects a run, replays it to its start time, and shows a presenter note taken
from evaluation truth: normal with a load step; knock then a load change;
misfire; misfire with recovery; accelerated valve degradation; a waveform gap;
and a missed mild-knock challenge run (a stated limitation).

## Panels

For `core_v2` the v2 model reports each fault separately: one calibrated
probability per fault, a "Faults detected" count, one alert banner per active
fault (each confirmed and cleared on its own), and a per-fault decision
timeline (dot detected, x uncertain, square active alert). The truth overlay
shades each fault's own active intervals.


- Permanent "SYNTHETIC DATA — RECORDED REPLAY" banner; model/feature versions
- Accepted decision, classifier class, calibrated confidence (v1) or decision
  margin (temporary), anomaly score vs threshold, quality, processing and
  server-side render time, alert status
- Last 2 s of audio and vibration (display decimated 8×) and the 5-10 kHz
  vibration impulse envelope; Welch PSD of the current window
- Vibration spectrogram of the last 10 s received (STFT 2,048/512 from config)
- RPM, load, throttle, coolant, and fuel trims with freshness status
- Decision timeline with active alerts; optional evaluation-truth overlay (off by
  default, loaded separately, never passed to the engine)
- Three largest robust deviations from the normal reference of the same RPM/load
  bin (v1)
- Evidence panel per active alert (`runtime/evidence.py`): a fault-specific claim
  for sound, vibration, and RPM (e.g. misfire: irregular firing rhythm, repeated
  abnormal pulses, accompanying RPM fluctuation), each rated supports / weak / not
  seen from robust z-scores of the model's own features, plus an overall assessment.
  The weak, missing, or burst firings and RPM dips are shaded in the panel's chart
  and, in the same colors, on the waveform and RPM charts. RPM is "not expected" for
  knock and valve faults. Valve-clearance evidence is usually weak, matching the model

## Fault-injection lab

Sidebar **Mode → Fault-injection lab** (needs core_v1 runs, their evaluation truth,
and the v1 model). Judges set, then press **Apply and replay**:

- Engine RPM: picks a clean healthy core_v1 recording (tuning, calibration, or test
  engine; never a classifier training engine) nearest the chosen RPM
- Injected fault: none, misfire, or knock-like bursts, with severity, onset, and
  cylinder. Injection (`src/engine_diagnostics/lab.py`) reuses the generator's source
  model and settings on the base run's own combustion schedule: misfire removes
  85-97% of the rendered combustion transient on the chosen cylinder and adds the
  RPM dip; knock adds damped 5.5-8.5 kHz bursts 10-20 degrees after combustion.
  Stroke probability and burst strength follow the generator's severity mapping.
  Valve-clearance injection is not offered (needs the full generator)
- Background noise: total sensor noise 1-5x the recording's (2x = challenge runs)
- Sensor availability: microphone, accelerometer, or RPM telemetry stop at a chosen
  time, producing `missing_waveform` or `stale_telemetry` (no prediction, alert held)
- Random seed for reproducible injections

Everything modified is labeled: a dashed "FAULT-INJECTION LAB — INJECTED / SIMULATED
SIGNALS" banner listing each change, "LAB: injected/modified" on waveform titles, and
the injected interval and sensor-failure time on the timeline (known to the lab,
hidden from the model). A **Lab response** box compares detection latency and
decisions with a full replay of the unmodified recording. The modified run goes
through the unchanged replay path (features, model, alerts, evidence panels).
Approximation: per-engine sensor gain/resonance variation is not reproduced, and a
misfire's RPM dip does not re-time later events.
- Simulated valve-wear indicator: 5 s rolling mean of the calibrated
  valve-clearance probability (accelerated simulation, not remaining life)
- "Hold out this run's engine" (integration dataset, training engines only): replays with a model
  refitted without that engine (`fit_holdout_model`, version suffix
  `-without-<engine>`), so a fault on an unseen engine can be watched live.
  Needs `data/processed/integration_v1_features.parquet`; refitting takes about a
  second and is cached per engine
- Cross-engine check: per-engine and pooled leave-one-engine-out macro-F1 and
  recall for fused and waveform-only features, plus the pooled fused confusion
  matrix, read from `reports/generated/temporary_model_report.json`

Play, pause, restart, speed, and seek are in the sidebar. Seek resets buffers and
alert state and re-processes from the start. Predictions are computed live by
`ReplayEngine` from 40 ms packets; display panels show only data already
received.

Measured on the macOS development machine: processing p95 13 ms per decision;
server-side render 112 ms median / 177 ms max per 0.5 s tick (AppTest; browser
drawing not included).
