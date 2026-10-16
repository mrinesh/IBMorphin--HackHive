# Replay dashboard

```powershell
uv run --all-extras streamlit run dashboard/app.py
```

`.streamlit/config.toml` keeps the server on `localhost`, disables usage
statistics, sets the dark soft-UI theme, and turns on static serving for
`dashboard/static` (bundled Space Grotesk, Inter and JetBrains Mono fonts, OFL; and
Three.js 0.185.1, MIT), so the dashboard needs no internet connection. Restart
the server after editing `theme.py`, `explain.py` or `engine3d.py`: a running
server keeps the modules it already imported. The app lists every dataset whose
runs and model exist:

| Dataset | Runs | Model |
|---|---|---|
| `core_v1` (final) | `data/raw/core_v1` | `artifacts/models/diagnostic_v1.joblib` (`engine-diagnostics train`) |
| `core_v2` (multi-fault) | `data/raw/core_v2` | `artifacts/models/multifault_v2.joblib` (`engine-diagnostics train-multi`) |
| `integration_v1` | `data/raw` | `artifacts/models/temporary.joblib` (`engine-diagnostics train-temporary`) |

## Files

| File | Role |
|---|---|
| `app.py` | Layout, replay loop (unchanged `ReplayEngine` path), charts |
| `theme.py`, `hud.css` | Design tokens, soft-UI styles, Plotly template, HTML cards |
| `explain.py` | Diagnosis card text: uncertain reasons, evidence, evidence strength |
| `engine3d.py`, `static/engine3d/engine3d.js` | Three.js engine and car view (components v2) |

## Demo presets and scenario cards

The sidebar scenario cards (Normal, Knock, Misfire, Valve-train impacts) select
the first preset tagged with that `scenario`. Abnormal vibration is represented
by the valve-train impact class the model was trained on; there is no general
abnormal-vibration classifier. The full list stays under "Demo preset".

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
- 3D view (Three.js): a neon wireframe car that flies into a detailed inline-4
  when playback starts or an alert is raised (Car / Engine / Reset view buttons,
  drag or arrow keys to orbit). Pistons, rods, crankshaft, camshafts and valves
  follow slider-crank kinematics at the run's RPM, shown slowed by 1/25, firing
  in the configured order. Effects follow the model only: the alert class or
  accepted fault, scaled by the model's confidence; shake follows the window's
  vibration RMS relative to the normal reference. The model does not localize
  faults, so effects are engine-wide; the faulty cylinder appears only with the
  evaluation-truth overlay, labelled as truth. The scene is created once and
  updated with a small state dict on each rerun. Reduced motion shows a steady
  state glow; without WebGL a 2D schematic replaces it
- Diagnosis card: decision with icon and text, why a window is uncertain,
  calibrated confidence or margin and anomaly score against their thresholds,
  evidence strength (how many of the model's own criteria hold; not a fault
  severity), the three largest deviations in plain language, and general
  background on the class, marked as not model output
- Telemetry cards: last received RPM, load, throttle, coolant and fuel trims
  with freshness and a 10 s sparkline
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

## Health timeline (deterioration demonstration)

Below the decision timeline, a live chart tracks, relative to the healthy baseline of
the same RPM/load range (`runtime/health.py`): firing-rhythm irregularity (sound),
high-frequency impulsiveness (vibration), and the model's overall anomaly score. Each
line is smoothed over the last 2 s of good windows (causal) and divided by its
developing threshold, so 1.0 is the boundary. A stage ribbon shows healthy →
developing anomaly (a line stays at or above 1 without a confirmed alert) →
persistent fault (the model's confirmed alert), plus no data for failed-quality
windows, and a text summary lists the transitions. Thresholds (z = 4 for both
features; the model's own anomaly threshold) were set on core_v1 tuning engines.
All progression is simulated, so the chart is labeled a deterioration
demonstration and makes no forecast or remaining-useful-life claim.

Early warning: with a calibrated model (v1, v2) a fourth line tracks the fault
likelihood, the probability of the most likely fault, against a warning bar of
0.6, below the alert's 0.7 acceptance bar. A fault the model is becoming
confident about therefore shows as developing before the alert is confirmed,
and an **Early warning** line reports how many seconds ahead it was. The bar was
chosen on tuning engines (0.55 doubled false warnings; 0.65 lost most valve lead
time). Best demo of the full progression: core_v1 `run_00232` (valve degradation,
developing 17.0 s → persistent fault 22.0 s) or `run_00215` (3.5 s ahead); demo
preset 5 (`run_00216`) still goes straight from healthy to fault.

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
server-side render 105 ms median / 157 ms max per 0.5 s tick with the redesigned panels
(AppTest; browser drawing not included).

## Checking the 3D view

With the dashboard running, `node scripts/check_engine3d.mjs` (Node 22+, Google
Chrome) plays the knock scenario in headless Chrome and checks that the scene was
created once and never disposed across reruns, kept receiving updates, moved into
the engine view, and animated.
