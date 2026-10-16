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

`configs/demo_presets.json` holds thirteen presets on held-out test engines. Each
selects a run, replays it to its start time, and shows a presenter note taken
from evaluation truth:

- 1-7 (core_v1): normal with a load step; knock then a load change; misfire;
  misfire with recovery; accelerated valve degradation; a waveform gap; and a
  missed mild-knock challenge run (a stated limitation).
- 8-10 (core_v2): simultaneous faults.
- 11-13 (core_v3, crank wheel and cylinder temperatures): misfire against the
  ECU's own monitor; knock confirmed by a hot cylinder head; three faults seen
  three independent ways.

## Panels

**Crank speed** and **Exhaust temperatures** tabs (core_v3, after
`engine-diagnostics calibrate-crosscheck`):

- Crank speed tab:
  - Crank speed per tooth and per 90° stroke, with the misfire strokes marked.
  - Per-cylinder roughness against the crank detector's threshold.
  - The simulated OBD-II monitor: type A/B block progress, per-cylinder counts
    and codes, next to this replay's acoustic alert time.
- Exhaust temperatures tab: a thermal-camera-style view of the eight EGT/CHT
  readings on an engine outline (not a camera image), and their trends.
- Alert banners add "confirmed independently by crank speed / EGT/CHT" when
  those paths agree.
- The 3D engine's exhaust runners glow with measured EGT on the same inferno
  scale.

For `core_v2` the v2 model reports each fault separately: one calibrated
probability per fault, a "Detected this window" count, one alert banner per
active fault (each confirmed and cleared on its own), and a per-fault decision
timeline (dot detected, x uncertain, square active alert). The truth overlay
shades each fault's own active intervals.

The **Decision** tile, the diagnosis card title and the 3D view all name the
same faults as the alert banners: every confirmed alert plus every fault
detected in the current window (`explain.reported_faults`). Alerts are
debounced (confirmed over several updates, held through brief dropouts), so a
fault can stay alerted while it is missing from one 2 s window, or be detected
before its alert is confirmed; a grey note under the tile says which ("Knock-like
held (not in this window)", "Misfire awaiting confirmation"). "Best guess" and
"Detected this window" stay per window.


- Permanent "SYNTHETIC DATA — RECORDED REPLAY" banner; model/feature versions
- 3D view (Three.js): a neon wireframe car that flies into a detailed inline-4
  when playback starts or an alert is raised (Car / Engine / Reset view buttons,
  drag or arrow keys to orbit). Pistons, rods, crankshaft, camshafts and valves
  follow slider-crank kinematics at the run's RPM, shown slowed by 1/25, firing
  in the configured order. Effects follow the model only: every reported fault
  (alerts and this window's detections), each scaled by the model's confidence
  for it; shake follows the window's vibration RMS relative to the normal
  reference. When the cylinder localizer
  (`artifacts/models/cylinder_localizer.joblib`) names a fault's cylinder, that
  fault's effect is confined to it, its piston glows red and a ring in the
  fault's color marks it, so two faults on two cylinders show side by side;
  a fault without a cylinder (or without the localizer) shows engine-wide.
  Faults on one cylinder compose (misfire weakens or skips the firing, knock
  adds sparks, valve-train impacts add seat ticks). The HUD names each fault
  with its own confidence, status and cylinder. The evaluation truth cylinder
  appears only with the truth overlay, as a separate amber ring labelled as
  truth. The scene is created once and updated with a small state dict on each
  rerun. Pause freezes the engine; when the replay reaches its end the engine
  keeps running on the final decision (labelled as such in the HUD) so the
  results can be studied with the animation on; this runs in the browser only,
  with no server reruns. Reduced motion shows a steady state glow; without WebGL
  a 2D schematic replaces it
- WebGL budget: browsers keep only a few live WebGL contexts per page and drop
  the oldest when more are created. Plotly's WebGL traces (`Scattergl`) each
  open several and were re-created on every 0.5 s update, so the long-lived 3D
  view was the one dropped ("3D view paused…"). All charts are therefore SVG
  (`go.Scatter`; traces thinned to 2,000 points by keeping each bin's min and
  max, so short knock and impact spikes survive), and a test fails if a WebGL
  trace returns. If the 3D context is still lost (GPU reset), the view waits for
  the browser to restore it, then rebuilds its renderer with the same scene,
  showing a notice only if the loss lasts
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
- Last 2 s of audio and vibration (min/max-thinned to 2,000 points) and the 5-10 kHz
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
- Cylinder location per active alert (`runtime/localization.py`, needs
  `train-localizer`): the alert banner names the cylinder ("Misfire detected in
  cylinder 3"), and the evidence panel shows its confidence, how many decision
  windows support it, and the accumulated probability of each cylinder. Runs
  without the cam phase reference show a firing slot (A-D) instead, labelled as
  such; until enough windows agree the panel says it is still locating.

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

## Engine profiles and history

Optional. The sidebar's **Engine profile** section defaults to *None (history
off)*: nothing is saved and the dashboard behaves as without the feature.

1. Choose **+ New engine profile**, pick a make and model from
   `configs/engine_catalog.json` (specs fill in and can be edited) or choose
   *Other* and enter the specs by hand, then enter a serial/ID and an optional
   nickname. Specs are validated as you type; engines that are not inline
   4-cylinder 4-stroke can be saved but show a warning that diagnostic results
   may not be reliable.
2. With the profile selected, replay any run. When the replay reaches the end
   it is saved once to that engine's history; **Save run so far** saves a
   partial replay. Restart, seek, or a new run starts a new analysis.
3. The **Engine history** panel below the live view lists the engine's runs
   (newest first: source, analysed time, predicted fault, alerts, max
   P(fault), peak anomaly score, final health stage, whether raw data is still
   kept), the per-window decisions of a chosen run, and notes. Add a note about
   the engine or one run (e.g. "replaced injector 3").

History lives in `data/engine_history/` (`ENGINE_HISTORY_DIR`); raw signals are
kept for the newest 10 runs per engine (`ENGINE_HISTORY_RAW_KEEP`), results and
notes permanently. Format and leakage rules: `docs/DATA_CONTRACT.md`, "Engine
history". Profile specs are stored metadata only: the analysis still uses
`configs/mvp.json`, and recorded replays are synthetic runs, not measurements of
the profiled engine.

## Possible causes (Diagnose) and the engine report

Shown only when an engine profile is selected, in the **Diagnosis and report**
section below the engine history.

**Possible causes.** **Diagnose** ranks likely causes of the engine's condition
and recommends next steps, using the engine's saved history (save the current
replay first to include it). The dashboard builds one compact context
(`engine_diagnostics.diagnosis_context`): profile and specs (with where they
came from and the non-4-cylinder warning), the newest saved run and up to seven
earlier ones (predicted fault, likelihoods, alerts, health stages, operating
point), cross-run trends, notes, and signal summaries computed locally from the
raw data history still keeps (per-cylinder crank-speed roughness, EGT/CHT
spread, fuel-trim drift, audio band energy). Raw waveforms, lab injection
settings and recording ids are never sent. Every item has an ID (`SPEC`, `R5`,
`N3`, `S5-crank`, `T-misfire`) that each cause must cite.

The context goes to Google Gemini (`gemini-3.8-flash`, free tier; model and
limits in `configs/diagnosis.json`) with structured JSON output. The answer is
validated: malformed output counts as a failure, and a cause citing IDs that do
not exist is removed (or keeps only its real IDs, with the rest shown as
ignored). With no key, no network, a timeout, a rate limit (429) or invalid
output, a short message explains why and a **basic** rule-based diagnosis in the
same format is shown instead (model alerts, likelihoods, trends, crank/thermal/
fuel-trim summaries and repair notes mapped to cheap-first checks). The panel
shows the summary, causes with confidence and named evidence, actions grouped
as stop now / soon / monitor, data gaps, and a decision-support disclaimer.
Each diagnosis is saved to the engine's history (inputs digest, output, model,
whether it was the basic version) and can be reopened from **Saved diagnoses**.

Demo engine: `uv run --all-extras python scripts/seed_demo_engine.py` creates
*Demo engine (DRIVEX Sim Synthetic 1.6 I4 · DEMO-058)* with four saved core_v3
runs of one test engine (two healthy, a cylinder 3 misfire, then the misfire
again with knock) and notes, including a spark-plug replacement that did not fix
it. It replays the runs through the dashboard, so they are saved exactly as a
user's replays (needs core_v3 and `calibrate-crosscheck`).

Key setup: create a free key at <https://aistudio.google.com/apikey>, then set
`GEMINI_API_KEY` in the environment or put `GEMINI_API_KEY=...` in a `.env`
file at the repository root (git-ignored; see `.env.example`).

**Report.** Turn on **Show report** for a preview of the engine report:
profile and specs, the latest run (summary, health timeline, likelihoods), the
latest saved diagnosis (or a prompt to run one; building a report never calls
Gemini), the run history table and trend across runs, notes, a generation time
and the disclaimer. **Export PDF** downloads the same report. Preview and PDF
draw from one `EngineReport` (`engine_diagnostics.report`); the PDF uses fpdf2
with matplotlib charts, so it needs no browser or system install.

## Checking the 3D view

With the dashboard running, `node scripts/check_engine3d.mjs [url]` (Node 22+,
Google Chrome) drives headless Chrome through two passes. The knock scenario
checks that the scene was created once and never disposed across reruns, kept
receiving updates, moved into the engine view, and animated. Multi-fault preset 9,
played to the end, checks that the Decision tile names both faults whenever two
alerts show, that the 3D view marks both localized cylinders, and that the engine
keeps running after the replay ends. Throughout, no WebGL context may be lost or
evicted and charts may create none.
