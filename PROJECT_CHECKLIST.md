# Honeywell Engine Diagnostics — Project Checklist

Execution tracker. Checked items have been completed; unchecked items remain future work. The current milestone is project initialization, not a working diagnostic prototype.

**Recommended stack:** MATLAB for synthetic scenario generation and reference analysis; Python/SciPy/scikit-learn for the shared analysis and inference pipeline; Streamlit/Plotly for the dashboard. Use file exports between MATLAB and Python so the demonstration does not require a running MATLAB session.

**MVP:** normal, intermittent misfire, knock-like combustion bursts, and valve-clearance-like impacts; synthetic audio, one vibration axis, and numerical telemetry; normal-baseline detection plus one SVM classifier; an evidence dashboard with recorded replay and a simulated degradation trend.

**MATLAB assumption:** MATLAB is available. Check Simulink and individual toolboxes separately. A MATLAB event-and-response generator is the default; building a complete Simulink engine model is optional.

## 1. Scope, ownership, and environment

**Owner:** technical lead. **Dependency:** none. **Timing:** first 5% of project effort.

- [ ] Assign owners for MATLAB/data generation, signal processing, ML/evaluation, and dashboard/integration; combine roles if needed.
- [ ] Record the actual deadline, available effort, demo machine, and storage budget.
- [ ] Confirm the four MVP conditions and record excluded features: real acquisition, current sensing, CNNs, combined-fault classification, and remaining-useful-life prediction.
- [ ] Record assumptions: four-cylinder four-stroke engine; 800–3,500 RPM; normalized load 0.2–0.8; warm operation.
- [ ] Confirm organizer expectations for team-generated synthetic data, MATLAB simulation, and recorded-data replay; track unanswered questions without blocking independent work.
- [x] Record MATLAB release and installed products using `ver`; verify required functions can actually run under the available licenses. See `docs/TESTING.md`.
- [x] Check Signal Processing Toolbox and Simulink availability; check Powertrain Blockset only if considering an existing powertrain example. Signal installed; Simulink licensed, not installed.
- [ ] Establish a pinned Python environment with NumPy, SciPy, pandas, scikit-learn, Streamlit, Plotly, PyArrow, and pytest. The configuration/test environment is installed; the full stack is locked and enabled with `uv sync --locked --all-extras --group quality`.
- [x] Choose one authoritative configuration format, preferably JSON readable by both MATLAB and Python. See `configs/mvp.json` and its schema.
- [x] Create an assumption/decision log and a deferred-feature backlog. See `docs/DECISIONS.md` and Section 10.

**Artifacts:** scope document, role assignments, environment specifications, MATLAB product inventory, initial configuration.

**Exit check:** scope and ownership are agreed. Missing optional toolboxes have a documented Python/base-MATLAB fallback rather than blocking the project.

## 2. Data contracts and MATLAB–Python handoff

**Owners:** MATLAB and integration leads. **Dependency:** Phase 1. **Timing:** before substantive generator work.

- [x] Define one run bundle: waveform file, telemetry table, public metadata, and separate private simulation truth. See `docs/DATA_CONTRACT.md`.
- [x] Freeze audio at 24 kHz and single-axis vibration at 25.6 kHz; use 45-second core runs. See `configs/mvp.json`.
- [x] Define acoustic units as normalized sensor units and vibration as nominal simulated m/s²; document that neither is calibrated against a real engine.
- [x] Define telemetry fields and rates: RPM 50 Hz; throttle/load 10 Hz; coolant 1 Hz; short-/long-term fuel trims 2 Hz.
- [x] Define elapsed-time origin, sample indexing, start offsets, clock corrections, missing-data flags, and field freshness limits. See `docs/DATA_CONTRACT.md`.
- [x] Export waveform arrays as numeric `single` arrays in per-run MAT files explicitly saved as Version 7 (`-v7`).
- [x] Export telemetry as CSV and metadata as JSON; Python can convert telemetry to Parquet after validation.
- [x] Keep fault schedules, severity, source-event times, affected cylinders, and seeds in evaluation-only files. Specified in `docs/DATA_CONTRACT.md`; enforced when the generator exists.
- [x] Define run IDs, engine IDs, configuration versions, schema versions, and file checksums. See `docs/DATA_CONTRACT.md`.
- [ ] Define ordered model feature names and prediction-record fields, including window time, status, score, alert, and model version.
- [x] Plan a round-trip check for shape, dtype, sample rate, timestamps, and numeric values after Python import. Uses `waveform_summary` in metadata.
- [x] Document orientation explicitly: one sample dimension per channel; convert MATLAB column vectors consistently in Python. See `docs/DATA_CONTRACT.md`.

**Compatibility note:** MATLAB supports explicit MAT-file versions. SciPy's `loadmat` supports versions through 7.2 but does not implement the HDF5-based 7.3 interface. Per-run Version 7 numeric arrays avoid adding a separate HDF5 reader. See [MATLAB save documentation](https://www.mathworks.com/help/matlab/ref/save.html) and [SciPy loadmat documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html).

**Artifacts:** data dictionary, schemas, handoff specification, configuration template.

**Exit check:** both sides agree on the contract before bulk generation. No MATLAB-specific table or object serialization is required by Python.

## 3. First end-to-end milestone

**Owners:** MATLAB, ML, and dashboard leads. **Dependency:** Phase 2. **Timing:** achieve within the first 20% of elapsed project time.

- [x] Create a small integration batch: approximately 12 runs of 20 seconds, covering normal operation and simple examples of each modeled fault. Expanded to 16 runs on 4 engines so each training engine has normal and every fault class; engine_004 held out.
- [x] Use a shared event schedule to keep audio, vibration, and telemetry synchronized.
- [x] Export from MATLAB and verify the round trip in Python. `engine-diagnostics validate-runs`.
- [x] Implement a minimal common Python feature path using RMS, envelope variation, and broad band energies. `pipeline/features.py`, version 0.1.0, 31 features.
- [x] Train a temporary baseline/classifier using separate runs; do not present early scores as final evaluation. `train-temporary`; engine_004 held out.
- [x] Replay one run at 1× speed using trailing two-second windows and 0.5-second updates. `engine-diagnostics replay <run> --speed 1`.
- [x] Display telemetry, two waveforms, one spectrum, and actual computed predictions. `dashboard/app.py`.
- [x] Confirm inference has no access to labels or hidden simulation variables. Tests check imports, feature names, and future-sample independence.
- [x] Measure the first processing-latency estimate. ~12 ms p50 / 13 ms p95 per update (fast replay), 24/34 ms at 1× on the macOS dev machine.

**Artifacts:** integration batch, temporary model, working replay dashboard, handoff check report.

**Exit check:** one complete normal-to-fault demonstration works. Improve fidelity after this path exists.

## 4. MATLAB synthetic-data generator

**Owner:** MATLAB/data-generation lead. **Dependency:** Phase 3 contracts proven. **Timing:** approximately 20% of effort.

- [x] Build the generator as reusable MATLAB functions driven by configuration, not an undocumented interactive workspace.
- [x] Generate RPM/load trajectories for steady operation, idle-like operation, gradual ramps, and load transitions.
- [x] Integrate RPM into crank angle and schedule combustion events across the 720-degree four-stroke cycle.
- [ ] Include normal combustion variability, shaft-order components, ordinary valve impacts, structural ringing, and colored noise.
- [ ] Render acoustic and vibration channels through different gains, delays, filters, and resonances; avoid scaled copies.
- [ ] Model misfire by weakening/suppressing selected combustion events and sharing their mechanical consequences across modalities.
- [ ] Model knock-like faults with short post-combustion damped bursts, initially within 5–9 kHz.
- [ ] Model valve-clearance-like faults by changing recurring impact timing, shape, and variability; avoid assuming amplitude must always increase.
- [ ] Generate slow coolant and fuel-trim behavior; do not assign a unique telemetry value to each fault class.
- [x] Add randomized fault onset, severity, intermittence, and selected recovery intervals.
- [x] Add one accelerated valve-like deterioration scenario; label its time scale as simulated, not service life.
- [ ] Add sensor gain/placement variation, resonance variation, background noise, unrelated impacts, drift, clipping, and missing-data cases.
- [ ] Add timing jitter, small clock offsets, and drift as explicit robustness settings.
- [x] Band-limit rendered impulses before downsampling; verify aliasing is controlled.
- [x] Use reproducible independent random streams for operating state, fault events, and each sensor's noise.
- [ ] Save resolved configuration, generator version, engine/run identifiers, and seed references for every run.
- [ ] Document which equations are physically motivated, which parameter mappings are heuristic, and which require measurements.

**MATLAB/Simulink cut line:** if a suitable licensed Simulink example is already available, time-box a timing/dynamics comparison. Do not delay the MVP to build a full engine model. Simulink output still needs a justified acoustic/vibration rendering model.

**Artifacts:** MATLAB generator functions, scenario configurations, seed manifest, generator model card.

**Exit check:** all four conditions can be rendered reproducibly across varied operating conditions without class-specific metadata shortcuts.

## 5. Generator validation and dataset freeze

**Owners:** signal-processing and data leads. **Dependency:** Phase 4. **Timing:** approximately 15% of effort, including generation.

- [ ] Check array lengths, units, duration, timestamps, finite values, and expected sample rates.
- [x] Verify normal event/firing frequency at several fixed RPM values.
- [ ] Verify consistent event timing across sensor streams and correct clock correction.
- [ ] Verify intended fault interventions using private source-event truth.
- [ ] Check bandwidth, clipping, filter transients, and discontinuities at onset/recovery.
- [x] Verify repeated seeds reproduce outputs within the pinned MATLAB environment.
- [ ] Inspect parameter sweeps: severity, RPM, load, gain, noise, and resonance changes.
- [x] Match gain and operating distributions across classes; include healthy high-load and faulty low-load runs.
- [ ] Ensure normal signals contain structural energy in fault-analysis bands; avoid unique fault frequencies.
- [x] Freeze engine-level splits before creating overlapping windows: 36 training, 8 tuning, 8 calibration, and 8 final-test engines. Scaled to 18/4/4/4.
- [x] Generate the core target of 60 engines × 8 runs × 45 seconds; use two runs per class per engine. Scaled by decision to 30 engines × 8 runs × 30 s.
- [ ] Reserve operating/noise/generation settings for 32 additional challenge runs on final-test engines.
- [ ] Reserve approximately 8–10 GB working storage; core float32 waveform payload is approximately 4.29 GB before MAT compression and metadata.
- [x] Validate every run; quarantine failures with reasons rather than silently skipping them.
- [ ] Record labels for normal, observable fault, transition/ambiguous windows, and event intervals.
- [ ] Lock dataset checksums and final-test manifests before model tuning.

**Scope fallback:** reduce the number of engines proportionally if necessary, preserving independent training/tuning/calibration/test groups and coverage. Do not replace group separation with random window splitting.

**Artifacts:** validated dataset, split manifest, dataset summary, generator validation report, exclusion log.

**Exit check:** no engine overlaps partitions; retained runs pass checks; private truth is excluded from model inputs.

## 6. Signal processing and signatures

**Owner:** signal-processing lead. **Dependency:** contracts and validated development runs. **Timing:** approximately 10% of effort; can overlap Phase 5.

- [ ] Establish one authoritative Python feature implementation for training and replay; use MATLAB for reference checks.
- [ ] Implement continuous causal filtering with state; do not use future samples or whole-run zero-phase filtering for replay features.
- [ ] Start with acoustic 20 Hz–10 kHz and vibration 5 Hz–10 kHz analysis bands; verify responses.
- [ ] Add a high-band envelope branch for impulsive behavior and retain amplitude-sensitive features.
- [ ] Implement trailing two-second windows with 0.5-second steps.
- [ ] Use Hann full-window FFTs, Welch PSD with 8,192 samples/50% overlap, and STFT with 2,048 samples/hop 512.
- [ ] Verify frequency axes, PSD units, band integration, and window scaling against known signals.
- [ ] Extract RMS, crest factor, kurtosis, envelope variability, burst density, band energies/ratios, and spectral summaries.
- [ ] Add broad RPM-normalized order features using observed RPM; never use hidden crank-angle truth.
- [ ] Combine per-modality features with observed RPM/load/throttle context; keep slow telemetry primarily for display initially.
- [ ] Add quality rules for clipping, missing waveforms, stale telemetry, and unsupported transients.
- [x] Verify MATLAB/Python reference agreement with explicit numerical tolerances and matched settings.
- [ ] Verify offline and packet-replay features agree after identical causal processing.
- [ ] Freeze feature definitions, order, units, and version.

**Artifacts:** feature dictionary, analytical verification results, extraction pipeline, reference plots.

**Exit check:** features preserve intended signatures and operating context; training and inference consume identical definitions.

## 7. Baseline, SVM, and evaluation

**Owner:** ML/evaluation lead. **Dependency:** frozen features and development partitions. **Timing:** approximately 15% of effort.

- [x] Fit robust normal reference statistics conditioned on RPM/load; tune its anomaly threshold on tuning runs.
- [x] Train one RBF SVM for the four classes with training-only preprocessing.
- [x] Thin training windows and cap the initial training set at approximately 6,000–8,000 representative samples.
- [x] Use a small documented hyperparameter search; keep final-test data untouched.
- [x] Compare telemetry-only, audio-plus-context, vibration-plus-context, and fused models using identical partitions.
- [x] Audit suspiciously strong telemetry-only or amplitude-only results for generator shortcuts.
- [x] Calibrate probabilities on independent calibration engines if supported; otherwise display labeled decision scores.
- [x] Implement data-quality, unsupported-operation, and uncertain-prediction states. Unsupported operation is not separate from quality/uncertain.
- [x] Start alert persistence at two consistent accepted fault updates; clear after three accepted normal updates; tune on development runs.
- [x] Freeze model, preprocessing, calibration, thresholds, and persistence before final evaluation.
- [x] Report macro-F1, balanced accuracy, per-class precision/recall, support, and confusion matrices.
- [x] Report rejected-data coverage and unresolved predictions alongside accepted-only results.
- [x] Measure false alert episodes per normal minute and event recall, misses, and detection latency during actual replay.
- [ ] Evaluate unseen engines/seeds, reserved RPM ranges, noise, sensor responses, timing failures, and mild/intermittent faults.
- [x] Evaluate the degradation indicator on held-out trajectories without using severity truth as an input.
- [ ] Save failure examples and document that synthetic tests do not establish real-engine accuracy.

**Proposed goals, not promised results:** core macro-F1 ≥0.80; each fault recall ≥0.75; challenge macro-F1 ≥0.65; ≤0.2 false alerts per normal minute; p95 processing below 250 ms; p95 alert latency ≤4 seconds for predefined sustained moderate/severe cases. Report intermittent cases separately.

**Artifacts:** model bundle, ablation report, confusion matrices, latency logs, model card, final evaluation report.

**Exit check:** achieved and missed targets are recorded honestly. Any test-informed redesign requires fresh evaluation data.

## 8. Dashboard and complete demonstration

**Owner:** dashboard/integration lead. **Dependency:** Phase 3 shell; final integration depends on Phase 7. **Timing:** approximately 10% of effort; build in parallel where possible.

- [x] Show a permanent “SYNTHETIC DATA — RECORDED REPLAY” banner and run/model versions.
- [x] Display RPM, load, throttle, coolant, fuel trims, and freshness/quality status.
- [x] Display raw audio and vibration with a short-time zoom.
- [x] Display spectra/PSD, a spectrogram, and an impulse envelope.
- [x] Show predicted condition, appropriately labeled confidence/score, and alert status.
- [x] Explain two or three feature deviations against a matched normal reference; avoid claiming a peak proves a fault. Reference is training normal windows, not yet RPM/load-matched.
- [x] Add prediction/fault timeline and a separate optional truth overlay.
- [x] Show the accelerated simulated degradation indicator without a remaining-life claim.
- [x] Add scenario selection, play/pause, restart, and seek with buffer/filter reset and warm-up.
- [x] Confirm current predictions never consume future replay samples. `tests/test_replay.py`.
- [x] Decimate display traces while preserving full-rate analytical processing.
- [x] Measure inference and display latency on the declared demo machine.
- [x] Optional engine profiles with per-engine history: make/model/serial with catalog or hand-entered specs (validated; warning for engines that are not 4-cylinder 4-stroke), saved runs, results, notes, and raw data pruned beyond the newest 10 runs. Off by default; see `docs/DATA_CONTRACT.md`, "Engine history".
- [ ] Rehearse normal → RPM/load change → fault onset → signature change → alert → evidence.

**Artifacts:** integrated local dashboard, demo presets, prediction logs, performance report.

**Exit check:** the entire scenario runs at 1× without manual intervention; invalid or uncertain inputs have visible states.

## 9. Documentation, rehearsal, and submission

**Owners:** technical lead and presentation owner. **Dependency:** integrated MVP. **Timing:** protect the final 20% of elapsed time for verification, documentation, and rehearsal; this overlaps the effort allocations above.

- [ ] Freeze features when the final 20% of elapsed time begins; allow correctness and demo-stability fixes only.
- [ ] Document MATLAB generation/export and Python setup/training/evaluation/dashboard commands.
- [ ] Ensure replay works from exported files without MATLAB running; verify this explicitly.
- [ ] Include architecture diagram, data dictionary, generator assumptions, model card, and limitations.
- [ ] Record MATLAB release/toolbox requirements and pinned Python dependencies.
- [ ] Package a small validated sample dataset and model, plus full regeneration instructions.
- [ ] Prepare result slides with denominators, test conditions, and clearly labeled synthetic metrics.
- [ ] Prepare screenshots, static analysis figures, and a complete fallback video.
- [ ] Complete three consecutive offline rehearsals on the demo machine.
- [ ] Check actual organizer rules, naming, file-size limits, and submission requirements.
- [ ] Verify every claimed capability has a corresponding working artifact or measured result.
- [ ] Submit the required artifacts and retain the exact submitted configuration/model versions.

**Exit check:** another teammate can run the demo from the README; a recording is clearly labeled if used as fallback.

## 10. Critical path, optional work, and first actions

**Critical path:** contracts → MATLAB export round trip → early end-to-end replay → generator validation → dataset freeze → frozen features/model → final evaluation/integration → rehearsal.

**Parallel work:** dashboard shell after contracts; DSP reference checks before bulk generation; documentation throughout; independent scenario/configuration review while the main pipeline is built.

Optional work starts only after the integrated MVP passes its exit checks:

- [ ] Compare a licensed Simulink engine-timing/misfire example with the generator.
- [ ] Add three-axis vibration and re-run modality/placement evaluation.
- [ ] Improve order tracking using a suitable observed speed/tachometer signal.
- [ ] Evaluate verified, appropriately licensed real-engine recordings.
- [ ] Compare a small CNN against the frozen SVM baseline.
- [ ] Explore combined faults as an explicitly separate evaluation task.

**First five actions:** assign owners; inventory MATLAB products; freeze the data contract; create the small MATLAB integration batch; prove Python import and one complete replay path.

**Responsible demo claim:** the system recognizes specified simulated patterns under documented tests and shows their evidence. Real-engine diagnostic accuracy, calibrated physical wear, and remaining useful life remain unvalidated.
