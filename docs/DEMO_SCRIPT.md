# Demo video script (about 5 minutes)

For the presenter recording the submission video. Every claim below is backed by
a measured result in [PROJECT_REPORT.md](../PROJECT_REPORT.md) §11; do not add
claims that are not.

## Before recording

1. Close MATLAB. The demo runs from exported files only.
2. Start the dashboard from the repository root:
   `uv run --all-extras streamlit run dashboard/app.py`
3. Open `http://localhost:8501`, set the browser to full screen, speed 1×.
4. Keep "Show evaluation truth overlay" **off** until the step that says to turn it on.
5. Have `PROJECT_REPORT.md` §1 (results table) open in a second tab.

## 0:00 – 0:40 · Problem and approach

Say:
- "Engines change sound and vibration with speed and load, so a louder signal alone isn't a fault."
- "We built a MATLAB simulator of a four-cylinder engine that produces synchronized microphone, accelerometer, and ECU-style telemetry for normal operation and three faults: misfire, knock-like bursts, and valve-clearance-like impacts."
- "Everything you'll see is synthetic and replayed from recorded files. The red banner says so."

Show: the banner and the sidebar.

## 0:40 – 1:20 · Normal operation (preset 1)

Select **"1. Normal operation with a load step"**, press **Play**.

Say:
- "Every half-second the system takes the last two seconds of audio and vibration, computes 31 features, and classifies them. The model was trained on 18 virtual engines; this run is from an engine it has never seen."
- Point at RPM/load: "At about 9 seconds the load steps up. Decisions stay normal, with an occasional 'uncertain' where the model is not confident, and no alert is raised."
- Point at the processing time: "Each decision takes about 12 milliseconds."

## 1:20 – 2:30 · Fault onset, alert, evidence (preset 2)

Select **"2. Knock-like bursts, then a load change"**, press **Play** (starts at 6 s).

Say, as it happens:
- Around 12–15 s: "Short high-frequency bursts appear after combustion. You can see them in the vibration envelope and as a band near 6–7 kHz in the spectrogram."
- When the red alert appears (~15 s): "Two consecutive confident knock decisions confirm an alert, about 2.5 seconds after the fault began."
- Point at the evidence table: "The dashboard shows why: high-band envelope statistics are far above the normal reference for this same speed and load range. That's evidence to inspect, not proof of a cause."
- Point at calibrated confidence: "Confidence is calibrated on separate engines."
- Turn on **truth overlay**: "Only now do we reveal the simulation's ground truth: the shaded region is where the fault was actually injected. The model never sees this."

## 2:30 – 3:10 · Misfire with recovery (preset 4)

Turn the truth overlay off. Select **"4. Misfire with a recovery interval"**, press **Play**.

Say:
- "This misfire comes and goes. The alert is raised, clears after three normal decisions during the recovery, and returns when the misfire does."
- Point at evidence: "The top evidence is irregular firing-peak strength: some power strokes are missing."

## 3:10 – 3:50 · Accelerated degradation (preset 5)

Select **"5. Valve clearance: accelerated degradation trend"**, press **Play** at 2× speed.

Say:
- "Here a simulated valve-clearance fault grows steadily. The wear indicator, a rolling average of the model's valve probability, rises with it. On held-out degradation runs it tracks the true simulated severity with a median rank correlation of 0.91."
- "This is an accelerated simulation, not a remaining-life prediction."

## 3:50 – 4:15 · Data quality (preset 6)

Select **"6. Data-quality state: waveform gap"**, press **Play**.

Say:
- "At 12.5 seconds a short audio dropout occurs. Windows that contain it are marked missing waveform; the system makes no prediction for them instead of guessing, and the alert state is held."

## 4:15 – 4:45 · Honest limits (preset 7)

Select **"7. Limitation: mild knock at 3,300 rpm (challenge)"**, press **Play**.

Say:
- "This is a challenge run: very high RPM, double the noise, and a mild fault. The model misses it. On these reserved conditions macro-F1 drops to 0.38, and knock is never detected."

## 4:45 – 5:15 · Results and close

Show the results table in PROJECT_REPORT.md §1.

Say:
- "On four held-out virtual engines: macro-F1 0.835; zero false alerts during normal replay; confirmed alerts for detected moderate-to-severe faults within 2.5 seconds at p95; 13 millisecond processing."
- "Misses: valve and knock recall are just below our 0.75 goal, mild faults are often missed, and the challenge conditions are not handled."
- "A model given only RPM, load and throttle scores at chance, so the classifier is using the sensor signatures, not the operating conditions."
- "Everything is reproducible from the repository: MATLAB generation, validation, training, evaluation, and this dashboard. The next step is real engine recordings."

## Claims to avoid

- Real-engine accuracy, physical wear, remaining useful life, or root-cause certainty.
- Calling the run "live" sensor data; it is recorded synthetic replay.
- Any number not in PROJECT_REPORT.md §11.
