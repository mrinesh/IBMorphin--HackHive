# Replay dashboard

```powershell
uv run --all-extras engine-diagnostics build-features
uv run --all-extras engine-diagnostics train-temporary
uv run --all-extras streamlit run dashboard/app.py
```

`.streamlit/config.toml` keeps the server on `localhost` and disables usage
statistics. The app needs exported runs in `data/raw` and the temporary model in
`artifacts/models/temporary.joblib`.

## External dataset check

The sidebar's **External dataset check** section accepts a CSV and displays its
columns, missing values, and `Engine Condition` counts. It compares the upload
with the linked notebook's reported schema. It does not run the waveform model
on tabular data: the notebook dataset has no audio/vibration signals and uses a
different label set.

## What it shows

- Permanent "SYNTHETIC DATA — RECORDED REPLAY" banner, model and feature versions
- Play/pause, restart, speed, and seek. Seek resets buffers and alert state and
  re-processes from the start, so the trailing window is warmed up honestly
- Current accepted decision, SVM class, anomaly score vs threshold, quality,
  processing time, and alert status
- Last 2 s of audio and vibration (display decimated 8×; analysis is full rate)
- Welch PSD of the current window
- RPM, load, and throttle telemetry received so far
- Decision timeline with active alerts; optional evaluation-truth overlay (off by
  default, loaded separately, never passed to the engine)
- Three largest robust deviations from the training normal reference

Predictions are computed live by `ReplayEngine` from replayed 40 ms packets.

## Not yet implemented

Spectrogram, impulse envelope, coolant/fuel-trim panels, freshness indicators,
simulated degradation trend, and measured display latency (checklist §8).
