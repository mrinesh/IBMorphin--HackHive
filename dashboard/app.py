"""Local replay dashboard: recorded synthetic runs through the live inference path.

Run from the repository root:
    uv run --all-extras streamlit run dashboard/app.py

Predictions come from ReplayEngine (shared features, model, alerts) as packets
are replayed; nothing is precomputed. Display traces are decimated; analysis
uses full-rate samples. Display panels may show any data already received
(up to the current stream time). The optional truth overlay is evaluation data
and is never passed to the engine.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import spectrogram, welch

from engine_diagnostics.config import load_config
from engine_diagnostics.data.run_io import find_runs, load_run, validate_run
from engine_diagnostics.pipeline.features import HIGH_ENVELOPE_HZ, MODALITIES, _band_envelope
from engine_diagnostics.runtime.replay import ReplayEngine, packets

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "core_v1": {"raw": ROOT / "data/raw/core_v1", "truth": ROOT / "evaluation_truth/core_v1",
                "model": ROOT / "artifacts/models/diagnostic_v1.joblib",
                "splits": ROOT / "configs/core_v1.json"},
    "integration_v1": {"raw": ROOT / "data/raw", "truth": ROOT / "evaluation_truth",
                       "model": ROOT / "artifacts/models/temporary.joblib",
                       "splits": ROOT / "configs/integration_batch.json"},
}
PRESETS_PATH = ROOT / "configs/demo_presets.json"
DISPLAY_DECIMATION = 8
SPECTROGRAM_S = 10.0
TICK_S = 0.5
CONTEXT_SIGNALS = ("rpm", "load", "throttle")
SLOW_SIGNALS = ("coolant_temp", "short_term_fuel_trim", "long_term_fuel_trim")
CLASS_ORDER = ["normal", "misfire", "knock_like", "valve_clearance_like", "uncertain"]
VALVE = "valve_clearance_like"
DEGRADATION_WINDOW = 10

st.set_page_config(page_title="Engine replay", layout="wide")


@st.cache_resource
def load_config_cached():
    return load_config(ROOT / "configs/mvp.json", ROOT / "schemas/project-config.schema.json")


@st.cache_resource
def load_model(path: str):
    return joblib.load(path) if Path(path).is_file() else None


@st.cache_resource
def engine_split_map(path: str) -> dict:
    splits = json.loads(Path(path).read_text(encoding="utf-8"))["engine_splits"]
    return {engine: split for split, engines in splits.items() for engine in engines}


@st.cache_resource
def load_bundle(raw_dir: str, run_id: str):
    config = load_config_cached()
    bundle = load_run(Path(raw_dir) / run_id)
    issues = validate_run(bundle, config)
    return bundle, issues, list(packets(bundle, config["runtime"]["packet_s"]))


@st.cache_resource
def run_engines(raw_dir: str) -> dict[str, str]:
    """run_id -> engine_id from each run's small metadata file."""
    return {d.name: json.loads((d / "metadata.json").read_text(encoding="utf-8"))["engine_id"]
            for d in find_runs(Path(raw_dir))}


def available_datasets() -> list[str]:
    return [name for name, d in DATASETS.items()
            if d["raw"].is_dir() and find_runs(d["raw"]) and d["model"].is_file()]


def start_run(dataset: str, run_id: str) -> None:
    config = load_config_cached()
    model = load_model(str(DATASETS[dataset]["model"]))
    bundle, _, _ = load_bundle(str(DATASETS[dataset]["raw"]), run_id)
    st.session_state.update(dataset=dataset, run_id=run_id,
                            engine=ReplayEngine(model, config, bundle.metadata),
                            next_packet=0, records=[], stream_t=0.0, playing=False)


def advance_to(target_s: float) -> None:
    """Push every packet that has fully arrived by stream time target_s."""
    state = st.session_state
    _, _, run_packets = load_bundle(str(DATASETS[state.dataset]["raw"]), state.run_id)
    while (state.next_packet < len(run_packets)
           and run_packets[state.next_packet].t_end_s <= target_s + 1e-9):
        packet = run_packets[state.next_packet]
        state.records += state.engine.push(packet)
        state.stream_t = packet.t_end_s
        state.next_packet += 1


def seek(target_s: float) -> None:
    """Reset buffers and alerts, then re-process from the start (warm-up included)."""
    start_run(st.session_state.dataset, st.session_state.run_id)
    advance_to(target_s)


config = load_config_cached()
datasets = available_datasets()

st.markdown(
    "<div style='padding:0.6rem 1rem;border-radius:0.4rem;background:#8a1c1c;color:white;"
    "font-weight:700;letter-spacing:0.05em'>SYNTHETIC DATA — RECORDED REPLAY</div>",
    unsafe_allow_html=True,
)
if not datasets:
    st.error("Missing inputs. Generate runs in MATLAB, then run `engine-diagnostics "
             "build-features` and `engine-diagnostics train` (or `train-temporary`).")
    st.stop()

presets = []
if PRESETS_PATH.is_file():
    presets = [p for p in json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
               if p["dataset"] in datasets]

with st.sidebar:
    st.header("Replay")
    preset_names = ["(none)"] + [p["name"] for p in presets]
    preset_name = st.selectbox("Demo preset", preset_names)
    preset = next((p for p in presets if p["name"] == preset_name), None)
    if preset and st.session_state.get("applied_preset") != preset_name:
        start_run(preset["dataset"], preset["run_id"])
        if preset.get("start_s"):
            advance_to(preset["start_s"])
        st.session_state.applied_preset = preset_name
    if preset:
        st.info(preset.get("note", ""))

    dataset = st.selectbox("Dataset", datasets,
                           index=datasets.index(st.session_state.get("dataset", datasets[0]))
                           if st.session_state.get("dataset") in datasets else 0)
    paths = DATASETS[dataset]
    split_of = engine_split_map(str(paths["splits"]))
    engines = run_engines(str(paths["raw"]))
    run_ids = sorted(engines)
    current = st.session_state.get("run_id") if st.session_state.get("dataset") == dataset else None
    run_id = st.selectbox(
        "Run", run_ids, index=run_ids.index(current) if current in run_ids else 0,
        format_func=lambda r: f"{r} · {engines[r]} · {split_of.get(engines[r], '?')}")
    if st.session_state.get("run_id") != run_id or st.session_state.get("dataset") != dataset:
        start_run(dataset, run_id)
    model = load_model(str(paths["model"]))
    bundle, issues, run_packets = load_bundle(str(paths["raw"]), run_id)
    duration = bundle.metadata["duration_s"]
    speed = st.select_slider("Speed", options=[0.5, 1.0, 2.0, 4.0], value=1.0)
    play_col, restart_col = st.columns(2)
    if play_col.button("Pause" if st.session_state.playing else "Play", width="stretch"):
        st.session_state.playing = not st.session_state.playing
        st.session_state.anchor = time.monotonic() - st.session_state.stream_t / speed
    if restart_col.button("Restart", width="stretch"):
        start_run(dataset, run_id)
    target = st.slider("Seek (s)", 0.0, float(duration), 0.0, step=0.5)
    if st.button("Seek", width="stretch"):
        seek(target)
        st.session_state.anchor = time.monotonic() - st.session_state.stream_t / speed
    show_truth = st.checkbox("Show evaluation truth overlay", value=False,
                             help="Simulation truth for review only; never used by the model.")
    st.caption(f"Model {model.model_version} · features {model.feature_version} · "
               f"packets {config['runtime']['packet_s'] * 1000:.0f} ms · "
               f"window {config['analysis']['window_s']:g} s / "
               f"step {config['analysis']['step_s']:g} s")
    if issues:
        st.error("Run failed validation:\n" + "\n".join(issues))


def reference_stats(record):
    """Normal reference for this window: operating bin (v1) or global (temporary)."""
    if hasattr(model, "bin_stats"):
        ctx = record["context"]
        key = model.operating_bin(ctx["rpm"], ctx["load"])
        return model.bin_stats.get(key, model.global_stats), key in model.bin_stats
    return (model.baseline_median, model.baseline_scale), False


def evidence_table(record) -> tuple[pd.DataFrame, bool]:
    names = model.baseline_names
    values = np.array([record["features"][model.feature_names.index(n)] for n in names])
    (median, scale), matched = reference_stats(record)
    z = (values - median) / scale
    top = np.argsort(-np.abs(z))[:3]
    table = pd.DataFrame({"feature": [names[k] for k in top], "value": values[top].round(4),
                          "normal median": median[top].round(4), "robust z": z[top].round(1)})
    return table, matched


def freshness(signal: str, t_now: float) -> str:
    rows = bundle.telemetry[(bundle.telemetry["signal"] == signal)
                            & (bundle.telemetry["valid"] == 1) & (bundle.telemetry["t_s"] < t_now)]
    if rows.empty:
        return "no data"
    runtime = config["runtime"]
    limits = {"rpm": runtime["rpm_max_age_s"], "load": runtime["load_throttle_max_age_s"],
              "throttle": runtime["load_throttle_max_age_s"]}
    limit = limits.get(signal, 2 / config["streams"]["telemetry_rates_hz"][signal])
    age = t_now - rows["t_s"].iloc[-1]
    return f"{'fresh' if age <= limit else 'STALE'} ({age:.2f} s)"


@st.fragment(run_every=TICK_S)
def live_view():
    rendered = time.perf_counter()
    state = st.session_state
    if state.playing:
        advance_to(min((time.monotonic() - state.anchor) * speed, duration))
        if state.next_packet >= len(run_packets):
            state.playing = False
    t_now = state.stream_t
    records = state.records
    latest = records[-1] if records else None
    has_probability = bool(latest and "probability" in latest)

    cols = st.columns(7)
    cols[0].metric("Stream time", f"{t_now:.2f} s")
    cols[1].metric("Accepted", latest["accepted"] if latest else "warming up")
    cols[2].metric("Classifier", (latest["predicted"] or "-") if latest else "-")
    if has_probability:
        cols[3].metric("Calibrated confidence", f"{latest['probability']:.2f}",
                       help=f"Accepted only if ≥ {model.probability_min:.2f}; calibrated on "
                            "separate calibration engines (synthetic).")
    else:
        cols[3].metric("Decision margin", f"{latest['margin']:.2f}" if latest else "-")
    cols[4].metric("Anomaly score", f"{latest['anomaly_score']:.2f}" if latest else "-",
                   help=f"Threshold {model.anomaly_threshold:.2f}")
    cols[5].metric("Quality", latest["quality"] if latest else "-")
    cols[6].metric("Processing / display",
                   f"{latest['processing_ms']:.0f} / {state.get('render_ms', 0):.0f} ms"
                   if latest else "-")
    confirm = getattr(model, "confirm_updates", config["runtime"]["confirm_fault_updates"])
    if latest and latest["alert"]:
        st.error(f"ALERT: {latest['alert']} (confirmed by {confirm} consistent updates)")
    else:
        st.success("No active alert")

    engine = state.engine
    window_s = config["analysis"]["window_s"]
    left, right = st.columns(2)
    waves = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                          subplot_titles=[f"{m} (last {window_s:g} s, display decimated "
                                          f"{DISPLAY_DECIMATION}x)" for m in MODALITIES]
                          + ["vibration high-band envelope (5-10 kHz)"])
    psd = go.Figure()
    for row, m in enumerate(MODALITIES, start=1):
        fs = engine.rates[m]
        x = engine.buffers[m][-round(window_s * fs):]
        t = t_now - (x.size - np.arange(x.size)) / fs
        waves.add_trace(go.Scattergl(x=t[::DISPLAY_DECIMATION], y=x[::DISPLAY_DECIMATION],
                                     name=m, line={"width": 1}), row=row, col=1)
        if x.size >= config["analysis"]["welch_segment_samples"]:
            clean = np.nan_to_num(x.astype(np.float64))
            f, p = welch(clean, fs=fs, window="hann",
                         nperseg=config["analysis"]["welch_segment_samples"])
            psd.add_trace(go.Scatter(x=f / 1000, y=10 * np.log10(p + 1e-20), name=m))
            if m == "vibration":
                env = _band_envelope(clean, fs, HIGH_ENVELOPE_HZ)
                waves.add_trace(go.Scattergl(x=t[::DISPLAY_DECIMATION],
                                             y=env[::DISPLAY_DECIMATION], name="envelope",
                                             line={"width": 1}), row=3, col=1)
    waves.update_layout(height=460, showlegend=False, margin={"t": 40, "b": 20})
    psd.update_layout(height=460, title="PSD of current window (dB)", xaxis_title="kHz",
                      margin={"t": 40, "b": 20})
    left.plotly_chart(waves, width="stretch")
    right.plotly_chart(psd, width="stretch")

    # Spectrogram of received vibration (display only; STFT settings from config).
    fs = engine.rates["vibration"]
    received = bundle.vibration[max(0, round((t_now - SPECTROGRAM_S) * fs)):round(t_now * fs)]
    seg = config["analysis"]["stft_segment_samples"]
    if received.size >= seg:
        f, ts, sxx = spectrogram(np.nan_to_num(received.astype(np.float64)), fs=fs, window="hann",
                                 nperseg=seg, noverlap=seg - config["analysis"]["stft_hop_samples"])
        offset = t_now - received.size / fs
        spec = go.Figure(go.Heatmap(x=ts + offset, y=f / 1000, z=10 * np.log10(sxx + 1e-20),
                                    colorscale="Viridis", colorbar={"title": "dB"}))
        spec.update_layout(height=280, title=f"Vibration spectrogram (last {SPECTROGRAM_S:g} s)",
                           xaxis_title="s", yaxis_title="kHz", margin={"t": 40, "b": 20})
        st.plotly_chart(spec, width="stretch")

    signals = CONTEXT_SIGNALS + SLOW_SIGNALS
    titles = ["RPM", "Load (%)", "Throttle (%)", "Coolant (°C)", "Short-term trim (%)",
              "Long-term trim (%)"]
    tele = make_subplots(rows=2, cols=3, subplot_titles=[
        f"{title} · {freshness(s, t_now)}" for title, s in zip(titles, signals, strict=True)])
    received_tele = bundle.telemetry[(bundle.telemetry["t_s"] < t_now)
                                     & (bundle.telemetry["valid"] == 1)]
    for k, signal in enumerate(signals):
        rows = received_tele[received_tele["signal"] == signal]
        tele.add_trace(go.Scatter(x=rows["t_s"], y=rows["value"], name=signal),
                       row=k // 3 + 1, col=k % 3 + 1)
    tele.update_xaxes(range=[0, duration])
    tele.update_layout(height=380, showlegend=False, margin={"t": 50, "b": 20})
    st.plotly_chart(tele, width="stretch")

    timeline = go.Figure()
    if records:
        log = pd.DataFrame([{k: r[k] for k in ("t_end_s", "accepted", "alert")}
                            for r in records])
        timeline.add_trace(go.Scatter(x=log["t_end_s"], y=log["accepted"], mode="markers",
                                      name="accepted decision"))
        alerted = log[log["alert"].notna()]
        timeline.add_trace(go.Scatter(x=alerted["t_end_s"], y=alerted["alert"], mode="markers",
                                      marker={"symbol": "square", "size": 12, "opacity": 0.4},
                                      name="active alert"))
    if show_truth:
        from engine_diagnostics.pipeline.labels import load_truth  # evaluation only

        truth = load_truth(DATASETS[dataset]["truth"], run_id)
        intervals = truth.get("active_intervals") or []
        if intervals and not isinstance(intervals[0], list):
            intervals = [intervals]
        for a, b in intervals:
            timeline.add_vrect(x0=a, x1=b, fillcolor="orange", opacity=0.15, line_width=0)
        note = (f"truth: {truth['condition']}" if intervals else "truth: normal run")
        timeline.add_annotation(x=duration / 2, y=1.08, yref="paper", text=note, showarrow=False)
    timeline.update_layout(height=260, title="Decision timeline",
                           xaxis={"range": [0, duration], "title": "s"},
                           yaxis={"categoryorder": "array", "categoryarray": CLASS_ORDER},
                           margin={"t": 40, "b": 20})
    st.plotly_chart(timeline, width="stretch")

    lower_left, lower_right = st.columns(2)
    if latest and latest["quality"] == "ok":
        table, matched = evidence_table(latest)
        lower_left.subheader("Evidence: deviations from the normal reference")
        lower_left.dataframe(table, hide_index=True)
        lower_left.caption(
            ("Reference: training normal windows in the same RPM/load bin. " if matched else
             "Reference: all training normal windows. ")
            + "A deviation shows what changed; it does not prove a fault.")
    if has_probability:
        valve = pd.Series([r.get("probabilities", {}).get(VALVE, np.nan) for r in records])
        indicator = valve.rolling(DEGRADATION_WINDOW, min_periods=1).mean()
        trend = go.Figure(go.Scatter(x=[r["t_end_s"] for r in records], y=indicator))
        trend.update_layout(height=260, title="Simulated valve-wear indicator",
                            xaxis={"range": [0, duration], "title": "s"},
                            yaxis={"range": [0, 1], "title": "rolling P(valve)"},
                            margin={"t": 40, "b": 20})
        lower_right.plotly_chart(trend, width="stretch")
        lower_right.caption("5 s rolling mean of the calibrated valve-clearance probability. "
                            "Accelerated simulation; not a remaining-life estimate.")
    state.render_ms = 1000 * (time.perf_counter() - rendered)


live_view()
