"""Local replay dashboard: recorded synthetic runs through the live inference path.

Run from the repository root:
    uv run --all-extras streamlit run dashboard/app.py

Predictions come from ReplayEngine (shared features, temporary model, alerts) as
packets are replayed; nothing is precomputed. Display traces are decimated;
analysis uses full-rate samples. The optional truth overlay is evaluation data
and is never passed to the engine.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import welch

from engine_diagnostics.config import load_config
from engine_diagnostics.data.run_io import find_runs, load_run, validate_run
from engine_diagnostics.pipeline.features import MODALITIES
from engine_diagnostics.runtime.replay import ReplayEngine, packets

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw"
TRUTH_DIR = ROOT / "evaluation_truth"
MODEL_PATH = ROOT / "artifacts/models/temporary.joblib"
DISPLAY_DECIMATION = 8
TICK_S = 0.5
CONTEXT_SIGNALS = ("rpm", "load", "throttle")
CLASS_ORDER = ["normal", "misfire", "knock_like", "valve_clearance_like", "uncertain"]

st.set_page_config(page_title="Engine replay", layout="wide")


@st.cache_resource
def load_project():
    config = load_config(ROOT / "configs/mvp.json", ROOT / "schemas/project-config.schema.json")
    model = joblib.load(MODEL_PATH) if MODEL_PATH.is_file() else None
    return config, model


@st.cache_resource
def load_bundle(run_id: str):
    config, _ = load_project()
    bundle = load_run(RAW_DIR / run_id)
    issues = validate_run(bundle, config)
    return bundle, issues, list(packets(bundle, config["runtime"]["packet_s"]))


def start_run(run_id: str) -> None:
    config, model = load_project()
    bundle, _, _ = load_bundle(run_id)
    st.session_state.update(run_id=run_id, engine=ReplayEngine(model, config, bundle.metadata),
                            next_packet=0, records=[], stream_t=0.0, playing=False)


def advance_to(target_s: float) -> None:
    """Push every packet that has fully arrived by stream time target_s."""
    _, _, run_packets = load_bundle(st.session_state.run_id)
    engine = st.session_state.engine
    while (st.session_state.next_packet < len(run_packets)
           and run_packets[st.session_state.next_packet].t_end_s <= target_s + 1e-9):
        packet = run_packets[st.session_state.next_packet]
        st.session_state.records += engine.push(packet)
        st.session_state.stream_t = packet.t_end_s
        st.session_state.next_packet += 1


def seek(target_s: float) -> None:
    """Reset buffers and alerts, then re-process from the start (warm-up included)."""
    start_run(st.session_state.run_id)
    advance_to(target_s)


config, model = load_project()
run_ids = [p.name for p in find_runs(RAW_DIR)] if RAW_DIR.is_dir() else []

section = st.sidebar.radio("Section", ["Recorded replay", "External dataset check"])
if section == "External dataset check":
    st.title("External engine health dataset")
    st.markdown(
        "Review the [GitHub notebook](https://github.com/cozzolinoac11/"
        "automotive-vehicles-engine-health/blob/main/automotive_vehicles_engine_health.ipynb) "
        "dataset separately from the recorded waveform replay. Its notebook output reports "
        "19,535 rows with engine RPM, oil/fuel/coolant pressure, oil/coolant temperature, "
        "and an `Engine Condition` label. The CSV itself is not stored in that repository; "
        "the notebook loads it from Google Drive."
    )
    st.warning(
        "The current temporary model cannot run on these columns. It requires 33 features "
        "computed from synchronized audio and vibration windows plus RPM, load, and throttle. "
        "This table has no audio or vibration waveforms, and its binary health label does "
        "not map to the model's normal/misfire/knock/valve classes. No predictions are made here."
    )
    st.subheader("Inspect a CSV copy")
    uploaded = st.file_uploader("Upload the dataset CSV", type=["csv"], key="external_engine_csv")
    if uploaded is not None:
        try:
            external = pd.read_csv(uploaded)
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            st.error(f"Could not read this CSV: {error}")
        else:
            st.write(f"{len(external):,} rows · {len(external.columns)} columns")
            st.dataframe(external.head(20), hide_index=True, width="stretch")
            st.subheader("Dataset checks")
            checks = pd.DataFrame({
                "column": external.columns,
                "dtype": [str(dtype) for dtype in external.dtypes],
                "missing": [int(value) for value in external.isna().sum()],
            })
            st.dataframe(checks, hide_index=True, width="stretch")
            label = "Engine Condition"
            if label in external:
                st.caption("Label counts")
                st.dataframe(external[label].value_counts(dropna=False).rename("rows"),
                             width="stretch")
            expected = {
                "Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure",
                "lub oil temp", "Coolant temp", label,
            }
            missing = sorted(expected - set(external.columns))
            if missing:
                st.info("This CSV does not match the notebook's reported columns. Missing: "
                        + ", ".join(missing))
            else:
                st.success("The CSV matches the notebook's reported seven-column schema. "
                           "That confirms dataset identity, not compatibility with this model.")
    st.stop()

st.markdown(
    "<div style='padding:0.6rem 1rem;border-radius:0.4rem;background:#8a1c1c;color:white;"
    "font-weight:700;letter-spacing:0.05em'>SYNTHETIC DATA — RECORDED REPLAY</div>",
    unsafe_allow_html=True,
)
if model is None or not run_ids:
    st.error("Missing inputs. Generate runs in MATLAB, then run "
             "`engine-diagnostics build-features` and `engine-diagnostics train-temporary`.")
    st.stop()

with st.sidebar:
    st.header("Replay")
    run_id = st.selectbox("Run", run_ids)
    if st.session_state.get("run_id") != run_id:
        start_run(run_id)
    bundle, issues, run_packets = load_bundle(run_id)
    duration = bundle.metadata["duration_s"]
    speed = st.select_slider("Speed", options=[0.5, 1.0, 2.0, 4.0], value=1.0)
    play_col, restart_col = st.columns(2)
    if play_col.button("Pause" if st.session_state.playing else "Play", width="stretch"):
        st.session_state.playing = not st.session_state.playing
        st.session_state.anchor = time.monotonic() - st.session_state.stream_t / speed
    if restart_col.button("Restart", width="stretch"):
        start_run(run_id)
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


def evidence_table(record) -> pd.DataFrame:
    """Largest deviations from the training normal reference for the latest window."""
    names = model.baseline_names
    values = np.array([record["features"][model.feature_names.index(n)] for n in names])
    z = (values - model.baseline_median) / model.baseline_scale
    top = np.argsort(-np.abs(z))[:3]
    return pd.DataFrame({"feature": [names[k] for k in top],
                         "value": values[top].round(4),
                         "normal median": model.baseline_median[top].round(4),
                         "robust z": z[top].round(1)})


@st.fragment(run_every=TICK_S)
def live_view():
    state = st.session_state
    if state.playing:
        advance_to(min((time.monotonic() - state.anchor) * speed, duration))
        if state.next_packet >= len(run_packets):
            state.playing = False
    t_now = state.stream_t
    records = state.records
    latest = records[-1] if records else None

    cols = st.columns(6)
    cols[0].metric("Stream time", f"{t_now:.2f} s")
    cols[1].metric("Accepted", latest["accepted"] if latest else "warming up")
    cols[2].metric("SVM class", (latest["predicted"] or "-") if latest else "-")
    cols[3].metric("Anomaly score",
                   f"{latest['anomaly_score']:.2f}" if latest else "-",
                   help=f"Threshold {model.anomaly_threshold:.2f}")
    cols[4].metric("Quality", latest["quality"] if latest else "-")
    cols[5].metric("Processing", f"{latest['processing_ms']:.1f} ms" if latest else "-")
    if latest and latest["alert"]:
        st.error(f"ALERT: {latest['alert']} (confirmed by "
                 f"{config['runtime']['confirm_fault_updates']} consistent updates)")
    else:
        st.success("No active alert")

    left, right = st.columns(2)
    engine = state.engine
    window_s = config["analysis"]["window_s"]
    waves = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                          subplot_titles=[f"{m} (last {window_s:g} s, display decimated "
                                          f"{DISPLAY_DECIMATION}x)" for m in MODALITIES])
    psd = go.Figure()
    for row, m in enumerate(MODALITIES, start=1):
        fs = engine.rates[m]
        x = engine.buffers[m][-round(window_s * fs):]
        t = t_now - (x.size - np.arange(x.size)) / fs
        waves.add_trace(go.Scattergl(x=t[::DISPLAY_DECIMATION], y=x[::DISPLAY_DECIMATION],
                                     name=m, line={"width": 1}), row=row, col=1)
        if x.size >= config["analysis"]["welch_segment_samples"]:
            f, p = welch(x.astype(np.float64), fs=fs, window="hann",
                         nperseg=config["analysis"]["welch_segment_samples"])
            psd.add_trace(go.Scatter(x=f / 1000, y=10 * np.log10(p + 1e-20), name=m))
    waves.update_layout(height=380, showlegend=False, margin={"t": 40, "b": 20})
    psd.update_layout(height=380, title="PSD of current window (dB)", xaxis_title="kHz",
                      margin={"t": 40, "b": 20})
    left.plotly_chart(waves, width="stretch")
    right.plotly_chart(psd, width="stretch")

    telemetry = bundle.telemetry[(bundle.telemetry["t_s"] < t_now)
                                 & bundle.telemetry["signal"].isin(CONTEXT_SIGNALS)]
    tele = make_subplots(rows=1, cols=3, subplot_titles=["RPM", "Load (%)", "Throttle (%)"])
    for col, signal in enumerate(CONTEXT_SIGNALS, start=1):
        rows = telemetry[telemetry["signal"] == signal]
        tele.add_trace(go.Scatter(x=rows["t_s"], y=rows["value"], name=signal), row=1, col=col)
    tele.update_xaxes(range=[0, duration])
    tele.update_layout(height=250, showlegend=False, margin={"t": 40, "b": 20})
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

        truth = load_truth(TRUTH_DIR, run_id)
        if truth.get("onset_s") is not None:
            timeline.add_vline(x=truth["onset_s"], line_dash="dash",
                               annotation_text=f"truth onset: {truth['condition']}")
        else:
            timeline.add_annotation(x=duration / 2, y=0, text="truth: normal run",
                                    showarrow=False)
    timeline.update_layout(height=260, title="Decision timeline",
                           xaxis={"range": [0, duration], "title": "s"},
                           yaxis={"categoryorder": "array", "categoryarray": CLASS_ORDER},
                           margin={"t": 40, "b": 20})
    st.plotly_chart(timeline, width="stretch")

    if latest and latest["quality"] == "ok":
        st.subheader("Evidence: largest deviations from the normal reference")
        st.dataframe(evidence_table(latest), hide_index=True)
        st.caption("Robust z-scores against training normal windows. A deviation shows what "
                   "changed; it does not prove a fault.")


live_view()
