"""Local replay dashboard: recorded synthetic runs through the live inference path.

Run from the repository root:
    uv run --all-extras streamlit run dashboard/app.py

Predictions come from ReplayEngine (shared features, model, alerts) as packets
are replayed; nothing is precomputed. Display traces are decimated; analysis
uses full-rate samples. Display panels may show any data already received
(up to the current stream time). The optional truth overlay is evaluation data
and is never passed to the engine. For the integration dataset, "Hold out this
run's engine" replays with a temporary model refitted without that engine, and
the cross-engine panel shows the leave-one-engine-out report from train-temporary.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import spectrogram, welch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `dashboard` package (theme, explain) for `streamlit run`
    sys.path.insert(0, str(ROOT))

from dashboard import engine3d, explain, theme  # noqa: E402
from engine_diagnostics.config import load_config  # noqa: E402
from engine_diagnostics.data.run_io import find_runs, load_run, validate_run  # noqa: E402
from engine_diagnostics.models.temporary import fit_holdout_model  # noqa: E402
from engine_diagnostics.pipeline.features import (  # noqa: E402
    HIGH_ENVELOPE_HZ,
    MODALITIES,
    _band_envelope,
)
from engine_diagnostics.runtime.replay import ReplayEngine, packets  # noqa: E402

DATASETS = {
    "core_v1": {"raw": ROOT / "data/raw/core_v1", "truth": ROOT / "evaluation_truth/core_v1",
                "model": ROOT / "artifacts/models/diagnostic_v1.joblib",
                "splits": ROOT / "configs/core_v1.json"},
    "integration_v1": {"raw": ROOT / "data/raw", "truth": ROOT / "evaluation_truth",
                       "model": ROOT / "artifacts/models/temporary.joblib",
                       "splits": ROOT / "configs/integration_batch.json"},
}
PRESETS_PATH = ROOT / "configs/demo_presets.json"
HOLDOUT_DATASET = "integration_v1"
FEATURES_PATH = ROOT / "data/processed/integration_v1_features.parquet"
REPORT_PATH = ROOT / "reports/generated/temporary_model_report.json"
DISPLAY_DECIMATION = 8
SPECTROGRAM_S = 10.0
TICK_S = 0.5
CONTEXT_SIGNALS = ("rpm", "load", "throttle")
SLOW_SIGNALS = ("coolant_temp", "short_term_fuel_trim", "long_term_fuel_trim")
CLASS_ORDER = ["normal", "misfire", "knock_like", "valve_clearance_like", "uncertain"]
VALVE = "valve_clearance_like"
DEGRADATION_WINDOW = 10
SPARKLINE_S = 10.0
# Scenario cards map to tagged demo presets. "Abnormal vibration" is represented by
# the valve-train impact class the model was trained on, not a general detector.
SCENARIOS = {
    "normal": ":material/check_circle: Normal",
    "knock_like": ":material/bolt: Knock",
    "misfire": ":material/local_fire_department: Misfire",
    "valve_clearance_like": ":material/graphic_eq: Valve-train impacts",
}
TELEMETRY_CARDS = [  # signal, label, value format, unit
    ("rpm", "RPM", ",.0f", "rpm"), ("load", "Load", ".1f", "%"),
    ("throttle", "Throttle", ".1f", "%"), ("coolant_temp", "Coolant", ".1f", "°C"),
    ("short_term_fuel_trim", "Short-term trim", "+.1f", "%"),
    ("long_term_fuel_trim", "Long-term trim", "+.1f", "%"),
]

st.set_page_config(page_title="ENGINE//SENSE · replay", layout="wide",
                   initial_sidebar_state="expanded")
theme.register_plotly_template()


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


@st.cache_resource
def load_truth_cached(dataset: str, run_id: str) -> dict:
    """Evaluation truth for the review overlay only; never passed to the engine."""
    from engine_diagnostics.pipeline.labels import load_truth  # evaluation only

    return load_truth(DATASETS[dataset]["truth"], run_id)


@st.cache_resource
def load_holdout_model(engine_id: str):
    return fit_holdout_model(pd.read_parquet(FEATURES_PATH), engine_id)


def available_datasets() -> list[str]:
    return [name for name, d in DATASETS.items()
            if d["raw"].is_dir() and find_runs(d["raw"]) and d["model"].is_file()]


def start_run(dataset: str, run_id: str, held_out: str | None = None) -> None:
    config = load_config_cached()
    model = (load_holdout_model(held_out) if held_out
             else load_model(str(DATASETS[dataset]["model"])))
    bundle, _, _ = load_bundle(str(DATASETS[dataset]["raw"]), run_id)
    st.session_state.update(dataset=dataset, run_id=run_id, held_out=held_out, model=model,
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
    state = st.session_state
    start_run(state.dataset, state.run_id, state.held_out)
    advance_to(target_s)


config = load_config_cached()
datasets = available_datasets()

st.html(theme.css())
st.markdown(theme.banner_html(), unsafe_allow_html=True)
st.markdown(theme.brand_html(), unsafe_allow_html=True)
if not datasets:
    st.error("Missing inputs. Generate runs in MATLAB, then run `engine-diagnostics "
             "build-features` and `engine-diagnostics train` (or `train-temporary`).")
    st.stop()

presets = []
if PRESETS_PATH.is_file():
    presets = [p for p in json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
               if p["dataset"] in datasets]
preset_names = ["(none)"] + [p["name"] for p in presets]
scenario_options = [tag for tag in SCENARIOS if any(p.get("scenario") == tag for p in presets)]


def on_scenario_change() -> None:
    """A scenario card selects the first demo preset tagged with that scenario."""
    tag = st.session_state.get("scenario")
    match = next((p["name"] for p in presets if p.get("scenario") == tag), None)
    if match:
        st.session_state.preset_select = match


def on_preset_change() -> None:
    """Keep the scenario cards in step when a preset is picked from the list."""
    preset = next((p for p in presets if p["name"] == st.session_state.preset_select), None)
    tag = preset.get("scenario") if preset else None
    st.session_state.scenario = tag if tag in scenario_options else None


with st.sidebar:
    st.subheader("Scenario")
    if scenario_options:
        st.segmented_control("Scenario", scenario_options, format_func=SCENARIOS.get,
                             key="scenario", on_change=on_scenario_change,
                             label_visibility="collapsed")
        st.caption("Each scenario replays a recorded held-out run. Abnormal vibration is "
                   "shown as valve-train impacts, the fault class the model was trained on.")
    preset_name = st.selectbox("Demo preset", preset_names, key="preset_select",
                               on_change=on_preset_change)
    preset = next((p for p in presets if p["name"] == preset_name), None)
    if preset and st.session_state.get("applied_preset") != preset_name:
        start_run(preset["dataset"], preset["run_id"])
        if preset.get("start_s"):
            advance_to(preset["start_s"])
        st.session_state.applied_preset = preset_name
    if preset:
        st.info(preset.get("note", ""))

    with st.expander("Run selection", expanded=preset is None):
        dataset = st.selectbox(
            "Dataset", datasets,
            index=datasets.index(st.session_state.get("dataset", datasets[0]))
            if st.session_state.get("dataset") in datasets else 0)
        paths = DATASETS[dataset]
        split_of = engine_split_map(str(paths["splits"]))
        engines = run_engines(str(paths["raw"]))
        run_ids = sorted(engines)
        current = (st.session_state.get("run_id")
                   if st.session_state.get("dataset") == dataset else None)
        run_id = st.selectbox(
            "Run", run_ids, index=run_ids.index(current) if current in run_ids else 0,
            format_func=lambda r: f"{r} · {engines[r]} · {split_of.get(engines[r], '?')}")
        holdable = (dataset == HOLDOUT_DATASET and FEATURES_PATH.is_file()
                    and split_of.get(engines[run_id]) == "train")
        hold_out = st.checkbox(
            "Hold out this run's engine", disabled=not holdable,
            help="Integration dataset only: replay with a temporary model refitted without "
                 "this engine, so the engine is unseen. Test engines are already unseen.")
    held_out = engines[run_id] if hold_out and holdable else None
    state_key = tuple(st.session_state.get(k) for k in ("dataset", "run_id", "held_out"))
    if state_key != (dataset, run_id, held_out):
        start_run(dataset, run_id, held_out)
    model = st.session_state.model
    bundle, issues, run_packets = load_bundle(str(paths["raw"]), run_id)
    duration = bundle.metadata["duration_s"]

    st.subheader("Playback")
    speed = st.select_slider("Speed", options=[0.5, 1.0, 2.0, 4.0], value=1.0, key="speed")

    def toggle_play() -> None:
        # A callback runs before the rerun, so the button label below is already current.
        state = st.session_state
        state.playing = not state.playing
        state.anchor = time.monotonic() - state.stream_t / state.speed

    play_col, restart_col = st.columns(2)
    play_col.button("Pause" if st.session_state.playing else "Play", width="stretch",
                    type="primary", on_click=toggle_play)
    if restart_col.button("Restart", width="stretch"):
        start_run(dataset, run_id, held_out)
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

if held_out:
    st.warning(f"Unseen-engine replay: the model was refitted without {held_out}, so "
               f"this run's engine is new to it. Synthetic check, not a real-world claim.")


def evidence_table(record) -> tuple[pd.DataFrame, bool]:
    """Three largest robust deviations (same rows as the diagnosis card's 'why')."""
    rows, matched = explain.evidence(record, model)
    table = pd.DataFrame({"feature": [r["feature"] for r in rows],
                          "meaning": [explain.describe_feature(r["feature"]) for r in rows],
                          "value": [round(r["value"], 4) for r in rows],
                          "normal median": [round(r["median"], 4) for r in rows],
                          "robust z": [round(r["z"], 1) for r in rows]})
    return table, matched


def freshness_info(signal: str, t_now: float) -> tuple[str, float | None, pd.DataFrame]:
    """Status ('fresh', 'STALE', 'no data'), age in s, and the valid rows received so far."""
    rows = bundle.telemetry[(bundle.telemetry["signal"] == signal)
                            & (bundle.telemetry["valid"] == 1) & (bundle.telemetry["t_s"] < t_now)]
    if rows.empty:
        return "no data", None, rows
    runtime = config["runtime"]
    limits = {"rpm": runtime["rpm_max_age_s"], "load": runtime["load_throttle_max_age_s"],
              "throttle": runtime["load_throttle_max_age_s"]}
    limit = limits.get(signal, 2 / config["streams"]["telemetry_rates_hz"][signal])
    age = t_now - rows["t_s"].iloc[-1]
    return ("fresh" if age <= limit else "STALE"), age, rows


def freshness(signal: str, t_now: float) -> str:
    status, age, _ = freshness_info(signal, t_now)
    return "no data" if age is None else f"{status} ({age:.2f} s)"


def telemetry_cards(t_now: float) -> list[dict]:
    """Last received valid value per telemetry signal, its freshness, and recent history."""
    cards = []
    for signal, label, fmt, unit in TELEMETRY_CARDS:
        status, age, rows = freshness_info(signal, t_now)
        recent = rows[rows["t_s"] >= t_now - SPARKLINE_S]["value"]
        cards.append({"label": label, "fmt": fmt, "unit": unit, "status": status,
                      "age_s": age, "value": None if rows.empty else float(rows["value"].iloc[-1]),
                      "history": recent.iloc[-80:].astype(float).tolist()})
    return cards


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
    confirm = getattr(model, "confirm_updates", config["runtime"]["confirm_fault_updates"])

    if latest and latest["alert"]:
        st.error(f"ALERT: {latest['alert']} (confirmed by {confirm} consistent updates)",
                 icon=":material/warning:")
    else:
        st.success("No active alert", icon=":material/check_circle:")

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

    cards = telemetry_cards(t_now)
    view_col, diag_col = st.columns([3, 2], gap="medium")
    with view_col, st.container(key="viewport"):
        engine3d.render(engine3d.scene_state(
            latest, model, t_now=t_now, playing=bool(state.playing),
            rpm=next(c["value"] for c in cards if c["label"] == "RPM"),
            firing_order=config["engine"]["firing_order"],
            truth=load_truth_cached(dataset, run_id) if show_truth else None))
    with diag_col:
        st.html(theme.status_card_html(explain.diagnose(latest, model, confirm)))

    st.html(theme.telemetry_cards_html(cards))

    signals_tab, frequency_tab, telemetry_tab, timeline_tab = st.tabs([
        ":material/show_chart: Signals", ":material/equalizer: Frequency · FFT / STFT",
        ":material/speed: Telemetry", ":material/timeline: Timeline & evidence"])

    engine = state.engine
    window_s = config["analysis"]["window_s"]
    waves = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                          subplot_titles=[f"{m} (last {window_s:g} s, display decimated "
                                          f"{DISPLAY_DECIMATION}x)" for m in MODALITIES]
                          + ["vibration high-band envelope (5-10 kHz)"])
    psd = go.Figure()
    for row, m in enumerate(MODALITIES, start=1):
        fs = engine.rates[m]
        x = engine.buffers[m][-round(window_s * fs):]
        t = t_now - (x.size - np.arange(x.size)) / fs
        color = theme.MODALITY_COLORS[m]
        waves.add_trace(go.Scattergl(x=t[::DISPLAY_DECIMATION], y=x[::DISPLAY_DECIMATION],
                                     name=m, line={"width": 1, "color": color}), row=row, col=1)
        if x.size >= config["analysis"]["welch_segment_samples"]:
            clean = np.nan_to_num(x.astype(np.float64))
            f, p = welch(clean, fs=fs, window="hann",
                         nperseg=config["analysis"]["welch_segment_samples"])
            psd.add_trace(go.Scatter(x=f / 1000, y=10 * np.log10(p + 1e-20), name=m,
                                     line={"width": 1.5, "color": color}))
            if m == "vibration":
                env = _band_envelope(clean, fs, HIGH_ENVELOPE_HZ)
                waves.add_trace(go.Scattergl(x=t[::DISPLAY_DECIMATION],
                                             y=env[::DISPLAY_DECIMATION], name="envelope",
                                             line={"width": 1,
                                                   "color": theme.MODALITY_COLORS["envelope"]}),
                                row=3, col=1)
    waves.update_layout(height=470, showlegend=False, margin={"t": 40, "b": 20})
    waves.update_xaxes(title_text="stream time (s)", row=3, col=1)
    psd.update_layout(height=380, title="Welch PSD of the current window (FFT, dB)",
                      xaxis_title="kHz", yaxis_title="dB", margin={"t": 50, "b": 20},
                      legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"})
    signals_tab.plotly_chart(waves, width="stretch")

    # Spectrogram of received vibration (display only; STFT settings from config).
    fs = engine.rates["vibration"]
    received = bundle.vibration[max(0, round((t_now - SPECTROGRAM_S) * fs)):round(t_now * fs)]
    seg = config["analysis"]["stft_segment_samples"]
    psd_col, spec_col = frequency_tab.columns(2)
    psd_col.plotly_chart(psd, width="stretch")
    if received.size >= seg:
        f, ts, sxx = spectrogram(np.nan_to_num(received.astype(np.float64)), fs=fs, window="hann",
                                 nperseg=seg, noverlap=seg - config["analysis"]["stft_hop_samples"])
        offset = t_now - received.size / fs
        spec = go.Figure(go.Heatmap(x=ts + offset, y=f / 1000, z=10 * np.log10(sxx + 1e-20),
                                    colorscale="Inferno", colorbar={"title": "dB"}))
        spec.update_layout(height=380,
                           title=f"Vibration STFT spectrogram (last {SPECTROGRAM_S:g} s)",
                           xaxis_title="s", yaxis_title="kHz", margin={"t": 50, "b": 20})
        spec_col.plotly_chart(spec, width="stretch")
    else:
        spec_col.info("Spectrogram appears once one STFT segment of vibration has arrived.")

    signals = CONTEXT_SIGNALS + SLOW_SIGNALS
    titles = ["RPM", "Load (%)", "Throttle (%)", "Coolant (°C)", "Short-term trim (%)",
              "Long-term trim (%)"]
    tele = make_subplots(rows=2, cols=3, subplot_titles=[
        f"{title} · {freshness(s, t_now)}" for title, s in zip(titles, signals, strict=True)])
    received_tele = bundle.telemetry[(bundle.telemetry["t_s"] < t_now)
                                     & (bundle.telemetry["valid"] == 1)]
    for k, signal in enumerate(signals):
        rows = received_tele[received_tele["signal"] == signal]
        tele.add_trace(go.Scatter(x=rows["t_s"], y=rows["value"], name=signal,
                                  line={"width": 1.5, "color": theme.TOKENS["cyan"]}),
                       row=k // 3 + 1, col=k % 3 + 1)
    tele.update_xaxes(range=[0, duration])
    tele.update_layout(height=400, showlegend=False, margin={"t": 50, "b": 20})
    telemetry_tab.plotly_chart(tele, width="stretch")

    timeline = go.Figure()
    if records:
        log = pd.DataFrame([{k: r[k] for k in ("t_end_s", "accepted", "alert")}
                            for r in records])
        timeline.add_trace(go.Scatter(
            x=log["t_end_s"], y=log["accepted"], mode="markers", name="accepted decision",
            marker={"size": 7, "color": [theme.class_meta(a)["color"] for a in log["accepted"]],
                    "line": {"width": 0}}))
        alerted = log[log["alert"].notna()]
        timeline.add_trace(go.Scatter(x=alerted["t_end_s"], y=alerted["alert"], mode="markers",
                                      marker={"symbol": "square-open", "size": 13,
                                              "color": theme.TOKENS["rose"]},
                                      name="active alert"))
    if show_truth:
        truth = load_truth_cached(dataset, run_id)
        intervals = truth.get("active_intervals") or []
        if intervals and not isinstance(intervals[0], list):
            intervals = [intervals]
        for a, b in intervals:
            timeline.add_vrect(x0=a, x1=b, fillcolor=theme.TOKENS["amber"], opacity=0.12,
                               line_width=0)
        note = (f"evaluation truth: {truth['condition']}" if intervals
                else "evaluation truth: normal run")
        timeline.add_annotation(x=duration / 2, y=1.1, yref="paper", text=note, showarrow=False)
    timeline.update_layout(height=280, title="Decision timeline (model output)",
                           xaxis={"range": [0, duration], "title": "s"},
                           yaxis={"categoryorder": "array", "categoryarray": CLASS_ORDER},
                           legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
                           margin={"t": 60, "b": 20})
    timeline_tab.plotly_chart(timeline, width="stretch")

    lower_left, lower_right = timeline_tab.columns(2)
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
        trend = go.Figure(go.Scatter(x=[r["t_end_s"] for r in records], y=indicator,
                                     line={"color": theme.TOKENS["violet"], "width": 2}))
        trend.update_layout(height=260, title="Simulated valve-wear indicator",
                            xaxis={"range": [0, duration], "title": "s"},
                            yaxis={"range": [0, 1], "title": "rolling P(valve)"},
                            margin={"t": 40, "b": 20})
        lower_right.plotly_chart(trend, width="stretch")
        lower_right.caption("5 s rolling mean of the calibrated valve-clearance probability. "
                            "Accelerated simulation; not a remaining-life estimate.")
    state.render_ms = 1000 * (time.perf_counter() - rendered)


live_view()


def cross_engine_panel():
    st.subheader("Cross-engine check (leave-one-engine-out, temporary model)")
    report = (json.loads(REPORT_PATH.read_text(encoding="utf-8"))
              if REPORT_PATH.is_file() else {})
    cross = report.get("cross_domain")
    if cross is None:
        st.info("No cross-engine results. Run `engine-diagnostics train-temporary`.")
        return
    st.caption(f"{cross['status']}. Protocol: {cross['protocol']}.")
    rows = []
    for protocol, part in cross["results"].items():
        for engine_id, metrics in [*part["engines"].items(), ("pooled", part["pooled"])]:
            rows.append({"features": protocol, "held-out engine": engine_id,
                         "windows": metrics["windows"],
                         "macro-F1": round(metrics["macro_f1"], 3),
                         **{f"recall {c}": round(v, 2) for c, v in metrics["recall"].items()}})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    pooled = cross["results"]["fused"]["pooled"]
    labels = pooled["confusion_labels"]
    st.dataframe(pd.DataFrame(pooled["confusion"], index=[f"true {c}" for c in labels],
                              columns=[f"predicted {c}" for c in labels]))
    st.caption("Pooled confusion of the fused SVM over all held-out engines. Report from "
               f"model {report['model_version']}; regenerate with train-temporary after "
               "changing data.")


if dataset == HOLDOUT_DATASET:
    cross_engine_panel()
