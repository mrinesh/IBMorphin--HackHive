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
from engine_diagnostics.lab import (  # noqa: E402
    LAB_FAULTS,
    LabSettings,
    base_runs,
    build_lab_bundle,
)
from engine_diagnostics.models.temporary import fit_holdout_model  # noqa: E402
from engine_diagnostics.pipeline.features import (  # noqa: E402
    HIGH_ENVELOPE_HZ,
    MODALITIES,
    _band_envelope,
    _firing_rate_hz,
    window_slice,
)
from engine_diagnostics.runtime.evidence import EvidencePanel, build_evidence  # noqa: E402
from engine_diagnostics.runtime.health import STAGES, HealthTimeline, health_timeline  # noqa: E402
from engine_diagnostics.runtime.replay import ReplayEngine, packets, replay_run  # noqa: E402

DATASETS = {
    "core_v1": {"raw": ROOT / "data/raw/core_v1", "truth": ROOT / "evaluation_truth/core_v1",
                "model": ROOT / "artifacts/models/diagnostic_v1.joblib",
                "splits": ROOT / "configs/core_v1.json"},
    "core_v2": {"raw": ROOT / "data/raw/core_v2", "truth": ROOT / "evaluation_truth/core_v2",
                "model": ROOT / "artifacts/models/multifault_v2.joblib",
                "splits": ROOT / "configs/core_v2.json"},
    "integration_v1": {"raw": ROOT / "data/raw", "truth": ROOT / "evaluation_truth",
                       "model": ROOT / "artifacts/models/temporary.joblib",
                       "splits": ROOT / "configs/integration_batch.json"},
}
# Cylinder localizer (train-localizer): names the affected cylinder of each active alert
# from the vibration signature and the cam phase reference. Optional.
LOCALIZER_PATH = ROOT / "artifacts/models/cylinder_localizer.joblib"
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
FAULT_COLORS = {"misfire": "#d62728", "knock_like": "#ff7f0e", "valve_clearance_like": "#9467bd"}
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
MODALITY_TITLES = {"audio": "Sound", "vibration": "Vibration"}
TELEMETRY_CARDS = [  # signal, label, value format, unit
    ("rpm", "RPM", ",.0f", "rpm"), ("load", "Load", ".1f", "%"),
    ("throttle", "Throttle", ".1f", "%"), ("coolant_temp", "Coolant", ".1f", "°C"),
    ("short_term_fuel_trim", "Short-term trim", "+.1f", "%"),
    ("long_term_fuel_trim", "Long-term trim", "+.1f", "%"),
]
# Evidence panels: each signal's statement and its highlighted segments share a color.
SIGNAL_COLORS = {"sound": "#1f77b4", "vibration": "#2ca02c", "rpm": "#d62728"}
SIGNAL_TAGS = {"sound": "blue", "vibration": "green", "rpm": "red"}
SIGNAL_NAMES = {"sound": "Sound", "vibration": "Vibration", "rpm": "RPM"}
WAVE_ROWS = {"sound": [1], "vibration": [2, 3]}  # rows of the waveform figure
# Fault-injection lab: healthy core_v1 recordings modified on request, replayed unchanged.
LAB_DATASET = "core_v1"
LAB_COLOR = "#b35900"
LAB_FAULT_NAMES = {"none": "No fault", "misfire": "Misfire", "knock_like": "Knock-like bursts"}
LAB_NOISE = [1.0, 1.5, 2.0, 3.0, 5.0]  # x recorded noise; 2x is the challenge-run level
LAB_SENSORS = {"audio": "Microphone (audio)", "vibration": "Accelerometer (vibration)",
               "rpm": "RPM telemetry"}
STAGE_COLORS = dict(zip(STAGES, ("#2ca02c", "#ff9f1c", "#d62728", "#bbbbbb"), strict=True))

st.set_page_config(page_title="DRIVEX · EngineSense", layout="wide",
                   initial_sidebar_state="auto")
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


@st.cache_resource
def lab_bases() -> pd.DataFrame:
    d = DATASETS[LAB_DATASET]
    return base_runs(d["raw"], d["truth"], d["splits"])


@st.cache_resource
def lab_truth(run_id: str) -> dict:
    path = DATASETS[LAB_DATASET]["truth"] / f"{run_id}_events.json"
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource
def lab_generator() -> dict:
    batch = json.loads(DATASETS[LAB_DATASET]["splits"].read_text(encoding="utf-8"))
    return batch["generator_params"]


@st.cache_resource
def lab_bundle(settings: LabSettings):
    base, _, _ = load_bundle(str(DATASETS[LAB_DATASET]["raw"]), settings.base_run)
    bundle, log = build_lab_bundle(base, lab_truth(settings.base_run), settings, lab_generator())
    return bundle, log, list(packets(bundle, load_config_cached()["runtime"]["packet_s"]))


@st.cache_resource
def unmodified_alerts(run_id: str) -> list[tuple[float, str]]:
    """Alert onsets when the unmodified base recording is replayed (for comparison)."""
    bundle, _, _ = load_bundle(str(DATASETS[LAB_DATASET]["raw"]), run_id)
    records = replay_run(bundle, load_model(str(DATASETS[LAB_DATASET]["model"])),
                         load_config_cached())
    return alert_onsets(records)


def alert_onsets(records) -> list[tuple[float, str]]:
    onsets, previous = [], None
    for r in records:
        if r["alert"] and r["alert"] != previous:
            onsets.append((r["t_end_s"], r["alert"]))
        previous = r["alert"]
    return onsets


def lab_available() -> bool:
    d = DATASETS[LAB_DATASET]
    return LAB_DATASET in available_datasets() and d["truth"].is_dir()


def start_run(dataset: str, run_id: str, held_out: str | None = None,
              lab: LabSettings | None = None) -> None:
    config = load_config_cached()
    model = (load_holdout_model(held_out) if held_out
             else load_model(str(DATASETS[dataset]["model"])))
    if lab:
        bundle, lab_log, run_packets = lab_bundle(lab)
    else:
        (bundle, _, run_packets), lab_log = load_bundle(str(DATASETS[dataset]["raw"]), run_id), None
    localizer = load_model(str(LOCALIZER_PATH))
    st.session_state.update(dataset=dataset, run_id=run_id, held_out=held_out, model=model,
                            lab=lab, lab_log=lab_log, bundle=bundle, packets=run_packets,
                            engine=ReplayEngine(model, config, bundle.metadata,
                                                localizer=localizer),
                            next_packet=0, records=[], stream_t=0.0, playing=False)


def advance_to(target_s: float) -> None:
    """Push every packet that has fully arrived by stream time target_s."""
    state = st.session_state
    run_packets = state.packets
    while (state.next_packet < len(run_packets)
           and run_packets[state.next_packet].t_end_s <= target_s + 1e-9):
        packet = run_packets[state.next_packet]
        state.records += state.engine.push(packet)
        state.stream_t = packet.t_end_s
        state.next_packet += 1


def seek(target_s: float) -> None:
    """Reset buffers and alerts, then re-process from the start (warm-up included)."""
    state = st.session_state
    start_run(state.dataset, state.run_id, state.held_out, state.get("lab"))
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


def lab_controls() -> None:
    """Sidebar form for the fault-injection lab; applies settings on submit only."""
    bases = lab_bases()
    current: LabSettings | None = st.session_state.get("lab")
    labels = {r.run_id: f"~{r.rpm:,.0f} rpm · {r.profile} · {r.run_id} ({r.split})"
              for r in bases.itertuples()}
    default = current.base_run if current else bases.run_id.iloc[len(bases) // 2]
    with st.form("lab_form"):
        st.caption("Healthy recordings from engines the classifier was not trained on.")
        base = st.select_slider("Engine RPM (base recording)", options=list(bases.run_id),
                                value=default, format_func=lambda r: labels[r])
        fault = st.selectbox("Injected fault", LAB_FAULTS, format_func=LAB_FAULT_NAMES.get,
                             index=LAB_FAULTS.index(current.fault) if current else 1)
        severity = st.slider("Fault severity", 0.1, 1.0, current.severity if current else 0.5, 0.05)
        onset = st.slider("Fault onset (s)", 3.0, 25.0, current.onset_s if current else 8.0, 0.5)
        cylinder = st.selectbox("Affected cylinder", [1, 2, 3, 4],
                                index=(current.cylinder - 1) if current else 1)
        noise = st.select_slider("Background noise", options=LAB_NOISE,
                                 value=current.noise_scale if current else 1.0,
                                 format_func=lambda k: "recorded" if k == 1 else f"{k:g}x",
                                 help="Total sensor noise relative to the recording. 2x is the "
                                      "level of the reserved challenge runs.")
        st.markdown("**Sensor availability**")
        available = {s: st.checkbox(label, value=s not in (current.failed if current else ()),
                                    key=f"lab_sensor_{s}") for s, label in LAB_SENSORS.items()}
        fail_s = st.slider("Unavailable sensors fail at (s)", 3.0, 28.0,
                           current.fail_s if current else 14.0, 0.5)
        seed = st.number_input("Random seed", 1, 9999, current.seed if current else 1)
        submitted = st.form_submit_button("Apply and replay", width="stretch", type="primary")
    settings = LabSettings(base_run=base, fault=fault, severity=float(severity),
                           onset_s=float(onset), cylinder=int(cylinder), noise_scale=float(noise),
                           failed=tuple(s for s, ok in available.items() if not ok),
                           fail_s=float(fail_s), seed=int(seed))
    if submitted or current is None:
        start_run(LAB_DATASET, base, lab=settings)
        if submitted:
            st.session_state.playing = True
            st.session_state.anchor = time.monotonic()



with st.sidebar:
    st.html(f'<div class="sidebar-brand">{theme.wordmark_html("small")}'
            '<span class="sub">EngineSense · diagnostics</span></div>')
    modes = ["Recorded replay"] + (["Fault-injection lab"] if lab_available() else [])
    mode = st.radio("Mode", modes, horizontal=True)
    lab_mode = mode == "Fault-injection lab"
    if lab_mode:
        st.header("Fault-injection lab")
        lab_controls()
    else:
        st.header("Scenario")
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

        st.subheader("Run selection")
        dataset = st.selectbox("Dataset", datasets,
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
            help="Integration dataset only: replay with a temporary model refitted without this "
                 "engine, so the engine is unseen. Test engines are already unseen.")
        held_out = engines[run_id] if hold_out and holdable else None
        state_key = tuple(st.session_state.get(k) for k in ("dataset", "run_id", "held_out", "lab"))
        if state_key != (dataset, run_id, held_out, None):
            start_run(dataset, run_id, held_out)
    state = st.session_state
    dataset, run_id, held_out = state.dataset, state.run_id, state.held_out
    model, bundle, run_packets = state.model, state.bundle, state.packets
    issues = [] if lab_mode else load_bundle(str(DATASETS[dataset]["raw"]), run_id)[1]
    duration = bundle.metadata["duration_s"]

    st.header("Playback")
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
        start_run(dataset, run_id, held_out, state.get("lab"))
    target = st.slider("Seek (s)", 0.0, float(duration), 0.0, step=0.5)
    if st.button("Seek", width="stretch"):
        seek(target)
        st.session_state.anchor = time.monotonic() - st.session_state.stream_t / speed
    show_truth = not lab_mode and st.checkbox(
        "Show evaluation truth overlay", value=False,
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
lab_log = state.get("lab_log") if lab_mode else None
if lab_log:
    items = "".join(f"<li>{line}</li>" for line in lab_log.lines)
    st.markdown(
        f"<div style='margin-top:0.4rem;padding:0.6rem 1rem;border-radius:0.4rem;"
        f"border:3px dashed {LAB_COLOR};background:rgba(179,89,0,0.12)'>"
        f"<b style='color:{LAB_COLOR};letter-spacing:0.05em'>FAULT-INJECTION LAB — "
        "INJECTED / SIMULATED SIGNALS</b><ul style='margin:0.3rem 0 0.2rem 1rem'>"
        f"{items}</ul><small>Injected with the MATLAB generator's source model in Python "
        "(per-engine sensor variation not reproduced). The injection schedule is known to the "
        "lab and shown on the timeline; the model only sees the modified signals.</small></div>",
        unsafe_allow_html=True)


def evidence_table(record) -> tuple[pd.DataFrame, bool]:
    """Three largest robust deviations (same rows as the diagnosis card's 'why')."""
    rows, matched = explain.evidence(record, model)
    table = pd.DataFrame({"feature": [r["feature"] for r in rows],
                          "meaning": [explain.describe_feature(r["feature"]) for r in rows],
                          "value": [round(r["value"], 4) for r in rows],
                          "normal median": [round(r["median"], 4) for r in rows],
                          "robust z": [round(r["z"], 1) for r in rows]})
    return table, matched


def feature_z(record) -> dict[str, float]:
    """Robust z of every baseline feature against the model's normal reference."""
    names = model.baseline_names
    values = np.array([record["features"][model.feature_names.index(n)] for n in names])
    (median, scale), _ = explain.reference_stats(record, model)
    return dict(zip(names, (values - median) / scale, strict=True))


def active_alerts(record) -> list[str]:
    if record is None:
        return []
    if hasattr(model, "fault_classes"):
        return list(record.get("alerts", []))
    return [record["alert"]] if record["alert"] else []


def alert_evidence(records, fault: str) -> EvidencePanel | None:
    """Evidence from the latest decision window with good quality."""
    record = next((r for r in reversed(records) if r["quality"] == "ok"), None)
    if record is None:
        return None
    t_end, window_s = record["t_end_s"], config["analysis"]["window_s"]
    rates = {m: getattr(bundle, f"{m}_fs_hz") for m in MODALITIES}
    waves = {m: getattr(bundle, m)[window_slice(t_end, rates[m], getattr(bundle, f"{m}_t0_s"),
                                                config)] for m in MODALITIES}
    tel = bundle.telemetry
    rpm = tel[(tel["signal"] == "rpm") & (tel["valid"] == 1)
              & (tel["t_s"] >= t_end - window_s) & (tel["t_s"] < t_end)]
    firing_hz = _firing_rate_hz(record["context"]["rpm"], config["engine"])
    return build_evidence(fault, waves, rates, t_end, window_s, rpm["t_s"].to_numpy(),
                          rpm["value"].to_numpy(), firing_hz, feature_z(record))


def evidence_figure(panel: EvidencePanel) -> go.Figure:
    window_s = config["analysis"]["window_s"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[f"{SIGNAL_NAMES[i.signal]}: {i.trace_label}"
                                        for i in panel.indicators])
    for row, ind in enumerate(panel.indicators, start=1):
        fig.add_trace(go.Scattergl(x=ind.trace_t, y=ind.trace_y, name=ind.trace_label,
                                   line={"width": 1, "color": "#555"}), row=row, col=1)
        for a, b in ind.segments:
            fig.add_vrect(x0=a, x1=b, fillcolor=SIGNAL_COLORS[ind.signal], opacity=0.3,
                          line_width=0, row=row, col=1)
    fig.update_xaxes(range=[panel.t_end_s - window_s, panel.t_end_s])
    fig.update_xaxes(title="s", row=3, col=1)
    fig.update_layout(height=420, showlegend=False, margin={"t": 40, "b": 20})
    return fig


def location_phrase(record, fault: str) -> str:
    """' in cylinder 3', ' in firing slot B', or a locating note, for an alert banner."""
    if record is None or "localization" not in record:
        return ""
    where = record["localization"].get(fault)
    if where and where.get("cylinder") is not None:
        return f" in cylinder {where['cylinder']}"
    if where and where.get("slot") is not None:
        return f" in firing slot {where['slot']} (no cam reference)"
    return " (locating cylinder…)"


def localization_figure(where: dict, fault: str) -> go.Figure:
    """Accumulated P(cylinder) for one alert, cylinders in block order."""
    labels = sorted(where["probabilities"], key=str)
    values = [where["probabilities"][k] for k in labels]
    named = where.get("cylinder") is not None or where.get("slot") is not None
    best = max(range(len(values)), key=values.__getitem__)
    colors = [FAULT_COLORS.get(fault, "#888") if named and k == best else "#b8bec6"
              for k in range(len(values))]
    prefix = "Cylinder" if where.get("phase") == "cam_sync" else "Slot"
    fig = go.Figure(go.Bar(x=[str(k) for k in labels], y=values, marker_color=colors,
                           text=[f"{v:.2f}" for v in values], textposition="outside",
                           hovertemplate=f"{prefix} %{{x}}: %{{y:.2f}}<extra></extra>"))
    fig.update_xaxes(title=prefix, type="category", tickangle=0)
    fig.update_yaxes(range=[0, 1.15], title="P(fault here)")
    fig.update_layout(height=200, showlegend=False, margin={"t": 10, "b": 10, "l": 10, "r": 10})
    return fig


def render_localization(where: dict | None, fault: str) -> None:
    if where is None:
        return
    if where.get("cylinder") is not None:
        st.markdown(f"**Location:** cylinder **{where['cylinder']}** — confidence "
                    f"{where['confidence']:.2f} over {where['windows']} decision windows")
    elif where.get("slot") is not None:
        st.markdown(f"**Location:** firing slot **{where['slot']}** — no cam phase reference "
                    "in this recording, so the cylinder number cannot be named")
    else:
        st.markdown("**Location:** still accumulating evidence for a cylinder")
    if where.get("probabilities"):
        st.plotly_chart(localization_figure(where, fault), width="stretch",
                        key=f"localization_{fault}")
    st.caption("Strokes are attributed to cylinders by crank angle from the cam phase "
               "pulse; the localizer compares each cylinder's stroke signature with the "
               "others. It is trained on synthetic engines.")


def render_evidence(panel: EvidencePanel, where: dict | None = None) -> None:
    color = FAULT_COLORS.get(panel.fault, "#888")
    with st.container(border=True):
        st.markdown(f"<h4 style='margin:0;color:{color}'>{panel.headline}</h4>",
                    unsafe_allow_html=True)
        render_localization(where, panel.fault)
        st.caption(f"Window {panel.t_end_s - config['analysis']['window_s']:.1f}-"
                   f"{panel.t_end_s:.1f} s. Shaded segments below and on the waveform and RPM "
                   "charts match each signal's color.")
        for ind in panel.indicators:
            count = f" · {len(ind.segments)} highlighted" if ind.segments else ""
            st.markdown(f":{SIGNAL_TAGS[ind.signal]}[■ **{SIGNAL_NAMES[ind.signal]}:**] "
                        f"{ind.claim} — **{ind.status}**{count}  \n"
                        f"<small>{ind.detail}</small>", unsafe_allow_html=True)
        st.markdown(f"**Assessment:** {panel.assessment}")
        st.plotly_chart(evidence_figure(panel), width="stretch", key=f"evidence_{panel.fault}")
        st.caption("Indicators compare this window with the model's normal reference for the same "
                   "operating range. They show what changed; they do not prove a cause.")


def lab_response(records, t_now: float) -> None:
    """How predictions responded to the lab changes, beside the unmodified recording."""
    settings: LabSettings = st.session_state.lab
    onset = settings.onset_s if settings.fault != "none" else None
    lab_alerts = alert_onsets(records)
    base_alerts = unmodified_alerts(settings.base_run)
    with st.container(border=True):
        st.markdown(f"<b style='color:{LAB_COLOR}'>Lab response</b> "
                    f"<small>(this replay up to {t_now:.1f} s vs the unmodified recording)</small>",
                    unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            st.markdown("**This lab replay**")
            if onset is not None:
                early = [a for a in lab_alerts if a[0] < onset]
                hits = [a for a in lab_alerts if a[0] >= onset]
                hit = next((a for a in hits if a[1] == settings.fault), None)
                if hit:
                    st.markdown(f"Injected {settings.fault} detected: alert at {hit[0]:.1f} s, "
                                f"**{hit[0] - onset:.1f} s after onset**")
                elif t_now < onset:
                    st.markdown(f"Fault starts at {onset:g} s.")
                else:
                    st.markdown(f"Injected {settings.fault} **not detected** so far.")
                other = [f"{c} at {t:.1f} s" for t, c in hits if c != settings.fault]
                if other:
                    st.markdown(f"Other alerts after onset: {', '.join(other)}")
                if early:
                    listed = ", ".join(f"{c} at {t:.1f} s" for t, c in early)
                    st.markdown(f"Alerts before onset: {listed}")
            else:
                st.markdown("No fault injected. Alerts: " + (", ".join(
                    f"{c} at {t:.1f} s" for t, c in lab_alerts) or "none so far"))
            since = [r for r in records if onset is None or r["t_end_s"] >= onset]
            counts = pd.Series([r["accepted"] if r["quality"] == "ok" else r["quality"]
                                for r in since]).value_counts()
            if not counts.empty:
                st.caption("Decisions" + (" since onset: " if onset is not None else ": ")
                           + ", ".join(f"{k} {v}" for k, v in counts.items()))
        with right:
            st.markdown("**Unmodified recording** (full replay, same model)")
            st.markdown(", ".join(f"{c} alert at {t:.1f} s" for t, c in base_alerts)
                        or "No alerts during the whole run.")
            st.caption("Compare to see which changes the injected signals caused.")


def run_health(records) -> HealthTimeline:
    return health_timeline(
        [r["t_end_s"] for r in records],
        pd.DataFrame([feature_z(r) if r["quality"] == "ok" else {} for r in records]),
        [r["anomaly_score"] for r in records], model.anomaly_threshold,
        [bool(active_alerts(r)) for r in records], [r["quality"] == "ok" for r in records])


def health_figure(health: HealthTimeline) -> go.Figure:
    step = config["analysis"]["step_s"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22],
                        vertical_spacing=0.06)
    top = max(2.0, float(np.nanmax(health.level.to_numpy(), initial=0)) * 1.1)
    fig.add_hrect(y0=0, y1=1, fillcolor=STAGE_COLORS["healthy"], opacity=0.08, line_width=0,
                  row=1, col=1)
    fig.add_hline(y=1, line={"color": STAGE_COLORS["developing anomaly"], "dash": "dash"},
                  annotation_text="developing threshold", annotation_position="top left",
                  row=1, col=1)
    for label in health.level.columns:
        fig.add_trace(go.Scatter(x=health.t, y=health.level[label], name=label, mode="lines"),
                      row=1, col=1)
    for stage in STAGES:
        hit = [s == stage for s in health.stage]
        if any(hit):
            fig.add_trace(go.Bar(x=health.t[hit], y=[1] * sum(hit), width=step, name=stage,
                                 marker_color=STAGE_COLORS[stage], showlegend=True,
                                 legendgroup="stage"), row=2, col=1)
    fig.update_yaxes(title="x threshold", range=[0, top], row=1, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)
    fig.update_xaxes(range=[0, duration], title="s", row=2, col=1)
    fig.update_layout(height=380, bargap=0, margin={"t": 40, "b": 20},
                      title="Health timeline vs healthy baseline: deterioration demonstration "
                            "(simulated data)",
                      legend={"orientation": "h", "y": -0.25})
    return fig


def single_timeline(records) -> go.Figure:
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
    timeline.update_layout(height=280, title="What the model decided over time",
                           yaxis={"categoryorder": "array", "categoryarray": CLASS_ORDER,
                                  "tickvals": CLASS_ORDER,
                                  "ticktext": [label(c) for c in CLASS_ORDER]},
                           legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
                           margin={"t": 60, "b": 20})
    return timeline


def multi_timeline(records) -> go.Figure:
    """One row per fault: detections, borderline (uncertain) windows, and alerts."""
    timeline = go.Figure()
    faults = model.fault_classes
    for fault in faults:
        rows = [r for r in records if "fault_decisions" in r]
        for kind, marker, name in (("fault", {"size": 8}, "detected"),
                                   ("uncertain", {"symbol": "x", "size": 7}, "uncertain")):
            hits = [r["t_end_s"] for r in rows if r["fault_decisions"][fault] == kind]
            timeline.add_trace(go.Scatter(
                x=hits, y=[fault] * len(hits), mode="markers", name=f"{fault} {name}",
                marker={**marker, "color": FAULT_COLORS[fault]}, showlegend=False))
        alerted = [r["t_end_s"] for r in records if fault in r.get("alerts", [])]
        timeline.add_trace(go.Scatter(
            x=alerted, y=[fault] * len(alerted), mode="markers", showlegend=False,
            marker={"symbol": "square", "size": 14, "opacity": 0.3,
                    "color": FAULT_COLORS[fault]}, name=f"{fault} alert"))
    timeline.update_layout(height=260, margin={"t": 40, "b": 20},
                           title="Per-fault timeline (dot detected, x uncertain, "
                                 "square active alert)",
                           yaxis={"categoryorder": "array", "categoryarray": list(faults)})
    return timeline


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


def label(class_name: str | None) -> str:
    """Readable class name (e.g. knock_like -> Knock-like) for display only."""
    return explain.names(class_name)


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
    multi = hasattr(model, "fault_classes")  # v2: one detector and alert per fault
    has_probability = bool(latest and "probability" in latest)
    confirm = getattr(model, "confirm_updates", config["runtime"]["confirm_fault_updates"])

    # Sections whose element count changes during playback (alert banners, lab
    # response, evidence panels) each live in one fixed container, so elements after
    # them keep their place in the page and the 3D view is not remounted (which would
    # rebuild the scene and reset its camera) whenever an alert appears or clears.
    active = active_alerts(latest)
    with st.container(key="alerts"):
        if active:
            for fault in active:
                n = model.confirm_updates[fault] if multi else confirm
                st.error(f"ALERT: {label(fault)} detected{location_phrase(latest, fault)} — "
                         f"confirmed by {n} consistent updates", icon=":material/warning:")
        else:
            st.success("No active alert — the engine sounds normal so far",
                       icon=":material/check_circle:")

    cols = st.container(key="readout").columns(7)
    cols[0].metric("Replay time", f"{t_now:.1f} s")
    cols[1].metric("Decision", label(latest["accepted"]) if latest else "Warming up",
                   help="What the model accepts for this window after its confidence and "
                        "anomaly checks.")
    cols[2].metric("Best guess", label(latest["predicted"]) if latest else "-",
                   help="The classifier's top class before the acceptance checks.")
    if multi:
        cols[3].metric("Faults detected", len(latest["detected"]) if latest else "-",
                       help="Each fault has its own calibrated detector and threshold, so "
                            "several can be reported in the same window.")
    elif has_probability:
        cols[3].metric("Calibrated confidence", f"{latest['probability']:.2f}",
                       help=f"Accepted only if ≥ {model.probability_min:.2f}; calibrated on "
                            "separate calibration engines (synthetic).")
    else:
        cols[3].metric("Decision margin", f"{latest['margin']:.2f}" if latest else "-")
    cols[4].metric("Anomaly score", f"{latest['anomaly_score']:.2f}" if latest else "-",
                   help=f"How unusual this window is compared with normal running. "
                        f"Threshold {model.anomaly_threshold:.2f}.")
    cols[5].metric("Signal quality",
                   ("Good" if latest["quality"] == "ok"
                    else latest["quality"].replace("_", " ").capitalize()) if latest else "-")
    cols[6].metric("Response time",
                   f"{latest['processing_ms']:.0f} / {state.get('render_ms', 0):.0f} ms"
                   if latest else "-",
                   help="Model processing per decision / dashboard update time (server).")

    if multi:
        fault_cols = st.columns(len(model.fault_classes))
        for col, fault in zip(fault_cols, model.fault_classes, strict=True):
            col.metric(f"P({fault})",
                       f"{latest['probabilities'][fault]:.2f}" if has_probability else "-",
                       help=f"Detected at ≥ {model.thresholds[fault]:.2f}; calibrated on "
                            "separate calibration engines (synthetic).")

    with st.container(key="lab_response"):
        if lab_log:
            lab_response(records, t_now)

    panels = [p for p in (alert_evidence(records, f) for f in active) if p]
    with st.container(key="evidence"):
        if panels:
            st.subheader("Evidence for active alerts")
            for col, panel in zip(st.columns(len(panels)), panels, strict=True):
                with col:
                    render_evidence(panel,
                                    (latest.get("localization") or {}).get(panel.fault))

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
        ":material/graphic_eq: Sound & vibration", ":material/equalizer: Frequencies (FFT / STFT)",
        ":material/speed: Engine data", ":material/timeline: History & evidence"])

    engine = state.engine
    window_s = config["analysis"]["window_s"]
    tag = " · LAB: injected/modified" if lab_log else ""
    waves = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                          subplot_titles=[f"{MODALITY_TITLES[m]}{tag} · last {window_s:g} s"
                                          for m in MODALITIES]
                          + [f"Sharp vibration impacts (5-10 kHz envelope){tag}"])
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
    for panel in panels:  # link: same segments and colors as the evidence panels
        for ind in panel.indicators:
            for row in WAVE_ROWS.get(ind.signal, []):
                for a, b in ind.segments:
                    waves.add_vrect(x0=a, x1=b, fillcolor=SIGNAL_COLORS[ind.signal],
                                    opacity=0.2, line_width=0, row=row, col=1)
    waves.update_layout(height=470, showlegend=False, margin={"t": 40, "b": 20})
    waves.update_xaxes(title_text="stream time (s)", row=3, col=1)
    psd.update_layout(height=380, title="Frequency content now (FFT power, dB)",
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
                           title=f"Vibration frequencies over the last {SPECTROGRAM_S:g} s (STFT)",
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
    for panel in panels:
        for a, b in next(i for i in panel.indicators if i.signal == "rpm").segments:
            tele.add_vrect(x0=a, x1=b, fillcolor=SIGNAL_COLORS["rpm"], opacity=0.3,
                           line_width=0, row=1, col=1)
    tele.update_xaxes(range=[0, duration])
    tele.update_layout(height=400, showlegend=False, margin={"t": 50, "b": 20})
    telemetry_tab.plotly_chart(tele, width="stretch")

    timeline = multi_timeline(records) if multi else single_timeline(records)
    if show_truth:
        from engine_diagnostics.pipeline.labels import fault_intervals  # evaluation only

        truth = load_truth_cached(dataset, run_id)
        intervals = fault_intervals(truth)
        for fault, spans in intervals.items():
            for a, b in spans:
                timeline.add_vrect(x0=a, x1=b,
                                   fillcolor=FAULT_COLORS.get(fault, theme.TOKENS["amber"]),
                                   opacity=0.12, line_width=0)
        note = (f"evaluation truth: {' + '.join(label(f) for f in sorted(intervals))}"
                if intervals else "evaluation truth: normal run")
        timeline.add_annotation(x=duration / 2, y=1.1, yref="paper", text=note, showarrow=False)
    if lab_log:
        if lab_log.fault_interval:
            a, b = lab_log.fault_interval
            timeline.add_vrect(x0=a, x1=b, fillcolor=LAB_COLOR, opacity=0.12,
                               line={"color": LAB_COLOR, "dash": "dash", "width": 1})
            timeline.add_annotation(x=a, y=1.1, yref="paper", xanchor="left", showarrow=False,
                                    text=f"injected {state.lab.fault} from {a:g} s "
                                         "(known to the lab, hidden from the model)")
        if lab_log.fail_s is not None:
            timeline.add_vline(x=lab_log.fail_s, line={"color": "black", "dash": "dot"})
            timeline.add_annotation(x=lab_log.fail_s, y=-0.25, yref="paper", showarrow=False,
                                    text=f"sensor failure: {', '.join(state.lab.failed)}")
    timeline.update_xaxes(range=[0, duration], title="s")
    timeline_tab.plotly_chart(timeline, width="stretch")

    if records:
        health = run_health(records)
        timeline_tab.plotly_chart(health_figure(health), width="stretch")
        timeline_tab.markdown(f"**Health stages so far:** {health.summary()}")
        timeline_tab.caption(
            "Each line is a smoothed (2 s, causal) deviation from the healthy baseline for the "
            "same operating range, divided by its developing threshold: firing-rhythm "
            "irregularity and high-frequency impulsiveness as robust z (threshold 4), and the "
            "model's anomaly score (its own threshold). Developing anomaly = a line stays at or "
            "above 1 without a confirmed alert; persistent fault = the model's confirmed alert. "
            "Thresholds were set on tuning engines; on held-out test engines about 9% of healthy "
            "windows still show as developing. All progression shown is simulated, so this is a "
            "deterioration demonstration: it describes the current state and makes no forecast "
            "or remaining-useful-life claim, which would need real run-to-failure data.")

    lower_left, lower_right = timeline_tab.columns(2)
    if latest and latest["quality"] == "ok":
        table, matched = evidence_table(latest)
        lower_left.subheader("Biggest differences from normal")
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
        trend.update_layout(height=260, title="Valve wear trend (simulated)",
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
