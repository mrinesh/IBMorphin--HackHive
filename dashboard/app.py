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

Engine profiles are optional. With none selected nothing is saved. With one
selected, each replay that reaches the end (or "Save run so far") is saved to
that engine's history (engine_diagnostics.history) and the history panel shows
its runs and notes. History never stores evaluation truth or preset text.
"""

import dataclasses
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

from dashboard import engine3d, explain, sensors, theme  # noqa: E402
from engine_diagnostics import report  # noqa: E402
from engine_diagnostics.config import load_config  # noqa: E402
from engine_diagnostics.data.run_io import find_runs, load_run, validate_run  # noqa: E402
from engine_diagnostics.diagnosis import DISCLAIMER as DIAGNOSIS_DISCLAIMER  # noqa: E402
from engine_diagnostics.diagnosis import run_diagnosis  # noqa: E402
from engine_diagnostics.history import (  # noqa: E402
    Diagnosis,
    HistoryStore,
    Profile,
    raw_from_bundle,
    summarize_run,
)
from engine_diagnostics.lab import (  # noqa: E402
    LAB_FAULTS,
    LabSettings,
    base_runs,
    build_lab_bundle,
)
from engine_diagnostics.models.obd_misfire import CrankCalibration  # noqa: E402
from engine_diagnostics.models.temporary import fit_holdout_model  # noqa: E402
from engine_diagnostics.pipeline.features import (  # noqa: E402
    HIGH_ENVELOPE_HZ,
    MODALITIES,
    _band_envelope,
    _firing_rate_hz,
    window_slice,
)
from engine_diagnostics.profiles import (  # noqa: E402
    load_catalog,
    lookup,
    reliability_warning,
    spec_source,
    validate_specs,
)
from engine_diagnostics.runtime.evidence import EvidencePanel, build_evidence  # noqa: E402
from engine_diagnostics.runtime.health import STAGES, HealthTimeline, health_timeline  # noqa: E402
from engine_diagnostics.runtime.replay import ReplayEngine, packets, replay_run  # noqa: E402
from engine_diagnostics.runtime.thermal import ThermalCalibration  # noqa: E402

DATASETS = {
    "core_v1": {"raw": ROOT / "data/raw/core_v1", "truth": ROOT / "evaluation_truth/core_v1",
                "model": ROOT / "artifacts/models/diagnostic_v1.joblib",
                "splits": ROOT / "configs/core_v1.json"},
    "core_v2": {"raw": ROOT / "data/raw/core_v2", "truth": ROOT / "evaluation_truth/core_v2",
                "model": ROOT / "artifacts/models/multifault_v2.joblib",
                "splits": ROOT / "configs/core_v2.json"},
    # core_v2's recordings plus the 60-2 crank wheel and cylinder temperatures (contract
    # 1.2); its audio and vibration are bit-identical to core_v2, so the v2 model applies.
    "core_v3": {"raw": ROOT / "data/raw/core_v3", "truth": ROOT / "evaluation_truth/core_v3",
                "model": ROOT / "artifacts/models/multifault_v2.joblib",
                "splits": ROOT / "configs/core_v3.json"},
    "integration_v1": {"raw": ROOT / "data/raw", "truth": ROOT / "evaluation_truth",
                       "model": ROOT / "artifacts/models/temporary.joblib",
                       "splits": ROOT / "configs/integration_batch.json"},
}
# Cylinder localizer (train-localizer): names the affected cylinder of each active alert
# from the vibration signature and the cam phase reference. Optional.
LOCALIZER_PATH = ROOT / "artifacts/models/cylinder_localizer.joblib"
# Independent cross-checks for runs with contract 1.2 sensors (calibrate-crosscheck).
CRANK_PATH = ROOT / "artifacts/models/crank_monitor.json"
THERMAL_PATH = ROOT / "artifacts/models/thermal_monitor.json"
PRESETS_PATH = ROOT / "configs/demo_presets.json"
HOLDOUT_DATASET = "integration_v1"
FEATURES_PATH = ROOT / "data/processed/integration_v1_features.parquet"
REPORT_PATH = ROOT / "reports/generated/temporary_model_report.json"
# Points per displayed trace (min/max per bin, so short spikes survive). Charts are SVG:
# WebGL (Scattergl) traces would compete with the 3D view for the browser's few WebGL
# contexts, and the browser then drops the oldest one, the 3D view's.
DISPLAY_POINTS = 2000
SPECTROGRAM_S = 10.0
TICK_S = 0.5
STATUS_SHOWN = {"weak": "partial"}  # evidence indicator status as displayed
CONTEXT_SIGNALS = ("rpm", "load", "throttle")
SLOW_SIGNALS = ("coolant_temp", "short_term_fuel_trim", "long_term_fuel_trim")
CLASS_ORDER = ["normal", "misfire", "knock_like", "valve_clearance_like", "uncertain"]
# Same decision colors as theme.CLASS_META and the 3D view.
FAULT_COLORS = {fault: theme.CLASS_META[fault]["color"]
                for fault in ("misfire", "knock_like", "valve_clearance_like")}
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
SIGNAL_COLORS = {"sound": "#0A84FF", "vibration": "#30D158", "rpm": "#FF453A"}
SIGNAL_TAGS = {"sound": "blue", "vibration": "green", "rpm": "red"}
SIGNAL_NAMES = {"sound": "Sound", "vibration": "Vibration", "rpm": "RPM"}
WAVE_ROWS = {"sound": [1], "vibration": [2, 3]}  # rows of the waveform figure
# Fault-injection lab: healthy core_v1 recordings modified on request, replayed unchanged.
LAB_DATASET = "core_v1"
LAB_COLOR = "#FF9F0A"
LAB_FAULT_NAMES = {"none": "No fault", "misfire": "Misfire", "knock_like": "Knock bursts"}
LAB_NOISE = [1.0, 1.5, 2.0, 3.0, 5.0]  # x recorded noise; 2x is the challenge-run level
LAB_SENSORS = {"audio": "Microphone (audio)", "vibration": "Accelerometer (vibration)",
               "rpm": "RPM telemetry"}
STAGE_COLORS = dict(zip(STAGES, ("#30D158", "#FF9F0A", "#FF453A", "#8E8E93"), strict=True))
# Engine profiles (optional): selecting one turns on saving to that engine's history.
NEW_PROFILE = "new"
OTHER = "Other (not in catalog)…"
MANUAL_SPECS = {"cylinders": 4, "strokes": 4, "displacement_l": 1.6, "firing_order": [1, 3, 4, 2],
                "rpm_min": 800, "rpm_max": 6000, "load_min": 0.1, "load_max": 0.9}
SPEC_SOURCE_NAMES = {"catalog": "from the engine catalog",
                     "catalog_edited": "from the catalog, edited by hand",
                     "manual": "entered by hand"}

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
def load_crosschecks() -> tuple[CrankCalibration | None, ThermalCalibration | None]:
    crank = CrankCalibration.load(CRANK_PATH) if CRANK_PATH.is_file() else None
    thermal = ThermalCalibration.load(THERMAL_PATH) if THERMAL_PATH.is_file() else None
    return crank, thermal


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


def valid_telemetry(bundle) -> dict[str, pd.DataFrame]:
    """Valid telemetry rows per signal, in time order (the run file is sorted by t_s)."""
    tel = bundle.telemetry[bundle.telemetry["valid"] == 1]
    return {signal: rows.reset_index(drop=True) for signal, rows in tel.groupby("signal")}


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
    crank, thermal = load_crosschecks()
    st.session_state.update(dataset=dataset, run_id=run_id, held_out=held_out, model=model,
                            lab=lab, lab_log=lab_log, bundle=bundle, packets=run_packets,
                            telemetry=valid_telemetry(bundle),
                            engine=ReplayEngine(model, config, bundle.metadata,
                                                localizer=localizer, crank=crank,
                                                thermal=thermal),
                            next_packet=0, records=[], stream_t=0.0, playing=False,
                            saved_history_run=None)


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
        # The thumb shows only the RPM (a long label ran off the sidebar); the profile and
        # run are in the help text and the lab log.
        base = st.select_slider("Engine RPM (base recording)", options=list(bases.run_id),
                                value=default, format_func=lambda r: labels[r].split(" · ")[0],
                                help="Base recordings, slowest to fastest: "
                                     + "; ".join(labels[r] for r in bases.run_id))
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


@st.cache_resource
def engine_catalog():
    return load_catalog()


def parse_firing_order(text: str):
    """'1-3-4-2' (or commas/spaces) -> [1, 3, 4, 2]; unparseable text is returned for validation."""
    try:
        return [int(p) for p in text.replace(",", " ").replace("-", " ").split()]
    except ValueError:
        return text


def spec_inputs(defaults: dict, key: str) -> dict:
    """Spec fields prefilled from `defaults`; returns the entered (unvalidated) specs."""
    left, right = st.columns(2)
    return {
        "cylinders": left.number_input("Cylinders", value=int(defaults["cylinders"]), step=1,
                                       key=f"{key}_cylinders"),
        "strokes": right.selectbox("Strokes", [4, 2], index=[4, 2].index(defaults["strokes"]),
                                   key=f"{key}_strokes"),
        "displacement_l": left.number_input("Displacement (L)",
                                            value=float(defaults["displacement_l"]), step=0.1,
                                            format="%.2f", key=f"{key}_displacement"),
        "firing_order": parse_firing_order(right.text_input(
            "Firing order", "-".join(map(str, defaults["firing_order"])), key=f"{key}_order")),
        "rpm_min": left.number_input("RPM min", value=float(defaults["rpm_min"]), step=50.0,
                                     format="%.0f", key=f"{key}_rpm_min"),
        "rpm_max": right.number_input("RPM max", value=float(defaults["rpm_max"]), step=50.0,
                                      format="%.0f", key=f"{key}_rpm_max"),
        "load_min": left.number_input("Load min (0-1)", value=float(defaults["load_min"]),
                                      step=0.05, format="%.2f", key=f"{key}_load_min"),
        "load_max": right.number_input("Load max (0-1)", value=float(defaults["load_max"]),
                                       step=0.05, format="%.2f", key=f"{key}_load_max"),
    }


def specs_valid(specs: dict, warn: bool = True) -> bool:
    """Show validation problems (and the reliability warning); True if the specs are valid."""
    problems = validate_specs(specs)
    if problems:
        st.error("Check the specs:\n" + "\n".join(f"- {p}" for p in problems))
    elif warn and reliability_warning(specs):
        st.warning(reliability_warning(specs), icon=":material/warning:")
    return not problems


def specs_text(profile: Profile) -> str:
    s = profile.specs
    return (f"{s['cylinders']} cyl · {s['strokes']}-stroke · {s['displacement_l']:g} L · "
            f"firing order {'-'.join(map(str, s['firing_order']))} · "
            f"{s['rpm_min']:,.0f}-{s['rpm_max']:,.0f} rpm · load {s['load_min']:g}-"
            f"{s['load_max']:g} · specs {SPEC_SOURCE_NAMES[profile.spec_source]}")


def new_profile_controls() -> None:
    """Identify an engine (make, model, serial); specs come from the catalog or by hand."""
    catalog = engine_catalog()
    make = st.selectbox("Make", sorted({e.make for e in catalog}) + [OTHER],
                        key="new_profile_make")
    if make == OTHER:
        make = st.text_input("Make", key="new_profile_make_text")
        model_name = st.text_input("Model", key="new_profile_model_text")
    else:
        model_name = st.selectbox("Model", [e.model for e in catalog if e.make == make] + [OTHER],
                                  key="new_profile_model")
        if model_name == OTHER:
            model_name = st.text_input("Model (not in catalog)", key="new_profile_model_text")
    serial = st.text_input("Serial / ID", key="new_profile_serial",
                           placeholder="e.g. engine number or VIN")
    nickname = st.text_input("Nickname (optional)", key="new_profile_nickname")
    entry = lookup(catalog, make, model_name)
    st.caption("Specs found in the engine catalog; edit them if this engine differs."
               if entry else "Not in the catalog: enter the specs by hand.")
    specs = spec_inputs(entry.engine if entry else MANUAL_SPECS,
                        key=f"new_specs_{entry.make}_{entry.model}" if entry else "new_specs")
    valid = specs_valid(specs)
    if st.button("Create profile", type="primary", width="stretch", disabled=not valid):
        try:
            created = store.create_profile(make, model_name, serial, specs,
                                           spec_source(entry, specs), nickname)
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state.pending_profile = created.id
            st.rerun()


def profile_controls() -> Profile | None:
    """Optional engine profile. None (the default) means nothing is saved."""
    state = st.session_state
    if "pending_profile" in state:  # just created: select it before the widget is drawn
        state.profile_choice = state.pop("pending_profile")
    profiles = {p.id: p for p in store.list_profiles()}
    if state.get("profile_choice") not in (None, NEW_PROFILE, *profiles):
        state.profile_choice = None
    choice = st.selectbox(
        "Engine", [None, *profiles, NEW_PROFILE], key="profile_choice",
        format_func=lambda c: ("None (history off)" if c is None else
                               "+ New engine profile" if c == NEW_PROFILE else profiles[c].label),
        help="Optional. Pick or create an engine to save each analysis to its history.")
    if choice == NEW_PROFILE:
        new_profile_controls()
        return None
    if choice is None:
        st.caption("No engine selected: analyses are not saved.")
        return None
    selected = profiles[choice]
    st.caption(specs_text(selected))
    if reliability_warning(selected.specs):
        st.warning(reliability_warning(selected.specs), icon=":material/warning:")
    with st.expander("Edit specs"):
        specs = spec_inputs(selected.specs, key=f"edit_specs_{selected.id}")
        valid = specs_valid(specs, warn=False)
        if st.button("Save specs", width="stretch", disabled=not valid,
                     key=f"save_specs_{selected.id}"):
            entry = lookup(engine_catalog(), selected.make, selected.model)
            store.update_specs(selected.id, specs, spec_source(entry, specs))
            st.rerun()
        st.caption("Saved runs keep the specs they were analysed with.")
    return selected


store = HistoryStore()  # nothing is written until a profile is created


def keep_choice(key: str) -> None:
    """Segmented controls can be clicked off; keep the last choice instead of none."""
    state = st.session_state
    if state[key] is None:
        state[key] = state[f"_{key}_last"]
    else:
        state[f"_{key}_last"] = state[key]


st.session_state.setdefault("_mode_last", "Replay")
st.session_state.setdefault("_speed_last", 1.0)

with st.sidebar:
    st.html(f'<div class="sidebar-brand">{theme.wordmark_html("small")}'
            '<span class="sub">EngineSense · diagnostics</span></div>')
    modes = ["Replay"] + (["Fault lab"] if lab_available() else [])
    mode = st.segmented_control("Mode", modes, default="Replay", key="mode",
                                on_change=keep_choice, args=("mode",)) or "Replay"
    lab_mode = mode == "Fault lab"
    if lab_mode:
        st.header("Fault-injection lab",
                  help="A healthy recording with a fault added on request, then replayed.")
        lab_controls()
    else:
        st.header("Scenario", help="Each scenario replays a recorded held-out run. Abnormal "
                  "vibration is shown as valve-train impacts, the fault class the model was "
                  "trained on.")
        if scenario_options:
            st.segmented_control("Scenario", scenario_options, format_func=SCENARIOS.get,
                                 key="scenario", on_change=on_scenario_change,
                                 label_visibility="collapsed")
        preset_name = st.selectbox("Demo preset", preset_names, key="preset_select",
                                   on_change=on_preset_change)
        preset = next((p for p in presets if p["name"] == preset_name), None)
        if preset and st.session_state.get("applied_preset") != preset_name:
            start_run(preset["dataset"], preset["run_id"])
            if preset.get("start_s"):
                advance_to(preset["start_s"])
            st.session_state.applied_preset = preset_name
        if preset:
            st.caption(preset.get("note", ""))

        st.header("Run selection")
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
    if "telemetry" not in state:  # session started before this key existed
        state.telemetry = valid_telemetry(bundle)
    issues = [] if lab_mode else load_bundle(str(DATASETS[dataset]["raw"]), run_id)[1]
    duration = bundle.metadata["duration_s"]

    st.header("Playback")
    speed = st.segmented_control("Speed", [0.5, 1.0, 2.0, 4.0], default=1.0,
                                 format_func=lambda v: f"{v:g}×", key="speed",
                                 on_change=keep_choice, args=("speed",)) or 1.0

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
    show_truth = not lab_mode and st.toggle(
        "Show evaluation truth overlay", value=False,
        help="Simulation truth for review only; never used by the model.")
    st.caption(f"Model {model.model_version} · features {model.feature_version} · "
               f"packets {config['runtime']['packet_s'] * 1000:.0f} ms · "
               f"window {config['analysis']['window_s']:g} s / "
               f"step {config['analysis']['step_s']:g} s")
    if issues:
        st.error("Run failed validation:\n" + "\n".join(issues))

    st.header("Engine profile")
    profile = profile_controls()

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
    rpm = received_rows("rpm", t_end)
    rpm = rpm.iloc[int(rpm["t_s"].searchsorted(t_end - window_s, side="left")):]
    firing_hz = _firing_rate_hz(record["context"]["rpm"], config["engine"])
    return build_evidence(fault, waves, rates, t_end, window_s, rpm["t_s"].to_numpy(),
                          rpm["value"].to_numpy(), firing_hz, feature_z(record))


def add_bands(fig: go.Figure, bands: list[tuple[float, float, str, float, int, int]]) -> None:
    """Shaded x-spans (x0, x1, color, opacity, row, col) added in one layout update.

    fig.add_vrect re-validates every existing shape on each call, which made a refresh
    with dozens of highlighted segments take ~0.5 s; one update keeps it a few ms.
    """
    if not bands:
        return
    shapes = list(fig.layout.shapes)
    for x0, x1, color, opacity, row, col in bands:
        subplot = fig.get_subplot(row, col)
        xref = subplot.xaxis.plotly_name.replace("axis", "")
        yref = subplot.yaxis.plotly_name.replace("axis", "")
        shapes.append({"type": "rect", "x0": x0, "x1": x1, "xref": xref, "y0": 0, "y1": 1,
                       "yref": f"{yref} domain", "fillcolor": color, "opacity": opacity,
                       "line": {"width": 0}, "layer": "below"})
    fig.update_layout(shapes=shapes)


def evidence_figure(panel: EvidencePanel) -> go.Figure:
    window_s = config["analysis"]["window_s"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[f"{SIGNAL_NAMES[i.signal]}: {i.trace_label}"
                                        for i in panel.indicators])
    for row, ind in enumerate(panel.indicators, start=1):
        x, y = sensors.peak_decimate(ind.trace_t, ind.trace_y, DISPLAY_POINTS)
        fig.add_trace(go.Scatter(x=x, y=y, name=ind.trace_label,
                                 line={"width": 1, "color": "#555"}), row=row, col=1)
    add_bands(fig, [(a, b, SIGNAL_COLORS[ind.signal], 0.3, row, 1)
                    for row, ind in enumerate(panel.indicators, start=1)
                    for a, b in ind.segments])
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
    colors = [FAULT_COLORS.get(fault, "#888") if named and k == best else "#636366"
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
                        f"{ind.claim} — **{STATUS_SHOWN.get(ind.status, ind.status)}**{count}  \n"
                        f"<small>{ind.detail}</small>", unsafe_allow_html=True)
        st.markdown(f"**Assessment:** {panel.assessment}")
        st.plotly_chart(evidence_figure(panel), width="stretch", key=f"evidence_{panel.fault}")
        st.caption("Indicators compare this window with the model's normal reference for the same "
                   "operating range. They show what changed; they do not prove a cause.")


def crank_panel(records, latest, t_now: float) -> None:
    """Independent crank-speed misfire path and the simulated ECU monitor."""
    crank = (latest or {}).get("crank")
    if crank is None:
        st.info("This recording has no crank trigger wheel (data contract 1.2, core_v3), or "
                "the crank calibration is missing (`engine-diagnostics calibrate-crosscheck`).")
        return
    if crank.get("status") == "unavailable":
        st.info("Crank wheel signal unavailable in this window.")
        return
    summary = sensors.obd_summary(records, crank)
    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.plotly_chart(sensors.crank_speed_figure(crank), width="stretch", key="crank_speed")
    with right:
        st.plotly_chart(sensors.roughness_figure(crank), width="stretch", key="crank_rough")
        if crank.get("alert"):
            st.error(f"Crank speed: misfire on cylinder {crank['cylinder']} "
                     "(independent of the sound/vibration model)", icon=":material/speed:")
        elif crank.get("phase") != "cam_sync":
            st.warning("No cam phase in this window: the wheel alone cannot name cylinders.")
        else:
            st.success("Crank speed: every cylinder pulls its weight", icon=":material/speed:")
    with st.container(border=True):
        st.markdown("**Benchmark: the ECU's own misfire monitor (simulated OBD-II)**")
        st.markdown(sensors.comparison_text(summary, t_now))
        st.dataframe(sensors.obd_table(summary), hide_index=True)
        st.caption(
            "A production ECU counts misfiring strokes from crankshaft speed in 200-revolution "
            "blocks (type A, catalyst damage: here 5 % of strokes) and 1000-revolution blocks "
            "(type B, emissions: 1.5 %) and sets P030N for one cylinder. It cannot see knock or "
            "valve faults. Thresholds were calibrated on healthy training engines; the 5 % type "
            "A rate is at the strict end of real calibrations, which favours the ECU. "
            "Synthetic data.")


def thermal_panel(latest, t_now: float) -> None:
    """Per-cylinder EGT/CHT: thermal-camera view, trends, and the thermal cross-check."""
    thermal = (latest or {}).get("thermal")
    if thermal is None:
        st.info("This recording has no per-cylinder exhaust gas / head temperatures (data "
                "contract 1.2, core_v3), or the thermal calibration is missing.")
        return
    left, right = st.columns([3, 2], gap="medium")
    with left:
        # theme=None keeps the inferno scale (Streamlit's theme would replace it), so
        # the colors match the 3D view's exhaust runners.
        st.plotly_chart(sensors.thermal_camera_figure(
            thermal, f"Head and exhaust temperatures at {t_now:.1f} s"),
            width="stretch", key="thermal_camera", theme=None)
        st.caption("Drawn from the eight thermocouple readings on an engine outline (top "
                   "view: exhaust ports above, head below), not a camera image.")
    with right:
        st.markdown("**Thermal cross-check**")
        for text in sensors.thermal_lines(thermal):
            st.markdown(f"- {text}")
        st.caption("Each cylinder is compared with the others (operating-point changes cancel) "
                   "and with its own level in the first seconds of the recording. Slow "
                   "(seconds), but independent of the sound/vibration path.")
    st.plotly_chart(sensors.thermal_trend_figure(bundle.telemetry, t_now, duration),
                    width="stretch", key="thermal_trend")


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


def fault_probability(record) -> float:
    """Calibrated probability of the most likely fault (NaN if the model has none)."""
    faults = [p for c, p in record.get("probabilities", {}).items() if c != "normal"]
    return max(faults) if faults else float("nan")


def run_health(records) -> HealthTimeline:
    return health_timeline(
        [r["t_end_s"] for r in records],
        pd.DataFrame([feature_z(r) if r["quality"] == "ok" else {} for r in records]),
        [r["anomaly_score"] for r in records], model.anomaly_threshold,
        [bool(active_alerts(r)) for r in records], [r["quality"] == "ok" for r in records],
        [fault_probability(r) for r in records])


def save_history_run(engine: Profile, complete: bool) -> int:
    """Save the current replay (decisions so far, results, received raw data) to history.

    The source names how the run was produced; it never includes evaluation truth
    or demo preset/scenario text, which describe the simulated condition.
    """
    state = st.session_state
    records = state.records
    health = run_health(records)
    lab = state.get("lab")
    source = {"mode": "lab" if lab else "recorded", "dataset": state.dataset,
              "recording_run_id": state.run_id, "held_out_engine": state.held_out,
              "config": "configs/mvp.json", "model_version": model.model_version,
              "feature_version": model.feature_version,
              # The user's own lab settings (shown on screen), not evaluation truth.
              "lab_injection": dataclasses.asdict(lab) if lab else None}
    summary = {**summarize_run(records, health.stage, health.summary(), health.warning_leads()),
               "anomaly_threshold": model.anomaly_threshold}
    run_id = store.save_run(engine.id, records, source=source, summary=summary,
                            feature_names=list(model.feature_names), duration_s=duration,
                            complete=complete, raw=raw_from_bundle(state.bundle, state.stream_t),
                            stages=health.stage)
    state.saved_history_run = run_id
    state.history_notice = f"Saved run #{run_id} to the history of {engine.label}"
    return run_id


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
    # The unit sits on the ticks ("5 s"), so the legend below has no axis title to cover.
    fig.update_xaxes(range=[0, duration], ticksuffix=" s", row=2, col=1)
    fig.update_layout(height=470, bargap=0, margin={"t": 50, "b": 130},
                      title="Health timeline vs healthy baseline: deterioration demonstration "
                            "(simulated data)",
                      legend={"orientation": "h", "y": -0.2, "yanchor": "top", "x": 0})
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
    """One row per fault: detections and alerts."""
    timeline = go.Figure()
    faults = model.fault_classes
    for fault in faults:
        rows = [r for r in records if "fault_decisions" in r]
        hits = [r["t_end_s"] for r in rows if r["fault_decisions"][fault] == "fault"]
        timeline.add_trace(go.Scatter(
            x=hits, y=[fault] * len(hits), mode="markers", name=f"{fault} detected",
            marker={"size": 8, "color": FAULT_COLORS[fault]}, showlegend=False))
        alerted = [r["t_end_s"] for r in records if fault in r.get("alerts", [])]
        timeline.add_trace(go.Scatter(
            x=alerted, y=[fault] * len(alerted), mode="markers", showlegend=False,
            marker={"symbol": "square", "size": 14, "opacity": 0.3,
                    "color": FAULT_COLORS[fault]}, name=f"{fault} alert"))
    timeline.update_layout(height=260, margin={"t": 40, "b": 20},
                           title="Per-fault timeline (dot detected, square active alert)",
                           yaxis={"categoryorder": "array", "categoryarray": list(faults)})
    return timeline


def received_rows(signal: str, t_now: float) -> pd.DataFrame:
    """Valid rows of one telemetry signal received before t_now (binary search, no scan)."""
    rows = st.session_state.telemetry.get(signal)
    if rows is None:
        return bundle.telemetry.iloc[:0]
    return rows.iloc[:int(rows["t_s"].searchsorted(t_now, side="left"))]


def freshness_info(signal: str, t_now: float) -> tuple[str, float | None, pd.DataFrame]:
    """Status ('fresh', 'STALE', 'no data'), age in s, and the valid rows received so far."""
    rows = received_rows(signal, t_now)
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
    """Readable class name (e.g. knock_like -> Knock) for display only."""
    return explain.names(class_name)


def live_view():
    rendered = time.perf_counter()
    state = st.session_state
    if state.playing:
        advance_to(min((time.monotonic() - state.anchor) * speed, duration))
        if state.next_packet >= len(run_packets):
            state.playing = False
            st.rerun()  # the sidebar's Pause button turns back into Play and ticking stops
    if (profile and state.records and state.next_packet >= len(run_packets)
            and state.get("saved_history_run") is None):
        save_history_run(profile, complete=True)  # replay finished: save it once
        st.rerun()  # refresh the history panel outside this fragment
    t_now = state.stream_t
    records = state.records
    latest = records[-1] if records else None
    multi = hasattr(model, "fault_classes")  # v2: one detector and alert per fault
    has_probability = bool(latest and "probability" in latest)
    confirm = getattr(model, "confirm_updates", config["runtime"]["confirm_fault_updates"])

    # Sections whose element count changes during playback or between models (alert
    # banners, per-fault probabilities, lab response, evidence panels) each live in one
    # fixed container, so elements after them keep their place in the page and the 3D
    # view is not remounted (which would rebuild the scene, reset its camera and open a
    # new WebGL context) whenever an alert appears or clears or the model type changes.
    active = active_alerts(latest)
    with st.container(key="alerts"):
        if active:
            for fault in active:
                n = model.confirm_updates[fault] if multi else confirm
                st.error(f"ALERT: {label(fault)} detected{location_phrase(latest, fault)} — "
                         f"confirmed by {n} consistent updates"
                         f"{sensors.confirmation_text(latest, fault)}",
                         icon=":material/warning:")
        else:
            st.success("No active alert",
                       icon=":material/check_circle:")

    # Segmented tabs below the alert: Overview holds the readout, the 3D view and the
    # diagnosis; the other views each get their own tab. Labels stay fixed so the
    # selected tab and the 3D view survive every rerun.
    overview_tab, signals_tab, frequency_tab, telemetry_tab, timeline_tab = st.tabs([
        ":material/grid_view: Overview", ":material/graphic_eq: Signals",
        ":material/equalizer: Frequencies", ":material/speed: Engine data",
        ":material/timeline: History"])
    overview = overview_tab.container()

    # Decision: every fault reported now (same faults as the alert banners), else this
    # window's class. A note marks faults held by their alert but not detected in this
    # window, or detected but not confirmed yet.
    reported = explain.reported_faults(latest)
    readout = overview.container(key="readout")
    cols = readout.columns(4) + readout.columns(3)  # four, then three per row
    cols[0].metric("Replay time", f"{t_now:.1f} s")
    cols[1].metric("Decision",
                   explain.reported_names(reported) if reported
                   else label(latest["accepted"]) if latest else "Warming up",
                   help="Faults the model reports now: confirmed alerts plus faults detected "
                        "in this window. Without a fault: this window's class after the "
                        "confidence and anomaly checks.")
    if multi:
        cols[2].metric("Detected this window", len(latest["detected"]) if latest else "-",
                       help="Faults whose own calibrated detector fired in this "
                            f"{config['analysis']['window_s']:g} s window. "
                            "Each fault has its own detector and threshold, so several can "
                            "be detected together; an alert can stay active through a "
                            "window where its fault is not detected.")
    elif has_probability:
        cols[2].metric("Calibrated confidence", f"{latest['probability']:.2f}",
                       help=f"Accepted only if ≥ {model.probability_min:.2f}; calibrated on "
                            "separate calibration engines (synthetic).")
    else:
        cols[2].metric("Decision margin", f"{latest['margin']:.2f}" if latest else "-")
    cols[3].metric("Anomaly score", f"{latest['anomaly_score']:.2f}" if latest else "-",
                   help=f"How unusual this window is compared with normal running. "
                        f"Threshold {model.anomaly_threshold:.2f}.")
    cols[4].metric("Signal quality",
                   ("Good" if latest["quality"] == "ok"
                    else latest["quality"].replace("_", " ").capitalize()) if latest else "-")
    cols[5].metric("Response time",
                   f"{latest['processing_ms']:.0f} / {state.get('render_ms', 0):.0f} ms"
                   if latest else "-",
                   help="Model processing per decision / dashboard update time (server).")

    with overview, st.container(key="fault_probabilities"):  # v2 only: fixed slot (see above)
        if multi:
            fault_cols = st.columns(len(model.fault_classes))
            for col, fault in zip(fault_cols, model.fault_classes, strict=True):
                col.metric(f"P({fault})",
                           f"{latest['probabilities'][fault]:.2f}" if has_probability else "-",
                           help=f"Detected at ≥ {model.thresholds[fault]:.2f}; calibrated on "
                                "separate calibration engines (synthetic).")

    with overview, st.container(key="lab_response"):
        if lab_log:
            lab_response(records, t_now)

    panels = [p for p in (alert_evidence(records, f) for f in active) if p]
    cards = telemetry_cards(t_now)
    view_col, diag_col = overview.columns([3, 2], gap="medium")
    with view_col, st.container(key="viewport"):
        engine3d.render(engine3d.scene_state(
            latest, model, t_now=t_now, playing=bool(state.playing),
            # At the end the engine keeps running on the final decision (browser-side
            # animation only; nothing reruns on the server).
            finished=bool(latest) and state.next_packet >= len(run_packets),
            rpm=next(c["value"] for c in cards if c["label"] == "RPM"),
            firing_order=config["engine"]["firing_order"],
            truth=load_truth_cached(dataset, run_id) if show_truth else None))
    with diag_col:
        st.html(theme.status_card_html(explain.diagnose(latest, model, confirm)))

    # Evidence below the engine view: the 3D view and diagnosis keep their place on screen
    # when an alert appears, and the evidence reads as the detail behind them.
    with overview, st.container(key="evidence"):
        if panels:
            st.subheader("Evidence for active alerts")
            for col, panel in zip(st.columns(len(panels)), panels, strict=True):
                with col:
                    render_evidence(panel,
                                    (latest.get("localization") or {}).get(panel.fault))

    telemetry_tab.html(theme.telemetry_cards_html(cards))

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
        shown_t, shown_x = sensors.peak_decimate(t, x, DISPLAY_POINTS)
        waves.add_trace(go.Scatter(x=shown_t, y=shown_x, name=m,
                                   line={"width": 1, "color": color}), row=row, col=1)
        if x.size >= config["analysis"]["welch_segment_samples"]:
            clean = np.nan_to_num(x.astype(np.float64))
            f, p = welch(clean, fs=fs, window="hann",
                         nperseg=config["analysis"]["welch_segment_samples"])
            psd.add_trace(go.Scatter(x=f / 1000, y=10 * np.log10(p + 1e-20), name=m,
                                     line={"width": 1.5, "color": color}))
            if m == "vibration":
                env = _band_envelope(clean, fs, HIGH_ENVELOPE_HZ)
                env_t, env_y = sensors.peak_decimate(t, env, DISPLAY_POINTS)
                waves.add_trace(go.Scatter(x=env_t, y=env_y, name="envelope",
                                           line={"width": 1,
                                                 "color": theme.MODALITY_COLORS["envelope"]}),
                                row=3, col=1)
    # Link: same segments and colors as the evidence panels.
    add_bands(waves, [(a, b, SIGNAL_COLORS[ind.signal], 0.2, row, 1)
                      for panel in panels for ind in panel.indicators
                      for row in WAVE_ROWS.get(ind.signal, []) for a, b in ind.segments])
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
                                    colorscale=theme.INFERNO, colorbar={"title": "dB"}))
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
    for k, signal in enumerate(signals):
        rows = received_rows(signal, t_now)
        tele.add_trace(go.Scatter(x=rows["t_s"], y=rows["value"], name=signal,
                                  line={"width": 1.5, "color": theme.TOKENS["cyan"]}),
                       row=k // 3 + 1, col=k % 3 + 1)
    add_bands(tele, [(a, b, SIGNAL_COLORS["rpm"], 0.3, 1, 1) for panel in panels
                     for a, b in next(i for i in panel.indicators if i.signal == "rpm").segments])
    tele.update_xaxes(range=[0, duration])
    tele.update_layout(height=400, showlegend=False, margin={"t": 50, "b": 20})
    telemetry_tab.plotly_chart(tele, width="stretch")
    # Per-cylinder views from the crank wheel and exhaust temperatures (core_v3 runs).
    with telemetry_tab:
        st.subheader("Crank speed")
        crank_panel(records, latest, t_now)
        st.subheader("Exhaust temperatures")
        thermal_panel(latest, t_now)

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
        leads = health.warning_leads()
        if leads:
            timeline_tab.markdown("**Early warning:** developing anomaly flagged "
                                  + ", ".join(f"{lead:.1f} s" for lead in leads)
                                  + " before the fault alert was confirmed.")
        timeline_tab.caption(
            "Each line is a smoothed (2 s, causal) deviation from the healthy baseline for the "
            "same operating range, divided by its developing threshold: firing-rhythm "
            "irregularity and high-frequency impulsiveness as robust z (threshold 4), the "
            "model's anomaly score (its own threshold), and the model's calibrated fault "
            "likelihood (warning bar 0.6, below the alert's 0.7). Developing anomaly = a line "
            "stays at or above 1 without a confirmed alert; persistent fault = the model's "
            "confirmed alert. Thresholds were set on tuning engines. On held-out test engines "
            "the developing stage came before the alert in misfire 6/8, knock 3/8 and valve "
            "3/8 runs (2 of 5 degradation runs, 3.5-5 s ahead). Progression is simulated: a "
            "deterioration demonstration of the current state, with no forecast or "
            "remaining-useful-life claim.")

    lower_left, lower_right = timeline_tab.columns(2)
    if latest and latest["quality"] == "ok":
        table, matched = evidence_table(latest)
        lower_left.subheader("Biggest differences from normal")
        lower_left.dataframe(table, hide_index=True)
        lower_left.caption(
            ("Reference: training normal windows in the same RPM/load bin. " if matched else
             "Reference: all training normal windows. ")
            + "Each row shows how far this window is from normal.")
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
        lower_right.caption("5 s rolling mean of the calibrated valve-clearance probability "
                            "(accelerated simulation).")
    state.render_ms = 1000 * (time.perf_counter() - rendered)


# Tick only while playing. A paused view has nothing new to show; re-rendering it every
# TICK_S in every open tab kept the server busy, so clicks such as Pause queued for seconds.
st.fragment(run_every=TICK_S if st.session_state.playing else None)(live_view)()


def history_table(runs) -> pd.DataFrame:
    """One row per saved run (newest first) with its predicted fault and key scores."""
    rows = []
    for run in runs:
        s, src = run.summary, run.source
        faults = [p for c, p in (s.get("max_probabilities") or {}).items() if c != "normal"]
        peak = s.get("peak_anomaly_score")
        rows.append({
            "run": f"#{run.id}",
            "saved (UTC)": run.created_utc[:16].replace("T", " "),
            "source": (f"lab · {src.get('recording_run_id')}" if src.get("mode") == "lab"
                       else f"{src.get('dataset')} · {src.get('recording_run_id')}"),
            "analysed": f"{run.analyzed_s:.1f} / {run.duration_s:g} s"
                        + ("" if run.complete else " (partial)"),
            "predicted fault": label(s["predicted_fault"]),
            "alerts": ", ".join(f"{label(a['fault'])} {a['onset_s']:.1f} s"
                                for a in s["alerts"]) or "none",
            "max P(fault)": round(max(faults), 2) if faults else None,
            "peak anomaly": round(peak, 2) if peak is not None else None,
            "final health": s.get("final_health_stage") or "-",
            "model": s.get("model_version"),
            "raw data": "kept" if run.raw_available else "pruned",
        })
    return pd.DataFrame(rows)


def engine_history_panel(engine: Profile) -> None:
    """Saved runs and notes of the selected engine, and a form to add a note."""
    state = st.session_state
    notice = state.pop("history_notice", None)
    if notice:
        st.toast(notice, icon=":material/save:")
    st.subheader(f"Engine history · {engine.label}")
    if reliability_warning(engine.specs):
        st.warning(reliability_warning(engine.specs), icon=":material/warning:")
    runs = store.list_runs(engine.id)
    saved = state.get("saved_history_run")
    info_col, save_col = st.columns([3, 1], vertical_alignment="center")
    info_col.caption(
        (f"This replay is saved as run #{saved}. " if saved else
         "This replay is saved automatically when it reaches the end. ")
        + f"Raw signals are kept for the newest {store.raw_keep} runs of this engine; results "
          "and notes are kept permanently. Recorded replays are synthetic runs, not "
          "measurements of this engine.")
    if save_col.button("Save run so far", width="stretch",
                       disabled=not state.records or saved is not None,
                       help="Save the decisions and signals received up to now."):
        save_history_run(engine, complete=False)
        st.rerun()
    runs_col, notes_col = st.columns([3, 2], gap="medium")
    with runs_col:
        st.markdown("**Saved runs** (newest first)")
        if runs:
            st.dataframe(history_table(runs), hide_index=True)
            with st.expander("Decisions of a saved run"):
                pick = st.selectbox("Run", [r.id for r in runs], format_func=lambda r: f"#{r}",
                                    key="history_run_detail")
                decisions = store.decisions(pick)
                st.dataframe(decisions[[c for c in ("t_end_s", "quality", "accepted", "alert",
                                                    "anomaly_score", "health_stage",
                                                    "localization") if c in decisions]],
                             hide_index=True)
        else:
            st.info("No saved runs yet. Play a run to the end, or use Save run so far.")
    with notes_col:
        st.markdown("**Notes**")
        run_ids = [None, *[r.id for r in runs]]
        with st.form("history_note", clear_on_submit=True):
            text = st.text_area("New note", placeholder="e.g. replaced injector 3", height=80)
            about = st.selectbox("About", run_ids,
                                 index=run_ids.index(saved) if saved in run_ids else 0,
                                 format_func=lambda r: "This engine (no specific run)"
                                 if r is None else f"Run #{r}")
            if st.form_submit_button("Add note", width="stretch"):
                try:
                    store.add_note(engine.id, text, about)
                except ValueError as error:
                    st.warning(str(error))
        notes = store.list_notes(engine.id)
        for note in notes:
            with st.container(border=True):
                st.caption(f"{note.created_utc[:16].replace('T', ' ')} UTC · "
                           + (f"run #{note.run_id}" if note.run_id else "engine"))
                st.markdown(note.text)
        if not notes:
            st.caption("No notes yet.")


URGENCY_TITLES = {"stop_now": ":material/dangerous: Stop now",
                  "soon": ":material/build: Soon", "monitor": ":material/visibility: Monitor"}
EVIDENCE_KINDS = {"crank": "crank speed", "thermal": "cylinder temperatures",
                  "fuel": "fuel trims", "spectrum": "sound spectrum"}


def evidence_label(item: str) -> str:
    """Readable name of a context ID (R5, N3, S5-crank, T-misfire, SPEC)."""
    if item == "SPEC":
        return "engine specs"
    if item.startswith("T-"):
        return f"{label(item[2:]) if item[2:] in FAULT_COLORS else item[2:]} trend across runs"
    if item.startswith("S") and "-" in item:
        run, kind = item[1:].split("-", 1)
        return f"run #{run} {EVIDENCE_KINDS.get(kind, kind)}"
    if item.startswith("R"):
        return f"run #{item[1:]}"
    if item.startswith("N"):
        return f"note #{item[1:]}"
    return item


def show_diagnosis(saved: Diagnosis) -> None:
    """Summary, ranked causes with evidence, actions by urgency, data gaps, disclaimer."""
    out = saved.output
    if saved.fallback:
        st.info(f"Basic version. {saved.fallback_reason or ''}", icon=":material/info:")
    st.caption(f"{saved.created_utc[:16].replace('T', ' ')} UTC · {saved.model}"
               + (f" · saved runs up to #{saved.latest_run_id}" if saved.latest_run_id else ""))
    st.markdown(out["summary"])
    st.markdown("**Causes** (ranked by confidence)")
    if not out["causes"]:
        st.caption("No cause is supported by the saved data.")
    for k, cause in enumerate(out["causes"], 1):
        with st.container(border=True):
            kind = "detected by the model" if cause["kind"] == "detected" else "suggested"
            st.markdown(f"**{k}. {cause['title']}** · {kind} · "
                        f"confidence **{cause['confidence']}**")
            st.markdown(cause["reasoning"])
            if cause.get("confidence_notes"):
                st.caption(cause["confidence_notes"])
            st.caption("Evidence: " + ", ".join(
                f"{evidence_label(i)} ({i})" for i in cause["evidence_ids"]))
            if cause.get("unverified_ids"):
                st.caption(":material/warning: Also cited but not found in this engine's "
                           "data, so ignored: " + ", ".join(cause["unverified_ids"]))
    if out.get("dropped_causes"):
        st.warning(f"{len(out['dropped_causes'])} cause(s) were removed because they cited "
                   "evidence that is not in this engine's data.", icon=":material/filter_alt:")
    st.markdown("**Recommended actions**")
    for urgency, title in URGENCY_TITLES.items():
        steps = [a for a in out["actions"] if a["urgency"] == urgency]
        if steps:
            st.markdown(f"{title}\n" + "\n".join(
                f"- {a['step']}" + (f" _(for: {', '.join(a['addresses'])})_"
                                    if a["addresses"] else "") for a in steps))
    if out["data_gaps"]:
        st.markdown("**Data gaps**\n" + "\n".join(f"- {g}" for g in out["data_gaps"]))
    st.caption(DIAGNOSIS_DISCLAIMER)


def diagnosis_panel(engine: Profile) -> None:
    """Diagnose on demand from the engine's history; saved diagnoses can be reopened."""
    state = st.session_state
    key = f"diagnosis_pick_{engine.id}"
    st.caption("Ranks causes and next steps from this engine's saved runs, notes and "
               "signal summaries. Save the current replay first to include it.")
    if st.button("Diagnose", type="primary", key=f"diagnose_{engine.id}",
                 icon=":material/troubleshoot:"):
        with st.spinner("Ranking causes from this engine's history..."):
            try:
                state[f"diagnosis_new_{engine.id}"] = run_diagnosis(store, engine, config).id
            except Exception as error:  # noqa: BLE001 - never show a stack trace in the demo
                st.error(f"Could not diagnose this engine ({type(error).__name__}). Its saved "
                         "history may be incomplete; the rest of the dashboard still works.")
    diagnoses = {d.id: d for d in store.list_diagnoses(engine.id)}
    if not diagnoses:
        st.caption("No diagnosis yet for this engine.")
        return
    new = state.pop(f"diagnosis_new_{engine.id}", None)
    if new in diagnoses or state.get(key) not in diagnoses:
        state[key] = new if new in diagnoses else next(iter(diagnoses))
    pick = st.selectbox(
        "Saved diagnoses", list(diagnoses), key=key,
        format_func=lambda d: f"#{d} · {diagnoses[d].created_utc[:16].replace('T', ' ')} UTC · "
                              + ("basic" if diagnoses[d].fallback else diagnoses[d].model))
    show_diagnosis(diagnoses[pick])


def report_timeline_figure(timeline: dict) -> go.Figure:
    """Report health timeline (same series as the PDF chart)."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2],
                        vertical_spacing=0.06)
    if timeline["probabilities"]:
        for fault, values in timeline["probabilities"].items():
            fig.add_trace(go.Scatter(x=timeline["t"], y=values, name=label(fault),
                                     line={"color": FAULT_COLORS[fault], "width": 2}),
                          row=1, col=1)
        fig.update_yaxes(range=[0, 1.02], title="likelihood", row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=timeline["t"], y=timeline["anomaly_ratio"],
                                 name="anomaly score / threshold"), row=1, col=1)
        fig.add_hline(y=1, line={"dash": "dot"}, row=1, col=1)
        fig.update_yaxes(title="anomaly / threshold", row=1, col=1)
    for stage in dict.fromkeys(timeline["stage"]):
        t = [x for x, s in zip(timeline["t"], timeline["stage"], strict=True) if s == stage]
        fig.add_trace(go.Bar(x=t, y=[1] * len(t), name=stage, width=0.5,
                             marker_color=STAGE_COLORS.get(stage, "#8E8E93")),
                      row=2, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)
    fig.update_xaxes(range=[0, timeline["duration_s"]], title="s", row=2, col=1)
    fig.update_layout(height=340, bargap=0, margin={"t": 20, "b": 20},
                      legend={"orientation": "h", "y": -0.25})
    return fig


def report_likelihoods_figure(likelihoods: dict) -> go.Figure:
    fig = go.Figure(go.Bar(x=likelihoods["values"], y=likelihoods["labels"], orientation="h",
                           marker_color=[FAULT_COLORS.get(k, "#7f7f7f")
                                         for k in likelihoods["keys"]],
                           text=[f"{v:.2f}" for v in likelihoods["values"]]))
    fig.update_layout(height=220, title=likelihoods["title"], xaxis={"range": [0, 1]},
                      yaxis={"autorange": "reversed"}, margin={"t": 40, "b": 20})
    return fig


def report_trends_figure(trends: dict) -> go.Figure:
    fig = go.Figure()
    if trends["max_probabilities"]:
        for fault, values in trends["max_probabilities"].items():
            fig.add_trace(go.Scatter(x=trends["runs"], y=values, name=label(fault),
                                     mode="lines+markers",
                                     line={"color": FAULT_COLORS[fault]}))
        fig.update_yaxes(range=[0, 1.02], title="max likelihood")
    else:
        fig.add_trace(go.Scatter(x=trends["runs"], y=trends["anomaly_ratio"],
                                 mode="lines+markers", name="peak anomaly / threshold"))
        fig.add_hline(y=1, line={"dash": "dot"})
        fig.update_yaxes(title="anomaly / threshold")
    fig.update_layout(height=260, margin={"t": 20, "b": 20},
                      xaxis={"title": "saved run (oldest to newest)", "type": "category"})
    return fig


def report_panel(engine: Profile) -> None:
    """Report preview and PDF export, both drawn from one EngineReport (no Gemini call)."""
    state = st.session_state
    if not st.toggle("Show report", key=f"show_report_{engine.id}",
                     help="Builds the report from this engine's saved history."):
        st.caption("Engine profile and specs, the latest run with its health timeline and "
                   "likelihoods, the latest saved diagnosis, and the run history, as a preview "
                   "and a PDF.")
        return
    try:
        rep = report.build_report(store, engine)
    except Exception as error:  # noqa: BLE001 - never show a stack trace in the demo
        st.error(f"Could not build the report ({type(error).__name__}).")
        return
    # The PDF is rebuilt only when the engine's history changes.
    signature = (engine.id, tuple(r["run"] for r in rep.history),
                 rep.diagnosis and rep.diagnosis["id"], len(rep.notes), str(engine.specs))
    cached = state.get("report_pdf")
    if not cached or cached[0] != signature:
        try:
            state.report_pdf = (signature, report.render_pdf(rep), rep.file_name())
        except Exception as error:  # noqa: BLE001
            state.report_pdf = (signature, None, None)
            st.error(f"Could not create the PDF ({type(error).__name__}); the preview below "
                     "still shows the report.")
    _, pdf_bytes, file_name = state.report_pdf
    st.download_button("Export PDF", data=pdf_bytes or b"", file_name=file_name or "report.pdf",
                       mime="application/pdf", disabled=pdf_bytes is None, type="primary",
                       icon=":material/picture_as_pdf:", key=f"export_pdf_{engine.id}")
    st.markdown(f"#### {rep.title}")
    st.caption(f"Generated {report.utc_text(rep.generated_utc)}. {rep.disclaimer}")

    e = rep.engine
    st.markdown("**1. Engine profile and specs**")
    st.dataframe(pd.DataFrame([("Make", e["make"]), ("Model", e["model"]),
                               ("Serial / ID", e["serial"]), ("Nickname", e["nickname"] or "-"),
                               *e["specs"], ("Specs source", e["spec_source"])],
                              columns=["Field", "Value"]), hide_index=True)
    if e["warning"]:
        st.warning(e["warning"], icon=":material/warning:")

    st.markdown("**2. Latest run**")
    cur = rep.current_run
    if cur is None:
        st.info("No saved runs yet. Play a run to the end (or use Save run so far).")
    else:
        st.dataframe(pd.DataFrame([
            ("History run", f"#{cur['id']}"), ("Source", cur["source"]),
            ("Analysed", cur["analysed"]), ("Predicted fault", cur["predicted_fault"]),
            ("Alerts", cur["alerts"]), ("Final health stage", cur["final_health_stage"]),
            ("Health stages", cur["health_summary"]), ("Early warning", cur["early_warning"]),
            ("Peak anomaly", cur["peak_anomaly"]), ("Cylinder", cur["localization"]),
            ("Model", cur["model_version"])], columns=["Field", "Value"]), hide_index=True)
        if rep.timeline:
            st.plotly_chart(report_timeline_figure(rep.timeline), width="stretch",
                            key="report_timeline")
        if rep.likelihoods:
            st.plotly_chart(report_likelihoods_figure(rep.likelihoods), width="stretch",
                            key="report_likelihoods")

    st.markdown("**3. Latest diagnosis**")
    d = rep.diagnosis
    if d is None:
        st.info("No diagnosis saved yet. Use Diagnose above to add ranked causes and "
                "actions to this report.", icon=":material/troubleshoot:")
    else:
        show_diagnosis(Diagnosis(d["id"], engine.id, d["latest_run_id"], d["created_utc"],
                                 d["model"], d["fallback"], d["fallback_reason"], "",
                                 d["output"]))

    st.markdown("**4. Run history and trends**")
    if rep.history:
        st.dataframe(pd.DataFrame(rep.history, columns=report.HISTORY_COLUMNS),
                     hide_index=True)
    if rep.trends and len(rep.trends["runs"]) > 1:
        st.plotly_chart(report_trends_figure(rep.trends), width="stretch", key="report_trends")
    if rep.notes:
        st.markdown("**5. Notes**\n" + "\n".join(
            f"- {report.utc_text(n['created_utc'])} · "
            + (f"run #{n['run_id']}" if n["run_id"] else "engine") + f": {n['text']}"
            for n in rep.notes))


def engine_tools_panel(engine: Profile) -> None:
    st.subheader(f"Diagnosis and report · {engine.label}")
    causes_tab, report_tab = st.tabs([":material/troubleshoot: Causes",
                                      ":material/description: Report"])
    with causes_tab:
        diagnosis_panel(engine)
    with report_tab:
        report_panel(engine)


if profile:
    engine_history_panel(profile)
    engine_tools_panel(profile)


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
