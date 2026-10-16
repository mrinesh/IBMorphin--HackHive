"""Run-bundle loading and validation against the data contract, using synthetic files."""

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")
scipy_io = pytest.importorskip("scipy.io")

from engine_diagnostics.config import load_config  # noqa: E402
from engine_diagnostics.data import run_io  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DURATION_S = 2


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/mvp.json", ROOT / "schemas/project-config.schema.json")


def _summary(x):
    x64 = x.astype(np.float64).ravel()
    return {"n_samples": x.size, "sum": float(x64.sum()), "sum_sq": float(np.square(x64).sum()),
            "first": float(x64[0]), "last": float(x64[-1])}


def write_bundle(directory: Path, config: dict, *, mutate=None, run_id="run_00001") -> Path:
    """Write a contract-conforming bundle; `mutate(parts)` may alter it before saving."""
    rng = np.random.default_rng(0)
    streams = config["streams"]
    rates = streams["telemetry_rates_hz"]
    waves = {
        name: (0.01 * rng.standard_normal((DURATION_S * streams[name]["sample_rate_hz"], 1)))
        .astype(np.float32)
        for name in run_io.WAVEFORM_STREAMS
    }
    rows = [
        (k / rates[signal], signal, 1.0 + k, unit, 1)
        for signal, unit in run_io.TELEMETRY_UNITS.items()
        for k in range(DURATION_S * rates[signal])
    ]
    rows.sort(key=lambda row: (row[0], row[1]))
    metadata = {
        "schema_version": "1.0", "run_id": run_id, "engine_id": "engine_001",
        "batch_id": "test_v1", "generator_version": "0.1.0", "config_schema_version": "1.0",
        "created_utc": "2026-10-15T00:00:00Z", "synthetic": True, "duration_s": DURATION_S,
        "streams": {
            name: {"fs_hz": streams[name]["sample_rate_hz"], "t0_s": 0,
                   "unit": streams[name]["unit"], "n_samples": waves[name].size,
                   "full_scale": 1.0}
            for name in run_io.WAVEFORM_STREAMS
        },
        "telemetry_signals": {
            signal: {"rate_hz": rates[signal], "unit": unit}
            for signal, unit in run_io.TELEMETRY_UNITS.items()
        },
        "clock": {name: {"offset_s": 0, "drift_ppm": 0}
                  for name in ("audio", "vibration", "telemetry")},
        "waveform_summary": {name: _summary(waves[name]) for name in run_io.WAVEFORM_STREAMS},
    }
    mat = {**waves, "audio_fs_hz": float(streams["audio"]["sample_rate_hz"]),
           "vibration_fs_hz": float(streams["vibration"]["sample_rate_hz"]),
           "audio_t0_s": 0.0, "vibration_t0_s": 0.0}
    parts = {"mat": mat, "rows": rows, "metadata": metadata}
    if mutate:
        mutate(parts)

    run_dir = directory / run_id
    run_dir.mkdir(parents=True)
    scipy_io.savemat(run_dir / "waveforms.mat", parts["mat"], format="5")
    lines = ["t_s,signal,value,unit,valid"] + [
        f"{t:.6f},{signal},{'' if value is None else value},{unit},{valid}"
        for t, signal, value, unit, valid in parts["rows"]
    ]
    (run_dir / "telemetry.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps(parts["metadata"]), encoding="utf-8")
    return run_dir


def issues_for(tmp_path, config, mutate):
    return run_io.validate_run(run_io.load_run(write_bundle(tmp_path, config, mutate=mutate)),
                               config)


def test_valid_bundle_round_trips(tmp_path, config):
    run_dir = write_bundle(tmp_path, config)
    bundle = run_io.load_run(run_dir)
    assert run_io.validate_run(bundle, config) == []
    assert bundle.audio.dtype == np.float32 and bundle.audio.ndim == 1
    assert bundle.vibration.size == DURATION_S * 25600
    assert bundle.audio_fs_hz == 24000.0
    assert set(bundle.telemetry["signal"]) == set(run_io.TELEMETRY_UNITS)


def test_row_vector_is_rejected(tmp_path, config):
    def mutate(parts):
        parts["mat"]["audio"] = parts["mat"]["audio"].T
    with pytest.raises(ValueError, match="column vector"):
        run_io.load_run(write_bundle(tmp_path, config, mutate=mutate))


def test_double_waveform_is_rejected(tmp_path, config):
    def mutate(parts):
        parts["mat"]["vibration"] = parts["mat"]["vibration"].astype(np.float64)
    with pytest.raises(ValueError, match="float32"):
        run_io.load_run(write_bundle(tmp_path, config, mutate=mutate))


@pytest.mark.parametrize(
    ("description", "mutate", "expected"),
    [
        ("short audio", lambda p: p["mat"].update(audio=p["mat"]["audio"][:-10]), "samples"),
        ("too many NaN", lambda p: p["mat"]["vibration"].__setitem__(slice(0, 6000), np.nan),
         "non-finite"),
        ("unrecorded NaN", lambda p: p["mat"]["vibration"].__setitem__((5, 0), np.nan),
         "missing samples"),
        ("infinite", lambda p: p["mat"]["audio"].__setitem__((5, 0), np.inf), "infinite"),
        ("changed value", lambda p: p["mat"]["audio"].__setitem__((100, 0), 0.5), "sum"),
        ("missing signal", lambda p: p.update(rows=[r for r in p["rows"] if r[1] != "load"]),
         "'load' missing"),
        ("wrong unit", lambda p: p.update(rows=[
            (t, s, v, "kPa" if s == "load" else u, ok) for t, s, v, u, ok in p["rows"]]), "unit"),
        ("unsorted", lambda p: p["rows"].reverse(), "sorted"),
        ("valid but empty", lambda p: p["rows"].__setitem__(0, (*p["rows"][0][:2], None,
                                                                 *p["rows"][0][3:])), "empty"),
        ("wrong rate", lambda p: p["mat"].update(audio_fs_hz=48000.0), "sample rate"),
        ("not synthetic", lambda p: p["metadata"].update(synthetic=False), "synthetic"),
        ("missing key", lambda p: p["metadata"].pop("clock"), "missing clock"),
    ],
)
def test_contract_violations_are_reported(tmp_path, config, description, mutate, expected):
    issues = issues_for(tmp_path, config, mutate)
    assert any(expected in issue for issue in issues), (description, issues)


def test_missing_value_marked_invalid_is_accepted(tmp_path, config):
    def mutate(parts):
        t, signal, _, unit, _ = parts["rows"][0]
        parts["rows"][0] = (t, signal, None, unit, 0)
    assert issues_for(tmp_path, config, mutate) == []


def test_declared_gap_is_valid_and_clipping_is_a_warning(tmp_path, config):
    def mutate(parts):
        audio = parts["mat"]["audio"]
        audio[1000:1500] = np.nan
        audio[0:1000] = 1.0
        summary = _summary(audio)
        summary["n_missing"] = 500
        summary["sum"] = float(np.nansum(audio.astype(np.float64)))
        summary["sum_sq"] = float(np.nansum(np.square(audio.astype(np.float64))))
        parts["metadata"]["waveform_summary"]["audio"] = summary
    bundle = run_io.load_run(write_bundle(tmp_path, config, mutate=mutate))
    assert run_io.validate_run(bundle, config) == []
    warnings = run_io.run_warnings(bundle, config)
    assert any("clipped" in w for w in warnings) and any("missing" in w for w in warnings)


def test_manifest_records_hashes(tmp_path, config):
    run_dir = write_bundle(tmp_path, config)
    manifest = json.loads(run_io.write_manifest(tmp_path, [run_dir]).read_text())
    entry = manifest["runs"]["run_00001"]["waveforms.mat"]
    assert len(entry["sha256"]) == 64
    assert entry["bytes"] == (run_dir / "waveforms.mat").stat().st_size


def test_loader_does_not_reference_evaluation_truth():
    source = Path(run_io.__file__).read_text(encoding="utf-8")
    code = source.split('"""', 2)[2]  # skip the module docstring that names it
    assert "evaluation_truth" not in code and "_events.json" not in code


def with_cam(parts, pulses=None, **overrides):
    """Upgrade a bundle to contract 1.1 with a cam phase reference at 1500 rpm."""
    pulses = np.arange(0.01, DURATION_S, 120 / 1500) if pulses is None else pulses
    parts["mat"]["cam_sync_s"] = np.asarray(pulses, dtype=np.float64).reshape(-1, 1)
    parts["metadata"]["schema_version"] = "1.1"
    parts["metadata"]["streams"]["cam_sync"] = {
        "unit": "s", "clock": "vibration", "reference": "cylinder_1_firing_tdc",
        "angle_deg": 0, "n_pulses": len(pulses), **overrides}


def test_cam_sync_round_trips(tmp_path, config):
    bundle = run_io.load_run(write_bundle(tmp_path, config, mutate=with_cam))
    assert run_io.validate_run(bundle, config) == []
    assert bundle.cam_sync_s.ndim == 1 and bundle.cam_sync_s.size == 25
    assert run_io.run_warnings(bundle, config) == []


def test_contract_1_0_bundle_has_no_cam_reference(tmp_path, config):
    bundle = run_io.load_run(write_bundle(tmp_path, config))
    assert bundle.cam_sync_s is None
    assert run_io.validate_run(bundle, config) == []


@pytest.mark.parametrize(
    ("description", "mutate", "expected"),
    [
        ("1.1 without pulses", lambda p: p["metadata"].update(schema_version="1.1"),
         "needs both"),
        ("count mismatch", lambda p: with_cam(p, n_pulses=3), "metadata n_pulses"),
        ("wrong angle", lambda p: with_cam(p, angle_deg=90), "angle_deg"),
        ("wrong clock", lambda p: with_cam(p, clock="audio"), "clock"),
        ("not increasing", lambda p: with_cam(p, pulses=[0.5, 0.4, 0.9]), "strictly"),
        ("outside run", lambda p: with_cam(p, pulses=[0.5, 0.9, DURATION_S + 1]), "outside"),
        ("undeclared pulses", lambda p: p["mat"].update(cam_sync_s=np.ones((3, 1))),
         "no streams.cam_sync"),
        ("unknown version", lambda p: p["metadata"].update(schema_version="2.0"),
         "schema_version"),
    ],
)
def test_cam_sync_violations(tmp_path, config, description, mutate, expected):
    issues = issues_for(tmp_path, config, mutate)
    assert any(expected in issue for issue in issues), (description, issues)


def test_missing_cam_pulses_are_a_warning_not_a_violation(tmp_path, config):
    pulses = np.delete(np.arange(0.01, DURATION_S, 120 / 1500), [5, 12])
    bundle = run_io.load_run(write_bundle(tmp_path, config,
                                          mutate=lambda p: with_cam(p, pulses=pulses)))
    assert run_io.validate_run(bundle, config) == []
    assert any("2 gaps" in w for w in run_io.run_warnings(bundle, config))


def test_speed_ramp_does_not_warn_about_cam_gaps(tmp_path, config):
    # Cycle period shrinking from 0.12 s to 0.045 s over the run: no pulse is missing.
    pulses = np.cumsum(np.linspace(0.12, 0.045, 24))
    pulses = pulses[pulses < DURATION_S]
    bundle = run_io.load_run(write_bundle(tmp_path, config,
                                          mutate=lambda p: with_cam(p, pulses=pulses)))
    assert run_io.validate_run(bundle, config) == []
    assert run_io.run_warnings(bundle, config) == []


def wheel_edges(rpm=1500.0, duration=DURATION_S, first_deg=270.0, teeth=60, missing=2):
    """Ideal 60-2 falling-edge times at constant speed (crank angle 0 at t = 0)."""
    angles = [first_deg + 360 * rev + 360 / teeth * k
              for rev in range(-1, int(duration * rpm / 60) + 2)
              for k in range(teeth - missing)]
    times = np.asarray(sorted(angles)) / (6 * rpm)
    return times[(times >= 0) & (times < duration)]


def with_sensors(parts, edges=None, thermal=True, **overrides):
    """Upgrade a bundle to contract 1.2: cam reference, crank wheel, temperatures."""
    with_cam(parts)
    parts["metadata"]["schema_version"] = "1.2"
    edges = wheel_edges() if edges is None else np.asarray(edges, dtype=np.float64)
    parts["mat"]["crank_tooth_s"] = edges.reshape(-1, 1)
    parts["metadata"]["streams"]["crank_teeth"] = {
        "unit": "s", "clock": "vibration", "reference": "cylinder_1_firing_tdc",
        "edge": "falling", "teeth_per_rev": 60, "missing_teeth": 2,
        "first_tooth_angle_deg": 270, "timer_resolution_s": 1e-6,
        "n_edges": edges.size, **overrides}
    if thermal:
        rates = {"egt": 10, "cht": 2}
        for kind, rate in rates.items():
            for c in range(1, 5):
                name = f"{kind}_cyl{c}"
                parts["rows"] += [(k / rate, name, 600.0, "degC", 1)
                                  for k in range(DURATION_S * rate)]
                parts["metadata"]["telemetry_signals"][name] = {"rate_hz": rate, "unit": "degC"}
        parts["rows"].sort(key=lambda row: (row[0], row[1]))


def test_contract_1_2_bundle_round_trips(tmp_path, config):
    bundle = run_io.load_run(write_bundle(tmp_path, config, mutate=with_sensors))
    assert run_io.validate_run(bundle, config) == []
    assert run_io.run_warnings(bundle, config) == []
    assert bundle.crank_tooth_s.ndim == 1 and bundle.crank_tooth_s.size == wheel_edges().size
    assert {"egt_cyl3", "cht_cyl1"} <= set(bundle.telemetry["signal"])
    teeth = 58
    assert np.all(np.diff(run_io.wheel_gaps(bundle.crank_tooth_s)) == teeth)


def test_contract_1_1_bundle_has_no_wheel_or_temperatures(tmp_path, config):
    bundle = run_io.load_run(write_bundle(tmp_path, config, mutate=with_cam))
    assert bundle.crank_tooth_s is None
    assert run_io.validate_run(bundle, config) == []


def _drop_signal(parts, name):
    parts["rows"] = [r for r in parts["rows"] if r[1] != name]


@pytest.mark.parametrize(
    ("description", "mutate", "expected"),
    [
        ("1.2 vector without metadata", lambda p: (
            with_sensors(p), p["metadata"]["streams"].pop("crank_teeth")), "needs both"),
        ("wheel in a 1.1 bundle", lambda p: (
            with_sensors(p), p["metadata"].update(schema_version="1.1")), "older than"),
        ("count mismatch", lambda p: with_sensors(p, n_edges=3), "metadata n_edges"),
        ("wrong wheel", lambda p: with_sensors(p, teeth_per_rev=36), "teeth_per_rev"),
        ("wrong edge", lambda p: with_sensors(p, edge="rising"), "edge"),
        ("not increasing", lambda p: with_sensors(p, edges=wheel_edges()[::-1]), "strictly"),
        ("outside run", lambda p: with_sensors(p, edges=np.r_[wheel_edges(), DURATION_S + 1]),
         "outside"),
        ("no gap", lambda p: with_sensors(p, edges=np.arange(0.001, DURATION_S, 60 / 1500 / 60)),
         "no missing-tooth gap"),
        ("temperature missing", lambda p: (with_sensors(p), _drop_signal(p, "cht_cyl2")),
         "'cht_cyl2' missing"),
        ("undeclared temperature", lambda p: (
            with_sensors(p), [p["metadata"]["telemetry_signals"].pop(f"{k}_cyl{c}")
                              for k in ("egt", "cht") for c in range(1, 5)]),
         "unknown telemetry signals"),
    ],
)
def test_contract_1_2_violations(tmp_path, config, description, mutate, expected):
    issues = issues_for(tmp_path, config, mutate)
    assert any(expected in issue for issue in issues), (description, issues)


def test_lost_teeth_are_a_warning_not_a_violation(tmp_path, config):
    edges = np.delete(wheel_edges(), [200, 201])
    bundle = run_io.load_run(write_bundle(tmp_path, config,
                                          mutate=lambda p: with_sensors(p, edges=edges)))
    assert run_io.validate_run(bundle, config) == []
    assert any("irregular gap" in w for w in run_io.run_warnings(bundle, config))
