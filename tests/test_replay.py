"""Replay equals offline extraction, never uses future samples, and resets cleanly."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from engine_diagnostics.config import load_config  # noqa: E402
from engine_diagnostics.data.run_io import TELEMETRY_UNITS, RunBundle  # noqa: E402
from engine_diagnostics.pipeline.features import extract_run, feature_names  # noqa: E402
from engine_diagnostics.runtime.replay import ReplayEngine, packets, replay_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DURATION_S = 4


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/mvp.json", ROOT / "schemas/project-config.schema.json")


class StubModel:
    """Labels a window as knock when vibration RMS is high; records nothing else."""

    model_version = "stub"
    feature_version = "test"
    feature_names = feature_names()  # replay passes each model only its own columns

    def predict(self, x):
        loud = np.atleast_2d(x)[0, feature_names().index("vibration_log_rms")] > 0
        label = "knock_like" if loud else "normal"
        return [{"predicted": label, "margin": 1.0, "anomaly_score": 0.0,
                 "anomalous": loud, "accepted": label}]


def make_bundle(config, seed=0, loud_after_s=None):
    rng = np.random.default_rng(seed)
    streams = config["streams"]
    waves = {}
    for m in ("audio", "vibration"):
        fs = streams[m]["sample_rate_hz"]
        x = 0.05 * rng.standard_normal(DURATION_S * fs)
        if loud_after_s is not None and m == "vibration":
            x[round(loud_after_s * fs):] *= 100
        waves[m] = x.astype(np.float32)
    rows = []
    for signal, unit in TELEMETRY_UNITS.items():
        rate = streams["telemetry_rates_hz"][signal]
        rows += [(k / rate, signal, 1500.0 if signal == "rpm" else 40.0, unit, 1)
                 for k in range(DURATION_S * rate)]
    telemetry = pd.DataFrame(rows, columns=["t_s", "signal", "value", "unit", "valid"])
    telemetry = telemetry.sort_values(["t_s", "signal"], kind="stable").reset_index(drop=True)
    metadata = {"duration_s": DURATION_S, "streams": {
        m: {"fs_hz": streams[m]["sample_rate_hz"], "t0_s": 0, "full_scale": 1e6}
        for m in ("audio", "vibration")}}
    return RunBundle("run_00001", waves["audio"], waves["vibration"],
                     float(streams["audio"]["sample_rate_hz"]),
                     float(streams["vibration"]["sample_rate_hz"]), 0.0, 0.0, telemetry, metadata)


def test_replay_features_equal_offline_features(config):
    bundle = make_bundle(config)
    records = replay_run(bundle, StubModel(), config)
    offline = extract_run(bundle, config)
    assert [r["t_end_s"] for r in records] == offline["t_end_s"].tolist()
    np.testing.assert_array_equal(np.vstack([r["features"] for r in records]),
                                  offline[feature_names()].to_numpy())


def test_decisions_never_use_future_samples(config):
    calm = replay_run(make_bundle(config), StubModel(), config)
    changed = replay_run(make_bundle(config, loud_after_s=3.0), StubModel(), config)
    for before, after in zip(calm, changed, strict=True):
        if before["t_end_s"] <= 3.0:
            np.testing.assert_array_equal(before["features"], after["features"])
        else:
            assert not np.array_equal(before["features"], after["features"])


def test_alert_follows_persistence_and_reset_clears(config):
    bundle = make_bundle(config, loud_after_s=2.0)
    records = replay_run(bundle, StubModel(), config)
    # windows ending 2.5, 3.0 contain loud samples; alert confirmed on the second
    assert [r["alert"] for r in records] == [None, None, "knock_like", "knock_like", "knock_like"]
    engine = ReplayEngine(StubModel(), config, bundle.metadata)
    for packet in packets(bundle, config["runtime"]["packet_s"]):
        engine.push(packet)
    assert engine.alerts.active == "knock_like"
    engine.reset()
    assert engine.alerts.active is None and engine.next_decision == 0
    assert all(buffer.size == 0 for buffer in engine.buffers.values())


def cam_bundle(config, misfire_cylinder=None):
    """make_bundle with an engine-like vibration signal and a cam phase reference."""
    from test_cylinders import engine_signal  # same synthetic engine as the unit tests

    bundle = make_bundle(config)
    x, pulses = engine_signal(seconds=DURATION_S, misfire=misfire_cylinder)
    bundle.vibration = x.astype(np.float32)
    bundle.cam_sync_s = pulses[(pulses >= 0) & (pulses < DURATION_S)]
    bundle.metadata["streams"]["cam_sync"] = {"unit": "s", "clock": "vibration"}
    bundle.telemetry.loc[bundle.telemetry["signal"] == "rpm", "value"] = 1800.0
    return bundle


def test_replay_cylinder_signatures_equal_offline(config):
    from engine_diagnostics.pipeline.cylinders import by_cylinder, cylinder_columns

    bundle = cam_bundle(config, misfire_cylinder=3)
    offline = extract_run(bundle, config)
    records = replay_run(bundle, StubModel(), config)
    assert set(offline["cyl_phase"]) == {"cam_sync"}
    for record, (_, row) in zip(records, offline.iterrows(), strict=True):
        values = by_cylinder(record["cylinders"], config["engine"])
        np.testing.assert_allclose([values[c] for c in cylinder_columns(config["engine"])],
                                   row[cylinder_columns(config["engine"])].to_numpy(float))


def test_replay_localizes_the_alerted_cylinder(config):
    from test_cylinders import fake_localizer

    class MisfireModel(StubModel):
        def predict(self, x):
            return [{"predicted": "misfire", "margin": 1.0, "anomaly_score": 0.0,
                     "anomalous": True, "accepted": "misfire"}]

    records = replay_run(cam_bundle(config, misfire_cylinder=3), MisfireModel(), config,
                         localizer=fake_localizer())
    assert records[-1]["alert"] == "misfire"
    assert records[-1]["localization"]["misfire"]["cylinder"] == 3
    assert "localization" not in replay_run(cam_bundle(config), StubModel(), config)[-1]


def wheel_bundle(config, misfire_cylinder=None):
    """make_bundle plus a 60-2 crank wheel and cam pulses at 1500 rpm (contract 1.2)."""
    from test_crankspeed import synthetic_wheel

    bundle = make_bundle(config)
    edges, cam, _, _ = synthetic_wheel(seconds=DURATION_S, misfire_cylinder=misfire_cylinder,
                                       misfire_every=2, jitter_s=1e-6)
    bundle.crank_tooth_s, bundle.cam_sync_s = edges, cam
    bundle.metadata["streams"]["crank_teeth"] = {"unit": "s", "clock": "vibration"}
    bundle.metadata["streams"]["cam_sync"] = {"unit": "s", "clock": "vibration"}
    return bundle


def test_crank_path_rides_along_without_changing_acoustic_decisions(config):
    from engine_diagnostics.models.obd_misfire import CrankCalibration

    edges = config["obd_misfire"]["rpm_bin_edges"]
    calibration = CrankCalibration(edges, [400.0] * (len(edges) - 1), 1.5)
    bundle = wheel_bundle(config, misfire_cylinder=2)
    plain = replay_run(bundle, StubModel(), config)
    with_crank = replay_run(bundle, StubModel(), config, crank=calibration)
    assert [r["alert"] for r in plain] == [r["alert"] for r in with_crank]
    assert all("crank" not in r for r in plain)
    assert with_crank[-1]["crank"]["alert"] and with_crank[-1]["crank"]["cylinder"] == 2
    # Each stroke reaches the simulated ECU exactly once, in time order.
    strokes = with_crank[-1]["crank"]["obd"]["strokes"]
    lag = config["crank_detector"]["stroke_lag_s"]
    assert strokes == pytest.approx((DURATION_S - lag) * 1500 / 30, abs=4)
    # A run without the wheel gets no crank record even when a calibration is given.
    assert all("crank" not in r for r in replay_run(make_bundle(config), StubModel(), config,
                                                    crank=calibration))


def test_crank_path_is_causal(config):
    from engine_diagnostics.models.obd_misfire import CrankCalibration

    edges = config["obd_misfire"]["rpm_bin_edges"]
    calibration = CrankCalibration(edges, [400.0] * (len(edges) - 1), 1.5)
    calm = wheel_bundle(config)
    late = wheel_bundle(config)
    late.crank_tooth_s = np.where(late.crank_tooth_s < 3.0, late.crank_tooth_s,
                                  late.crank_tooth_s + 1e-4)  # timing changes only after 3 s
    a = replay_run(calm, StubModel(), config, crank=calibration)
    b = replay_run(late, StubModel(), config, crank=calibration)
    for x, y in zip(a, b, strict=True):
        if x["t_end_s"] <= 3.0:
            assert x["crank"]["excess"] == y["crank"]["excess"]
