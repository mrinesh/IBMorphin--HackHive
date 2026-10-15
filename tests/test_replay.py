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
