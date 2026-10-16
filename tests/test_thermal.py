"""Per-cylinder EGT/CHT cross-check: patterns, baseline, staleness and causality."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from engine_diagnostics.config import load_config  # noqa: E402
from engine_diagnostics.runtime.thermal import (  # noqa: E402
    ThermalCalibration,
    ThermalPath,
    latest_readings,
)

ROOT = Path(__file__).resolve().parents[1]
DURATION_S = 20


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/mvp.json", ROOT / "schemas/project-config.schema.json")


def temperatures(egt_step=None, cht_step=None, onset=8.0, seed=0, ramp_c=0.0):
    """Long-form EGT (10 Hz) / CHT (2 Hz) rows; steps on one cylinder follow a 1.5/8 s lag.

    egt_step / cht_step are (cylinder, degC); ramp_c raises every cylinder over the
    run (an operating-point change that must cancel across cylinders).
    """
    rng = np.random.default_rng(seed)
    rows = []
    offsets = {"egt": [650, 665, 640, 655], "cht": [170, 172, 168, 171]}
    for kind, rate, tau, step, noise in (("egt", 10, 1.5, egt_step, 1.5),
                                         ("cht", 2, 8.0, cht_step, 0.3)):
        t = np.arange(DURATION_S * rate) / rate
        for c in range(1, 5):
            value = offsets[kind][c - 1] + ramp_c * t / DURATION_S
            if step and step[0] == c:
                value = value + step[1] * np.clip(1 - np.exp(-(t - onset) / tau), 0, None)
            value = value + rng.normal(0, noise, t.size)
            rows += [(tk, f"{kind}_cyl{c}", round(v, 1), "degC", 1)
                     for tk, v in zip(t, value, strict=True)]
    frame = pd.DataFrame(rows, columns=["t_s", "signal", "value", "unit", "valid"])
    return frame.sort_values(["t_s", "signal"], kind="stable").reset_index(drop=True)


CALIBRATION = ThermalCalibration(spread_c={"egt": 2.0, "cht": 0.5},
                                 thresholds={"cold": 8.0, "hot": 6.0, "not_hot": 3.0})


def replay(config, telemetry):
    path = ThermalPath(CALIBRATION, config)
    return [path.update(t, telemetry) for t in np.arange(2.0, DURATION_S + 0.01, 0.5)]


def test_dead_cylinder_runs_cold(config):
    records = replay(config, temperatures(egt_step=(3, -150), cht_step=(3, -30)))
    assert records[0]["status"] == "warming_up"
    times = np.arange(2.0, DURATION_S + 0.01, 0.5)
    assert not any(r["alerts"] for r, t in zip(records, times, strict=True) if t < 8.0)
    last = records[-1]
    assert last["alerts"] == ["misfire"] and last["patterns"]["cold"]["cylinder"] == 3
    first = next(2.0 + 0.5 * k for k, r in enumerate(records) if "misfire" in r["alerts"])
    assert 8.0 < first < 12.0  # seconds, not milliseconds: thermal lag plus confirmation


def test_hot_head_with_normal_exhaust_is_knock(config):
    records = replay(config, temperatures(cht_step=(2, 25), egt_step=(2, -8)))
    assert records[-1]["alerts"] == ["knock_like"]
    assert records[-1]["patterns"]["head_hot"]["cylinder"] == 2


def test_hot_exhaust_with_normal_head_is_valve(config):
    records = replay(config, temperatures(egt_step=(4, 30)))
    assert records[-1]["alerts"] == ["valve_clearance_like"]
    assert records[-1]["patterns"]["exhaust_hot"]["cylinder"] == 4


def test_healthy_engine_and_operating_change_raise_nothing(config):
    for seed in range(3):
        records = replay(config, temperatures(seed=seed, ramp_c=120.0))
        assert not any(r["alerts"] for r in records)


def test_fixed_cylinder_offsets_are_learned_not_alerted(config):
    telemetry = temperatures()
    hot = telemetry["signal"] == "egt_cyl1"
    telemetry.loc[hot, "value"] += 60.0  # a cylinder that always runs 60 C hotter
    assert not any(r["alerts"] for r in replay(config, telemetry))


def test_stale_and_missing_temperatures(config):
    telemetry = temperatures()
    readings = latest_readings(telemetry, 5.0, 4, CALIBRATION.max_age_s)
    assert np.isfinite(readings["egt"]).all() and np.isfinite(readings["cht"]).all()
    dropped = telemetry[~((telemetry["signal"] == "egt_cyl2") & (telemetry["t_s"] > 4.0))]
    path = ThermalPath(CALIBRATION, config)
    assert path.update(6.0, dropped)["status"] == "stale"
    assert path.update(6.0, telemetry.iloc[0:0])["status"] == "unavailable"


def test_thermal_path_is_causal(config):
    calm = temperatures()
    changed = temperatures()
    late = changed["t_s"] >= 12.0
    changed.loc[late & (changed["signal"] == "egt_cyl3"), "value"] -= 200.0
    a, b = replay(config, calm), replay(config, changed)
    for x, y, t in zip(a, b, np.arange(2.0, DURATION_S + 0.01, 0.5), strict=True):
        if t <= 12.0:
            assert x == y
