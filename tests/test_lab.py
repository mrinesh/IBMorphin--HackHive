"""Fault-injection lab: injections land where intended and nothing else changes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine_diagnostics.data.run_io import RunBundle
from engine_diagnostics.lab import LabSettings, _render, build_lab_bundle

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = json.loads((ROOT / "configs/core_v1.json").read_text(encoding="utf-8"))[
    "generator_params"]
RATES = {"audio": 24000.0, "vibration": 25600.0}
DURATION = 4.0
RPM = 1500.0
FIRING_S = 60 / RPM / 2  # four cylinders, four strokes
ORDER = (1, 3, 4, 2)


def fake_run():
    events = [{"time_s": 0.005 + k * FIRING_S, "cylinder": ORDER[k % 4], "kind": "combustion",
               "intensity": 1.0, "faulted": False}
              for k in range(int((DURATION - 0.05) / FIRING_S))]
    sources = [("combustion", e["time_s"], 1.0, 0.5) for e in events]
    rng = np.random.default_rng(0)
    waves = {}
    for m, fs in RATES.items():
        n = round(DURATION * fs)
        noise = GENERATOR["sensors"][m]["white_noise_rms"] * rng.standard_normal(n)
        waves[m] = (_render(sources, GENERATOR["sensors"][m], GENERATOR, fs, 0.0, n, m)
                    + noise).astype(np.float32)
    rows = [(t, "rpm", RPM, "rpm", 1) for t in np.arange(0, DURATION, 0.02)]
    rows += [(t, "load", 50.0, "percent", 1) for t in np.arange(0, DURATION, 0.1)]
    telemetry = pd.DataFrame(rows, columns=["t_s", "signal", "value", "unit", "valid"])
    bundle = RunBundle(run_id="run_x", audio=waves["audio"], vibration=waves["vibration"],
                       audio_fs_hz=RATES["audio"], vibration_fs_hz=RATES["vibration"],
                       audio_t0_s=0.0, vibration_t0_s=0.0, telemetry=telemetry,
                       metadata={"run_id": "run_x", "duration_s": DURATION})
    truth = {"engine_id": "engine_x", "events": events}
    return bundle, truth


def energy(x, fs, t0, t1):
    return float(np.sum(np.asarray(x[round(t0 * fs):round(t1 * fs)], dtype=float) ** 2))


@pytest.fixture(scope="module")
def run():
    return fake_run()


def lab(run, **kwargs):
    bundle, truth = run
    return build_lab_bundle(bundle, truth, LabSettings(base_run="run_x", **kwargs), GENERATOR)


def test_no_changes_reproduce_the_recording(run):
    bundle, _ = run
    out, log = lab(run, fault="none")
    np.testing.assert_array_equal(out.audio, bundle.audio)
    np.testing.assert_array_equal(out.vibration, bundle.vibration)
    pd.testing.assert_frame_equal(out.telemetry, bundle.telemetry)
    assert out.run_id == "lab:run_x" and out.metadata["lab"]["settings"]["fault"] == "none"
    assert log.fault_interval is None and len(log.lines) == 1


def test_misfire_removes_combustion_only_on_the_affected_cylinder_after_onset(run):
    bundle, truth = run
    out, log = lab(run, fault="misfire", severity=1.0, onset_s=2.0, cylinder=2)
    assert log.event_times and min(log.event_times) >= 2.0
    assert log.fault_interval == (2.0, DURATION)
    hit = set(log.event_times)
    for m, fs in RATES.items():
        delay = GENERATOR["sensors"][m]["delay_s"]
        for e in truth["events"]:
            t = e["time_s"] + delay
            before = energy(getattr(bundle, m), fs, t, t + 0.006)
            after = energy(getattr(out, m), fs, t, t + 0.006)
            if e["time_s"] in hit:
                assert e["cylinder"] == 2
                assert after < 0.3 * before, (m, e["time_s"])
            elif e["time_s"] < 2.0 - 0.03 or e["cylinder"] not in (2,):
                assert after == pytest.approx(before, rel=0.35), (m, e["time_s"])
        n = round(1.9 * fs)  # untouched before onset
        np.testing.assert_array_equal(getattr(out, m)[:n], getattr(bundle, m)[:n])


def test_misfire_dips_rpm_telemetry_at_misfired_strokes(run):
    bundle, _ = run
    out, log = lab(run, fault="misfire", severity=1.0, onset_s=2.0)
    rpm = out.telemetry[out.telemetry["signal"] == "rpm"].set_index("t_s")["value"]
    dip_t = log.event_times[0]
    near = rpm[(rpm.index > dip_t - 0.03) & (rpm.index < dip_t + 0.03)]
    assert near.min() <= RPM - 5
    assert (rpm[rpm.index < 1.9] == RPM).all()


def test_knock_adds_high_band_bursts_after_onset_only(run):
    bundle, _ = run
    out, log = lab(run, fault="knock_like", severity=0.9, onset_s=2.0, cylinder=3)
    assert log.event_times and min(log.event_times) > 2.0
    for m, fs in RATES.items():
        diff = np.asarray(getattr(out, m), float) - np.asarray(getattr(bundle, m), float)
        assert not diff[:round(2.0 * fs)].any()
        spectrum = np.abs(np.fft.rfft(diff))
        freqs = np.fft.rfftfreq(diff.size, 1 / fs)
        band = spectrum[(freqs >= 5000) & (freqs <= 9000)].sum()
        assert band > 0.8 * spectrum.sum()


def test_severity_raises_the_number_of_affected_strokes(run):
    low = len(lab(run, fault="knock_like", severity=0.1, onset_s=0.5)[1].event_times)
    high = len(lab(run, fault="knock_like", severity=1.0, onset_s=0.5)[1].event_times)
    assert high > low


def test_noise_scale_sets_total_noise(run):
    bundle, _ = run
    out, _ = lab(run, fault="none", noise_scale=3.0, seed=5)
    for m in RATES:
        added = np.asarray(getattr(out, m), float) - np.asarray(getattr(bundle, m), float)
        sensor = GENERATOR["sensors"][m]
        recorded = np.hypot(sensor["colored_noise_rms"], sensor["white_noise_rms"])
        assert np.std(added) == pytest.approx(np.sqrt(3.0 ** 2 - 1) * recorded, rel=0.15)


def test_sensor_failure_removes_data_from_the_failure_time(run):
    out, log = lab(run, fault="none", failed=("audio", "rpm"), fail_s=2.5)
    k = round(2.5 * RATES["audio"])
    assert np.isnan(out.audio[k:]).all() and np.isfinite(out.audio[:k]).all()
    assert np.isfinite(out.vibration).all()
    rpm = out.telemetry[out.telemetry["signal"] == "rpm"]
    assert rpm["t_s"].max() < 2.5
    assert (out.telemetry["signal"] == "load").sum() == (run[0].telemetry["signal"] == "load").sum()
    assert log.fail_s == 2.5 and "SENSOR FAILURE" in log.lines[-1]


def test_same_seed_same_injection_and_no_truth_in_the_bundle(run):
    a, _ = lab(run, fault="misfire", severity=0.6, onset_s=1.0, seed=7)
    b, _ = lab(run, fault="misfire", severity=0.6, onset_s=1.0, seed=7)
    c, _ = lab(run, fault="misfire", severity=0.6, onset_s=1.0, seed=8)
    np.testing.assert_array_equal(a.audio, b.audio)
    assert not np.array_equal(a.audio, c.audio)
    assert "events" not in json.dumps(a.metadata)


GENERATOR_V3 = json.loads((ROOT / "configs/core_v3.json").read_text(encoding="utf-8"))[
    "generator_params"]


def run_v3():
    """fake_run plus a 60-2 wheel at constant speed and flat cylinder temperatures."""
    bundle, truth = fake_run()
    # Cylinder 1 firing TDC at t = 0.005 s - 8 degrees; edges every 6 degrees but the gap.
    angles = np.array([270 + 360 * r + 6 * k for r in range(-1, 60) for k in range(58)])
    tdc0 = 0.005 - 8 / (6 * RPM)
    bundle.crank_tooth_s = np.sort(tdc0 + angles / (6 * RPM))
    bundle.crank_tooth_s = bundle.crank_tooth_s[(bundle.crank_tooth_s >= 0)
                                                & (bundle.crank_tooth_s < DURATION)]
    rows = [(t, f"{k}_cyl{c}", 650.0 if k == "egt" else 170.0, "degC", 1)
            for k, rate in (("egt", 10), ("cht", 2)) for c in range(1, 5)
            for t in np.arange(0, DURATION, 1 / rate)]
    extra = pd.DataFrame(rows, columns=["t_s", "signal", "value", "unit", "valid"])
    bundle.telemetry = (pd.concat([bundle.telemetry, extra], ignore_index=True)
                        .sort_values(["t_s", "signal"], kind="stable").reset_index(drop=True))
    return bundle, truth


def lab_v3(**kwargs):
    bundle, truth = run_v3()
    out, log = build_lab_bundle(bundle, truth, LabSettings(base_run="run_x", **kwargs),
                                GENERATOR_V3)
    return bundle, out, log


def test_misfire_slows_the_crank_on_misfired_strokes_and_cools_that_cylinder():
    bundle, out, log = lab_v3(fault="misfire", severity=1.0, onset_s=1.0, cylinder=3)
    assert out.crank_tooth_s.size == bundle.crank_tooth_s.size
    before, after = np.diff(bundle.crank_tooth_s), np.diff(out.crank_tooth_s)
    mid = (out.crank_tooth_s[:-1] + out.crank_tooth_s[1:]) / 2
    slower = after > before * 1.002  # tooth periods at least 0.2 % longer
    near = np.array([np.any((m - np.asarray(log.event_times) > 0)
                            & (m - np.asarray(log.event_times) < 0.03)) for m in mid])
    assert slower[near].mean() > 0.5 and slower[~near & (mid < 1.0)].sum() == 0
    # The wheel stays in phase with the recorded events (slow drift removed).
    assert np.max(np.abs(out.crank_tooth_s - bundle.crank_tooth_s)) < 30 / RPM / 4
    tel = out.telemetry
    egt = tel[tel["signal"] == "egt_cyl3"].set_index("t_s")["value"]
    other = tel[tel["signal"] == "egt_cyl1"].set_index("t_s")["value"]
    assert egt.loc[:0.9].eq(650.0).all() and other.eq(650.0).all()
    assert egt.iloc[-1] < 650.0 - 100  # dead cylinder runs cold within seconds


def test_knock_heats_the_head_not_the_exhaust():
    _, out, _ = lab_v3(fault="knock_like", severity=1.0, onset_s=1.0, cylinder=2)
    tel = out.telemetry
    cht = tel[tel["signal"] == "cht_cyl2"]["value"].to_numpy()
    egt = tel[tel["signal"] == "egt_cyl2"]["value"].to_numpy()
    assert cht[-1] > 170.0 + 3 and egt[-1] <= 650.0


def test_crank_and_thermal_sensors_can_fail():
    _, out, _ = lab_v3(failed=("crank", "thermal"), fail_s=2.0)
    assert out.crank_tooth_s.max() < 2.0
    temps = out.telemetry["signal"].str.match(r"^(egt|cht)_cyl")
    assert out.telemetry.loc[temps, "t_s"].max() < 2.0
