"""Crank wheel decoding, stroke speeds, the OBD monitor and the crank-speed path."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from engine_diagnostics.config import load_config  # noqa: E402
from engine_diagnostics.models.obd_misfire import (  # noqa: E402
    CrankCalibration,
    ObdMisfireMonitor,
    calibrate,
)
from engine_diagnostics.pipeline import crankspeed as C  # noqa: E402
from engine_diagnostics.runtime.crosscheck import CrankPath, cylinder_excess  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIRING_ORDER = (1, 3, 4, 2)


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/mvp.json", ROOT / "schemas/project-config.schema.json")


def synthetic_wheel(seconds=2.0, rpm=(1500.0, 1500.0), misfire_cylinder=None, misfire_every=1,
                    theta0=100.0, ripple_rpm=3.0, drop_rpm=30.0, recovery_s=0.03,
                    jitter_s=0.0, seed=0):
    """60-2 edges and cam pulses from an angle-domain speed model.

    Speed = linear ramp + healthy per-stroke ripple + an exponentially recovering
    deficit after each misfire of `misfire_cylinder` (every `misfire_every`-th of
    its strokes). Crank angle 0 mod 720 is cylinder 1 firing TDC; the cam pulse is
    at 0 mod 720 and the tooth after the gap at 270 mod 360.
    Returns (edges, cam, true_angle_at(t), misfire_tdc_times).
    """
    step = 0.5
    theta = np.arange(theta0, theta0 + 6 * max(rpm) * seconds * 1.2 + 720, step)
    # First pass: time from the ramp alone, used to place the deficits in time.
    base = np.interp(theta, [theta[0], theta[-1]], rpm)
    t = np.r_[0.0, np.cumsum(np.diff(theta) / (6 * base[:-1]))] - 0.05
    phase = np.mod(theta, 180.0)
    stroke = np.floor(theta / 180.0).astype(int)
    position = np.mod(stroke, 4)
    cylinder = np.asarray(FIRING_ORDER)[position]
    x = np.pi * phase / 180
    speed = base + ripple_rpm * ((1 - np.cos(x)) / 2 - x / np.pi) / 0.105
    misfires = []
    if misfire_cylinder is not None:
        mine = np.unique(stroke[cylinder == misfire_cylinder])
        for k, s in enumerate(mine):
            if k % misfire_every:
                continue
            start = s * 180.0
            in_stroke = stroke == s
            speed[in_stroke] -= drop_rpm * (1 - np.cos(x[in_stroke])) / 2
            end_t = np.interp(start + 180, theta, t)
            after = theta >= start + 180
            speed[after] -= drop_rpm * np.exp(-(t[after] - end_t) / recovery_s)
            misfires.append(start)
    t = np.r_[0.0, np.cumsum(np.diff(theta) / (6 * speed[:-1]))] - 0.05
    misfires = [float(np.interp(a, theta, t)) for a in misfires]
    revs = np.arange(np.floor((theta[0] - 270) / 360), np.ceil((theta[-1] - 270) / 360) + 1)
    angles = (270 + 360 * revs[:, None] + 6 * np.arange(58)[None, :]).ravel()
    angles = angles[(angles > theta[0]) & (angles < theta[-1])]
    edges = np.interp(angles, theta, t)
    cam_angles = 720 * np.arange(np.ceil(theta[0] / 720), theta[-1] // 720 + 1)
    cam = np.interp(cam_angles, theta, t)
    if jitter_s:
        edges = edges + np.random.default_rng(seed).normal(0, jitter_s, edges.size)
    keep = (edges >= 0) & (edges < seconds)
    cam = cam[(cam >= 0) & (cam < seconds)]
    return edges[keep], cam, (lambda times: np.interp(times, t, theta)), np.asarray(misfires)


def test_decode_gives_exact_angle_with_cam_phase(config):
    edges, cam, true_angle, _ = synthetic_wheel()
    decoded = C.decode_teeth(edges, config["streams"]["crank_wheel"], cam, 0.0)
    assert decoded.phase == C.PHASE_CAM
    error = decoded.angle_deg - true_angle(decoded.edges_s)
    # Same absolute angle mod 720 (the unwrapped count may differ by whole cycles).
    offset = np.round(np.median(error) / 720) * 720
    assert np.max(np.abs(error - offset)) < 0.05
    times = np.linspace(0.3, 1.7, 50)
    angle = C.crank_angle_from_teeth(times, decoded)
    assert np.max(np.abs(angle - offset - true_angle(times))) < 0.2


def test_without_cam_the_wheel_is_known_mod_360(config):
    edges, _, true_angle, _ = synthetic_wheel()
    decoded = C.decode_teeth(edges, config["streams"]["crank_wheel"], None, 0.0)
    assert decoded.phase == C.PHASE_WHEEL
    error = np.mod(decoded.angle_deg - true_angle(decoded.edges_s) + 180, 360) - 180
    assert np.max(np.abs(error)) < 0.05


def test_lost_tooth_only_misplaces_edges_until_the_next_gap(config):
    edges, cam, true_angle, _ = synthetic_wheel()
    edges = np.delete(edges, 100)  # a tooth lost early in the second revolution
    decoded = C.decode_teeth(edges, config["streams"]["crank_wheel"], cam, 0.0)
    error = decoded.angle_deg - true_angle(decoded.edges_s)
    offset = np.round(np.median(error) / 720) * 720
    late = decoded.edges_s > 0.3  # after the following gap
    assert np.max(np.abs(error[late] - offset)) < 0.05


def test_no_gap_means_no_decode(config):
    edges = np.arange(0, 1, 1 / 1500)
    assert C.decode_teeth(edges, config["streams"]["crank_wheel"], None, 0.0) is None
    assert C.stroke_speeds(edges, None, config).phase == C.PHASE_NONE


def test_misfire_strokes_score_high_on_the_right_cylinder(config):
    edges, cam, _, misfires = synthetic_wheel(misfire_cylinder=3, misfire_every=3,
                                              jitter_s=1e-6)
    strokes = C.stroke_speeds(edges, cam, config)
    assert strokes.phase == C.PHASE_CAM
    near = np.array([np.min(np.abs(misfires - t)) < 0.005 for t in strokes.time_s])
    finite = np.isfinite(strokes.score)
    assert set(strokes.cylinder[near & finite]) == {3}
    assert np.min(strokes.score[near & finite]) > 2 * np.max(strokes.score[~near & finite])
    assert np.median(strokes.rpm) == pytest.approx(1500, rel=0.02)


def test_speed_ramp_is_not_rough(config):
    edges, cam, _, _ = synthetic_wheel(rpm=(1200.0, 3200.0), jitter_s=1e-6)
    strokes = C.stroke_speeds(edges, cam, config)
    excess = cylinder_excess(strokes, 4)
    assert np.all(np.isfinite(excess)) and np.max(np.abs(excess)) < 1.0
    # The ramp's steady deceleration term cancels in the detrended drop.
    omega = np.deg2rad(6 * strokes.rpm)
    raw = omega[:-1] ** 2 - omega[1:] ** 2
    assert np.nanmedian(np.abs(strokes.drop)) < 0.1 * np.median(np.abs(raw))


def test_window_features_name_the_misfiring_cylinder(config):
    edges, cam, _, _ = synthetic_wheel(misfire_cylinder=2, misfire_every=2, jitter_s=1e-6)
    features = C.crank_window_features(C.stroke_speeds(edges, cam, config))
    assert list(features) == C.crank_feature_names()
    flagged = [features[f"crank_cyl{c}_flagged"] for c in range(1, 5)]
    assert int(np.argmax(flagged)) + 1 == 2 and flagged[1] > 0.3


def _calibration(config, stroke_threshold=5000.0, cylinder_threshold=1.5):
    edges = config["obd_misfire"]["rpm_bin_edges"]
    return CrankCalibration(edges, [stroke_threshold] * (len(edges) - 1), cylinder_threshold)


def test_obd_type_a_block_sets_cylinder_code(config):
    monitor = ObdMisfireMonitor(_calibration(config), config)
    n = monitor.blocks["A"]["strokes"]
    assert n == 400  # 200 revolutions x 2 firings per revolution
    cylinder = np.tile(FIRING_ORDER, n // 4)
    drop = np.zeros(n)
    drop[cylinder == 4] = np.where(np.arange(n // 4) % 3 == 0, 9000.0, 0.0)  # 1 in 3 strokes
    new = monitor.update(np.arange(n) * 0.02, cylinder, np.full(n, 1500.0), drop)
    assert len(new) == 1 and new[0]["code"] == "P0304" and new[0]["type"] == "A"
    assert new[0]["t_s"] == pytest.approx((n - 1) * 0.02)
    assert new[0]["rate_pct"] == pytest.approx(100 * 34 / 400)


def test_obd_low_rate_and_spread_misfire(config):
    monitor = ObdMisfireMonitor(_calibration(config), config)
    n = monitor.blocks["B"]["strokes"]
    cylinder = np.tile(FIRING_ORDER, n // 4)
    drop = np.zeros(n)
    drop[::50] = 9000.0  # 2 % of strokes, spread over every cylinder
    new = monitor.update(np.arange(n) * 0.02, cylinder, np.full(n, 1500.0), drop)
    assert [d["type"] for d in new] == ["B"]  # below the type A rate, above type B
    assert new[0]["code"] == "P0300" and new[0]["cylinder"] is None


def test_obd_never_counts_unphased_strokes(config):
    monitor = ObdMisfireMonitor(_calibration(config), config)
    n = monitor.blocks["A"]["strokes"]
    assert monitor.update(np.arange(n) * 0.02, np.full(n, -1), np.full(n, 1500.0),
                          np.full(n, 9000.0)) == []
    assert monitor.snapshot()["blocks"]["A"]["progress"] == 0.0


def test_calibration_thresholds_scale_and_fill_empty_bins(config):
    rng = np.random.default_rng(3)
    rpm = rng.uniform(1300, 1500, 20000)
    calib = calibrate([{"rpm": rpm, "drop": rng.normal(0, 100, rpm.size)}],
                      rng.normal(0, 0.2, 500), config, {"note": "test"})
    edges = np.asarray(config["obd_misfire"]["rpm_bin_edges"])
    populated = int(np.searchsorted(edges, 1400)) - 1
    values = np.asarray(calib.stroke_thresholds)
    assert values[populated] == pytest.approx(1.25 * np.quantile(
        rng.normal(0, 100, 20000), 0.9999), rel=0.3)
    assert np.all(np.diff(values[populated:]) > 0)  # higher speed bins borrow, scaled up
    assert calib.cylinder_threshold > 0 and calib.training["healthy_windows"] == 500


def test_crank_path_alerts_then_feeds_each_stroke_to_the_ecu_once(config):
    edges, cam, _, _ = synthetic_wheel(seconds=8.0, misfire_cylinder=4, misfire_every=2,
                                       jitter_s=1e-6)
    # A 30 rpm loss at 1500 rpm is a drop of about 2 omega d_omega = 1000 rad^2/s^2.
    path = CrankPath(_calibration(config, stroke_threshold=400.0), config)
    records = [path.update(t, edges, cam) for t in np.arange(2.0, 8.01, 0.5)]
    assert records[-1]["alert"] and records[-1]["cylinder"] == 4
    assert not records[0]["alert"]  # confirmation needs two windows
    stroke_count = records[-1]["obd"]["strokes"]
    expected = (8.0 - config["crank_detector"]["stroke_lag_s"]) * 1500 / 30
    assert stroke_count == pytest.approx(expected, abs=4)
    assert records[-1]["obd"]["misfires_total"][3] > 0
    assert sum(records[-1]["obd"]["misfires_total"][:3]) == 0
    assert len(records[-1]["speed_trace"]["rpm"]) <= C.SPEED_TRACE_POINTS


def test_crank_path_without_wheel_is_unavailable(config):
    path = CrankPath(_calibration(config), config)
    assert path.update(2.0, None, None) == {"status": "unavailable"}
