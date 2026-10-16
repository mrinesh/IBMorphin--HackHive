"""Cylinder attribution: crank phase from cam pulses, signatures, localizer, runtime state."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from engine_diagnostics.config import load_config  # noqa: E402
from engine_diagnostics.evaluation import (  # noqa: E402
    localization_episodes,
    summarize_localization,
)
from engine_diagnostics.models import localizer as L  # noqa: E402
from engine_diagnostics.pipeline import cylinders as C  # noqa: E402
from engine_diagnostics.pipeline import features as F  # noqa: E402
from engine_diagnostics.runtime.localization import (  # noqa: E402
    LOCALIZED,
    MIN_WINDOWS,
    RELATIVE_ONLY,
    LocalizationState,
    SlotAligner,
)

ROOT = Path(__file__).resolve().parents[1]
FS = 25600
RPM = 1800.0
ORDER = [1, 3, 4, 2]


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/mvp.json", ROOT / "schemas/project-config.schema.json")


def engine_signal(seconds=2.0, theta0=200.0, misfire=None, knock=None, t_start=0.0, seed=0):
    """Vibration with a 600 Hz combustion burst at TDC + 8 deg of every stroke.

    Crank angle theta = theta0 + 6 rpm t (cylinder 1 firing TDC at 0 mod 720);
    `misfire` weakens that cylinder's bursts, `knock` adds 7 kHz ringing 15 deg
    later. Returns the signal and the cam pulse times (theta = 0 mod 720).
    """
    rng = np.random.default_rng(seed)
    t = t_start + np.arange(round(seconds * FS)) / FS
    x = 0.002 * rng.standard_normal(t.size)
    deg_per_s = 6 * RPM
    first_cycle = np.floor((theta0 + deg_per_s * (t_start - 0.1)) / 720)
    last_cycle = np.ceil((theta0 + deg_per_s * (t[-1] + 0.1)) / 720)
    for cycle in np.arange(first_cycle, last_cycle + 1):
        for p, cylinder in enumerate(ORDER):
            angle = 720 * cycle + 180 * p + 8
            onset = (angle - theta0) / deg_per_s
            dt = t - onset
            on = (dt >= 0) & (dt < 0.03)
            amp = 0.05 if cylinder == misfire else 1.0
            x[on] += amp * np.exp(-dt[on] / 0.004) * np.sin(2 * np.pi * 600 * dt[on])
            if cylinder == knock:
                dk = dt - 15 / deg_per_s
                hit = (dk >= 0) & (dk < 0.006)
                x[hit] += 0.5 * np.exp(-dk[hit] / 0.001) * np.sin(2 * np.pi * 7000 * dk[hit])
    cycles = np.arange(first_cycle, last_cycle + 1)
    pulses = (720 * cycles - theta0) / deg_per_s
    return x, pulses


def window(config, x, pulses, t_end=2.0, t_start=0.0, with_cam=True):
    waves = {"audio": np.zeros(round(2 * 24000)), "vibration": x}
    rates = {"audio": 24000, "vibration": FS}
    context = {"rpm": RPM, "load": 40.0, "throttle": 42.0}
    cam = C.window_pulses(pulses, t_start, t_end) if with_cam else None
    return F.extract_window(waves, rates, context, True, {"audio": 1.0, "vibration": 500.0},
                            t_end, config, cam_sync_s=cam, vibration_start_s=t_start)


# --- crank phase -------------------------------------------------------------------


def test_crank_angle_interpolates_and_extrapolates_between_pulses():
    period = 120 / RPM
    pulses = 0.03 + period * np.arange(5)
    times = np.array([0.0, 0.03, 0.03 + period / 2, pulses[-1] + period / 4])
    angle = C.crank_angle(times, pulses, 0.0, RPM)
    np.testing.assert_allclose(angle, [-0.03 * 6 * RPM, 0, 360, 4 * 720 + 180], atol=1e-6)


def test_missing_cam_pulse_counts_as_a_skipped_cycle():
    period = 120 / RPM
    pulses = np.delete(period * np.arange(8), 3)  # pulse 3 lost
    angle = C.crank_angle(np.array([period * 3.5, period * 6]), pulses, 0.0, RPM)
    np.testing.assert_allclose(angle, [3.5 * 720, 6 * 720], atol=1e-6)


def test_single_pulse_uses_rpm_and_far_times_are_unphased():
    angle = C.crank_angle(np.array([0.5 + 120 / RPM / 2, 0.5 + 3 * 120 / RPM]),
                          np.array([0.5]), 90.0, RPM)
    assert angle[0] == pytest.approx(90 + 360)
    assert np.isnan(angle[1])
    assert np.isnan(C.crank_angle(np.array([1.0]), np.array([]), 0.0, RPM)).all()


def test_firing_positions_follow_tdc_and_cam_angle():
    period = 120 / RPM
    pulses = period * np.arange(4)
    onsets = (np.arange(8) * 180 + 10) / (6 * RPM)  # 10 deg after each TDC
    assert C.firing_positions(onsets, pulses, 0.0, RPM, 4).tolist() == [0, 1, 2, 3] * 2
    # The same strokes with a cam pulse mounted 180 deg after cylinder 1's TDC: the
    # configured angle compensates, so the attribution does not change.
    late = C.firing_positions(onsets, pulses + 180 / (6 * RPM), 180.0, RPM, 4)
    assert late.tolist() == [0, 1, 2, 3] * 2
    # Ignoring the mounting angle would shift every stroke by one position.
    wrong = C.firing_positions(onsets, pulses + 180 / (6 * RPM), 0.0, RPM, 4)
    assert wrong.tolist() == [3, 0, 1, 2] * 2


# --- signatures ----------------------------------------------------------------------


@pytest.mark.parametrize("theta0", [0.0, 200.0, 517.0])
def test_misfire_signature_points_to_the_weak_cylinder(config, theta0):
    x, pulses = engine_signal(theta0=theta0, misfire=4)
    sig = window(config, x, pulses).cylinders
    assert sig.phase == C.PHASE_CAM
    comb = sig.contrast[:, C.signature_names().index("comb_0_15")]
    assert ORDER[int(comb.argmin())] == 4 and comb.min() < -1
    assert sig.strokes.sum() == pytest.approx(2 * RPM / 30, abs=5)  # edge strokes


def test_knock_signature_points_to_the_ringing_cylinder(config):
    x, pulses = engine_signal(knock=3)
    sig = window(config, x, pulses).cylinders
    high = sig.contrast[:, C.signature_names().index("high_0_15")] \
        + sig.contrast[:, C.signature_names().index("high_15_30")]
    assert ORDER[int(high.argmax())] == 3


def test_signature_does_not_change_detector_features(config):
    x, pulses = engine_signal(misfire=2)
    with_cam = window(config, x, pulses)
    without = window(config, x, pulses, with_cam=False)
    np.testing.assert_array_equal(with_cam.features, without.features)
    assert without.cylinders.phase == C.PHASE_RELATIVE


def test_window_without_pulses_has_no_signature(config):
    x, _ = engine_signal()
    assert window(config, x, np.array([])).cylinders is None


def test_by_cylinder_round_trips_through_table_columns(config):
    x, pulses = engine_signal(misfire=1)
    sig = window(config, x, pulses).cylinders
    row = C.by_cylinder(sig, config["engine"])
    assert list(row) == C.cylinder_columns(config["engine"])
    np.testing.assert_allclose(C.contrast_from_row(row, config["engine"]), sig.contrast)


# --- localizer -----------------------------------------------------------------------


def synthetic_contrasts(n, rng, offset=0):
    """Windows whose faulty position p has a dip in value 0 and a rise at p+offset."""
    positions = rng.integers(0, 4, n)
    contrasts = 0.05 * rng.standard_normal((n, 4, len(C.signature_names())))
    contrasts[np.arange(n), positions, 0] -= 1.0
    contrasts[np.arange(n), (positions + offset) % 4, 5] += 0.8
    return contrasts, positions


def test_candidate_matrix_rotates_the_candidate_first():
    contrast = np.arange(8, dtype=float).reshape(4, 2)
    m = L.candidate_matrix(contrast)
    assert m.shape == (4, 16)
    np.testing.assert_array_equal(m[1, :8], [2, 3, 4, 5, 6, 7, 0, 1])


def test_localizer_learns_cross_position_patterns_and_is_rotation_equivariant():
    rng = np.random.default_rng(0)
    train_c, train_p = synthetic_contrasts(400, rng, offset=2)
    scorer = L._fit_scorer(train_c, train_p, 1.0)
    test_c, test_p = synthetic_contrasts(200, rng, offset=2)
    proba = L.softmax(L._all_scores(scorer, test_c))
    assert (proba.argmax(axis=1) == test_p).mean() > 0.97
    rolled = L.softmax(L._all_scores(scorer, np.roll(test_c[:1], 1, axis=1)))
    np.testing.assert_allclose(rolled[0], np.roll(proba[0], 1), atol=1e-9)


def test_temperature_softens_overconfident_scores():
    rng = np.random.default_rng(1)
    positions = rng.integers(0, 4, 500)
    scores = rng.standard_normal((500, 4))
    scores[np.arange(500), positions] += 1.0  # weakly informative...
    temperature = L._fit_temperature(10 * scores, positions)  # ...but scaled up 10x
    assert 5 < temperature < 20


def fake_localizer(min_confidence=0.6):
    rng = np.random.default_rng(2)
    train_c, train_p = synthetic_contrasts(300, rng)
    scorer = L._fit_scorer(train_c, train_p, 1.0)
    return L.CylinderLocalizer(
        faults=["misfire", "knock_like", "valve_clearance_like"], firing_order=ORDER,
        signature_names=C.signature_names(),
        scorers=dict.fromkeys(("misfire", "knock_like", "valve_clearance_like"), scorer),
        temperatures=dict.fromkeys(("misfire", "knock_like", "valve_clearance_like"), 1.0),
        min_confidence=dict.fromkeys(("misfire", "knock_like", "valve_clearance_like"),
                                     min_confidence))


def signature_for(position, phase=C.PHASE_CAM, times=None):
    contrast = np.zeros((4, len(C.signature_names())))
    contrast[position, 0] = -1.0
    times = np.arange(8) * 0.017 if times is None else times
    return C.CylinderSignature(contrast=contrast, strokes=np.full(4, 2), phase=phase,
                               stroke_times_s=times, stroke_positions=np.arange(times.size) % 4)


# --- runtime state -------------------------------------------------------------------


def test_state_names_a_cylinder_after_consistent_windows():
    state = LocalizationState(fake_localizer(), ORDER)
    fault = {"misfire": "fault"}
    first = state.update(fault, ["misfire"], signature_for(2), "ok")["misfire"]
    assert first["status"] != LOCALIZED and first["windows"] == 1 < MIN_WINDOWS
    second = state.update(fault, ["misfire"], signature_for(2), "ok")["misfire"]
    assert second["status"] == LOCALIZED and second["cylinder"] == ORDER[2]
    assert set(second["probabilities"]) == {str(c) for c in ORDER}


def test_state_holds_on_bad_windows_and_clears_on_normal():
    state = LocalizationState(fake_localizer(), ORDER)
    for _ in range(3):
        state.update({"misfire": "fault"}, ["misfire"], signature_for(1), "ok")
    held = state.update({"misfire": "fault"}, ["misfire"], None, "missing_waveform")
    assert held["misfire"]["cylinder"] == ORDER[1]
    state.update({"misfire": "normal"}, ["misfire"], signature_for(3), "ok")  # alert still on
    assert state.evidence["misfire"].windows == 3
    state.update({"misfire": "normal"}, [], signature_for(3), "ok")  # alert cleared
    assert state.evidence["misfire"].windows == 0


def test_state_rejects_a_different_firing_order():
    with pytest.raises(ValueError, match="firing order"):
        LocalizationState(fake_localizer(), [1, 2, 3, 4])


def test_aligner_keeps_slots_stable_across_overlapping_windows():
    aligner = SlotAligner()
    period = 30 / RPM
    first = signature_for(0, C.PHASE_RELATIVE, times=period * np.arange(8))
    assert aligner.align(first) == (0, True)
    # Next window starts three strokes later: its position 0 is run slot 3.
    later = signature_for(0, C.PHASE_RELATIVE, times=period * (np.arange(8) + 3) + 1e-4)
    assert aligner.align(later) == (3, False)
    gap = signature_for(0, C.PHASE_RELATIVE, times=10 + period * np.arange(8))
    assert aligner.align(gap) == (0, True) and aligner.epoch == 1


def test_relative_phase_reports_a_slot_but_never_a_cylinder():
    state = LocalizationState(fake_localizer(), ORDER)
    period = 30 / RPM
    for k in range(3):
        times = period * (np.arange(8) + 4 * k)  # whole cycles: positions do not rotate
        report = state.update({"knock_like": "fault"}, ["knock_like"],
                              signature_for(1, C.PHASE_RELATIVE, times), "ok")["knock_like"]
    assert report["status"] == RELATIVE_ONLY and report["slot"] == "B"
    assert report["cylinder"] is None


# --- evaluation ----------------------------------------------------------------------


def localization_record(t, active, cylinder=None):
    where = {"misfire": {"cylinder": cylinder}} if active else {}
    return {"t_end_s": t, "alerts": ["misfire"] if active else [], "localization": where}


def test_localization_episodes_score_first_and_final_answers():
    rec = localization_record
    records = [rec(2.0, False), rec(2.5, True), rec(3.0, True, 4), rec(3.5, True, 4),
               rec(4.0, True, 2), rec(4.5, False), rec(5.0, True, 4)]
    found = localization_episodes(records, "misfire", true_cylinder=4)
    assert len(found) == 2
    assert found[0]["first"] == 4 and found[0]["final"] == 2 and found[0]["changes"] == 1
    assert found[0]["delay_s"] == pytest.approx(0.5)
    summary = summarize_localization(found)
    assert summary["localized_fraction"] == 1.0 and summary["first_correct"] == 1.0
    assert summary["final_correct"] == 0.5 and summary["episodes_with_changes"] == 1
    false_alert = summarize_localization(localization_episodes(records, "misfire", None))
    assert false_alert["true_alert_episodes"] == 0
    assert false_alert["false_alert_episodes"] == false_alert["false_alert_episodes_localized"] == 2


def test_answers_before_onset_are_not_judged_as_the_faults_location():
    rec = localization_record
    # Alert raised at 2.5 s by something else (named cylinder 1); the fault starts at 4 s.
    records = [rec(2.5, True, 1), rec(3.0, True, 1), rec(4.5, True, 3), rec(5.0, True, 3),
               rec(6.0, False), rec(7.0, True, 2)]
    found = localization_episodes(records, "misfire", true_cylinder=3, intervals=[(4.0, 5.5)])
    assert [e["true_alert"] for e in found] == [True, False]
    first = found[0]
    assert (first["first_any"], first["first"], first["final"]) == (1, 3, 3)
    assert first["delay_s"] == pytest.approx(0.0)  # named at the first decision after onset
    summary = summarize_localization(found)
    assert summary["first_correct"] == 1.0 and summary["first_any_correct"] == 0.0
    assert summary["false_alert_episodes"] == 1
