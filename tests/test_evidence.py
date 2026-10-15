"""Evidence panels: indicators, highlighted segments, and the overall assessment."""

import numpy as np
import pytest

from engine_diagnostics.runtime.evidence import FAULT_EVIDENCE, build_evidence

FS = 24000.0
WINDOW_S = 2.0
T_END = 10.0
FIRING_HZ = 40.0  # 1,200 rpm on a four-cylinder four-stroke engine


def pulse_train(weak=(), bursts=(), seed=0):
    """One 1.5 kHz pulse per firing; `weak` firings at 10%, `bursts` add 7 kHz ringing."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(WINDOW_S * FS)) / FS
    x = 0.01 * rng.standard_normal(t.size)
    for k, start in enumerate(np.arange(0.01, WINDOW_S - 0.02, 1 / FIRING_HZ)):
        local = t - start
        on = (local >= 0) & (local < 0.01)
        gain = 0.1 if k in weak else 1.0
        x[on] += gain * np.exp(-local[on] / 0.002) * np.sin(2 * np.pi * 1500 * local[on])
        x[on] += 0.05 * np.exp(-local[on] / 0.001) * np.sin(2 * np.pi * 7000 * local[on])
        if k in bursts:
            x[on] += 1.0 * np.exp(-local[on] / 0.001) * np.sin(2 * np.pi * 7000 * local[on])
    return x


def firing_time(k):
    return T_END - WINDOW_S + 0.01 + k / FIRING_HZ


def steady_rpm(dips=()):
    t = np.arange(T_END - WINDOW_S, T_END, 0.02)
    v = 1200 + 0.5 * np.sin(np.arange(t.size))
    for d in dips:
        v[np.argmin(np.abs(t - d))] -= 40
    return t, v


def evidence(fault, audio, vibration, rpm=None, z=None):
    rpm_t, rpm_v = rpm if rpm is not None else steady_rpm()
    return build_evidence(fault, {"audio": audio, "vibration": vibration},
                          {"audio": FS, "vibration": FS}, T_END, WINDOW_S, rpm_t, rpm_v,
                          FIRING_HZ, z or {})


def covers(segments, t):
    return any(a <= t <= b for a, b in segments)


def test_misfire_marks_weak_pulses_and_accompanying_rpm_dip():
    weak = (20, 45)
    signal = pulse_train(weak=weak)
    z = {"audio_firing_peak_cv": 8.0, "vibration_firing_peak_cv": 4.0}
    panel = evidence("misfire", signal, signal, steady_rpm(dips=[firing_time(20)]), z)
    sound, vibration, rpm = panel.indicators
    assert [i.signal for i in panel.indicators] == ["sound", "vibration", "rpm"]
    assert sound.claim == "irregular firing rhythm" and sound.status == "supports"
    assert vibration.status == "supports"
    for k in weak:
        assert covers(sound.segments, firing_time(k))
    assert not covers(sound.segments, firing_time(30))
    assert rpm.status == "supports" and covers(rpm.segments, firing_time(20))
    assert panel.supporting == 3 and panel.expected == 3
    assert panel.assessment.startswith("Multiple signals support this alert")


def test_rpm_dip_without_abnormal_pulses_is_only_weak_support():
    signal = pulse_train()
    panel = evidence("misfire", signal, signal, steady_rpm(dips=[firing_time(30)]),
                     {"audio_firing_peak_cv": 0.5})
    rpm = panel.indicators[2]
    assert rpm.status == "weak"
    assert panel.indicators[0].status == "not seen" and not panel.indicators[0].segments


def test_knock_marks_burst_firings_and_does_not_expect_rpm_change():
    bursts = (5, 12, 33, 60)
    signal = pulse_train(bursts=bursts)
    panel = evidence("knock_like", signal, signal,
                     z={"audio_env_kurtosis_high": 9.0, "vibration_env_cv_high": 2.5})
    sound, vibration, rpm = panel.indicators
    for k in bursts:
        assert covers(sound.segments, firing_time(k))
    assert len(sound.segments) == len(bursts)
    assert sound.status == "supports" and vibration.status == "weak"
    assert rpm.status == "not expected"
    assert panel.expected == 2 and panel.supporting == 1
    assert "Only sound supports" in panel.assessment


def test_healthy_window_has_no_segments_and_no_support():
    signal = pulse_train()
    for fault in FAULT_EVIDENCE:
        panel = evidence(fault, signal, signal, z={})
        assert all(not i.segments for i in panel.indicators), fault
        assert panel.supporting == 0
        assert "inspect manually" in panel.assessment


def test_falling_valve_signature_counts_but_falling_misfire_feature_does_not():
    signal = pulse_train()
    valve = evidence("valve_clearance_like", signal, signal, z={"audio_env_cv_mid": -4.0})
    misfire = evidence("misfire", signal, signal, z={"audio_firing_peak_cv": -4.0})
    assert valve.indicators[0].status == "supports"
    assert misfire.indicators[0].status == "not seen"


def test_unknown_firing_rate_degrades_gracefully():
    signal = pulse_train()
    panel = build_evidence("misfire", {"audio": signal, "vibration": signal},
                           {"audio": FS, "vibration": FS}, T_END, WINDOW_S,
                           np.empty(0), np.empty(0), float("nan"), {})
    assert "firing period unknown" in panel.indicators[0].detail
    assert panel.indicators[2].status == "not seen"


@pytest.mark.parametrize("fault", list(FAULT_EVIDENCE))
def test_traces_cover_the_decision_window(fault):
    signal = pulse_train()
    panel = evidence(fault, signal, signal)
    for ind in panel.indicators[:2]:
        assert ind.trace_t[0] == pytest.approx(T_END - WINDOW_S)
        assert ind.trace_t[-1] < T_END
        assert ind.trace_t.size == ind.trace_y.size
