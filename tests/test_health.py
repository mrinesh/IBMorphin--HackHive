"""Health timeline: stages move healthy -> developing -> persistent fault causally."""

import numpy as np
import pandas as pd
import pytest

from engine_diagnostics.runtime.health import (
    DEVELOPING,
    FAULT,
    HEALTHY,
    NO_DATA,
    OVERALL,
    SMOOTH_WINDOWS,
    TRACKED,
    health_timeline,
)

FIRING, IMPULSE = (name for name, _, _ in TRACKED)
T = np.arange(2.0, 30.0, 0.5)


def timeline(firing=None, impulse=None, score=None, alert=None, ok=None, threshold=2.0):
    n = T.size
    z = pd.DataFrame({FIRING: np.zeros(n) if firing is None else firing,
                      IMPULSE: np.zeros(n) if impulse is None else impulse})
    return health_timeline(T, z, np.ones(n) if score is None else score, threshold,
                           np.zeros(n, bool) if alert is None else alert,
                           np.ones(n, bool) if ok is None else ok)


def test_ramp_moves_healthy_then_developing_then_persistent_fault():
    firing = np.clip((T - 8) * 0.5, 0, None)  # crosses z = 4 at 16 s
    alert = T >= 22
    tl = timeline(firing=firing, alert=alert)
    steps = tl.transitions()
    assert [s for _, s in steps] == [HEALTHY, DEVELOPING, FAULT]
    developing_at = steps[1][0]
    assert 16.0 <= developing_at <= 16.0 + SMOOTH_WINDOWS * 0.5  # smoothing delays, never leads
    assert steps[2][0] == 22.0
    assert tl.summary() == (f"healthy → developing anomaly at {developing_at:.1f} s → "
                            "persistent fault at 22.0 s")


def test_single_spike_is_smoothed_away():
    firing = np.zeros(T.size)
    firing[10] = 50.0
    assert set(timeline(firing=firing).stage) == {HEALTHY}


def test_lines_are_relative_to_their_own_threshold():
    tl = timeline(firing=np.full(T.size, 4.0), score=np.full(T.size, 1.0), threshold=2.0)
    assert tl.level[TRACKED[0][1]].iloc[-1] == pytest.approx(1.0)
    assert tl.level[OVERALL].iloc[-1] == pytest.approx(0.5)
    assert set(tl.stage) == {DEVELOPING}


def test_anomaly_score_above_its_threshold_alone_is_developing():
    tl = timeline(score=np.where(T >= 10, 3.0, 1.0), threshold=2.0)
    assert [s for _, s in tl.transitions()] == [HEALTHY, DEVELOPING]


def test_bad_quality_windows_are_no_data_and_do_not_feed_smoothing():
    firing = np.where(np.arange(T.size) % 2 == 0, 0.0, 100.0)
    ok = np.arange(T.size) % 2 == 0  # every high value is in a bad window
    tl = timeline(firing=firing, ok=ok)
    assert set(np.array(tl.stage)[~ok]) == {NO_DATA}
    assert set(np.array(tl.stage)[ok]) == {HEALTHY}
    assert all(s != NO_DATA for _, s in tl.transitions())


def test_stage_at_t_ignores_later_windows():
    firing = np.clip((T - 8) * 0.5, 0, None)
    full = timeline(firing=firing).stage
    k = 20
    z = pd.DataFrame({FIRING: firing[:k + 1], IMPULSE: np.zeros(k + 1)})
    prefix = health_timeline(T[:k + 1], z, np.ones(k + 1), 2.0, np.zeros(k + 1, bool),
                             np.ones(k + 1, bool)).stage
    assert prefix == full[:k + 1]


def test_missing_feature_columns_fall_back_to_anomaly_score():
    tl = health_timeline(T, pd.DataFrame(index=range(T.size)), np.full(T.size, 5.0), 2.0,
                         np.zeros(T.size, bool), np.ones(T.size, bool))
    assert set(tl.stage) == {DEVELOPING}
