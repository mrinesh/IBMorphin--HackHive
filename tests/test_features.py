"""Feature definitions against known signals, causality, labels, and context freshness."""

import ast
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from engine_diagnostics.config import load_config  # noqa: E402
from engine_diagnostics.pipeline import features as F  # noqa: E402
from engine_diagnostics.pipeline.labels import TRANSITION, window_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RATES = {"audio": 24000, "vibration": 25600}
SCALES = {"audio": 1.0, "vibration": 500.0}
CONTEXT = {"rpm": 1500.0, "load": 40.0, "throttle": 42.0}


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/mvp.json", ROOT / "schemas/project-config.schema.json")


def tone(freq, amplitude, fs, seconds=2.0):
    t = np.arange(round(seconds * fs)) / fs
    return amplitude * np.sin(2 * np.pi * freq * t)


def features_of(waves, config, context=CONTEXT, fresh=True):
    result = F.extract_window(waves, RATES, context, fresh, SCALES, 2.0, config)
    return dict(zip(F.all_feature_names(), result.features, strict=True)), result.quality


def test_feature_names_are_unique_and_exclude_bookkeeping():
    names = F.feature_names()
    assert len(names) == len(set(names)) == 2 * len(F.PER_MODALITY) + len(F.CONTEXT)
    forbidden = ("run", "engine", "label", "seed", "severity", "cylinder", "onset")
    assert not any(word in name for name in names for word in forbidden)


def test_sine_statistics_and_band_placement(config):
    # 6037 Hz avoids sampling a sine at a few fixed phases (6 kHz at 24 kHz is 4 points).
    waves = {m: tone(6037, 0.5, RATES[m]) for m in F.MODALITIES}
    values, quality = features_of(waves, config)
    assert quality == "ok"
    for m in F.MODALITIES:
        assert values[f"{m}_log_rms"] == pytest.approx(np.log10(0.5 / np.sqrt(2)), abs=1e-3)
        assert values[f"{m}_crest"] == pytest.approx(np.sqrt(2), rel=1e-3)
        assert values[f"{m}_kurtosis"] == pytest.approx(-1.5, abs=1e-2)
        # 6.04 kHz falls in band 4 (5-7 kHz); Welch power equals the sine power.
        assert values[f"{m}_band_frac_4"] > 0.99
        assert values[f"{m}_log_band_power"] == pytest.approx(np.log10(0.125), abs=0.01)


def test_impulsive_high_band_raises_envelope_kurtosis(config):
    fs = RATES["vibration"]
    steady = tone(6000, 0.1, fs)
    bursty = steady.copy()
    for start in range(0, bursty.size - 100, fs // 25):
        bursty[start:start + 60] += np.sin(2 * np.pi * 7000 * np.arange(60) / fs)
    calm, _ = features_of({"audio": tone(6000, 0.1, 24000), "vibration": steady}, config)
    hit, _ = features_of({"audio": tone(6000, 0.1, 24000), "vibration": bursty}, config)
    assert hit["vibration_env_kurtosis_high"] > calm["vibration_env_kurtosis_high"] + 5


def test_missing_firings_raise_firing_peak_cv(config):
    fs = RATES["audio"]
    firing_hz = 1500 / 60 * 2
    pulses = np.zeros(2 * fs)
    weak = np.zeros(2 * fs)
    burst = np.sin(2 * np.pi * 1200 * np.arange(120) / fs) * np.exp(-np.arange(120) / 30)
    for k, start in enumerate(np.arange(0, 2 * fs - 200, fs / firing_hz).astype(int)):
        pulses[start:start + 120] += burst
        weak[start:start + 120] += burst * (0.1 if k % 4 == 1 else 1.0)
    vib = tone(500, 0.01, RATES["vibration"])
    even, _ = features_of({"audio": pulses, "vibration": vib}, config)
    missing, _ = features_of({"audio": weak, "vibration": vib}, config)
    assert even["audio_firing_peak_cv"] < 0.1 < missing["audio_firing_peak_cv"]


def test_quality_states(config):
    waves = {m: tone(1000, 0.1, RATES[m]) for m in F.MODALITIES}
    short = {**waves, "audio": waves["audio"][:-100]}
    assert features_of(short, config)[1] == "missing_waveform"
    clipped = {**waves, "vibration": np.full(RATES["vibration"] * 2, 500.0)}
    assert features_of(clipped, config)[1] == "clipped"
    assert features_of(waves, config, fresh=False)[1] == "stale_telemetry"


def test_window_slice_is_causal_and_never_wraps(config):
    assert F.window_slice(2.0, 24000, 0.0, config) == slice(0, 48000)
    assert F.window_slice(5.5, 24000, 0.0, config) == slice(84000, 132000)
    early = F.window_slice(1.0, 24000, 0.0, config)
    assert early.start == 0 and early.stop - early.start < 48000
    np.testing.assert_allclose(F.window_ends(20, config)[[0, -1]], [2.0, 20.0])
    assert len(F.window_ends(20, config)) == 37


def test_telemetry_context_uses_only_past_fresh_readings(config):
    telemetry = pd.DataFrame({
        "t_s": [0.0, 0.0, 0.0, 1.96, 1.98, 2.0],
        "signal": ["load", "throttle", "rpm", "rpm", "rpm", "rpm"],
        "value": [40.0, 42.0, 1400.0, 1490.0, 1500.0, 9999.0],
        "unit": ["percent", "percent", "rpm", "rpm", "rpm", "rpm"],
        "valid": [1, 1, 1, 1, 1, 1],
    })
    values, fresh = F.telemetry_context(telemetry, 2.0, config)
    assert values["rpm"] == 1500.0  # the reading at exactly t is in the future
    assert not fresh  # load/throttle are 2 s old, beyond the 0.3 s limit


def test_window_labels_mark_transitions():
    truth = {"label_timeline": [
        {"start_s": 0, "end_s": 7.5, "label": "normal"},
        {"start_s": 7.5, "end_s": 8.5, "label": "transition"},
        {"start_s": 8.5, "end_s": 20, "label": "misfire"},
    ]}
    labels = window_labels(truth, np.array([7.5, 8.0, 10.0, 10.5]), 2.0)
    assert labels == ["normal", "transition", TRANSITION, "misfire"]
    single = {"label_timeline": {"start_s": 0, "end_s": 20, "label": "normal"}}
    assert window_labels(single, np.array([2.0]), 2.0) == ["normal"]


def test_inference_modules_never_import_labels_or_truth():
    for module in ("features.py",):
        tree = ast.parse((ROOT / "src/engine_diagnostics/pipeline" / module).read_text())
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imported |= {a.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                     for a in node.names}
        assert not any("labels" in name or "dataset" in name for name in imported if name)


def test_stroke_sync_excess_locates_a_one_cylinder_burst():
    """Combustion pulses every 180 degrees; a 7 kHz burst ~20 degrees later on one cylinder."""
    fs, rpm = 25600, 1800.0
    period = 30 / rpm  # one firing per 180 crank degrees
    t = np.arange(round(2.0 * fs)) / fs
    x = np.zeros_like(t)
    rng = np.random.default_rng(1)
    for k, start in enumerate(np.arange(0.003, 2.0 - period, period)):
        rel = t - start
        on = rel >= 0
        x[on] += np.sin(2 * np.pi * 600 * rel[on]) * np.exp(-rel[on] / 0.004)
        if k % 4 == 1:  # the same cylinder every fourth stroke
            burst = rel - 20 / (6 * rpm)
            hit = burst >= 0
            x[hit] += 0.5 * np.sin(2 * np.pi * 7000 * burst[hit]) * np.exp(-burst[hit] / 0.0008)
    x += 0.001 * rng.normal(size=x.size)
    names = F.sync_feature_names()
    values = dict(zip(names, F._stroke_sync_features(x, fs, rpm), strict=True))
    high = {n: v for n, v in values.items() if n.startswith("sync_high_")}
    assert max(high, key=high.get) == "sync_high_cyl_excess_15_30"
    assert values["sync_ref_jitter"] < 0.05
    # Without the burst no stroke group stands out.
    assert F._stroke_sync_features(x * 0, fs, rpm).tolist() == [0.0] * len(names)
    assert np.isnan(F._stroke_sync_features(x, fs, float("nan"))).sum() == 0
