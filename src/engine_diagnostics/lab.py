"""Fault-injection lab: modify a recorded healthy run, then replay it unchanged.

A judge picks a healthy base recording (by RPM), and the lab can inject a
misfire or knock-like fault from a chosen onset and severity, add background
noise, and make sensors fail. Injection reuses the MATLAB generator's source
model (damped modes, sensor delays, mode gains, resonance, gain) and the base
run's own combustion event schedule from its evaluation truth, so injected
changes land on real firing times. The lab plays the simulator's role: the
event schedule and injection log are never passed to the model, which sees
only the modified waveforms and telemetry through the normal replay path.

Approximation, stated in the dashboard: per-engine sensor gain and resonance
variation are not reproduced, and a misfire's RPM dip does not re-time later
events. Valve-clearance injection is not offered because its signature (impact
timing and shape changes) needs the full generator.
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

from engine_diagnostics.data.run_io import RunBundle

LAB_FAULTS = ("none", "misfire", "knock_like")
SENSORS = ("audio", "vibration", "rpm")
CYLINDERS = (1, 2, 3, 4)


@dataclass(frozen=True)
class LabSettings:
    base_run: str
    fault: str = "none"
    severity: float = 0.5
    onset_s: float = 8.0
    cylinder: int = 2
    noise_scale: float = 1.0  # total background noise relative to the recording's own
    failed: tuple[str, ...] = ()  # sensors that stop delivering data at fail_s
    fail_s: float = 12.0
    seed: int = 1


@dataclass
class InjectionLog:
    lines: list[str] = field(default_factory=list)
    fault_interval: tuple[float, float] | None = None
    event_times: list[float] = field(default_factory=list)
    fail_s: float | None = None


def base_runs(raw_dir: Path, truth_dir: Path, splits_config: Path,
              splits: tuple[str, ...] = ("tuning", "calibration", "test")) -> pd.DataFrame:
    """Clean healthy runs on engines the classifier was not fitted on, with mean RPM."""
    engine_splits = json.loads(splits_config.read_text(encoding="utf-8"))["engine_splits"]
    split_of = {e: s for s, engines in engine_splits.items() for e in engines}
    rows = []
    for path in sorted(truth_dir.glob("run_*_events.json")):
        truth = json.loads(path.read_text(encoding="utf-8"))
        disturbances = truth.get("disturbances", {})
        if (truth.get("condition") != "normal" or truth.get("challenge")
                or split_of.get(truth["engine_id"]) not in splits
                or disturbances.get("missing_stream") or disturbances.get("dropout_signal")
                or disturbances.get("clip_gain", 1) != 1
                or not (raw_dir / truth["run_id"]).is_dir()):
            continue
        tel = pd.read_csv(raw_dir / truth["run_id"] / "telemetry.csv")
        rpm = tel.loc[(tel["signal"] == "rpm") & (tel["valid"] == 1), "value"]
        rows.append({"run_id": truth["run_id"], "engine_id": truth["engine_id"],
                     "split": split_of[truth["engine_id"]], "profile": truth["profile_kind"],
                     "rpm": float(rpm.mean())})
    return pd.DataFrame(rows).sort_values("rpm").reset_index(drop=True)


def build_lab_bundle(bundle: RunBundle, truth: dict, settings: LabSettings,
                     generator: dict) -> tuple[RunBundle, InjectionLog]:
    """Return a modified copy of `bundle` and a description of every change."""
    rng = np.random.default_rng(settings.seed)
    duration = bundle.metadata["duration_s"]
    waves = {m: getattr(bundle, m).astype(np.float64) for m in ("audio", "vibration")}
    rates = {m: getattr(bundle, f"{m}_fs_hz") for m in waves}
    starts = {m: getattr(bundle, f"{m}_t0_s") for m in waves}
    telemetry = bundle.telemetry.copy()
    log = InjectionLog(lines=[f"Base recording {bundle.run_id} ({truth['engine_id']}), healthy"])
    sensors = generator["sensors"]

    if settings.fault != "none":
        transients, rpm_dips, times = _fault_events(truth, settings, generator, telemetry, rng)
        for m in waves:
            delta = _render(transients, sensors[m], generator, rates[m], starts[m],
                            waves[m].size, m)
            waves[m] += delta
        if rpm_dips:
            telemetry = _apply_rpm_dips(telemetry, rpm_dips, generator)
        log.fault_interval = (settings.onset_s, duration)
        log.event_times = times
        name = {"misfire": "misfire", "knock_like": "knock-like bursts"}[settings.fault]
        log.lines.append(f"INJECTED {name} on cylinder {settings.cylinder} from "
                         f"{settings.onset_s:g} s, severity {settings.severity:.2f} "
                         f"({len(times)} affected strokes)")

    if settings.noise_scale > 1:
        extra = np.sqrt(settings.noise_scale ** 2 - 1)  # total noise = scale x recorded noise
        for m in waves:
            waves[m] += extra * _noise(sensors[m], rates[m], waves[m].size, rng)
        log.lines.append(f"ADDED background noise: {settings.noise_scale:g}x the recorded "
                         "sensor noise")

    for m in waves:
        full = sensors[m]["full_scale"]
        waves[m] = np.clip(waves[m], -full, full)
    for sensor in settings.failed:
        if sensor == "rpm":
            gone = (telemetry["signal"] == "rpm") & (telemetry["t_s"] >= settings.fail_s)
            telemetry = telemetry.loc[~gone].reset_index(drop=True)
        else:
            first = max(0, round((settings.fail_s - starts[sensor]) * rates[sensor]))
            waves[sensor][first:] = np.nan
        log.fail_s = settings.fail_s
    if settings.failed:
        log.lines.append(f"SENSOR FAILURE from {settings.fail_s:g} s: "
                         f"{', '.join(settings.failed)} stop delivering data")

    metadata = copy.deepcopy(bundle.metadata)
    metadata["run_id"] = f"lab:{bundle.run_id}"
    metadata["lab"] = {"settings": settings.__dict__ | {"failed": list(settings.failed)}}
    lab = RunBundle(run_id=metadata["run_id"],
                    audio=waves["audio"].astype(np.float32),
                    vibration=waves["vibration"].astype(np.float32),
                    audio_fs_hz=bundle.audio_fs_hz, vibration_fs_hz=bundle.vibration_fs_hz,
                    audio_t0_s=bundle.audio_t0_s, vibration_t0_s=bundle.vibration_t0_s,
                    telemetry=telemetry, metadata=metadata,
                    cam_sync_s=None if bundle.cam_sync_s is None else bundle.cam_sync_s.copy())
    return lab, log


def _fault_events(truth, settings, generator, telemetry, rng):
    """Transients to add (source time, kind, amplitude, freq, tau) and RPM dips."""
    faults, sources = generator["faults"], generator["sources"]
    combustion = [e for e in truth["events"]
                  if e["kind"] == "combustion" and e["cylinder"] == settings.cylinder
                  and e["time_s"] >= settings.onset_s]
    load = _interp_signal(telemetry, "load")
    rpm = _interp_signal(telemetry, "rpm")
    transients, dips, times = [], [], []
    sev = settings.severity
    if settings.fault == "misfire":
        f = faults["misfire"]
        p = min(1.0, f["base_probability"] + f["probability_per_severity"] * sev)
        for e in combustion:
            if rng.random() >= p:
                continue
            residual = rng.uniform(*f["residual_range"])
            # Remove (1 - residual) of the recorded combustion transient.
            transients.append(("combustion", e["time_s"], (residual - 1) * e["intensity"],
                               load(e["time_s"])))
            dips.append(e["time_s"])
            times.append(e["time_s"])
    elif settings.fault == "knock_like":
        f, src = faults["knock_like"], sources["knock_burst"]
        p = min(1.0, f["base_probability"] + f["probability_per_severity"] * sev)
        engine_freq = rng.uniform(*generator["engine_variation"]["knock_freq_range_hz"])
        for e in combustion:
            if rng.random() >= p:
                continue
            delay_s = rng.uniform(*f["delay_range_deg"]) / (6 * max(rpm(e["time_s"]), 1.0))
            freq = np.clip(engine_freq + f["freq_jitter_hz"] * rng.standard_normal(),
                           *src["freq_range_hz"])
            tau = rng.uniform(*src["tau_range_s"])
            intensity = sev * (0.6 + 0.8 * rng.random()) * e["intensity"]
            transients.append(("knock_burst", e["time_s"] + delay_s, intensity, freq, tau))
            times.append(e["time_s"] + delay_s)
    return transients, dips, times


def _render(transients, sensor, generator, fs, t0, n, modality) -> np.ndarray:
    """Generator rendering of the given transients: modes, resonance, gain."""
    x = np.zeros(n)
    t = t0 + np.arange(n) / fs
    for item in transients:
        if item[0] == "combustion":
            _, time_s, scale, load = item
            src = generator["sources"]["combustion"]
            amps = np.asarray(sensor["combustion_gains"]) * (0.5 + load) * scale
            freqs, taus, ramp = src["freq_hz"], src["tau_s"], src["ramp_s"]
        else:
            _, time_s, intensity, freq, tau = item
            amps = [intensity * sensor["knock_gain"]]
            freqs, taus = [freq], [tau]
            ramp = generator["sources"]["knock_burst"]["ramp_s"]
        _add_transient(x, t, fs, time_s + sensor["delay_s"], amps, freqs, taus, ramp)
    lo, hi = sensor["resonance_hz"]
    b, a = butter(2, [lo, hi], btype="bandpass", fs=fs)
    x = x + sensor["resonance_mix"] * lfilter(b, a, x)
    return sensor["gain"] * x


def _add_transient(x, t, fs, t0, amps, freqs, taus, ramp_s) -> None:
    first = max(0, int(np.ceil((t0 - t[0]) * fs)))
    last = min(x.size, int(np.floor((t0 + 6 * max(taus) - t[0]) * fs)) + 1)
    if first >= last:
        return
    dt = t[first:last] - t0
    onset = 0.5 * (1 - np.cos(np.pi * np.minimum(dt / ramp_s, 1)))
    burst = sum(a * np.exp(-dt / tau) * np.sin(2 * np.pi * f * dt)
                for a, f, tau in zip(amps, freqs, taus, strict=True))
    x[first:last] += onset * burst


def _noise(sensor, fs, n, rng) -> np.ndarray:
    """The generator's background noise model: low-passed colored plus white noise."""
    b, a = butter(2, sensor["noise_lowpass_hz"], fs=fs)
    colored = lfilter(b, a, rng.standard_normal(n))
    colored *= sensor["colored_noise_rms"] / (np.std(colored) + 1e-20)
    return colored + sensor["white_noise_rms"] * rng.standard_normal(n)


def _apply_rpm_dips(telemetry, dip_times, generator) -> pd.DataFrame:
    f = generator["faults"]["misfire"]
    rows = (telemetry["signal"] == "rpm") & (telemetry["valid"] == 1)
    t = telemetry.loc[rows, "t_s"].to_numpy()[:, None]
    dips = f["rpm_dip_rpm"] * np.exp(-0.5 * ((t - np.asarray(dip_times)[None, :])
                                            / f["rpm_dip_sigma_s"]) ** 2)
    telemetry = telemetry.copy()
    telemetry.loc[rows, "value"] = (telemetry.loc[rows, "value"] - dips.sum(axis=1)).round()
    return telemetry


def _interp_signal(telemetry, signal):
    rows = telemetry[(telemetry["signal"] == signal) & (telemetry["valid"] == 1)]
    t, v = rows["t_s"].to_numpy(), rows["value"].to_numpy(dtype=float)
    if signal == "load":
        v = v / 100  # telemetry percent -> generator's normalized load
    return lambda time: float(np.interp(time, t, v)) if t.size else 0.5
