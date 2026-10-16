"""Compact, citable context about one engine for Diagnose (possible causes).

Built only from the engine's history: profile and specs, saved runs (results and
per-window decisions), notes, and signal summaries computed here from the raw
data history still keeps. Raw waveforms never leave this module; only numbers
derived from them do.

Every item has a stable ID the diagnosis must cite:

    SPEC            the engine profile and specs
    R<id>           a saved run (history run id)
    N<id>           a note
    S<id>-<kind>    a signal summary of run <id>: crank, thermal, fuel, spectrum
    T-<name>        a trend across runs: T-<fault>, T-anomaly, T-health

Leakage: the context holds what the dashboard user saw and wrote. It never reads
evaluation truth, and it leaves out the lab injection settings and the recording
id, which describe the simulated condition (docs/DATA_CONTRACT.md).
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engine_diagnostics.history import HistoryStore, Profile, RawData, RunEntry
from engine_diagnostics.profiles import reliability_warning

ROOT = Path(__file__).resolve().parents[2]
FAULTS = ("misfire", "knock_like", "valve_clearance_like")
CLASSES = ("normal", *FAULTS)
SPEC_SOURCE_TEXT = {"catalog": "from the engine catalog",
                    "catalog_edited": "from the engine catalog, edited by hand",
                    "manual": "entered by hand"}
AUDIO_BANDS_HZ = {"below_1k": (0, 1000), "1k_5k": (1000, 5000), "knock_5k_9k": (5000, 9000)}
TREND_STEP = 0.1  # change in mean max-likelihood (newer half vs older half) called a trend


def _r(value, digits: int = 3):
    """Round for a compact context; non-finite numbers become None."""
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def run_id(run: RunEntry) -> str:
    return f"R{run.id}"


# -- runs ---------------------------------------------------------------------------------
def run_item(store: HistoryStore, run: RunEntry) -> dict[str, Any]:
    """Results of one saved run, with likelihoods and health stages from its decisions."""
    s = run.summary
    decisions = store.decisions(run.id)
    ok = decisions[decisions["quality"] == "ok"] if not decisions.empty else decisions
    mean_p: dict[str, float | None] = {}
    if not ok.empty and "probabilities" in ok:
        probs = pd.DataFrame([p or {} for p in ok["probabilities"]])
        mean_p = {c: _r(probs[c].mean()) for c in probs.columns}
    stages = (decisions["health_stage"].dropna().value_counts().to_dict()
              if "health_stage" in decisions else {})
    operating: dict[str, Any] = {}
    if not ok.empty and "context" in ok:
        ctx = pd.DataFrame([c or {} for c in ok["context"]])
        for key in ("rpm", "load"):
            if key in ctx and ctx[key].notna().any():
                operating[f"{key}_mean"] = _r(ctx[key].mean(), 2)
                operating[f"{key}_range"] = [_r(ctx[key].min(), 2), _r(ctx[key].max(), 2)]
    windows = s.get("windows") or 0
    return {
        "id": run_id(run),
        "saved_utc": run.created_utc,
        "mode": ("fault-injection lab (signals modified on purpose)"
                 if run.source.get("mode") == "lab" else "recorded replay"),
        "complete": run.complete,
        "analysed_s": _r(run.analyzed_s, 1), "duration_s": _r(run.duration_s, 1),
        "predicted_fault": s.get("predicted_fault"),
        "alerts": [{"fault": a["fault"], "onset_s": _r(a["onset_s"], 1)}
                   for a in s.get("alerts", [])],
        "max_likelihoods": {c: _r(p) for c, p in (s.get("max_probabilities") or {}).items()},
        "mean_likelihoods": mean_p,
        "final_health_stage": s.get("final_health_stage"),
        "health_stage_counts": {str(k): int(v) for k, v in stages.items()},
        "early_warning_leads_s": [_r(v, 1) for v in s.get("warning_leads_s") or []],
        "peak_anomaly_score": _r(s.get("peak_anomaly_score")),
        "anomaly_threshold": _r(s.get("anomaly_threshold")),
        "ok_window_fraction": _r(s.get("ok_windows", 0) / windows) if windows else None,
        "decision_counts": s.get("decision_counts", {}),
        "localization": s.get("localization"),
        "operating_point": operating,
        "model_version": s.get("model_version"),
        "raw_data": "kept" if run.raw_available else "pruned",
    }


# -- signal summaries ---------------------------------------------------------------------
def _telemetry(raw: RawData, signal: str) -> pd.DataFrame:
    tel = raw.telemetry
    if tel.empty:
        return tel
    return tel[(tel["signal"] == signal) & (tel["valid"] == 1)].dropna(subset=["value"])


def crank_summary(raw: RawData, config: dict[str, Any]) -> dict[str, Any] | None:
    """Per-cylinder crank-speed deviation and roughness from the 60-2 wheel."""
    from engine_diagnostics.pipeline.crankspeed import FLAG_SCORE, stroke_speeds

    if raw.crank_tooth_s is None or len(raw.crank_tooth_s) == 0:
        return None
    strokes = stroke_speeds(raw.crank_tooth_s, raw.cam_sync_s, config)
    if strokes.time_s.size == 0:
        return None
    phased = strokes.phase == "cam_sync"
    keys = strokes.cylinder if phased else strokes.position
    mean_rpm = float(np.nanmean(strokes.rpm))
    per: dict[str, Any] = {}
    for key in sorted({int(k) for k in keys}):
        sel = keys == key
        score = strokes.score[sel]
        finite = score[np.isfinite(score)]
        per[f"cyl{key}" if phased else f"firing_position{key}"] = {
            "strokes": int(sel.sum()),
            "speed_deviation_pct": _r(100 * (np.nanmean(strokes.rpm[sel]) / mean_rpm - 1), 3),
            "mean_roughness": _r(finite.mean(), 2) if finite.size else None,
            "flagged_stroke_pct": _r(100 * (finite >= FLAG_SCORE).mean(), 1)
            if finite.size else None,
        }
    roughest = max(per, key=lambda k: per[k]["mean_roughness"]
                   if per[k]["mean_roughness"] is not None else -math.inf)
    return {"kind": "crank_speed", "phase": strokes.phase, "mean_rpm": _r(mean_rpm, 1),
            "roughness_flag": FLAG_SCORE, "per_cylinder": per, "roughest": roughest,
            "note": "roughness = speed drop across a power stroke over the run's robust spread; "
                    "a misfiring cylinder decelerates (high roughness, flagged strokes)."}


def thermal_summary(raw: RawData, window_s: float) -> dict[str, Any] | None:
    """Per-cylinder EGT/CHT: late level vs the other cylinders, and change since the start."""
    out: dict[str, Any] = {"kind": "cylinder_temperatures", "window_s": window_s}
    for kind in ("egt", "cht"):
        per: dict[str, dict[str, float]] = {}
        c = 1
        while not (rows := _telemetry(raw, f"{kind}_cyl{c}")).empty:
            start, end = rows["t_s"].min(), rows["t_s"].max()
            early = rows[rows["t_s"] < start + window_s]["value"].mean()
            late = rows[rows["t_s"] > end - window_s]["value"].mean()
            per[f"cyl{c}"] = {"late": late, "change": late - early}
            c += 1
        if not per:
            continue
        median = float(np.median([v["late"] for v in per.values()]))
        out[kind] = {name: {"late_c": _r(v["late"], 1), "vs_median_c": _r(v["late"] - median, 1),
                            "change_since_start_c": _r(v["change"], 1)}
                     for name, v in per.items()}
        late = [v["late"] for v in per.values()]
        out[f"{kind}_spread_c"] = _r(max(late) - min(late), 1)
    return out if ("egt" in out or "cht" in out) else None


def fuel_summary(raw: RawData, window_s: float) -> dict[str, Any] | None:
    """Short- and long-term fuel trim levels and drift over the run."""
    out: dict[str, Any] = {"kind": "fuel_trims", "unit": "percent"}
    for signal in ("short_term_fuel_trim", "long_term_fuel_trim"):
        rows = _telemetry(raw, signal)
        if rows.empty:
            continue
        start, end = rows["t_s"].min(), rows["t_s"].max()
        first = rows[rows["t_s"] < start + window_s]["value"].mean()
        last = rows[rows["t_s"] > end - window_s]["value"].mean()
        out[signal] = {"mean": _r(rows["value"].mean(), 2), "start": _r(first, 2),
                       "end": _r(last, 2), "drift": _r(last - first, 2)}
    return out if len(out) > 2 else None


def spectrum_summary(raw: RawData) -> dict[str, Any] | None:
    """Audio band-energy fractions and audio/vibration level, first vs last third of the run."""
    from scipy.signal import welch

    audio, vib = np.asarray(raw.audio, dtype=float), np.asarray(raw.vibration, dtype=float)
    if audio.size < 3 * 4096 or vib.size < 3 * 256:
        return None

    def band_power(x: np.ndarray) -> dict[str, float]:
        f, p = welch(x, fs=raw.audio_fs_hz, nperseg=4096)
        return {"total": float(p.sum()), **{name: float(p[(f >= lo) & (f < hi)].sum())
                                            for name, (lo, hi) in AUDIO_BANDS_HZ.items()}}

    def fractions(power: dict[str, float]) -> dict[str, float | None]:
        return {name: _r(power[name] / power["total"]) if power["total"] > 0 else None
                for name in AUDIO_BANDS_HZ}

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x))))

    def ratio(x: np.ndarray, n: int):
        return _r(rms(x[-n:]) / rms(x[:n])) if rms(x[:n]) > 0 else None

    a, v = audio.size // 3, vib.size // 3
    early, late = band_power(audio[:a]), band_power(audio[-a:])
    change = {name: _r(10 * math.log10(late[name] / early[name]), 1)
              if early[name] > 0 and late[name] > 0 else None for name in AUDIO_BANDS_HZ}
    return {"kind": "spectrum", "audio_band_fraction_first_third": fractions(early),
            "audio_band_fraction_last_third": fractions(late),
            "audio_band_energy_change_db": change,
            "audio_rms_ratio_last_vs_first": ratio(audio, a),
            "vibration_rms_ratio_last_vs_first": ratio(vib, v)}


def signal_summaries(store: HistoryStore, run: RunEntry, config: dict[str, Any] | None,
                     window_s: float) -> list[dict[str, Any]]:
    """Summaries of a run's kept raw data (none once raw data is pruned)."""
    if not run.raw_available:
        return []
    try:
        raw = store.load_raw(run.id)
    except (OSError, ValueError, KeyError):
        return []
    if raw is None:
        return []
    found = {"crank": crank_summary(raw, config) if config else None,
             "thermal": thermal_summary(raw, window_s),
             "fuel": fuel_summary(raw, window_s),
             "spectrum": spectrum_summary(raw)}
    return [{"id": f"S{run.id}-{name}", "run": run_id(run), **summary}
            for name, summary in found.items() if summary is not None]


# -- trends -------------------------------------------------------------------------------
def _direction(values: list[float | None]) -> str:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return "not enough runs"
    half = len(vals) // 2
    older, newer = np.mean(vals[:half]), np.mean(vals[-half:])
    if newer - older >= TREND_STEP:
        return "rising"
    if older - newer >= TREND_STEP:
        return "falling"
    return "steady"


def trends(runs_oldest_first: list[RunEntry]) -> list[dict[str, Any]]:
    """Fault likelihood, anomaly and health trends across runs (oldest to newest)."""
    ids = [run_id(r) for r in runs_oldest_first]
    out = []
    for fault in FAULTS:
        values = [_r((r.summary.get("max_probabilities") or {}).get(fault))
                  for r in runs_oldest_first]
        alerted = [fault in {a["fault"] for a in r.summary.get("alerts", [])}
                   for r in runs_oldest_first]
        streak = 0
        for hit in reversed(alerted):
            if not hit:
                break
            streak += 1
        out.append({"id": f"T-{fault}", "fault": fault, "runs": ids,
                    "max_likelihood_by_run": values, "direction": _direction(values),
                    "runs_alerted": int(sum(alerted)), "latest_alert_streak": streak})
    ratio = []
    for r in runs_oldest_first:
        peak, threshold = r.summary.get("peak_anomaly_score"), r.summary.get("anomaly_threshold")
        ratio.append(_r(peak / threshold) if peak is not None and threshold else None)
    out.append({"id": "T-anomaly", "runs": ids, "peak_anomaly_over_threshold_by_run": ratio,
                "direction": _direction(ratio)})
    out.append({"id": "T-health", "runs": ids,
                "final_health_stage_by_run": [r.summary.get("final_health_stage")
                                              for r in runs_oldest_first]})
    return out


# -- whole context ------------------------------------------------------------------------
def build_context(store: HistoryStore, profile: Profile, *, max_runs: int = 8,
                  max_notes: int = 20, window_s: float = 5.0,
                  config: dict[str, Any] | None = None) -> dict[str, Any]:
    """The context object for one engine. `config` (configs/mvp.json) enables crank summaries."""
    all_runs = store.list_runs(profile.id)  # newest first
    runs = all_runs[:max_runs]
    all_notes = store.list_notes(profile.id)
    notes = all_notes[:max_notes]
    items = [run_item(store, r) for r in runs]
    summaries = [s for r in runs for s in signal_summaries(store, r, config, window_s)]
    return {
        "engine": {
            "id": "SPEC", "make": profile.make, "model": profile.model,
            "serial": profile.serial, "nickname": profile.nickname, "specs": profile.specs,
            "spec_source": profile.spec_source,
            "spec_source_text": SPEC_SOURCE_TEXT.get(profile.spec_source, profile.spec_source),
            "reliability_warning": reliability_warning(profile.specs),
        },
        "detection_classes": list(CLASSES),
        "current_run": items[0] if items else None,
        "past_runs": items[1:],
        "trends": trends(list(reversed(runs))) if runs else [],
        "notes": [{"id": f"N{n.id}", "created_utc": n.created_utc,
                   "run": f"R{n.run_id}" if n.run_id else None, "text": n.text} for n in notes],
        "signal_summaries": summaries,
        "coverage": {"runs_saved": len(all_runs), "runs_included": len(runs),
                     "runs_with_raw_data": sum(r.raw_available for r in runs),
                     "notes_saved": len(all_notes), "notes_included": len(notes)},
    }


def context_ids(context: dict[str, Any]) -> set[str]:
    """Every ID a diagnosis may cite."""
    ids = {"SPEC"}
    runs = ([context["current_run"]] if context.get("current_run") else []) + context.get(
        "past_runs", [])
    for group in (runs, context.get("notes", []), context.get("signal_summaries", []),
                  context.get("trends", [])):
        ids |= {item["id"] for item in group}
    return ids


def context_digest(context: dict[str, Any]) -> str:
    """sha256 of the context (canonical JSON), saved with each diagnosis."""
    text = json.dumps(context, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
