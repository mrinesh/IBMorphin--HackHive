"""Per-engine history: profiles, saved analysis runs, results, notes, and raw data.

Storage is one SQLite file (stdlib `sqlite3`) plus one folder of raw data per
saved run, under a history directory (default data/engine_history, overridden
by ENGINE_HISTORY_DIR):

    history.sqlite                 profiles, runs, decisions, notes, diagnoses
    raw/profile_<id>/run_<id>/     telemetry.csv (long form) and signals.npz

Results (run summary and per-window decisions), notes, and saved diagnoses are
kept permanently.
Raw data is kept for the most recent `raw_keep` runs of each engine (default
10, ENGINE_HISTORY_RAW_KEEP); older raw folders are deleted when a run is
saved, and the run keeps its results. Nothing is written until the first
write call, so reading an empty history creates no files.

History holds what the engine's user saw and wrote. It never stores
evaluation truth, demo preset names, notes, or scenario tags (those describe
the simulated condition). See docs/DATA_CONTRACT.md, "Engine history".
"""

import json
import math
import os
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engine_diagnostics.profiles import SPEC_SOURCES, validate_specs
from engine_diagnostics.runtime.replay import localization_summary

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "data/engine_history"
DEFAULT_RAW_KEEP = 10
HISTORY_VERSION = "1.0"
TELEMETRY_COLUMNS = ["t_s", "signal", "value", "unit", "valid"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY,
    make TEXT NOT NULL, model TEXT NOT NULL, serial TEXT NOT NULL, nickname TEXT,
    identity_key TEXT NOT NULL UNIQUE,
    specs_json TEXT NOT NULL, spec_source TEXT NOT NULL,
    created_utc TEXT NOT NULL, updated_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    profile_id INTEGER NOT NULL REFERENCES profiles(id),
    created_utc TEXT NOT NULL,
    source_json TEXT NOT NULL, specs_json TEXT NOT NULL, summary_json TEXT NOT NULL,
    feature_names_json TEXT NOT NULL,
    complete INTEGER NOT NULL, analyzed_s REAL NOT NULL, duration_s REAL NOT NULL,
    raw_path TEXT, raw_pruned_utc TEXT);
CREATE INDEX IF NOT EXISTS runs_by_profile ON runs(profile_id, id);
CREATE TABLE IF NOT EXISTS decisions (
    run_id INTEGER NOT NULL REFERENCES runs(id), idx INTEGER NOT NULL,
    t_end_s REAL NOT NULL, quality TEXT NOT NULL, predicted TEXT, accepted TEXT,
    alert TEXT, anomaly_score REAL, margin REAL, probabilities_json TEXT,
    health_stage TEXT, localization TEXT, context_json TEXT, features_json TEXT,
    PRIMARY KEY (run_id, idx));
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY,
    profile_id INTEGER NOT NULL REFERENCES profiles(id),
    run_id INTEGER REFERENCES runs(id),
    created_utc TEXT NOT NULL, text TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS diagnoses (
    id INTEGER PRIMARY KEY,
    profile_id INTEGER NOT NULL REFERENCES profiles(id),
    latest_run_id INTEGER REFERENCES runs(id),
    created_utc TEXT NOT NULL, model TEXT NOT NULL, fallback INTEGER NOT NULL,
    fallback_reason TEXT, inputs_digest TEXT NOT NULL, output_json TEXT NOT NULL);
"""


@dataclass(frozen=True)
class Profile:
    id: int
    make: str
    model: str
    serial: str
    nickname: str | None
    specs: dict[str, Any]
    spec_source: str
    created_utc: str

    @property
    def label(self) -> str:
        name = f"{self.make} {self.model} · {self.serial}"
        return f"{self.nickname} ({name})" if self.nickname else name


@dataclass(frozen=True)
class RunEntry:
    id: int
    profile_id: int
    created_utc: str
    source: dict[str, Any]
    specs: dict[str, Any]
    summary: dict[str, Any]
    complete: bool
    analyzed_s: float
    duration_s: float
    raw_available: bool


@dataclass(frozen=True)
class Note:
    id: int
    profile_id: int
    run_id: int | None
    created_utc: str
    text: str


@dataclass(frozen=True)
class Diagnosis:
    """A saved diagnosis: the output (summary, causes, actions, data gaps) and how it was made."""
    id: int
    profile_id: int
    latest_run_id: int | None
    created_utc: str
    model: str
    fallback: bool
    fallback_reason: str | None
    inputs_digest: str
    output: dict[str, Any]


@dataclass
class RawData:
    """The signals the model received for one analysis (up to the analyzed time)."""
    audio: np.ndarray
    vibration: np.ndarray
    audio_fs_hz: float
    vibration_fs_hz: float
    telemetry: pd.DataFrame
    cam_sync_s: np.ndarray | None = None
    crank_tooth_s: np.ndarray | None = None


def raw_from_bundle(bundle, up_to_s: float) -> RawData:
    """Raw data a replay had received by stream time up_to_s (t0 = 0, as in replay)."""
    tel = bundle.telemetry
    cam = bundle.cam_sync_s
    teeth = getattr(bundle, "crank_tooth_s", None)
    return RawData(
        audio=np.asarray(bundle.audio[:round(up_to_s * bundle.audio_fs_hz)]),
        vibration=np.asarray(bundle.vibration[:round(up_to_s * bundle.vibration_fs_hz)]),
        audio_fs_hz=float(bundle.audio_fs_hz), vibration_fs_hz=float(bundle.vibration_fs_hz),
        telemetry=tel[tel["t_s"] < up_to_s][TELEMETRY_COLUMNS].reset_index(drop=True),
        cam_sync_s=None if cam is None else np.asarray(cam[cam < up_to_s]),
        crank_tooth_s=None if teeth is None else np.asarray(teeth[teeth < up_to_s]))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _clean(value):
    """JSON-safe copy: NaN/inf -> None, numpy scalars/arrays -> Python."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [_clean(v) for v in value]
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(value) else None
    return value


def _dumps(value) -> str:
    return json.dumps(_clean(value), allow_nan=False)


def _identity_key(make: str, model: str, serial: str) -> str:
    return "|".join(" ".join(s.split()).casefold() for s in (make, model, serial))


def summarize_run(records: list[dict[str, Any]], stages: list[str] | None = None,
                  health_summary: str | None = None,
                  warning_leads: list[float] | None = None) -> dict[str, Any]:
    """Run-level results from replay decision records (and the health timeline, if any).

    predicted_fault: the faults alerted during the run joined with '+', 'normal'
    when no alert was raised, or 'no data' when no window passed quality checks.
    """
    onsets: list[dict[str, Any]] = []
    previous: set[str] = set()
    for r in records:
        active = set(r.get("alerts") or ([r["alert"]] if r.get("alert") else []))
        onsets += [{"fault": f, "onset_s": float(r["t_end_s"])} for f in sorted(active - previous)]
        previous = active
    ok = [r for r in records if r["quality"] == "ok"]
    alerted = sorted({o["fault"] for o in onsets})
    max_p: dict[str, float] = {}
    for r in ok:
        for c, p in (r.get("probabilities") or {}).items():
            if p is not None and math.isfinite(p):
                max_p[c] = max(max_p.get(c, 0.0), float(p))
    scores = [r["anomaly_score"] for r in ok if math.isfinite(r["anomaly_score"])]
    counts = pd.Series([r["accepted"] if r["quality"] == "ok" else r["quality"]
                        for r in records], dtype=object).value_counts()
    assessed = [s for s in (stages or []) if s != "no data"]
    latest = records[-1] if records else {}
    return _clean({
        "predicted_fault": "+".join(alerted) if alerted else ("normal" if ok else "no data"),
        "alerts": onsets,
        "decision_counts": {str(k): int(v) for k, v in counts.items()},
        "windows": len(records), "ok_windows": len(ok),
        "max_probabilities": max_p,
        "peak_anomaly_score": max(scores) if scores else None,
        "mean_anomaly_score": float(np.mean(scores)) if scores else None,
        "final_health_stage": assessed[-1] if assessed else None,
        "health_summary": health_summary,
        "warning_leads_s": list(warning_leads or []),
        "localization": latest.get("localization"),
        "model_version": latest.get("model_version"),
        "feature_version": latest.get("feature_version"),
    })


class HistoryStore:
    """Engine profiles and their saved runs. Every method opens its own connection."""

    def __init__(self, root: Path | str | None = None, raw_keep: int | None = None):
        self.root = Path(root or os.environ.get("ENGINE_HISTORY_DIR") or DEFAULT_DIR)
        keep = raw_keep if raw_keep is not None else int(
            os.environ.get("ENGINE_HISTORY_RAW_KEEP", DEFAULT_RAW_KEEP))
        if keep < 0:
            raise ValueError("raw_keep must be 0 or more")
        self.raw_keep = keep
        self.db_path = self.root / "history.sqlite"

    # -- connection -------------------------------------------------------------------
    def exists(self) -> bool:
        return self.db_path.is_file()

    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.executescript(_SCHEMA)
        con.execute("INSERT OR IGNORE INTO meta VALUES ('history_version', ?)",
                    (HISTORY_VERSION,))
        return con

    def _read(self, sql: str, args=()) -> list[sqlite3.Row]:
        if not self.exists():
            return []
        with closing(self._connect()) as con:
            return con.execute(sql, args).fetchall()

    # -- profiles ---------------------------------------------------------------------
    def create_profile(self, make: str, model: str, serial: str, specs: dict[str, Any],
                       spec_source: str, nickname: str | None = None) -> Profile:
        make, model, serial = (" ".join(s.split()) for s in (make, model, serial))
        if not (make and model and serial):
            raise ValueError("make, model, and serial/ID are required")
        self._check_specs(specs, spec_source)
        key = _identity_key(make, model, serial)
        now = _now()
        with closing(self._connect()) as con, con:
            if con.execute("SELECT 1 FROM profiles WHERE identity_key = ?", (key,)).fetchone():
                raise ValueError(f"a profile for {make} {model} serial {serial} already exists")
            cur = con.execute(
                "INSERT INTO profiles (make, model, serial, nickname, identity_key, specs_json, "
                "spec_source, created_utc, updated_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (make, model, serial, (nickname or "").strip() or None, key,
                 _dumps(specs), spec_source, now, now))
        return self.get_profile(cur.lastrowid)

    def update_specs(self, profile_id: int, specs: dict[str, Any], spec_source: str) -> Profile:
        """Change a profile's specs. Saved runs keep the specs they were analysed with."""
        self._check_specs(specs, spec_source)
        self._require_profile(profile_id)
        with closing(self._connect()) as con, con:
            con.execute("UPDATE profiles SET specs_json = ?, spec_source = ?, updated_utc = ? "
                        "WHERE id = ?", (_dumps(specs), spec_source, _now(), profile_id))
        return self.get_profile(profile_id)

    def list_profiles(self) -> list[Profile]:
        return [_profile(r) for r in self._read("SELECT * FROM profiles ORDER BY id")]

    def get_profile(self, profile_id: int) -> Profile | None:
        rows = self._read("SELECT * FROM profiles WHERE id = ?", (profile_id,))
        return _profile(rows[0]) if rows else None

    def find_profile(self, make: str, model: str, serial: str) -> Profile | None:
        rows = self._read("SELECT * FROM profiles WHERE identity_key = ?",
                          (_identity_key(make, model, serial),))
        return _profile(rows[0]) if rows else None

    # -- runs -------------------------------------------------------------------------
    def save_run(self, profile_id: int, records: list[dict[str, Any]], *,
                 source: dict[str, Any], summary: dict[str, Any], feature_names: list[str],
                 duration_s: float, complete: bool, raw: RawData | None = None,
                 stages: list[str] | None = None) -> int:
        """Save one analysis run for an engine; returns its history run id.

        source: how the run was produced (mode, dataset, recording run_id, model,
        lab settings). It must not contain evaluation truth or preset/scenario text.
        """
        profile = self._require_profile(profile_id)
        if stages is not None and len(stages) != len(records):
            raise ValueError("stages must align with records")
        analyzed = float(records[-1]["t_end_s"]) if records else 0.0
        raw_dir = None
        with closing(self._connect()) as con:
            try:
                with con:
                    cur = con.execute(
                        "INSERT INTO runs (profile_id, created_utc, source_json, specs_json, "
                        "summary_json, feature_names_json, complete, analyzed_s, duration_s) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (profile_id, _now(), _dumps(source), _dumps(profile.specs),
                         _dumps(summary), _dumps(list(feature_names)), int(complete),
                         analyzed, float(duration_s)))
                    run_id = cur.lastrowid
                    con.executemany(
                        "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [_decision_row(run_id, k, r, stages[k] if stages else None)
                         for k, r in enumerate(records)])
                    if raw is not None:
                        raw_dir = self.root / "raw" / f"profile_{profile_id}" / f"run_{run_id}"
                        _write_raw(raw_dir, raw)
                        con.execute("UPDATE runs SET raw_path = ? WHERE id = ?",
                                    (raw_dir.relative_to(self.root).as_posix(), run_id))
            except BaseException:
                if raw_dir is not None:
                    shutil.rmtree(raw_dir, ignore_errors=True)
                raise
        self.prune_raw(profile_id)
        return run_id

    def list_runs(self, profile_id: int) -> list[RunEntry]:
        """Saved runs of one engine, newest first."""
        return [_run(r) for r in self._read(
            "SELECT * FROM runs WHERE profile_id = ? ORDER BY id DESC", (profile_id,))]

    def get_run(self, run_id: int) -> RunEntry | None:
        rows = self._read("SELECT * FROM runs WHERE id = ?", (run_id,))
        return _run(rows[0]) if rows else None

    def decisions(self, run_id: int) -> pd.DataFrame:
        """Per-window decisions of a saved run, in time order (features as lists)."""
        rows = self._read("SELECT * FROM decisions WHERE run_id = ? ORDER BY idx", (run_id,))
        frame = pd.DataFrame([dict(r) for r in rows])
        for column in ("probabilities", "context", "features"):
            if f"{column}_json" in frame:
                frame[column] = [json.loads(v) if v else None for v in frame.pop(f"{column}_json")]
        return frame

    def load_raw(self, run_id: int) -> RawData | None:
        """Raw data of a saved run, or None once it has been pruned."""
        rows = self._read("SELECT raw_path FROM runs WHERE id = ?", (run_id,))
        if not rows or rows[0]["raw_path"] is None:
            return None
        folder = self.root / rows[0]["raw_path"]
        with np.load(folder / "signals.npz") as z:
            cam = z["cam_sync_s"] if bool(z["has_cam_sync"]) else None
            # Runs saved before data contract 1.2 have no crank wheel fields.
            has_teeth = "has_crank_teeth" in z.files and bool(z["has_crank_teeth"])
            raw = RawData(audio=z["audio"], vibration=z["vibration"],
                          audio_fs_hz=float(z["audio_fs_hz"]),
                          vibration_fs_hz=float(z["vibration_fs_hz"]),
                          telemetry=pd.DataFrame(), cam_sync_s=cam,
                          crank_tooth_s=z["crank_tooth_s"] if has_teeth else None)
        raw.telemetry = pd.read_csv(folder / "telemetry.csv")
        return raw

    def prune_raw(self, profile_id: int) -> list[int]:
        """Delete raw data beyond the newest raw_keep runs of an engine; returns run ids."""
        rows = self._read("SELECT id, raw_path FROM runs WHERE profile_id = ? AND raw_path "
                          "IS NOT NULL ORDER BY id DESC", (profile_id,))
        old = rows[self.raw_keep:]
        if not old:
            return []
        with closing(self._connect()) as con, con:
            for row in old:
                shutil.rmtree(self.root / row["raw_path"], ignore_errors=True)
                con.execute("UPDATE runs SET raw_path = NULL, raw_pruned_utc = ? WHERE id = ?",
                            (_now(), row["id"]))
        return [row["id"] for row in old]

    # -- notes ------------------------------------------------------------------------
    def add_note(self, profile_id: int, text: str, run_id: int | None = None) -> Note:
        text = text.strip()
        if not text:
            raise ValueError("a note needs some text")
        self._require_profile(profile_id)
        if run_id is not None:
            run = self.get_run(run_id)
            if run is None or run.profile_id != profile_id:
                raise ValueError(f"run {run_id} does not belong to this engine")
        now = _now()
        with closing(self._connect()) as con, con:
            cur = con.execute("INSERT INTO notes (profile_id, run_id, created_utc, text) "
                              "VALUES (?, ?, ?, ?)", (profile_id, run_id, now, text))
        return Note(cur.lastrowid, profile_id, run_id, now, text)

    def list_notes(self, profile_id: int) -> list[Note]:
        """Notes of one engine, newest first."""
        return [Note(r["id"], r["profile_id"], r["run_id"], r["created_utc"], r["text"])
                for r in self._read("SELECT * FROM notes WHERE profile_id = ? ORDER BY id DESC",
                                    (profile_id,))]

    # -- diagnoses --------------------------------------------------------------------
    def save_diagnosis(self, profile_id: int, output: dict[str, Any], *, model: str,
                       fallback: bool, inputs_digest: str, latest_run_id: int | None = None,
                       fallback_reason: str | None = None) -> Diagnosis:
        """Save one diagnosis of an engine (kept permanently, like notes)."""
        self._require_profile(profile_id)
        now = _now()
        with closing(self._connect()) as con, con:
            cur = con.execute(
                "INSERT INTO diagnoses (profile_id, latest_run_id, created_utc, model, fallback, "
                "fallback_reason, inputs_digest, output_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (profile_id, latest_run_id, now, model, int(fallback), fallback_reason,
                 inputs_digest, _dumps(output)))
        return Diagnosis(cur.lastrowid, profile_id, latest_run_id, now, model, bool(fallback),
                         fallback_reason, inputs_digest, _clean(output))

    def list_diagnoses(self, profile_id: int) -> list[Diagnosis]:
        """Diagnoses of one engine, newest first."""
        return [_diagnosis(r) for r in self._read(
            "SELECT * FROM diagnoses WHERE profile_id = ? ORDER BY id DESC", (profile_id,))]

    def latest_diagnosis(self, profile_id: int) -> Diagnosis | None:
        rows = self._read("SELECT * FROM diagnoses WHERE profile_id = ? ORDER BY id DESC "
                          "LIMIT 1", (profile_id,))
        return _diagnosis(rows[0]) if rows else None

    # -- helpers ----------------------------------------------------------------------
    def _require_profile(self, profile_id: int) -> Profile:
        profile = self.get_profile(profile_id)
        if profile is None:
            raise ValueError(f"no engine profile {profile_id}")
        return profile

    @staticmethod
    def _check_specs(specs: dict[str, Any], spec_source: str) -> None:
        if spec_source not in SPEC_SOURCES:
            raise ValueError(f"spec_source must be one of {', '.join(SPEC_SOURCES)}")
        problems = validate_specs(specs)
        if problems:
            raise ValueError("invalid engine specs: " + "; ".join(problems))


def _profile(row: sqlite3.Row) -> Profile:
    return Profile(row["id"], row["make"], row["model"], row["serial"], row["nickname"],
                   json.loads(row["specs_json"]), row["spec_source"], row["created_utc"])


def _run(row: sqlite3.Row) -> RunEntry:
    return RunEntry(row["id"], row["profile_id"], row["created_utc"],
                    json.loads(row["source_json"]), json.loads(row["specs_json"]),
                    json.loads(row["summary_json"]), bool(row["complete"]), row["analyzed_s"],
                    row["duration_s"], row["raw_path"] is not None)


def _diagnosis(row: sqlite3.Row) -> Diagnosis:
    return Diagnosis(row["id"], row["profile_id"], row["latest_run_id"], row["created_utc"],
                     row["model"], bool(row["fallback"]), row["fallback_reason"],
                     row["inputs_digest"], json.loads(row["output_json"]))


def _decision_row(run_id: int, k: int, r: dict[str, Any], stage: str | None) -> tuple:
    def number(key):
        value = r.get(key)
        return float(value) if value is not None and math.isfinite(value) else None

    probabilities = r.get("probabilities")
    return (run_id, k, float(r["t_end_s"]), r["quality"], r.get("predicted"), r.get("accepted"),
            r.get("alert"), number("anomaly_score"), number("margin"),
            _dumps(probabilities) if probabilities else None, stage,
            localization_summary(r), _dumps(r.get("context") or {}),
            _dumps(r["features"]) if r.get("features") is not None else None)


def _write_raw(folder: Path, raw: RawData) -> None:
    folder.mkdir(parents=True, exist_ok=False)
    has_cam = raw.cam_sync_s is not None
    has_teeth = raw.crank_tooth_s is not None
    np.savez_compressed(folder / "signals.npz", audio=raw.audio, vibration=raw.vibration,
                        audio_fs_hz=raw.audio_fs_hz, vibration_fs_hz=raw.vibration_fs_hz,
                        has_cam_sync=has_cam,
                        cam_sync_s=raw.cam_sync_s if has_cam else np.empty(0),
                        has_crank_teeth=has_teeth,
                        crank_tooth_s=raw.crank_tooth_s if has_teeth else np.empty(0))
    raw.telemetry[TELEMETRY_COLUMNS].to_csv(folder / "telemetry.csv", index=False)
