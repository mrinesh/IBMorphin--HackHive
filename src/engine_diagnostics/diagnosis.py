"""Diagnose: ranked possible causes and next actions for one engine, from its history.

The context (engine_diagnostics.diagnosis_context) goes to Google Gemini through the
official google-genai SDK, asking for JSON that matches RESPONSE_SCHEMA. The answer
is validated here before anyone sees it: malformed output is a failure, and a cause
whose evidence IDs do not exist in the context is dropped (or keeps only its real
IDs, with the rest flagged). Without a key, without network, on a timeout, a rate
limit (429) or invalid output, `diagnose` returns the rule-based diagnosis from
engine_diagnostics.diagnosis_fallback in the same schema, marked as the basic version.

The model name and limits live in configs/diagnosis.json; the API key is read from
the GEMINI_API_KEY environment variable, or from a GEMINI_API_KEY= line in the
repository's git-ignored .env file (see .env.example).
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine_diagnostics.diagnosis_context import build_context, context_digest, context_ids
from engine_diagnostics.diagnosis_fallback import rule_based_diagnosis
from engine_diagnostics.history import Diagnosis, HistoryStore, Profile

ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "configs/diagnosis.json"
API_KEY_ENV = "GEMINI_API_KEY"
ENV_FILE = ROOT / ".env"
FALLBACK_MODEL = "rule-based (basic)"
KINDS = ("detected", "suggested")
CONFIDENCES = ("high", "medium", "low")
URGENCIES = ("stop_now", "soon", "monitor")
DISCLAIMER = ("Decision support only: these are possible causes ranked from the recorded data, "
              "not a substitute for a physical inspection by a qualified technician.")

SYSTEM_PROMPT = """\
You are an engine diagnostics assistant supporting a technician. You receive
structured data about one engine: specs (engine, ID "SPEC"), the current run's
model outputs (current_run), past runs (past_runs, newest first), technician
notes (notes), trends across runs (trends), and signal summaries computed from
the recorded signals (signal_summaries). Each item has an ID.

Your job: identify the most likely causes of the engine's current condition
and recommend what to do next.

How to reason:
- Base every cause only on the provided data. Cite the specific IDs
  (runs, notes, signals, trends, SPEC) that support it in evidence_ids. A cause
  you cannot support with at least one cited item does not belong in the output.
- Look across history, not just the latest run: recurring patterns, trends
  that are getting worse, and whether past repairs in the notes resolved or
  failed to resolve earlier faults.
- The detection models only output a fixed set of fault classes
  (detection_classes: normal, misfire, knock_like, valve_clearance_like). You may
  propose more specific mechanical root causes (e.g. a worn spark plug behind
  a misfire), but label these kind "suggested", while causes that match a model
  detection are kind "detected".
- Rate confidence as high, medium or low, and say in confidence_notes what
  evidence would raise or lower it. If data is thin, conflicting or missing, say
  so plainly in data_gaps rather than guessing.
- If engine.reliability_warning says the engine is not a 4-cylinder, 4-stroke
  engine, lower confidence on detection-based causes and note why.

Actions: concrete steps a technician can carry out, ordered by urgency
(stop_now / soon / monitor), each linked to the cause title(s) it addresses in
addresses. Prefer cheap, non-invasive checks first.

Write the summary in 2-3 plain-language sentences. Rank causes from most to
least likely. Respond only with JSON matching the provided schema.
"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "causes": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "kind": {"type": "string", "enum": list(KINDS)},
                "confidence": {"type": "string", "enum": list(CONFIDENCES)},
                "reasoning": {"type": "string"},
                "confidence_notes": {"type": "string"},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "kind", "confidence", "reasoning", "evidence_ids"],
        }},
        "actions": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "step": {"type": "string"},
                "urgency": {"type": "string", "enum": list(URGENCIES)},
                "addresses": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["step", "urgency", "addresses"],
        }},
        "data_gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "causes", "actions", "data_gaps"],
}

MESSAGES = {
    "no_key": "No Gemini API key is set (GEMINI_API_KEY).",
    "not_installed": "The google-genai package is not installed.",
    "auth": "Gemini rejected the API key.",
    "rate_limit": "The free Gemini quota is used up for the moment (rate limit). Try again in "
                  "a minute.",
    "timeout": "Gemini did not answer in time.",
    "network": "Gemini could not be reached (network or service problem).",
    "invalid": "Gemini's answer did not match the expected format.",
}


@dataclass(frozen=True)
class DiagnosisSettings:
    model: str = "gemini-3.8-flash"
    timeout_s: float = 30.0
    temperature: float = 0.2
    max_runs: int = 8
    max_notes: int = 20
    summary_window_s: float = 5.0


def load_settings(path: Path = SETTINGS_PATH) -> DiagnosisSettings:
    """Settings from configs/diagnosis.json (defaults when the file is missing)."""
    if not Path(path).is_file():
        return DiagnosisSettings()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = DiagnosisSettings.__dataclass_fields__
    return DiagnosisSettings(**{k: v for k, v in data.items() if k in fields})


class DiagnosisError(Exception):
    """Why the Gemini diagnosis could not be used; `reason` is a key of MESSAGES."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail

    @property
    def message(self) -> str:
        return MESSAGES.get(self.reason, MESSAGES["invalid"])


# -- Gemini -------------------------------------------------------------------------------
def api_key(env_file: Path | None = None) -> str:
    """GEMINI_API_KEY from the environment, else from the repository's .env (git-ignored)."""
    key = os.environ.get(API_KEY_ENV, "").strip()
    env_file = ENV_FILE if env_file is None else env_file
    if key or not env_file.is_file():
        return key
    for line in env_file.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and name.strip().removeprefix("export ").strip() == API_KEY_ENV:
            return value.strip().strip("'\"")
    return ""


def make_client(settings: DiagnosisSettings):
    """google-genai client with the key from GEMINI_API_KEY, the timeout, and no retries."""
    key = api_key()
    if not key:
        raise DiagnosisError("no_key")
    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise DiagnosisError("not_installed", str(error)) from error
    return genai.Client(api_key=key, http_options=types.HttpOptions(
        timeout=int(settings.timeout_s * 1000),
        retry_options=types.HttpRetryOptions(attempts=1)))


def _classify(error: Exception) -> str:
    """Map an SDK/transport exception to a reason key."""
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if code == 429:
        return "rate_limit"
    if code in (401, 403):
        return "auth"
    if code == 400 and "api key" in str(error).lower():
        return "auth"
    name = type(error).__name__.lower()
    if "timeout" in name or isinstance(error, TimeoutError):
        return "timeout"
    return "network"


def call_gemini(context: dict[str, Any], settings: DiagnosisSettings, client=None) -> str:
    """Send the context and return the response text (JSON). Raises DiagnosisError."""
    try:
        from google.genai import types
    except ImportError as error:
        raise DiagnosisError("not_installed", str(error)) from error
    client = client if client is not None else make_client(settings)
    payload = "Engine data (JSON):\n" + json.dumps(context, separators=(",", ":"), default=str)
    try:
        response = client.models.generate_content(
            model=settings.model, contents=payload,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, temperature=settings.temperature,
                response_mime_type="application/json",
                response_json_schema=RESPONSE_SCHEMA))
    except DiagnosisError:
        raise
    except Exception as error:  # SDK, HTTP, and transport errors all fall back
        raise DiagnosisError(_classify(error), f"{type(error).__name__}: {error}") from error
    text = getattr(response, "text", None)
    if not text:
        raise DiagnosisError("invalid", "empty response")
    return text


# -- validation ---------------------------------------------------------------------------
def _strings(value) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        return None
    return value


def validate_output(data: str | dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Checked diagnosis output. Raises DiagnosisError('invalid') when malformed.

    Causes citing no ID that exists in the context are dropped (listed in
    dropped_causes); a cause with some unknown IDs keeps the real ones and lists
    the others in unverified_ids. Actions that only addressed dropped causes go too.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as error:
            raise DiagnosisError("invalid", f"not JSON: {error}") from error
    if not isinstance(data, dict):
        raise DiagnosisError("invalid", "top level is not an object")
    summary, causes, actions = data.get("summary"), data.get("causes"), data.get("actions")
    gaps = _strings(data.get("data_gaps"))
    if not isinstance(summary, str) or not summary.strip() or not isinstance(causes, list) \
            or not isinstance(actions, list) or gaps is None:
        raise DiagnosisError("invalid", "missing or wrongly typed summary/causes/actions/"
                                        "data_gaps")
    known = context_ids(context)
    kept, dropped = [], []
    for cause in causes:
        if not isinstance(cause, dict):
            dropped.append({"title": "?", "reason": "malformed"})
            continue
        title = cause.get("title")
        ids = _strings(cause.get("evidence_ids"))
        if (not isinstance(title, str) or not title.strip() or cause.get("kind") not in KINDS
                or cause.get("confidence") not in CONFIDENCES
                or not isinstance(cause.get("reasoning"), str) or ids is None):
            dropped.append({"title": str(title or "?"), "reason": "malformed"})
            continue
        real = [i for i in ids if i in known]
        unknown = [i for i in ids if i not in known]
        if not real:
            dropped.append({"title": title, "reason": "no evidence ID exists in the data",
                            "evidence_ids": ids})
            continue
        item = {"title": title.strip(), "kind": cause["kind"],
                "confidence": cause["confidence"], "reasoning": cause["reasoning"],
                "confidence_notes": str(cause.get("confidence_notes") or ""),
                "evidence_ids": real}
        if unknown:
            item["unverified_ids"] = unknown
        kept.append(item)
    titles = {c["title"] for c in kept}
    dropped_titles = {d["title"] for d in dropped}
    checked_actions = []
    for action in actions:
        addresses = _strings(action.get("addresses")) if isinstance(action, dict) else None
        if (addresses is None or not isinstance(action.get("step"), str)
                or not action["step"].strip() or action.get("urgency") not in URGENCIES):
            continue
        linked = [a for a in addresses if a in titles]
        if addresses and not linked and set(addresses) & dropped_titles:
            continue  # only addressed causes that were dropped
        checked_actions.append({"step": action["step"].strip(), "urgency": action["urgency"],
                                "addresses": linked})
    return {"summary": summary.strip(), "causes": kept, "actions": checked_actions,
            "data_gaps": [g for g in gaps if g.strip()], "dropped_causes": dropped}


def gemini_diagnosis(context: dict[str, Any], settings: DiagnosisSettings,
                     client=None) -> dict[str, Any]:
    """Validated Gemini diagnosis. Raises DiagnosisError on any failure."""
    output = validate_output(call_gemini(context, settings, client), context)
    if output["dropped_causes"] and not output["causes"]:
        raise DiagnosisError("invalid", "every cause cited evidence that does not exist")
    return output


# -- orchestration ------------------------------------------------------------------------
@dataclass(frozen=True)
class DiagnosisOutcome:
    output: dict[str, Any]
    model: str
    fallback: bool
    reason: str | None  # short user-facing message when the basic version was used
    inputs_digest: str


def diagnose(context: dict[str, Any], settings: DiagnosisSettings | None = None,
             client=None) -> DiagnosisOutcome:
    """Gemini diagnosis, or the rule-based one on any failure. Never raises for Gemini."""
    settings = settings or load_settings()
    digest = context_digest(context)
    try:
        return DiagnosisOutcome(gemini_diagnosis(context, settings, client), settings.model,
                                False, None, digest)
    except DiagnosisError as error:
        reason = error.message
    except Exception:  # noqa: BLE001 - the demo must never stop on a diagnosis error
        reason = MESSAGES["invalid"]
    return DiagnosisOutcome(rule_based_diagnosis(context), FALLBACK_MODEL, True,
                            f"{reason} Showing the basic rule-based diagnosis.", digest)


def run_diagnosis(store: HistoryStore, profile: Profile, config: dict[str, Any] | None = None,
                  settings: DiagnosisSettings | None = None, client=None) -> Diagnosis:
    """Build the context, diagnose, and save the result to the engine's history."""
    settings = settings or load_settings()
    context = build_context(store, profile, max_runs=settings.max_runs,
                            max_notes=settings.max_notes, window_s=settings.summary_window_s,
                            config=config)
    outcome = diagnose(context, settings, client)
    latest = context["current_run"]
    return store.save_diagnosis(
        profile.id, outcome.output, model=outcome.model, fallback=outcome.fallback,
        inputs_digest=outcome.inputs_digest, fallback_reason=outcome.reason,
        latest_run_id=int(latest["id"][1:]) if latest else None)
