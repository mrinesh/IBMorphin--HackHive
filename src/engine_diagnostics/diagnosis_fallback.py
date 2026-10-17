"""Basic rule-based diagnosis in the same schema as the Gemini one.

Used when Gemini is unavailable (no key, no network, timeout, rate limit, invalid
output). It reads only the diagnosis context: the model's fault alerts and
likelihoods, health stages, trends across runs, signal summaries, and repair
notes, and maps them to fixed, cheap-first technician checks. Every cause cites
context IDs, so it passes the same evidence validation as a Gemini answer.
"""

from typing import Any

from engine_diagnostics.diagnosis_context import FAULTS
from engine_diagnostics.runtime.health import DEVELOPING, FAULT

LABELS = {"misfire": "Misfire", "knock_like": "Knock combustion",
          "valve_clearance_like": "Valve-train impacts (valve clearance)"}
REPAIR_WORDS = ("replac", "repair", "fixed", "changed", "swapped", "new ", "installed",
                "adjusted", "cleaned")
FAULT_WORDS = {
    "misfire": ("misfire", "plug", "coil", "injector", "ignition", "compression"),
    "knock_like": ("knock", "octane", "fuel", "timing", "carbon", "detonation"),
    "valve_clearance_like": ("valve", "lash", "clearance", "tappet", "lifter", "rocker"),
}
STOP_NOW = {
    "misfire": "Avoid sustained high load until the misfire is fixed: unburnt fuel from a "
               "persistent misfire can overheat the catalytic converter.",
    "knock_like": "Reduce load and avoid hard acceleration until checked: sustained knock can "
                  "damage pistons, rings and the head gasket.",
}
CHECKS = {
    "misfire": [
        "Read stored OBD misfire codes, then inspect the spark plug and ignition coil of the "
        "affected cylinder; swap the coil with a neighbouring cylinder and re-test to see "
        "whether the misfire moves with it.",
        "If plug and coil are good, check the injector connector and spray, then run a "
        "compression test on the affected cylinder.",
    ],
    "knock_like": [
        "Confirm the fuel octane matches the engine specification and ask about recent "
        "fuel changes.",
        "Check ignition timing and the knock sensor and its wiring; inspect for carbon "
        "deposits with a borescope through the spark-plug hole.",
    ],
    "valve_clearance_like": [
        "Listen at the valve cover with a mechanic's stethoscope for ticking, then measure "
        "valve clearance (lash) against the specification.",
    ],
}
CONFIDENCE_DOWN = {"high": "medium", "medium": "low", "low": "low"}
MISFIRE_FLAG_PCT = 10.0  # flagged crank strokes on one cylinder that suggest misfire
COLD_EGT_C = -30.0       # EGT this far below the other cylinders: weak combustion
HOT_CHT_C = 8.0          # CHT this far above the other cylinders: heat (e.g. knock)
LTFT_LIMIT = 10.0        # |long-term fuel trim| in percent
STFT_DRIFT = 5.0         # short-term fuel trim drift in percentage points


def _runs(context: dict[str, Any]) -> list[dict[str, Any]]:
    current = context.get("current_run")
    return ([current] if current else []) + list(context.get("past_runs", []))


def _summaries(context: dict[str, Any], run: str) -> dict[str, dict[str, Any]]:
    return {s["id"].split("-", 1)[1]: s for s in context.get("signal_summaries", [])
            if s["run"] == run}


def _alerted(run: dict[str, Any]) -> set[str]:
    return {a["fault"] for a in run.get("alerts", [])}


def _repair_notes(context: dict[str, Any], fault: str, before_utc: str,
                  fault_runs: list[str]) -> list[dict[str, Any]]:
    """Repair notes written before `before_utc` about this fault (by wording or by run)."""
    out = []
    for n in context.get("notes", []):
        text = n["text"].casefold()
        if n["created_utc"] > before_utc or not any(w in text for w in REPAIR_WORDS):
            continue
        if n.get("run") in fault_runs or any(w in text for w in FAULT_WORDS[fault]):
            out.append(n)
    return out


def _signal_causes(current: dict[str, Any], sig: dict[str, dict[str, Any]],
                   causes: list[dict[str, Any]], actions: list[dict[str, Any]]) -> None:
    """Suggested root causes from crank speed, cylinder temperatures and fuel trims."""
    crank = sig.get("crank")
    if crank and crank.get("phase") == "cam_sync":
        worst = crank["roughest"]
        flagged = crank["per_cylinder"][worst].get("flagged_stroke_pct") or 0.0
        if flagged >= MISFIRE_FLAG_PCT:
            cyl = worst.removeprefix("cyl")
            title = f"Weak combustion on cylinder {cyl} (ignition or fuel delivery)"
            causes.append({
                "title": title, "kind": "suggested", "confidence": "medium",
                "reasoning": f"In {current['id']}, {flagged:g}% of cylinder {cyl}'s power "
                             "strokes slowed the crankshaft more than the roughness limit, "
                             "the pattern of a cylinder that fails to fire.",
                "confidence_notes": "Higher if the misfire follows the coil when swapped; "
                                    "lower if the cylinder's EGT is normal.",
                "evidence_ids": [crank["id"]]})
            actions.append({"step": f"Inspect cylinder {cyl}'s spark plug, coil and injector "
                                    "first.", "urgency": "soon", "addresses": [title]})
    thermal = sig.get("thermal")
    for cyl, v in ((thermal or {}).get("egt") or {}).items():
        if (v.get("vs_median_c") or 0) <= COLD_EGT_C:
            n = cyl.removeprefix("cyl")
            title = f"Cylinder {n} running cold (exhaust temperature)"
            causes.append({
                "title": title, "kind": "suggested", "confidence": "medium",
                "reasoning": f"Cylinder {n}'s exhaust gas temperature is {v['vs_median_c']:g} "
                             "C from the other cylinders, which points to little or no "
                             "combustion in that cylinder.",
                "confidence_notes": "Higher if crank speed also shows cylinder "
                                    f"{n} decelerating; lower if the thermocouple is suspect.",
                "evidence_ids": [thermal["id"]]})
            actions.append({"step": f"Check cylinder {n}'s exhaust thermocouple reading, then "
                                    "its ignition and injector.", "urgency": "soon",
                            "addresses": [title]})
    for cyl, v in ((thermal or {}).get("cht") or {}).items():
        if (v.get("vs_median_c") or 0) >= HOT_CHT_C:
            n = cyl.removeprefix("cyl")
            title = f"Cylinder {n} head running hot"
            causes.append({
                "title": title, "kind": "suggested", "confidence": "low",
                "reasoning": f"Cylinder {n}'s head temperature is {v['vs_median_c']:g} C above "
                             "the other cylinders; knock or local cooling problems raise it.",
                "confidence_notes": "Higher if knock is detected in the same run.",
                "evidence_ids": [thermal["id"]]})
            actions.append({"step": f"Check coolant level and flow around cylinder {n}.",
                            "urgency": "monitor", "addresses": [title]})
    fuel = sig.get("fuel")
    if fuel:
        ltft = (fuel.get("long_term_fuel_trim") or {}).get("mean")
        stft = (fuel.get("short_term_fuel_trim") or {}).get("drift")
        if (ltft is not None and abs(ltft) >= LTFT_LIMIT) or (
                stft is not None and abs(stft) >= STFT_DRIFT):
            title = "Air-fuel mixture correction out of range (air leak, injector or sensor)"
            causes.append({
                "title": title, "kind": "suggested", "confidence": "low",
                "reasoning": f"Fuel trims: long-term mean {ltft}%, short-term drift {stft} "
                             "points over the run. Large corrections mean the ECU is "
                             "compensating for a lean or rich mixture.",
                "confidence_notes": "Higher if the trims stay out of range across runs.",
                "evidence_ids": [fuel["id"]]})
            actions.append({"step": "Smoke-test the intake for vacuum leaks and compare the "
                                    "MAF and oxygen sensor readings with specification.",
                            "urgency": "soon", "addresses": [title]})


def rule_based_diagnosis(context: dict[str, Any]) -> dict[str, Any]:
    """Diagnosis from fault classes, alerts, trends, signals and notes (no network)."""
    current = context.get("current_run")
    warning = (context.get("engine") or {}).get("reliability_warning")
    gaps = []
    if current is None:
        return {"summary": "This engine has no saved runs yet, so there is nothing to diagnose. "
                           "Play a run to the end (or use Save run so far) and try again.",
                "causes": [], "actions": [{"step": "Record and save a complete run on this "
                                                   "engine.", "urgency": "monitor",
                                           "addresses": []}],
                "data_gaps": ["No saved runs for this engine."], "dropped_causes": []}
    runs = _runs(context)
    trends = {t["id"]: t for t in context.get("trends", [])}
    sig = _summaries(context, current["id"])
    causes: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    stage = current.get("final_health_stage")
    for fault in FAULTS:
        trend = trends.get(f"T-{fault}", {})
        p = (current.get("max_likelihoods") or {}).get(fault)
        past_hits = [r["id"] for r in runs[1:] if fault in _alerted(r)]
        evidence = [current["id"], f"T-{fault}"]
        evidence += [sig[k]["id"] for k in {"misfire": ("crank", "thermal"),
                                            "knock_like": ("spectrum", "thermal"),
                                            "valve_clearance_like": ("spectrum",)}[fault]
                     if k in sig]
        if fault in _alerted(current):
            streak = trend.get("latest_alert_streak", 1)
            confidence = "high" if streak >= 2 or stage == FAULT else "medium"
            reasoning = (f"The model raised a {LABELS[fault].lower()} alert in the latest run "
                         f"{current['id']}" + (f" (likelihood up to {p:.2f})" if p else "")
                         + (f", and in {streak} consecutive runs" if streak >= 2 else "")
                         + (f"; final health stage: {stage}" if stage else "") + ".")
            repairs = _repair_notes(context, fault, current["saved_utc"], past_hits)
            if repairs:
                evidence += [n["id"] for n in repairs]
                reasoning += (" A related repair was noted before this run ("
                              + ", ".join(n["id"] for n in repairs)
                              + "), yet the alert persists.")
        elif past_hits:
            confidence = "low"
            evidence = [current["id"], f"T-{fault}", *past_hits]
            reasoning = (f"Earlier runs ({', '.join(past_hits)}) alerted {LABELS[fault].lower()}"
                         f", but the latest run {current['id']} did not.")
            repairs = _repair_notes(context, fault, current["saved_utc"], past_hits)
            if repairs:
                evidence += [n["id"] for n in repairs]
                reasoning += (" A related repair was noted since ("
                              + ", ".join(n["id"] for n in repairs)
                              + "), so it may have worked; confirm with another run.")
            else:
                reasoning += " It may be intermittent; no related repair is noted."
        elif (p is not None and p >= 0.5) or trend.get("direction") == "rising":
            confidence = "low"
            parts = [f"{p:.2f} in {current['id']}"] if p is not None else []
            if trend.get("direction") == "rising":
                parts.append("rising across runs")
            reasoning = (f"No alert yet, but the {LABELS[fault].lower()} likelihood is "
                         + " and ".join(parts) + ".")
        else:
            continue
        notes = "Higher if it repeats in the next run; lower if a physical check finds nothing."
        if warning:
            confidence = CONFIDENCE_DOWN[confidence]
            evidence.append("SPEC")
            notes += (" Confidence lowered: the detection models are tuned for 4-cylinder, "
                      "4-stroke engines.")
        title = LABELS[fault]
        causes.append({"title": title, "kind": "detected", "confidence": confidence,
                       "reasoning": reasoning, "confidence_notes": notes,
                       "evidence_ids": list(dict.fromkeys(evidence))})
        if fault in _alerted(current) and confidence == "high" and fault in STOP_NOW:
            actions.append({"step": STOP_NOW[fault], "urgency": "stop_now",
                            "addresses": [title]})
        for step in CHECKS[fault]:
            actions.append({"step": step, "urgency": "soon" if fault in _alerted(current)
                            else "monitor", "addresses": [title]})
    _signal_causes(current, sig, causes, actions)
    if stage == DEVELOPING and not _alerted(current):
        title = "Developing anomaly outside the trained fault classes"
        causes.append({"title": title, "kind": "suggested", "confidence": "low",
                       "reasoning": f"The latest run {current['id']} ended in the 'developing "
                                    "anomaly' stage without a confirmed fault alert.",
                       "confidence_notes": "Higher if the anomaly grows over the next runs.",
                       "evidence_ids": [current["id"], "T-anomaly"]})
        actions.append({"step": "Re-record at the same operating point within a few days and "
                                "compare.", "urgency": "monitor", "addresses": [title]})
    rank = {"high": 0, "medium": 1, "low": 2}
    causes.sort(key=lambda c: (rank[c["confidence"]], c["kind"] != "detected"))
    actions.append({"step": "After any repair, save a new run on this engine and diagnose "
                            "again to confirm the fault has cleared.", "urgency": "monitor",
                    "addresses": [c["title"] for c in causes]})
    order = {"stop_now": 0, "soon": 1, "monitor": 2}
    actions.sort(key=lambda a: order[a["urgency"]])

    coverage = context.get("coverage", {})
    if coverage.get("runs_saved", 0) < 3:
        gaps.append(f"Only {coverage.get('runs_saved', 0)} saved run(s): trends across runs "
                    "are weak.")
    if not current.get("complete"):
        gaps.append(f"The latest run {current['id']} was saved before the replay ended.")
    if not current.get("max_likelihoods"):
        gaps.append("The model used for the latest run gives no calibrated likelihoods.")
    pruned = [r["id"] for r in runs if r.get("raw_data") == "pruned"]
    if pruned:
        gaps.append(f"Raw signals were pruned for {', '.join(pruned)}: no signal summaries "
                    "for those runs.")
    if "crank" not in sig:
        gaps.append("No crank-wheel data in the latest run, so per-cylinder crank speed was "
                    "not checked.")
    if "thermal" not in sig:
        gaps.append("No per-cylinder exhaust/head temperatures in the latest run.")
    if not context.get("notes"):
        gaps.append("No technician notes: repair history is unknown.")
    if warning:
        gaps.append(warning)

    faults = sorted(_alerted(current))
    found = (f"The latest run ({current['id']}) shows "
             + (" and ".join(LABELS[f].lower() for f in faults) if faults else "no fault alert")
             + (f", ending in the '{stage}' stage" if stage else "") + ".")
    if causes:
        top = causes[0]
        found += (f" The most likely cause is {top['title'].lower()} ({top['confidence']} "
                  "confidence); start with the cheapest checks below.")
    else:
        found += " Nothing in the saved history points to a specific cause; keep monitoring."
    return {"summary": found, "causes": causes, "actions": actions, "data_gaps": gaps,
            "dropped_causes": []}
