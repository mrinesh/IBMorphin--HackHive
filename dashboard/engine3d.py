"""Three.js engine view as an in-page Streamlit (components v2) component.

The scene lives in the browser and is created once per page: `engine3d.js` keeps it
in a WeakMap keyed by the component root and animates it with requestAnimationFrame.
Each Streamlit rerun only sends the small `data` dict below; the browser eases toward
it. Three.js and the scene module are served from dashboard/static (no CDN), and the
browser caches the dynamic import, so reruns do not re-download or re-evaluate them.
"""

import math
from typing import Any

import streamlit as st

from dashboard import explain
from dashboard.theme import CLASS_META, TOKENS, class_meta

FAULTS = explain.FAULTS
EGT_RANGE_C = (250.0, 850.0)  # exhaust runner color scale (cold dark -> hot white)
HELD_INTENSITY = 0.45  # effect strength while a fault is only held by its alert


def scene_state(record: dict[str, Any] | None, model, *, t_now: float, playing: bool,
                rpm: float | None, firing_order: list[int], finished: bool = False,
                truth: dict[str, Any] | None = None) -> dict[str, Any]:
    """Small JSON state for the 3D view, derived only from model output and sensor data.

    `faults` lists every fault reported now (explain.reported_faults: active alerts
    plus this window's detections), each with the cylinder the localizer names for it
    (from sensor data and the cam phase reference only; None when not localized), its
    status (confirmed / held / unconfirmed), its class color, and an effect intensity:
    the model's calibrated probability for that fault (v2), its confidence (v1), the
    evidence level / 3 (temporary model), or 0.45 while the fault is only held by its
    alert. The view shows each fault on its own cylinder, or engine-wide when it has
    none. `effect` is the first fault (block outline color).
    `vibration_ratio` is this window's vibration RMS over the normal-reference RMS.
    `animate` keeps the engine running: while playing, and after the replay finished
    (on the final decision, for review). A mid-replay pause freezes it.
    `truth` (evaluation truth) is only passed when the overlay is on, and only its
    affected cylinder during an active fault interval reaches the browser.
    With data contract 1.2 sensors, `egt` (each runner's exhaust gas temperature
    scaled to EGT_RANGE_C, 0-1) colors the exhaust runners and `egt_c` labels them;
    `crank_cylinder` and `thermal_cylinder` are the cylinders the independent crank
    and thermal paths name (sensor data only).
    """
    state: dict[str, Any] = {"t": round(t_now, 2), "playing": playing, "finished": finished,
                             "animate": playing or finished,
                             "rpm": None if rpm is None else round(rpm, 1),
                             "firing_order": list(firing_order), "effect": None, "faults": [],
                             "vibration_ratio": 1.0, "alert": None,
                             "edge_color": TOKENS["cyan"], "status_text": "Model: warming up",
                             "truth_cylinder": None, "egt": None, "egt_c": None,
                             "crank_cylinder": None, "thermal_cylinder": None}
    if record is not None:
        quality, accepted = record["quality"], record["accepted"]
        faults = [fault_state(entry, record, model) for entry in explain.reported_faults(record)]
        effect = faults[0]["fault"] if faults else None
        meta = CLASS_META[effect] if effect else class_meta(accepted, quality)
        state.update(effect=effect, faults=faults, alert=record.get("alert"),
                     edge_color=meta["color"],
                     status_text=status_text(record, faults, "localization" in record),
                     vibration_ratio=_vibration_ratio(record, model))
        state.update(sensor_state(record))
    if truth is not None:
        intervals = truth.get("active_intervals") or []
        if intervals and not isinstance(intervals[0], list):
            intervals = [intervals]
        active = any(a <= t_now <= b for a, b in intervals)
        cylinder = truth.get("affected_cylinder")
        if active and truth.get("condition") in FAULTS and isinstance(cylinder, int):
            state["truth_cylinder"] = cylinder
    return state


def fault_state(entry: dict[str, str], record: dict[str, Any], model) -> dict[str, Any]:
    """One reported fault for the view: cylinder, status, color and effect intensity."""
    fault, status = entry["fault"], entry["status"]
    intensity = fault_probability(entry, record)
    if intensity is None:
        strength = explain.evidence_strength(record, model) if status != "held" else None
        intensity = strength["level"] / 3 if strength else HELD_INTENSITY
    cylinder = ((record.get("localization") or {}).get(fault) or {}).get("cylinder")
    return {"fault": fault, "status": status,
            "cylinder": int(cylinder) if cylinder is not None else None,
            "intensity": round(float(intensity), 3), "color": CLASS_META[fault]["color"]}


def fault_probability(entry: dict[str, str], record: dict[str, Any]) -> float | None:
    """The model's calibrated probability for a fault detected in this window, if any."""
    if entry["status"] == "held":  # not detected in this window: nothing to quote
        return None
    if "detected" in record:  # v2: one calibrated detector per fault
        return record["probabilities"][entry["fault"]]
    return record.get("probability")  # v1: confidence of the accepted class


def status_text(record: dict[str, Any], faults: list[dict[str, Any]], localizing: bool) -> str:
    """HUD line naming each reported fault with its own confidence, status and cylinder."""
    if not faults:
        if record["quality"] != "ok":
            return "Model: no decision (data quality)"
        return f"Model: {class_meta(record['accepted'])['label']}"
    parts = []
    for f in faults:
        p = fault_probability(f, record)
        fields = [CLASS_META[f["fault"]]["label"],
                  *([f"confidence {p:.2f}"] if p is not None else []),
                  *([STATUS_TEXT[f["status"]]] if f["status"] in STATUS_TEXT else [])]
        where = (record.get("localization") or {}).get(f["fault"])
        parts.append(", ".join(fields) + location_text(where, localizing))
    prefix = "Model (no decision this window): " if record["quality"] != "ok" else "Model: "
    return prefix + "; ".join(parts)


STATUS_TEXT = {"held": "alert held", "unconfirmed": "not confirmed yet"}


def sensor_state(record: dict[str, Any]) -> dict[str, Any]:
    """Exhaust temperatures and the crank/thermal paths' cylinders for the view."""
    out: dict[str, Any] = {}
    thermal = record.get("thermal") or {}
    egt = thermal.get("egt_c")
    if egt and all(v is not None for v in egt):
        lo, hi = EGT_RANGE_C
        out["egt"] = [round(min(max((v - lo) / (hi - lo), 0.0), 1.0), 2) for v in egt]
        out["egt_c"] = [round(v) for v in egt]
    named = [p.get("cylinder") for p in (thermal.get("patterns") or {}).values()
             if p.get("alert") and p.get("cylinder")]
    if named:
        out["thermal_cylinder"] = int(named[0])
    crank = record.get("crank") or {}
    if crank.get("alert") and crank.get("cylinder"):
        out["crank_cylinder"] = int(crank["cylinder"])
    return out


def location_text(where: dict[str, Any] | None, localizing: bool) -> str:
    """Suffix describing where the model places one reported fault."""
    if not localizing:
        return ", cylinder not identified (no localizer loaded)"
    if not where:
        return ", locating cylinder…"
    if where.get("cylinder") is not None:
        return f", cylinder {where['cylinder']} (location confidence {where['confidence']:.2f})"
    if where.get("slot") is not None:
        return f", firing slot {where['slot']} (no cam reference: cylinder unknown)"
    return ", locating cylinder…"


def _vibration_ratio(record: dict[str, Any], model) -> float:
    """Current vibration RMS relative to the normal reference (1.0 when unknown)."""
    name = "vibration_log_rms"
    if record["quality"] != "ok" or name not in model.baseline_names:
        return 1.0
    (median, _), _ = explain.reference_stats(record, model)
    value = record["features"][model.feature_names.index(name)]
    ratio = 10 ** (value - median[model.baseline_names.index(name)])
    return round(float(ratio), 3) if math.isfinite(ratio) else 1.0


_HTML = """
<div class="engine3d-root" tabindex="0" role="img"
     aria-label="3D engine view. Drag or use arrow keys to orbit, plus and minus to zoom.">
  <div class="engine3d-hud" aria-hidden="true"></div>
  <div class="engine3d-controls" role="group" aria-label="3D view">
    <button type="button" data-action="reset">Reset view</button>
  </div>
  <div class="engine3d-fallback" hidden></div>
</div>
"""

_CSS = """
/* Status text sits above the canvas on the card, never over the model. */
.engine3d-root {
  position: relative; width: 100%; height: clamp(440px, 44vw, 660px);
  display: flex; flex-direction: column; outline: none; border-radius: 14px;
}
.engine3d-root:focus-visible { box-shadow: 0 0 0 3px rgba(10, 132, 255, 0.65); }
.engine3d-root canvas {
  display: block; width: 100%; flex: 1; min-height: 0; touch-action: none; border-radius: 14px;
  background: radial-gradient(120% 90% at 50% 15%, #1d1d21 0%, #0b0b0d 80%);
}
.engine3d-hud {
  order: -1; margin: 4px 8px 12px; pointer-events: none;
  font: 500 13.5px/1.55 ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace; color: #a1a1a8;
}
.engine3d-hud div:nth-child(2) { font-weight: 600; margin-top: 2px; }
.engine3d-controls {
  position: absolute; right: 12px; bottom: 12px; display: flex; gap: 6px; flex-wrap: wrap;
  justify-content: flex-end;
}
.engine3d-controls button {
  min-height: 34px; min-width: 44px; padding: 0 16px; border-radius: 999px; cursor: pointer;
  font: 600 13px/1 -apple-system, BlinkMacSystemFont, "Inter", system-ui, sans-serif;
  color: #f5f5f7; background: rgba(40, 40, 44, 0.68); border: 0;
  -webkit-backdrop-filter: blur(28px) saturate(1.8); backdrop-filter: blur(28px) saturate(1.8);
  transition: background-color 150ms ease, border-color 150ms ease, color 150ms ease;
  touch-action: manipulation;
}
.engine3d-controls button:hover { filter: brightness(1.2); }
.engine3d-controls button[aria-pressed="true"] {
  background: #0a84ff; color: #fff;
}
.engine3d-controls button:focus-visible { outline: 3px solid rgba(10, 132, 255, 0.65); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) {
  .engine3d-controls button { transition: none; }
}
.engine3d-fallback[hidden], .engine3d-controls[hidden] { display: none; }
.engine3d-fallback-status {
  font: 500 13px/1.5 -apple-system, BlinkMacSystemFont, "Inter", system-ui, sans-serif; color: #a1a1a8;
}
.engine3d-fallback {
  position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center;
  justify-content: center; gap: 8px; padding: 16px; background: #0b0b0d;
  color: #f5f5f7; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Inter", system-ui, sans-serif; text-align: center;
}
"""

_JS = """
const MODULE = new URL('app/static/engine3d/engine3d.js', document.baseURI).href;
export default function (component) {
  const { data, parentElement } = component;
  const root = parentElement.querySelector('.engine3d-root');
  if (!root) return;
  import(MODULE)
    .then((mod) => mod.render(root, data))
    .catch((error) => {
      const fallback = root.querySelector('.engine3d-fallback');
      fallback.hidden = false;
      fallback.textContent = '3D view unavailable: ' + error.message;
    });
}
"""

_component = st.components.v2.component("engine3d", html=_HTML, css=_CSS, js=_JS)


def render(state: dict, *, key: str = "engine3d") -> None:
    """Mount (first call) or update (later calls) the 3D view with a small state dict.

    The height is responsive CSS (shorter on phones), so the wrapper sizes to content.
    """
    _component(key=key, data=state)
