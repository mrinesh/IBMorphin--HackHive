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

FAULTS = ("misfire", "knock_like", "valve_clearance_like")


def scene_state(record: dict[str, Any] | None, model, *, t_now: float, playing: bool,
                rpm: float | None, firing_order: list[int],
                truth: dict[str, Any] | None = None) -> dict[str, Any]:
    """Small JSON state for the 3D view, derived only from model output and sensor data.

    The effect is the confirmed alert class if there is one, else an accepted fault.
    Its intensity is the model's calibrated confidence for that class (v1), or the
    evidence level / 3 for the temporary model, or 0.45 while an alert is only held.
    `vibration_ratio` is this window's vibration RMS over the normal-reference RMS.
    `model_cylinder` is the cylinder the localizer names for the effect's alert (from
    sensor data and the cam phase reference only); the view then confines the effect
    to that cylinder. `truth` (evaluation truth) is only passed when the overlay is on,
    and only its affected cylinder during an active fault interval reaches the browser.
    """
    state: dict[str, Any] = {"t": round(t_now, 2), "playing": playing,
                             "rpm": None if rpm is None else round(rpm, 1),
                             "firing_order": list(firing_order), "effect": None,
                             "intensity": 0.0, "vibration_ratio": 1.0, "alert": None,
                             "edge_color": TOKENS["cyan"], "status_text": "Model: warming up",
                             "truth_cylinder": None, "model_cylinder": None}
    if record is not None:
        quality, accepted, alert = record["quality"], record["accepted"], record.get("alert")
        if "+" in (alert or ""):  # v2 fault set: animate the first fault
            alert = alert.split("+")[0]
        if "+" in (accepted or ""):
            accepted = accepted.split("+")[0]
        effect = alert if alert in FAULTS else accepted if accepted in FAULTS else None
        if effect and effect == accepted and quality == "ok":
            if "probability" in record:
                intensity = record["probability"]
            else:
                strength = explain.evidence_strength(record, model)
                intensity = strength["level"] / 3 if strength else 0.45
        else:
            intensity = 0.45 if effect else 0.0
        meta = CLASS_META[effect] if effect else class_meta(accepted, quality)
        if quality != "ok":
            status = "Model: no decision (data quality)"
        elif accepted in FAULTS:
            confidence = (f" · confidence {record['probability']:.2f}"
                          if "probability" in record else "")
            label = explain.names(record["accepted"])
            status = f"Model: {label}{confidence}"
        else:
            status = f"Model: {class_meta(accepted)['label']}"
        if alert and alert != accepted:
            status += f" · alert held: {CLASS_META.get(alert, {'label': alert})['label']}"
        where = (record.get("localization") or {}).get(effect) if effect else None
        if effect:
            status += location_text(where, "localization" in record)
        if where and where.get("cylinder") is not None:
            state["model_cylinder"] = int(where["cylinder"])
        state.update(effect=effect, intensity=round(float(intensity), 3), alert=alert,
                     edge_color=meta["color"], status_text=status,
                     vibration_ratio=_vibration_ratio(record, model))
    if truth is not None:
        intervals = truth.get("active_intervals") or []
        if intervals and not isinstance(intervals[0], list):
            intervals = [intervals]
        active = any(a <= t_now <= b for a, b in intervals)
        cylinder = truth.get("affected_cylinder")
        if active and truth.get("condition") in FAULTS and isinstance(cylinder, int):
            state["truth_cylinder"] = cylinder
    return state


def location_text(where: dict[str, Any] | None, localizing: bool) -> str:
    """Status suffix describing where the model places the alerted fault."""
    if not localizing:
        return " · cylinder not identified (no localizer loaded)"
    if not where:
        return " · locating cylinder…"
    if where.get("cylinder") is not None:
        return f" · cylinder {where['cylinder']} (confidence {where['confidence']:.2f})"
    if where.get("slot") is not None:
        return f" · firing slot {where['slot']} (no cam reference: cylinder unknown)"
    return " · locating cylinder…"


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
    <button type="button" data-view="car" aria-pressed="true">Car</button>
    <button type="button" data-view="engine" aria-pressed="false">Engine</button>
    <button type="button" data-action="reset">Reset view</button>
  </div>
  <div class="engine3d-fallback" hidden></div>
</div>
"""

_CSS = """
.engine3d-root {
  position: relative; width: 100%; height: clamp(300px, 46vw, 640px);
  border-radius: 12px; overflow: hidden; outline: none;
  border: 1px solid #2b333e;
  background: radial-gradient(120% 90% at 50% 25%, #1a212b 0%, #10141a 75%);
}
.engine3d-root:focus-visible { box-shadow: inset 0 0 0 2px #00e5e5; }
.engine3d-root canvas { display: block; width: 100%; height: 100%; touch-action: none; }
.engine3d-hud {
  position: absolute; left: 16px; top: 14px; right: 16px; pointer-events: none;
  font: 500 13px/1.5 "Inter", system-ui, sans-serif; color: #aeb4be;
}
.engine3d-controls {
  position: absolute; right: 10px; bottom: 10px; display: flex; gap: 6px; flex-wrap: wrap;
  justify-content: flex-end;
}
.engine3d-controls button {
  min-height: 38px; min-width: 44px; padding: 0 16px; border-radius: 8px; cursor: pointer;
  font: 600 13px/1 "Inter", system-ui, sans-serif;
  color: #e6e8eb; background: rgba(29, 35, 44, 0.92); border: 1px solid #2b333e;
  transition: background-color 150ms ease, border-color 150ms ease, color 150ms ease;
  touch-action: manipulation;
}
.engine3d-controls button:hover { border-color: #00e5e5; color: #00e5e5; }
.engine3d-controls button[aria-pressed="true"] {
  background: rgba(0, 229, 229, 0.12); border-color: #00e5e5; color: #00e5e5;
}
.engine3d-controls button:focus-visible { outline: 2px solid #00e5e5; outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) {
  .engine3d-controls button { transition: none; }
}
.engine3d-fallback[hidden], .engine3d-controls[hidden] { display: none; }
.engine3d-fallback-status {
  font: 500 13px/1.5 "Inter", system-ui, sans-serif; color: #aeb4be;
}
.engine3d-fallback {
  position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center;
  justify-content: center; gap: 8px; padding: 16px; background: #10141a;
  color: #e6e8eb; font: 14px/1.5 "Inter", system-ui, sans-serif; text-align: center;
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
