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
    `truth` (evaluation truth) is only passed when the overlay is on, and only its
    affected cylinder during an active fault interval reaches the browser.
    """
    state: dict[str, Any] = {"t": round(t_now, 2), "playing": playing,
                             "rpm": None if rpm is None else round(rpm, 1),
                             "firing_order": list(firing_order), "effect": None,
                             "intensity": 0.0, "vibration_ratio": 1.0, "alert": None,
                             "edge_color": TOKENS["cyan"], "status_text": "Model: warming up",
                             "truth_cylinder": None}
    if record is not None:
        quality, accepted, alert = record["quality"], record["accepted"], record.get("alert")
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
            status = f"Model: {CLASS_META[accepted]['label']}{confidence}"
        else:
            status = f"Model: {class_meta(accepted)['label']}"
        if alert and alert != accepted:
            status += f" · alert held: {CLASS_META.get(alert, {'label': alert})['label']}"
        if effect:
            status += " · cylinder not identified by the model"
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
  <div class="engine3d-fallback" hidden></div>
</div>
"""

_CSS = """
.engine3d-root {
  position: relative; width: 100%; height: var(--engine3d-height, 440px);
  border-radius: 6px; overflow: hidden; outline: none;
  background: radial-gradient(120% 90% at 50% 20%, #0d1b2e 0%, #070b14 70%);
}
.engine3d-root:focus-visible { box-shadow: inset 0 0 0 2px #22d3ee; }
.engine3d-root canvas { display: block; width: 100%; height: 100%; touch-action: none; }
.engine3d-hud {
  position: absolute; left: 12px; top: 10px; right: 12px; pointer-events: none;
  font: 600 11px/1.4 "Fira Code", ui-monospace, monospace; letter-spacing: 0.06em;
  color: #94a3b8; text-transform: uppercase;
}
.engine3d-fallback {
  position: absolute; inset: 0; display: grid; place-items: center; padding: 16px;
  color: #e6edf7; font: 14px/1.5 "Fira Sans", sans-serif; text-align: center;
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


def render(state: dict, *, key: str = "engine3d", height: int = 440) -> None:
    """Mount (first call) or update (later calls) the 3D view with a small state dict."""
    _component(key=key, data=state | {"height": height}, height=height)
