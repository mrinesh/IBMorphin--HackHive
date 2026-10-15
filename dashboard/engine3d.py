"""Three.js engine view as an in-page Streamlit (components v2) component.

The scene lives in the browser and is created once per page: `engine3d.js` keeps it
in a WeakMap keyed by the component root and animates it with requestAnimationFrame.
Each Streamlit rerun only sends the small `data` dict below; the browser eases toward
it. Three.js and the scene module are served from dashboard/static (no CDN), and the
browser caches the dynamic import, so reruns do not re-download or re-evaluate them.
"""

import streamlit as st

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
