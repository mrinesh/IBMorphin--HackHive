"""Dark HUD theme for the replay dashboard: tokens, CSS, Plotly template, HTML cards.

Colors mirror `.streamlit/config.toml`. Every status color is paired with an icon
and a text label, so state is never conveyed by color alone. The HTML builders are
pure functions of their inputs (easy to unit-test) and escape all text.
"""

from html import escape
from pathlib import Path
from urllib.parse import quote

import plotly.graph_objects as go
import plotly.io as pio

TOKENS = {
    "bg": "#070B14",
    "surface": "rgba(15, 23, 42, 0.72)",
    "surface_solid": "#0F172A",
    "line": "rgba(34, 211, 238, 0.28)",
    "grid": "rgba(148, 163, 184, 0.10)",
    "text": "#E6EDF7",
    "muted": "#94A3B8",
    "cyan": "#22D3EE",
    "magenta": "#F472B6",
    "violet": "#A78BFA",
    "amber": "#FBBF24",
    "rose": "#FB7185",
    "green": "#4ADE80",
    "slate": "#8B9BB4",
}

# Lucide icon paths (24x24, 2px stroke), inlined so the dashboard works offline.
ICONS = {
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/>',
    "flame": '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.07-2.14-.22-4.05 2-6 '
             '.5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.15.43-2.29 1-3a2.5 2.5 '
             '0 0 0 2.5 2.5z"/>',
    "zap": '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "help": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>'
            '<path d="M12 17h.01"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 '
             '1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "signal_off": '<path d="M2 2l20 20"/><path d="M8.5 16.5a5 5 0 0 1 7 0"/>'
                  '<path d="M2 8.82a15 15 0 0 1 4.17-2.65"/><path d="M10.66 5c4.01-.36 8.14.9 '
                  '11.34 3.76"/><path d="M16.85 11.25a10 10 0 0 1 2.22 1.68"/>'
                  '<path d="M5 13a10 10 0 0 1 5.24-2.76"/><path d="M12 20h.01"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" '
           'height="6"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
}

# Accepted decisions and how they are shown. The valve class is named for what it
# models (valve-train impacts); there is no general abnormal-vibration classifier.
CLASS_META = {
    "normal": {"label": "Normal", "color": TOKENS["green"], "icon": "check"},
    "misfire": {"label": "Misfire", "color": TOKENS["amber"], "icon": "flame"},
    "knock_like": {"label": "Knock-like", "color": TOKENS["rose"], "icon": "zap"},
    "valve_clearance_like": {"label": "Valve-train impacts", "color": TOKENS["violet"],
                             "icon": "activity"},
    "uncertain": {"label": "Uncertain", "color": TOKENS["slate"], "icon": "help"},
    "no_decision": {"label": "No decision", "color": TOKENS["slate"], "icon": "signal_off"},
}

# Modality colors used in every chart (audio vs vibration stay consistent).
MODALITY_COLORS = {"audio": TOKENS["magenta"], "vibration": TOKENS["cyan"],
                   "envelope": TOKENS["violet"]}


def _svg_uri(svg: str) -> str:
    return "data:image/svg+xml," + quote(svg, safe="")


def icon(name: str, size: int = 18) -> str:
    """Decorative icon, always next to a visible text label.

    st.html sanitizes inline <svg>, so the icon is a CSS mask over currentColor.
    """
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
           'stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
           f"{ICONS[name]}</svg>")
    return (f'<span class="hud-icon" aria-hidden="true" style="width:{size}px;height:{size}px;'
            f"--hud-icon:url('{_svg_uri(svg)}')\"></span>")


def class_meta(accepted: str | None, quality: str | None = None) -> dict:
    if quality not in (None, "ok"):
        return CLASS_META["no_decision"]
    return CLASS_META.get(accepted or "uncertain", CLASS_META["uncertain"])


def register_plotly_template() -> None:
    """Transparent dark template with subtle grids and monospace numbers.

    plotly_dark and the HUD overrides are merged once and registered as one template:
    a "plotly_dark+engine_hud" default would be re-merged for every figure (~25 ms each).
    """
    if pio.templates.default == "engine_hud":
        return
    axis = {"gridcolor": TOKENS["grid"], "zerolinecolor": TOKENS["grid"],
            "linecolor": "rgba(148, 163, 184, 0.25)", "tickfont": {"color": TOKENS["muted"]},
            "title": {"font": {"color": TOKENS["muted"]}}}
    overrides = go.layout.Template(layout={
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(7, 11, 20, 0.55)",
        "font": {"family": "DM Sans, system-ui, sans-serif", "color": TOKENS["text"],
                 "size": 12},
        "title": {"font": {"family": "Sora, DM Sans, sans-serif", "size": 14,
                           "color": TOKENS["text"]}, "x": 0.01},
        "colorway": [TOKENS["cyan"], TOKENS["magenta"], TOKENS["violet"], TOKENS["amber"],
                     TOKENS["green"], TOKENS["rose"]],
        "xaxis": axis, "yaxis": axis,
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": TOKENS["muted"]}},
        "hoverlabel": {"bgcolor": TOKENS["surface_solid"], "bordercolor": TOKENS["cyan"],
                       "font": {"family": "DM Sans, sans-serif", "color": TOKENS["text"]}},
        "annotationdefaults": {"font": {"color": TOKENS["muted"]}},
    })
    pio.templates["engine_hud"] = pio.templates.merge_templates(pio.templates["plotly_dark"],
                                                                overrides)
    pio.templates.default = "engine_hud"


_CSS_VARS = ("bg", "surface", "line", "text", "muted", "cyan", "magenta", "rose", "green",
             "amber", "violet")


def css() -> str:
    """<style> block: token variables from TOKENS, then dashboard/hud.css."""
    variables = "".join(f"--hud-{name}: {TOKENS[name]};" for name in _CSS_VARS)
    sheet = (Path(__file__).with_name("hud.css")).read_text(encoding="utf-8")
    return f"<style>:root {{{variables}}}\n{sheet}</style>"


def banner_html() -> str:
    return ('<div class="synthetic-banner" role="note">SYNTHETIC DATA — RECORDED REPLAY'
            '<small>Simulated engines and sensors, replayed from recordings. Results do not '
            'show real-engine accuracy.</small></div>')


def brand_html() -> str:
    return ('<div class="hud-brand"><h1 class="name">Engine<span class="accent">Sense</span></h1>'
            '<div class="tag">Listens to engine sound, vibration and sensor data to spot '
            'faults early</div></div>')


def _bar(fraction: float, color: str, marker: float | None = None) -> str:
    fraction = min(max(fraction, 0.0), 1.0)
    mark = (f'<b style="left:calc({min(max(marker, 0), 1) * 100:.1f}% - 1px)"></b>'
            if marker is not None else "")
    return (f'<div class="bar"><i style="width:{fraction * 100:.1f}%;background:{color}"></i>'
            f"{mark}</div>")


def status_card_html(diagnosis: dict) -> str:
    """Diagnosis panel from `explain.diagnose` output (model outputs only)."""
    meta = diagnosis["meta"]
    color = meta["color"]
    rows = []
    for item in diagnosis["metrics"]:
        bar = (_bar(item["fraction"], item.get("color", TOKENS["cyan"]), item.get("marker"))
               if item.get("fraction") is not None else "")
        rows.append(f'<div><div class="hud-label">{escape(item["label"])}</div>'
                    f'<div class="hud-num">{escape(item["value"])}</div>{bar}</div>')
    alert = diagnosis.get("alert")
    alert_pill = (f'<span class="pill" style="color:{TOKENS["rose"]}">{icon("alert", 14)}'
                  f'Alert: {escape(alert)}</span>' if alert else
                  f'<span class="pill" style="color:{TOKENS["green"]}">{icon("check", 14)}'
                  f"No active alert</span>")
    return (
        f'<section class="hud-card status" aria-label="Current diagnosis">'
        f'<div class="hud-label">Model decision <span title="model version">· '
        f'{escape(diagnosis["model_version"])}</span></div>'
        f'<div class="status-head"><div class="status-badge" style="color:{color}">'
        f'{icon(meta["icon"], 24)}</div><div><div class="status-title" style="color:{color}">'
        f'{escape(meta["label"])}</div><div class="status-sub">{escape(diagnosis["subtitle"])}'
        f"</div></div></div>"
        f"<div>{alert_pill}</div>"
        f'<div class="status-grid">{"".join(rows)}</div>'
        f"{_strength_html(diagnosis.get('strength'))}"
        f"{_why_html(diagnosis.get('why'), diagnosis.get('why_ref'))}"
        f"{_about_html(diagnosis.get('about'))}"
        f'<div class="status-note">{escape(diagnosis["note"])}</div>'
        f"</section>"
    )


def _strength_html(strength: dict | None) -> str:
    """Three-segment meter plus the criteria behind it (each with icon and text)."""
    if not strength:
        return ""
    level = strength["level"]
    color = {3: TOKENS["rose"], 2: TOKENS["amber"]}.get(level, TOKENS["slate"])
    if strength["for"] == "Normal":
        color = TOKENS["green"] if level == 3 else color
    segments = "".join(f'<i class="{"on" if k < level else ""}"></i>' for k in range(3))
    items = "".join(
        f'<li class="{"met" if ok else "unmet"}">{icon("check" if ok else "help", 13)}'
        f'<span>{escape(text)}</span><span class="sr">{"met" if ok else "not met"}</span></li>'
        for text, ok in strength["criteria"])
    return (f'<div class="strength" style="--strength:{color}">'
            f'<div class="strength-head"><span class="hud-label">Evidence strength for '
            f'{escape(strength["for"])}</span><span class="strength-label">'
            f'{escape(strength["label"])} · {level}/3</span></div>'
            f'<div class="segments" role="img" aria-label="{level} of 3 criteria met">'
            f"{segments}</div><ul>{items}</ul></div>")


def _why_html(why: list[str] | None, reference: str | None) -> str:
    if not why:
        return ""
    items = []
    for line in why:  # "... is far above normal (z = +9.0)": show the z-score quietly
        text, _, z = line.partition(" (z = ")
        score = f' <span class="z">z {escape(z.rstrip(")"))}</span>' if z else ""
        items.append(f"<li>{escape(text)}{score}</li>")
    return (f'<div class="why"><div class="hud-label">Why the model thinks so · biggest '
            f'differences from normal ({escape(reference or "")})</div>'
            f'<ol>{"".join(items)}</ol></div>')


def _about_html(about: dict | None) -> str:
    if not about:
        return ""
    checks = (f'<div class="checks"><b>Typical checks on a real engine:</b> '
              f'{escape(about["checks"])}</div>' if about.get("checks") else "")
    return (f'<div class="about"><div class="hud-label">What {escape(about["class"])} means · '
            f'general background, not model output</div><p>{escape(about["means"])}</p>'
            f"{checks}</div>")


def sparkline_svg(values: list[float]) -> str:
    """Recent-history sparkline as a background image (inline <svg> is sanitized)."""
    if len(values) < 2:
        return '<span class="spark" aria-hidden="true"></span>'
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    step = 100 / (len(values) - 1)
    points = " ".join(f"{k * step:.1f},{22 - 20 * (v - lo) / span:.1f}"
                      for k, v in enumerate(values))
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 24" '
           f'preserveAspectRatio="none"><polyline points="{points}" fill="none" '
           f'stroke="{TOKENS["cyan"]}" stroke-width="1.5" '
           'vector-effect="non-scaling-stroke"/></svg>')
    return (f'<span class="spark" aria-hidden="true" '
            f"style=\"background-image:url('{_svg_uri(svg)}')\"></span>")


def telemetry_cards_html(cards: list[dict]) -> str:
    """Cards of last received telemetry values: label, value, unit, freshness, sparkline."""
    items = []
    for card in cards:
        status = card["status"]
        css = {"fresh": "ok", "STALE": "stale"}.get(status, "none")
        text = {"fresh": "Live", "STALE": "Stale", "no data": "No data"}[status]
        age = (f' title="Last reading {card["age_s"]:.1f} s ago"'
               if card.get("age_s") is not None else "")
        value = "—" if card["value"] is None else f"{card['value']:{card['fmt']}}"
        items.append(
            f'<div class="hud-card tele"><div class="row"><span class="hud-label">'
            f'{escape(card["label"])}</span><span class="fresh {css}"{age}>{text}</span></div>'
            f'<div class="val">{value}<span class="unit">{escape(card["unit"])}</span></div>'
            f'{sparkline_svg(card["history"])}</div>')
    return (f'<div class="tele-grid" role="group" aria-label="Latest telemetry">'
            f'{"".join(items)}</div>')
