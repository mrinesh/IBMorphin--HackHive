"""Crank-speed and cylinder-temperature panels (data contract 1.2 sensors).

Everything here is drawn from replay records (sensor data and model/monitor
outputs). Evaluation truth is never read. The thermal-camera image is a
rendering of the measured exhaust and head temperatures on an engine
outline, not a simulated infrared camera.
"""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard import theme

CYLINDER_COLORS = ("#38bdf8", "#f472b6", "#fbbf24", "#a78bfa")
CAMERA_RANGE_C = (100.0, 760.0)
PATTERN_TEXT = {"cold": "runs cold (EGT and CHT low): not firing → misfire",
                "head_hot": "head hot, exhaust not: detonation heat → knock-like",
                "exhaust_hot": "exhaust hot, head not: hot gas past the valve → valve-like"}


def peak_decimate(t, y, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Thin a trace for display to at most `max_points`, keeping each bin's extremes.

    Plain striding (y[::k]) drops short spikes such as knock bursts and valve-train
    impacts; keeping both the minimum and the maximum of every bin preserves them.
    Charts are drawn as SVG, not WebGL (see the 3D view in dashboard/README.md), so
    traces are kept to a few thousand points. NaN gaps stay NaN.
    """
    t, y = np.asarray(t), np.asarray(y)
    if y.size <= max_points:
        return t, y
    bins = max(max_points // 2, 1)
    size = -(-y.size // bins)  # ceil: every sample falls in a bin
    values = np.pad(y.astype(np.float64), (0, size * bins - y.size), constant_values=np.nan)
    blocks = values.reshape(bins, size)
    nan = np.isnan(blocks)
    low = np.argmin(np.where(nan, np.inf, blocks), axis=1)
    high = np.argmax(np.where(nan, -np.inf, blocks), axis=1)
    start = np.arange(bins) * size
    index = np.unique(np.minimum(np.concatenate([start + low, start + high]), y.size - 1))
    return t[index], y[index]


def crank_speed_figure(crank: dict[str, Any]) -> go.Figure:
    """Crank speed over the last 0.5 s: per tooth (faint) and per stroke, TDCs marked.

    The per-stroke speed is the 90-degree segment centred on each firing TDC, the
    quantity the misfire monitors use; per-tooth speed also carries tooth-spacing
    error and timer jitter.
    """
    trace, marks = crank.get("speed_trace") or {}, crank.get("stroke_marks") or {}
    fig = go.Figure(go.Scatter(x=trace.get("t_s", []), y=trace.get("rpm", []), mode="lines",
                               line={"width": 1, "color": "rgba(0,229,229,0.35)"},
                               name="per tooth", hovertemplate="%{x:.3f} s: %{y:.0f} rpm"
                                                               "<extra>per tooth</extra>"))
    flagged = marks.get("flagged", [])
    fig.add_trace(go.Scatter(
        x=marks.get("t_s", []), y=marks.get("rpm", []), mode="lines+markers", name="per stroke",
        line={"width": 2, "color": "#e2e8f0"},
        marker={"size": [10 if f else 6 for f in flagged],
                "color": ["#ef4444" if f else "#e2e8f0" for f in flagged]},
        customdata=marks.get("cylinder", []),
        hovertemplate="cyl %{customdata} stroke: %{y:.0f} rpm<extra></extra>"))
    for t, cylinder, flagged in zip(marks.get("t_s", []), marks.get("cylinder", []),
                                    marks.get("flagged", []), strict=True):
        color = "#ef4444" if flagged else "rgba(148,163,184,0.45)"
        fig.add_vline(x=t, line={"width": 2 if flagged else 1, "color": color})
        if cylinder and cylinder > 0:
            fig.add_annotation(x=t, y=1.02, yref="paper", showarrow=False, text=str(cylinder),
                               font={"size": 10, "color": color if flagged else "#94a3b8"})
    fig.update_layout(height=300, margin={"t": 40, "b": 20},
                      title="Crank speed (60-2 wheel) · red: misfire stroke",
                      title_font={"size": 13}, xaxis_title="s", yaxis_title="rpm",
                      legend={"orientation": "h", "y": -0.2, "x": 1, "xanchor": "right"})
    return fig


def roughness_figure(crank: dict[str, Any]) -> go.Figure:
    """Each cylinder's mean roughness excess in the window against the alert threshold."""
    excess = [v if v is not None else 0.0 for v in crank.get("excess") or [0, 0, 0, 0]]
    threshold = crank.get("threshold")
    colors = ["#ef4444" if threshold is not None and v > threshold else "#64748b"
              for v in excess]
    fig = go.Figure(go.Bar(x=[f"Cyl {k}" for k in range(1, len(excess) + 1)], y=excess,
                           marker_color=colors, text=[f"{v:.2f}" for v in excess],
                           textposition="outside",
                           hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    if threshold is not None:
        fig.add_hline(y=threshold, line={"dash": "dash", "color": "#ef4444"},
                      annotation_text="alert threshold", annotation_position="top left")
    top = max([*excess, threshold or 0.0]) * 1.3 + 0.2
    fig.update_layout(height=280, showlegend=False, margin={"t": 40, "b": 40},
                      title="Roughness by cylinder (last 2 s)",
                      title_font={"size": 13}, yaxis={"range": [min(min(excess), 0) - 0.2, top]})
    return fig


def first_alert(records: list[dict[str, Any]], fault: str) -> float | None:
    """Time of the first acoustic alert for `fault` in this replay (model output only)."""
    return next((r["t_end_s"] for r in records if fault in (r.get("alerts")
                 or ([r["alert"]] if r.get("alert") else []))), None)


def obd_summary(records: list[dict[str, Any]], crank: dict[str, Any]) -> dict[str, Any]:
    """ECU monitor state and the comparison with this replay's acoustic misfire alert."""
    obd = crank.get("obd") or {}
    dtcs = obd.get("dtcs") or []
    acoustic = first_alert(records, "misfire")
    crank_alert = next((r["t_end_s"] for r in records if (r.get("crank") or {}).get("alert")),
                       None)
    ecu = {kind: next((d for d in dtcs if d["type"] == kind), None) for kind in ("A", "B")}
    lead = None
    if acoustic is not None and ecu["A"] is not None:
        lead = ecu["A"]["t_s"] - acoustic
    return {"acoustic_s": acoustic, "crank_s": crank_alert, "ecu": ecu, "lead_s": lead,
            "blocks": obd.get("blocks") or {}, "misfires_total": obd.get("misfires_total"),
            "strokes": obd.get("strokes", 0)}


def comparison_text(summary: dict[str, Any], t_now: float) -> str:
    acoustic, ecu_a = summary["acoustic_s"], summary["ecu"]["A"]
    if acoustic is None:
        return "No acoustic misfire alert in this replay so far."
    if ecu_a is None:
        progress = summary["blocks"].get("A", {}).get("progress", 0.0)
        return (f"Acoustic misfire alert at **{acoustic:.1f} s**. The ECU monitor has set "
                f"**no code yet** ({t_now - acoustic:.1f} s later; current 200-revolution "
                f"block {100 * progress:.0f} % complete).")
    lead = summary["lead_s"]
    verb = "earlier" if lead >= 0 else "later"
    return (f"Acoustic misfire alert **{acoustic:.1f} s** · ECU type A code "
            f"**{ecu_a['code']}** at **{ecu_a['t_s']:.1f} s** → acoustic alert "
            f"**{abs(lead):.1f} s {verb}**.")


def obd_table(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for kind, label in (("A", "Type A (catalyst damage)"), ("B", "Type B (emissions)")):
        block = summary["blocks"].get(kind, {})
        dtc = summary["ecu"][kind]
        rows.append({
            "monitor": label,
            "block": f"{block.get('revs', '?')} rev",
            "limit": f"{block.get('rate_limit_pct', 0):g} % of strokes",
            "progress": f"{100 * block.get('progress', 0):.0f} %",
            "misfires per cylinder (this block)": " / ".join(map(str, block.get("misfires", []))),
            "code": f"{dtc['code']} at {dtc['t_s']:.1f} s ({dtc['rate_pct']:.1f} %)"
                    if dtc else "none",
        })
    return pd.DataFrame(rows)


def thermal_camera_figure(thermal: dict[str, Any], title: str) -> go.Figure:
    """Top view of the head: each exhaust port glows with its EGT, the head with its CHT."""
    egt = [v if v is not None else np.nan for v in thermal.get("egt_c") or [np.nan] * 4]
    cht = [v if v is not None else np.nan for v in thermal.get("cht_c") or [np.nan] * 4]
    x = np.linspace(0, 4, 240)
    y = np.linspace(0, 1.6, 96)
    gx, gy = np.meshgrid(x, y)
    image = np.full(gx.shape, CAMERA_RANGE_C[0])
    head = (gy > 0.25) & (gy < 1.05) & (gx > 0.1) & (gx < 3.9)
    for k in range(4):
        cx = 0.5 + k
        if np.isfinite(cht[k]):  # cylinder head over this bore
            bore = np.exp(-((gx - cx) ** 2 / 0.09 + (gy - 0.65) ** 2 / 0.06))
            level = CAMERA_RANGE_C[0] + (cht[k] - CAMERA_RANGE_C[0]) * (0.55 + 0.45 * bore)
            mine = head & (np.abs(gx - cx) < 0.5)
            image = np.where(mine, np.maximum(image, level), image)
        if np.isfinite(egt[k]):  # exhaust port and runner leaving the head
            port = np.exp(-((gx - cx) ** 2 / 0.03 + (gy - 1.2) ** 2 / 0.02))
            runner = np.exp(-((gx - cx) ** 2 / 0.02)) * ((gy > 1.1) & (gy < 1.55))
            image = np.maximum(image, CAMERA_RANGE_C[0] + (egt[k] - CAMERA_RANGE_C[0])
                               * np.maximum(port, 0.85 * runner))
    fig = go.Figure(go.Heatmap(x=x, y=y, z=image, colorscale=theme.INFERNO,
                               zmin=CAMERA_RANGE_C[0], zmax=CAMERA_RANGE_C[1],
                               colorbar={"title": "°C"},
                               hovertemplate="%{z:.0f} °C<extra></extra>"))
    for k in range(4):
        text = f"Cyl {k + 1}<br>EGT {egt[k]:.0f} °C<br>CHT {cht[k]:.0f} °C" \
            if np.isfinite(egt[k]) and np.isfinite(cht[k]) else f"Cyl {k + 1}<br>no reading"
        fig.add_annotation(x=0.5 + k, y=0.12, text=text, showarrow=False,
                           font={"size": 11, "color": "#e2e8f0"})
    fig.update_layout(height=330, title=title, title_font={"size": 13},
                      margin={"t": 40, "b": 10, "l": 10, "r": 10},
                      xaxis={"visible": False}, yaxis={"visible": False, "scaleanchor": "x"})
    return fig


def thermal_trend_figure(telemetry: pd.DataFrame, t_now: float, duration: float) -> go.Figure:
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Exhaust gas temperature (°C)",
                                                        "Cylinder head temperature (°C)"])
    received = telemetry[(telemetry["t_s"] < t_now) & (telemetry["valid"] == 1)]
    for col, kind in enumerate(("egt", "cht"), start=1):
        for c in range(1, 5):
            rows = received[received["signal"] == f"{kind}_cyl{c}"]
            fig.add_trace(go.Scatter(x=rows["t_s"], y=rows["value"], name=f"Cyl {c}",
                                     legendgroup=str(c), showlegend=col == 1,
                                     line={"width": 1.5, "color": CYLINDER_COLORS[c - 1]}),
                          row=1, col=col)
    fig.update_xaxes(range=[0, duration], title="s")
    fig.update_layout(height=300, margin={"t": 40, "b": 20},
                      legend={"orientation": "h", "y": -0.25})
    return fig


def thermal_lines(thermal: dict[str, Any]) -> list[str]:
    """Plain statements of the thermal path's state for the panel."""
    status = thermal.get("status")
    if status == "warming_up":
        return ["Learning each cylinder's normal temperature offset (first seconds of the "
                "recording)."]
    if status in ("stale", "unavailable"):
        return ["Cylinder temperatures are missing or stale: no thermal assessment."]
    lines = []
    for pattern, info in (thermal.get("patterns") or {}).items():
        if info.get("alert"):
            lines.append(f"**Cylinder {info['cylinder']}** {PATTERN_TEXT[pattern]}.")
        elif info.get("candidate"):
            lines.append(f"Cylinder {info['candidate']} starting to look "
                         f"{PATTERN_TEXT[pattern].split(':')[0]} (not confirmed).")
    return lines or ["All cylinders within their normal temperature spread."]


THERMAL_PATTERN_OF = {fault: pattern for pattern, fault in
                      (("cold", "misfire"), ("head_hot", "knock_like"),
                       ("exhaust_hot", "valve_clearance_like"))}


def confirmation_text(record: dict[str, Any] | None, fault: str) -> str:
    """Which independent sensor paths currently agree with an acoustic alert."""
    if not record:
        return ""
    parts = []
    crank = record.get("crank") or {}
    if fault == "misfire" and crank.get("alert"):
        parts.append(f"crank speed (cylinder {crank['cylinder']})")
    pattern = ((record.get("thermal") or {}).get("patterns") or {}).get(
        THERMAL_PATTERN_OF.get(fault, ""), {})
    if pattern.get("alert"):
        parts.append(f"EGT/CHT (cylinder {pattern['cylinder']})")
    if parts:
        return " · confirmed independently by " + " and ".join(parts)
    if "crank" in record or "thermal" in record:
        return " · not yet confirmed by crank speed or EGT/CHT"
    return ""
