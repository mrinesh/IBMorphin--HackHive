"""Engine report: one data structure, shown in the dashboard and exported as a PDF.

`build_report` collects everything from the engine's history (profile and specs,
the newest saved run with its health timeline and likelihoods, the latest saved
diagnosis, the run history and trends, notes). The dashboard preview and
`render_pdf` both draw from the returned EngineReport, so they cannot disagree.
Building a report never calls Gemini: it shows the latest saved diagnosis or a
prompt to run one.

The PDF uses fpdf2 with the DejaVu fonts that ship with matplotlib (Unicode text)
and charts drawn with matplotlib's Agg backend, so it needs no browser, kaleido or
system install. Like history, a report never contains evaluation truth.
"""

import io
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine_diagnostics.diagnosis import DISCLAIMER
from engine_diagnostics.diagnosis_context import FAULTS, SPEC_SOURCE_TEXT
from engine_diagnostics.history import HistoryStore, Profile, RunEntry
from engine_diagnostics.profiles import reliability_warning

LABELS = {"normal": "Normal", "misfire": "Misfire", "knock_like": "Knock",
          "valve_clearance_like": "Valve-train impacts", "uncertain": "Monitoring",
          "no data": "No data"}
FAULT_COLORS = {"misfire": "#d62728", "knock_like": "#ff7f0e",
                "valve_clearance_like": "#9467bd", "normal": "#2ca02c"}
STAGE_COLORS = {"healthy": "#2ca02c", "developing anomaly": "#ff9f1c",
                "persistent fault": "#d62728", "no data": "#9e9e9e"}
URGENCY_NAMES = {"stop_now": "Stop now", "soon": "Soon", "monitor": "Monitor"}
HISTORY_COLUMNS = ["run", "saved (UTC)", "source", "predicted fault", "alerts",
                   "max P(fault)", "peak anomaly", "final health", "raw data"]
SPEC_ROWS = [("cylinders", "Cylinders"), ("strokes", "Strokes"),
             ("displacement_l", "Displacement (L)"), ("firing_order", "Firing order"),
             ("rpm_min", "RPM min"), ("rpm_max", "RPM max"),
             ("load_min", "Load min (0-1)"), ("load_max", "Load max (0-1)")]
MAX_TREND_RUNS = 20
MAX_NOTES = 10


def utc_text(stamp: str) -> str:
    """'2026-10-16T17:05:00+00:00' -> '2026-10-16 17:05 UTC'."""
    return f"{stamp[:16].replace('T', ' ')} UTC"


def label(name: str | None) -> str:
    if not name:
        return "-"
    return " + ".join(LABELS.get(part, part) for part in name.split("+"))


def _num(value, digits: int = 2):
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


@dataclass
class EngineReport:
    generated_utc: str
    engine: dict[str, Any]
    current_run: dict[str, Any] | None
    timeline: dict[str, Any] | None
    likelihoods: dict[str, Any] | None
    diagnosis: dict[str, Any] | None
    history: list[dict[str, Any]]
    trends: dict[str, Any] | None
    notes: list[dict[str, Any]] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    @property
    def title(self) -> str:
        return f"Engine report · {self.engine['label']}"

    def file_name(self) -> str:
        stem = re.sub(r"[^A-Za-z0-9]+", "_", f"{self.engine['label']}").strip("_")
        return f"engine_report_{stem}_{self.generated_utc[:10]}.pdf"


# -- building -----------------------------------------------------------------------------
def _source(run: RunEntry) -> str:
    src = run.source
    if src.get("mode") == "lab":
        return f"lab · {src.get('recording_run_id')}"
    return f"{src.get('dataset')} · {src.get('recording_run_id')}"


def history_rows(runs: list[RunEntry]) -> list[dict[str, Any]]:
    """One row per saved run, newest first (the dashboard history table's columns)."""
    rows = []
    for run in runs:
        s = run.summary
        faults = [p for c, p in (s.get("max_probabilities") or {}).items()
                  if c != "normal" and p is not None]
        rows.append({
            "run": f"#{run.id}",
            "saved (UTC)": run.created_utc[:16].replace("T", " "),
            "source": _source(run),
            "predicted fault": label(s.get("predicted_fault")),
            "alerts": ", ".join(f"{label(a['fault'])} {a['onset_s']:.1f} s"
                                for a in s.get("alerts", [])) or "none",
            "max P(fault)": _num(max(faults)) if faults else None,
            "peak anomaly": _num(s.get("peak_anomaly_score")),
            "final health": s.get("final_health_stage") or "-",
            "raw data": "kept" if run.raw_available else "pruned",
        })
    return rows


def _current(run: RunEntry) -> dict[str, Any]:
    s = run.summary
    leads = s.get("warning_leads_s") or []
    return {
        "id": run.id, "saved_utc": run.created_utc, "source": _source(run),
        "analysed": f"{run.analyzed_s:.1f} of {run.duration_s:g} s"
                    + ("" if run.complete else " (saved before the end)"),
        "predicted_fault": label(s.get("predicted_fault")),
        "alerts": ", ".join(f"{label(a['fault'])} from {a['onset_s']:.1f} s"
                            for a in s.get("alerts", [])) or "none",
        "final_health_stage": s.get("final_health_stage") or "-",
        "health_summary": s.get("health_summary") or "-",
        "early_warning": ", ".join(f"{v:.1f} s" for v in leads) + " before the alert"
                         if leads else "-",
        "peak_anomaly": (f"{s['peak_anomaly_score']:.2f} (threshold "
                         f"{s.get('anomaly_threshold', float('nan')):.2f})"
                         if s.get("peak_anomaly_score") is not None else "-"),
        "localization": s.get("localization") or "-",
        "model_version": s.get("model_version") or "-",
    }


def _timeline(store: HistoryStore, run: RunEntry) -> dict[str, Any] | None:
    decisions = store.decisions(run.id)
    if decisions.empty:
        return None
    threshold = run.summary.get("anomaly_threshold")
    probs = [p or {} for p in decisions.get("probabilities", [None] * len(decisions))]
    series = {f: [_num(p.get(f), 3) for p in probs] for f in FAULTS}
    has_p = any(v is not None for values in series.values() for v in values)
    anomaly = [_num(a / threshold, 3) if threshold and a is not None
               and math.isfinite(a) else None for a in decisions["anomaly_score"]]
    stages = [s if isinstance(s, str) else "no data"
              for s in decisions.get("health_stage", [None] * len(decisions))]
    return {"t": [float(t) for t in decisions["t_end_s"]], "stage": stages,
            "probabilities": series if has_p else {}, "anomaly_ratio": anomaly,
            "duration_s": float(run.duration_s)}


def _likelihoods(run: RunEntry) -> dict[str, Any] | None:
    s = run.summary
    p = {c: v for c, v in (s.get("max_probabilities") or {}).items() if v is not None}
    if p:
        order = [c for c in ("normal", *FAULTS) if c in p]
        return {"kind": "likelihood", "title": "Highest likelihood per class (latest run)",
                "labels": [LABELS.get(c, c) for c in order],
                "keys": order, "values": [_num(p[c], 3) for c in order]}
    counts = s.get("decision_counts") or {}
    total = sum(counts.values())
    if not total:
        return None
    order = sorted(counts, key=counts.get, reverse=True)
    return {"kind": "share", "title": "Share of decisions (latest run; no calibrated "
                                      "likelihoods from this model)",
            "labels": [LABELS.get(c, c) for c in order], "keys": order,
            "values": [_num(counts[c] / total, 3) for c in order]}


def _trends(runs_oldest_first: list[RunEntry]) -> dict[str, Any] | None:
    if not runs_oldest_first:
        return None
    max_p = {f: [_num((r.summary.get("max_probabilities") or {}).get(f), 3)
                 for r in runs_oldest_first] for f in FAULTS}
    ratio = []
    for r in runs_oldest_first:
        peak, thr = r.summary.get("peak_anomaly_score"), r.summary.get("anomaly_threshold")
        ratio.append(_num(peak / thr, 3) if peak is not None and thr else None)
    has_p = any(v is not None for values in max_p.values() for v in values)
    return {"runs": [f"#{r.id}" for r in runs_oldest_first],
            "max_probabilities": max_p if has_p else {}, "anomaly_ratio": ratio}


def build_report(store: HistoryStore, profile: Profile, now: str | None = None) -> EngineReport:
    """Everything the report shows, from history only (no Gemini call)."""
    runs = store.list_runs(profile.id)  # newest first
    current = runs[0] if runs else None
    latest = store.latest_diagnosis(profile.id)
    s = profile.specs
    specs = []
    for key, name in SPEC_ROWS:
        value = s.get(key)
        specs.append((name, "-".join(map(str, value)) if isinstance(value, list)
                      else f"{value:g}" if isinstance(value, float) else str(value)))
    return EngineReport(
        generated_utc=now or datetime.now(UTC).isoformat(timespec="seconds"),
        engine={"label": profile.label, "make": profile.make, "model": profile.model,
                "serial": profile.serial, "nickname": profile.nickname, "specs": specs,
                "spec_source": SPEC_SOURCE_TEXT.get(profile.spec_source, profile.spec_source),
                "warning": reliability_warning(profile.specs)},
        current_run=_current(current) if current else None,
        timeline=_timeline(store, current) if current else None,
        likelihoods=_likelihoods(current) if current else None,
        diagnosis=None if latest is None else {
            "id": latest.id, "created_utc": latest.created_utc, "model": latest.model,
            "fallback": latest.fallback, "fallback_reason": latest.fallback_reason,
            "latest_run_id": latest.latest_run_id, "output": latest.output},
        history=history_rows(runs),
        trends=_trends(list(reversed(runs[:MAX_TREND_RUNS]))),
        notes=[{"id": n.id, "created_utc": n.created_utc, "run_id": n.run_id, "text": n.text}
               for n in store.list_notes(profile.id)[:MAX_NOTES]],
    )


# -- charts (matplotlib, for the PDF) -----------------------------------------------------
def _figure(height_in: float):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7.2, height_in), dpi=150)
    fig.patch.set_facecolor("white")
    return fig


def _png(fig) -> bytes:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()


def _plot_values(ax, x, values, **kwargs) -> None:
    ax.plot(x, [math.nan if v is None else v for v in values], **kwargs)


def timeline_png(timeline: dict[str, Any]) -> bytes:
    """Fault likelihoods (or anomaly/threshold) over the run, with the health-stage strip."""
    fig = _figure(2.9)
    top, strip = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    t = timeline["t"]
    if timeline["probabilities"]:
        for fault, values in timeline["probabilities"].items():
            _plot_values(top, t, values, color=FAULT_COLORS[fault], label=LABELS[fault], lw=1.5)
        top.set_ylim(0, 1.02)
        top.set_ylabel("likelihood")
        top.axhline(0.7, color="#555", ls=":", lw=0.8)
    else:
        _plot_values(top, t, timeline["anomaly_ratio"], color="#1f77b4", lw=1.5,
                     label="anomaly score / threshold")
        top.axhline(1.0, color="#555", ls=":", lw=0.8)
        top.set_ylabel("anomaly / threshold")
    top.legend(loc="upper left", fontsize=7, frameon=False, ncol=3)
    top.grid(alpha=0.25)
    step = (t[1] - t[0]) if len(t) > 1 else 0.5
    for x, stage in zip(t, timeline["stage"], strict=True):
        strip.axvspan(x - step, x, color=STAGE_COLORS.get(stage, "#9e9e9e"), lw=0)
    strip.set_yticks([])
    strip.set_ylabel("health", rotation=0, ha="right", va="center", fontsize=8)
    strip.set_xlim(0, timeline["duration_s"])
    strip.set_xlabel("stream time (s)")
    seen = list(dict.fromkeys(timeline["stage"]))
    from matplotlib.patches import Patch

    strip.legend(handles=[Patch(color=STAGE_COLORS.get(s, "#9e9e9e"), label=s) for s in seen],
                 loc="upper center", bbox_to_anchor=(0.5, -0.9), ncol=4, fontsize=7,
                 frameon=False)
    return _png(fig)


def likelihoods_png(likelihoods: dict[str, Any]) -> bytes:
    fig = _figure(1.8)
    ax = fig.subplots()
    colors = [FAULT_COLORS.get(k, "#7f7f7f") for k in likelihoods["keys"]]
    ax.barh(likelihoods["labels"], likelihoods["values"], color=colors)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    for y, v in enumerate(likelihoods["values"]):
        ax.text(min(v + 0.01, 0.93), y, f"{v:.2f}", va="center", fontsize=7)
    ax.set_title(likelihoods["title"], fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    return _png(fig)


def trends_png(trends: dict[str, Any]) -> bytes:
    fig = _figure(2.2)
    ax = fig.subplots()
    x = list(range(len(trends["runs"])))
    if trends["max_probabilities"]:
        for fault, values in trends["max_probabilities"].items():
            _plot_values(ax, x, values, color=FAULT_COLORS[fault], marker="o", lw=1.5,
                         label=LABELS[fault])
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("max likelihood")
    else:
        _plot_values(ax, x, trends["anomaly_ratio"], color="#1f77b4", marker="o", lw=1.5,
                     label="peak anomaly / threshold")
        ax.axhline(1.0, color="#555", ls=":", lw=0.8)
        ax.set_ylabel("anomaly / threshold")
    ax.set_xticks(x, trends["runs"], fontsize=7)
    ax.set_xlabel("saved run (oldest to newest)")
    ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=3)
    ax.grid(alpha=0.25)
    return _png(fig)


# -- PDF ----------------------------------------------------------------------------------
def _fonts() -> tuple[Path, Path]:
    from matplotlib import get_data_path

    base = Path(get_data_path()) / "fonts" / "ttf"
    return base / "DejaVuSans.ttf", base / "DejaVuSans-Bold.ttf"


def render_pdf(report: EngineReport) -> bytes:
    """The report as PDF bytes (A4), with the same sections and data as the preview."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    regular, bold = _fonts()
    footer_text = "Decision support only; not a substitute for a physical inspection."

    class ReportPDF(FPDF):
        def footer(self):
            self.set_y(-12)
            self.set_font("DejaVu", "", 7)
            self.set_text_color(110)
            self.cell(0, 5, f"{footer_text}   Page {self.page_no()}/{{nb}}", align="C")

    pdf = ReportPDF(format="A4")
    pdf.set_title(report.title)
    pdf.set_creator("EngineSense diagnostics")
    pdf.add_font("DejaVu", "", str(regular))
    pdf.add_font("DejaVu", "B", str(bold))
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_page()
    width = pdf.epw

    def heading(text: str, size: int = 13) -> None:
        if pdf.get_y() > pdf.h - 45:
            pdf.add_page()
        pdf.ln(2)
        pdf.set_font("DejaVu", "B", size)
        pdf.set_text_color(20)
        pdf.multi_cell(width, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(200)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + width, pdf.get_y())
        pdf.ln(1.5)

    def para(text: str, size: float = 9, bold_text: bool = False, color: int = 30) -> None:
        pdf.set_font("DejaVu", "B" if bold_text else "", size)
        pdf.set_text_color(color)
        pdf.multi_cell(width, size * 0.5, text, align="L", new_x=XPos.LMARGIN,
                       new_y=YPos.NEXT)

    def table(rows: list[list[str]], widths: list[float], size: float = 8,
              header: bool = True) -> None:
        pdf.set_font("DejaVu", "", size)
        pdf.set_text_color(30)
        with pdf.table(col_widths=widths, text_align="LEFT", line_height=size * 0.55,
                       first_row_as_headings=header,
                       headings_style=None if not header else _heading_style()) as t:
            for row in rows:
                cells = t.row()
                for value in row:
                    cells.cell("-" if value is None else str(value))
        pdf.ln(2)

    def _heading_style():
        from fpdf.fonts import FontFace

        return FontFace(family="DejaVu", emphasis="BOLD", fill_color=(235, 238, 242))

    def chart(title: str, png: bytes, w: float | None = None) -> None:
        """A captioned chart, moved to a new page with its caption when it does not fit."""
        from PIL import Image

        w = w or width
        with Image.open(io.BytesIO(png)) as im:
            h = w * im.height / im.width
        if pdf.get_y() + h + 8 > pdf.h - pdf.b_margin:
            pdf.add_page()
        para(title, 10, bold_text=True)
        pdf.image(io.BytesIO(png), w=w)
        pdf.ln(1)

    # Title
    pdf.set_font("DejaVu", "B", 17)
    pdf.multi_cell(width, 9, report.title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    para(f"Generated {utc_text(report.generated_utc)}", 8, color=100)
    para(report.disclaimer, 8, color=100)

    # 1 Engine
    e = report.engine
    heading("1. Engine profile and specs")
    table([["Field", "Value"], ["Make", e["make"]], ["Model", e["model"]],
           ["Serial / ID", e["serial"]], ["Nickname", e["nickname"] or "-"],
           *[[n, v] for n, v in e["specs"]], ["Specs source", e["spec_source"]]],
          [45, 145])
    if e["warning"]:
        para("Warning: " + e["warning"], 9, bold_text=True, color=150)

    # 2 Current run
    heading("2. Latest run")
    cur = report.current_run
    if cur is None:
        para("No saved runs yet. Play a run to the end (or use Save run so far) with this "
             "engine selected.")
    else:
        table([["Field", "Value"], ["History run", f"#{cur['id']}"],
               ["Saved (UTC)", cur["saved_utc"][:16].replace("T", " ")],
               ["Source", cur["source"]], ["Analysed", cur["analysed"]],
               ["Predicted fault", cur["predicted_fault"]], ["Alerts", cur["alerts"]],
               ["Final health stage", cur["final_health_stage"]],
               ["Health stages", cur["health_summary"]],
               ["Early warning", cur["early_warning"]], ["Peak anomaly", cur["peak_anomaly"]],
               ["Cylinder", cur["localization"]], ["Model", cur["model_version"]]],
              [45, 145])
        if report.timeline:
            chart("Health timeline", timeline_png(report.timeline))
        if report.likelihoods:
            chart("Likelihoods", likelihoods_png(report.likelihoods), width * 0.85)

    # 3 Diagnosis
    heading("3. Latest diagnosis")
    diag = report.diagnosis
    if diag is None:
        para("No diagnosis saved for this engine yet. Use Diagnose in the dashboard to add "
             "ranked causes and recommended actions to this report.")
    else:
        out = diag["output"]
        para(f"{utc_text(diag['created_utc'])} · {diag['model']}"
             + (f" · saved runs up to #{diag['latest_run_id']}" if diag["latest_run_id"]
                else ""), 8, color=100)
        para(out["summary"], 9.5)
        pdf.ln(1)
        para("Causes (ranked by confidence)", 10, bold_text=True)
        if not out["causes"]:
            para("No cause is supported by the saved data.")
        for k, c in enumerate(out["causes"], 1):
            kind = "detected by the model" if c["kind"] == "detected" else "suggested"
            para(f"{k}. {c['title']}  ({kind}, {c['confidence']} confidence)", 9,
                 bold_text=True)
            para(c["reasoning"], 8.5)
            if c.get("confidence_notes"):
                para(c["confidence_notes"], 8, color=90)
            para("Evidence: " + ", ".join(c["evidence_ids"]), 8, color=90)
            pdf.ln(1)
        para("Recommended actions", 10, bold_text=True)
        for urgency, name in URGENCY_NAMES.items():
            steps = [a for a in out["actions"] if a["urgency"] == urgency]
            if steps:
                para(name, 9, bold_text=True)
                for a in steps:
                    para("•  " + a["step"] + (f"  (for: {', '.join(a['addresses'])})"
                                              if a["addresses"] else ""), 8.5)
        if out["data_gaps"]:
            para("Data gaps", 10, bold_text=True)
            for g in out["data_gaps"]:
                para("•  " + g, 8.5)

    # 4 History and trends
    heading("4. Run history and trends")
    if report.history:
        table([HISTORY_COLUMNS] + [[r[c] for c in HISTORY_COLUMNS] for r in report.history],
              [10, 22, 27, 22, 33, 15, 15, 30, 16], size=6.5)
    else:
        para("No saved runs.")
    if report.trends and len(report.trends["runs"]) > 1:
        chart("Trend across runs", trends_png(report.trends))
    elif report.trends:
        para("A trend needs at least two saved runs.", 8.5, color=90)
    if report.notes:
        heading("5. Notes", 12)
        for n in report.notes:
            para(f"{n['created_utc'][:16].replace('T', ' ')} UTC · "
                 + (f"run #{n['run_id']}" if n["run_id"] else "engine"), 8, color=100)
            para(n["text"], 9)
    return bytes(pdf.output())
