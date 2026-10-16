"""Create a demo engine profile with a history that Diagnose and Report can use.

Replays four core_v3 runs of one simulated test engine (engine_058) through the
dashboard itself (Streamlit AppTest), so each run is saved exactly as a user's
replay would be, and adds technician notes between them:

    R1  run_00244  healthy
    R2  run_00245  healthy
    R3  run_00246  misfire on cylinder 3        note: spark plug replaced
    R4  run_00249  misfire returns, plus knock  note: customer complaint

The run choice is demo curation (like the demo presets); nothing from
evaluation_truth/ is written to history. Needs data/raw/core_v3, the v2 model,
and the crank/thermal calibrations (calibrate-crosscheck).

    uv run --all-extras python scripts/seed_demo_engine.py [--history-dir DIR]
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard/app.py"
DATASET = "core_v3"
MAKE, MODEL, SERIAL = "DRIVEX Sim", "Synthetic 1.6 I4", "DEMO-058"
NICKNAME = "Demo engine"
STORY = [
    ("run_00244", None),
    ("run_00245", None),
    ("run_00246", "Rough running reported. Replaced the spark plug on cylinder 3."),
    ("run_00249", "Customer reports the misfire is back, plus pinging under load."),
]


def widget(widgets, label):
    return next(w for w in widgets if w.label == label)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--history-dir", type=Path,
                        help="history directory (default: ENGINE_HISTORY_DIR or "
                             "data/engine_history)")
    args = parser.parse_args()
    if args.history_dir:
        os.environ["ENGINE_HISTORY_DIR"] = str(args.history_dir.resolve())
    sys.path.insert(0, str(ROOT))

    from streamlit.testing.v1 import AppTest

    from engine_diagnostics.history import HistoryStore
    from engine_diagnostics.profiles import load_catalog, lookup

    missing = [r for r, _ in STORY if not (ROOT / "data/raw" / DATASET / r).is_dir()]
    if missing:
        print(f"missing {DATASET} runs {missing}: generate core_v3 first", file=sys.stderr)
        return 1
    store = HistoryStore()
    if store.find_profile(MAKE, MODEL, SERIAL):
        print(f"{MAKE} {MODEL} serial {SERIAL} already exists in {store.root}; nothing done. "
              "Use --history-dir for a fresh history.")
        return 0
    entry = lookup(load_catalog(), MAKE, MODEL)
    profile = store.create_profile(MAKE, MODEL, SERIAL, dict(entry.engine), "catalog", NICKNAME)

    app = AppTest.from_file(str(APP), default_timeout=300).run()
    widget(app.sidebar.selectbox, "Engine").set_value(profile.id).run()
    for run_id, note in STORY:
        widget(app.sidebar.selectbox, "Dataset").set_value(DATASET).run()
        widget(app.sidebar.selectbox, "Run").set_value(run_id).run()
        end = float(app.session_state["bundle"].metadata["duration_s"])
        widget(app.sidebar.slider, "Seek (s)").set_value(end).run()
        widget(app.sidebar.button, "Seek").click().run()  # the finished replay is saved
        if app.exception:
            print(f"{run_id}: dashboard error {app.exception}", file=sys.stderr)
            return 1
        saved = app.session_state["saved_history_run"]
        summary = store.get_run(saved).summary
        print(f"run #{saved} ({run_id}): {summary['predicted_fault']}, "
              f"final health {summary.get('final_health_stage')}")
        if note:
            store.add_note(profile.id, note, saved)
            print(f"  note: {note}")
    print(f"\nCreated '{profile.label}' in {store.root}. In the dashboard pick it under "
          "Engine profile, then use Diagnose and Report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
