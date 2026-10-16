#!/usr/bin/env bash
# Python pipeline for the multi-fault core_v2 dataset, after MATLAB generation:
#   addpath('matlab'); generate_dataset(Config="configs/core_v2.json");
# Usage: scripts/run_pipeline_v2.sh [--final]
#   --final also scores the test engines (once; refuses to overwrite the reports).
set -euo pipefail
cd "$(dirname "$0")/.."

run() { echo "+ $*"; "$@"; }

run uv run --all-extras engine-diagnostics validate-config
run uv run --all-extras engine-diagnostics validate-runs data/raw/core_v2
run uv run --all-extras engine-diagnostics build-features \
    --raw-dir data/raw/core_v2 --truth-dir evaluation_truth/core_v2 \
    --batch-config configs/core_v2.json --output data/processed/core_v2_features.parquet
run uv run --all-extras engine-diagnostics train-multi
run uv run --all-extras engine-diagnostics train-localizer
if [[ "${1:-}" == "--final" ]]; then
    run uv run --all-extras engine-diagnostics evaluate-multi
    run uv run --all-extras engine-diagnostics evaluate-localization
fi
echo "Dashboard: uv run --all-extras streamlit run dashboard/app.py (dataset core_v2)"
