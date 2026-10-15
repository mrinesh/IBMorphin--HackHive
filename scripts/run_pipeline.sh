#!/usr/bin/env bash
# Python pipeline for the core_v1 dataset, after MATLAB generation:
#   addpath('matlab'); generate_dataset();   (in MATLAB, from the repository root)
# Usage: scripts/run_pipeline.sh [--final]
#   --final also scores the test engines (once; refuses to overwrite the report).
set -euo pipefail
cd "$(dirname "$0")/.."

run() { echo "+ $*"; "$@"; }

run uv run --all-extras engine-diagnostics validate-config
run uv run --all-extras engine-diagnostics validate-runs data/raw/core_v1
run uv run --all-extras engine-diagnostics build-features \
    --raw-dir data/raw/core_v1 --truth-dir evaluation_truth/core_v1 \
    --batch-config configs/core_v1.json --output data/processed/core_v1_features.parquet
run uv run --all-extras engine-diagnostics train
if [[ "${1:-}" == "--final" ]]; then
    run uv run --all-extras engine-diagnostics evaluate
fi
echo "Dashboard: uv run --all-extras streamlit run dashboard/app.py"
