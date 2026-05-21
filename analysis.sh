#!/usr/bin/env bash
# Build error-CDF and phase plots from a benchmark run.
# Prerequisites:
#   - source .venv/bin/activate
#   - cli.sh has produced summary.parquet for the matching run_id under
#     scripts/benchmark/v2/outputs/.
# Usage: ./analysis.sh --configfile scripts/analysis/config/smoke-test-01.yaml

set -euo pipefail

python -m snakemake \
    -s scripts/analysis/Snakefile \
    --cores all \
    "$@"
