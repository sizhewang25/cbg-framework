#!/usr/bin/env bash
# Run the v2 CBG benchmark.
# Prerequisite: source .venv/bin/activate
# Usage: ./cli.sh --configfile scripts/benchmark/v2/config/smoke-test-01.yaml

set -euo pipefail

python -m snakemake \
    -s scripts/benchmark/v2/Snakefile \
    --cores all \
    "$@"
