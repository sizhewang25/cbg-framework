#!/usr/bin/env bash
# Prepare the synthetic smoke-test dataset.
# Prerequisite: source .venv/bin/activate

set -euo pipefail

mkdir -p datasets
python -m scripts.benchmark.v2.sources._make_smoke_csv ./datasets/smoke-test.csv

cat <<'EOF'

✔ smoke dataset written to datasets/smoke-test.csv

Next steps:
  ./cli.sh      --configfile scripts/benchmark/v2/config/smoke-test-01.yaml
  ./analysis.sh --configfile scripts/analysis/config/smoke-test-01.yaml

Both scripts assume your .venv is active (run `source .venv/bin/activate` first).
EOF
