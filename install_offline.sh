#!/usr/bin/env bash
# Install geoscale on an air-gapped Linux machine from the bundled wheels.
#
# Run this inside the extracted bundle directory. No network is contacted —
# pip resolves every dep from ./wheels/.

set -euo pipefail

PYTHON="${PYTHON:-python3.12}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

command -v "$PYTHON" >/dev/null || { echo "ERROR: $PYTHON not on PATH"; exit 1; }
[[ -d wheels ]] || { echo "ERROR: wheels/ directory missing"; exit 1; }
[[ -f requirements.txt ]] || { echo "ERROR: requirements.txt missing"; exit 1; }

# 1. Fresh venv at .venv/ (collocated with the repo, easy to delete).
rm -rf .venv
"$PYTHON" -m venv .venv
echo "✔ venv at $(pwd)/.venv"

# 2. Install everything from the local wheelhouse.
./.venv/bin/pip install \
    --no-index \
    --find-links wheels/ \
    --upgrade pip
./.venv/bin/pip install \
    --no-index \
    --find-links wheels/ \
    -r requirements.txt
echo "✔ deps installed from wheels/"

# 3. Smoke-check imports + run the unit suites.
./.venv/bin/python -c "
import numpy, pandas, pyarrow, matplotlib, shapely, snakemake, PIL
print('  imports ok — numpy', numpy.__version__, '· pyarrow', pyarrow.__version__,
      '· matplotlib', matplotlib.__version__)
"
./.venv/bin/python -m unittest discover -s scripts/framework/v2/tests -t . -v 2>&1 | tail -3
./.venv/bin/python -m unittest discover -s scripts/benchmark/v2/tests -t . -v 2>&1 | tail -3

cat <<'EOF'

✔ Install verified.

Run things via the venv's python (no activation needed):
  ./.venv/bin/python -m scripts.benchmark.v2.cli --help
  ./.venv/bin/snakemake -s scripts/benchmark/v2/Snakefile --configfile <your-config.yaml> -j 4

EOF
