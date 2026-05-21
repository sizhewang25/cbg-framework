#!/usr/bin/env bash
# Install geoscale on a Linux machine from the bundled wheels.
#
# Run this inside the extracted bundle directory.
#
# Source-selection knobs (env vars):
#   PIP_INDEX_URL=<url>   — also pull from this enterprise/internal pip registry.
#                           When set, wheels/ acts as a fallback (--find-links).
#   PIP_NO_INDEX=1        — force air-gapped mode (wheels/ only, no index).
#                           This is the default when PIP_INDEX_URL is unset.

set -euo pipefail

PYTHON="${PYTHON:-python3.12}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

command -v "$PYTHON" >/dev/null || { echo "ERROR: $PYTHON not on PATH"; exit 1; }
[[ -f requirements.txt ]] || { echo "ERROR: requirements.txt missing"; exit 1; }

# Decide pip's resolution strategy.
PIP_ARGS=()
if [[ -n "${PIP_INDEX_URL:-}" ]]; then
    PIP_ARGS+=( --index-url "$PIP_INDEX_URL" )
    echo "→ using pip index: $PIP_INDEX_URL"
    if [[ -d wheels ]]; then
        PIP_ARGS+=( --find-links wheels/ )
        echo "→ wheels/ as fallback ($(ls wheels/ 2>/dev/null | wc -l) wheels)"
    fi
else
    [[ -d wheels ]] || { echo "ERROR: wheels/ missing and no PIP_INDEX_URL set"; exit 1; }
    PIP_ARGS+=( --no-index --find-links wheels/ )
    echo "→ air-gapped mode; resolving from wheels/ only"
fi

# 1. Fresh venv at .venv/ (collocated with the repo, easy to delete).
rm -rf .venv
"$PYTHON" -m venv .venv
echo "✔ venv at $(pwd)/.venv"

# 2. Install everything.
./.venv/bin/pip install "${PIP_ARGS[@]}" --upgrade pip
./.venv/bin/pip install "${PIP_ARGS[@]}" -r requirements.txt
echo "✔ deps installed"

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
