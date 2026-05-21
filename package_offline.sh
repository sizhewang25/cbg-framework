#!/usr/bin/env bash
# Build an offline-installable bundle of geoscale.
#
# Run on a connected Linux host with the same arch/glibc as the air-gapped
# target. Produces wheels via `pip wheel` (compiles from sdist when no wheel
# is published) so the target needs no compiler / no network.
#
# Requirements on this build host:
#   - python3.12 (matching the target)
#   - python3.12 -m pip available
#   - GNU tar (for --exclude)
#
# Output: geoscale-offline-YYYYMMDD-HHMMSS.tar.gz in the repo root.

set -euo pipefail

PYTHON="${PYTHON:-python3.12}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

command -v "$PYTHON" >/dev/null || { echo "ERROR: $PYTHON not on PATH"; exit 1; }

# 1. Extract runtime deps from [project.dependencies] in pyproject.toml.
#    Uses tomllib (stdlib in 3.11+), so no extra deps on the build host.
"$PYTHON" - <<'PY' > requirements.txt
import tomllib
with open("pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)
for dep in pyproject["project"]["dependencies"]:
    # Strip the parenthesized spec form pip accepts; "numpy (>=1.26)" → "numpy>=1.26"
    print(dep.replace(" (", "").rstrip(")").replace(" ", ""))
PY
echo "✔ requirements.txt written ($(wc -l < requirements.txt) deps)"

# 2. Build wheels for the current platform. `pip wheel` compiles sdists locally
#    so the target gets pure-wheel installs (no build toolchain needed).
rm -rf wheels
mkdir -p wheels
"$PYTHON" -m pip wheel \
    --wheel-dir wheels/ \
    -r requirements.txt
echo "✔ wheels in $(pwd)/wheels ($(ls wheels/ | wc -l) files)"

# 3. Tarball the repo + wheels, excluding heavy / build-artifact dirs.
TARBALL="geoscale-offline-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar --exclude='./.git' \
    --exclude='./.venv' \
    --exclude='./venv' \
    --exclude='./datasets' \
    --exclude='./clickhouse_files' \
    --exclude='./.snakemake' \
    --exclude='./tasks' \
    --exclude='./analysis/results' \
    --exclude='./analysis/figures' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='./scripts/benchmark/v2/inputs' \
    --exclude='./scripts/benchmark/v2/outputs' \
    --exclude='./scripts/analysis/outputs' \
    --exclude="./$TARBALL" \
    --exclude='./geoscale-offline-*.tar.gz' \
    -czf "$TARBALL" \
    .

echo "✔ Bundle: $TARBALL  ($(du -h "$TARBALL" | cut -f1))"
echo
echo "Transfer this tarball to the target, then run install_offline.sh."
