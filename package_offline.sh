#!/usr/bin/env bash
# Build an offline-installable bundle of geoscale.
#
# Run on a connected Linux host with the same Linux distro / glibc / CPU
# arch / Python 3.12.x patch level as the air-gapped target. The build
# materializes the venv into ./.venv/ and then tars the repo + venv,
# excluding datasets/, .git, and other regenerable state.
#
# Requirements on this build host:
#   - python3.12 (matching the target)
#   - python3.12 -m pip + network access (this is the only step that fetches)
#   - GNU tar (for --exclude)

set -euo pipefail

PYTHON="${PYTHON:-python3.12}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

command -v "$PYTHON" >/dev/null || { echo "ERROR: $PYTHON not on PATH"; exit 1; }

# 1. Extract runtime deps from [project.dependencies] in pyproject.toml.
"$PYTHON" - <<'PY' > requirements.txt
import tomllib
with open("pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)
for dep in pyproject["project"]["dependencies"]:
    print(dep.replace(" (", "").rstrip(")").replace(" ", ""))
PY
echo "✔ requirements.txt written ($(wc -l < requirements.txt) deps)"

# 2. Build a fresh venv at .venv/ with --copies so the python binary is a
#    real file (not a symlink to the host's system python). pip-install
#    every dep into it.
rm -rf .venv
"$PYTHON" -m venv --copies .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
echo "✔ venv at $(pwd)/.venv ($(du -sh .venv | cut -f1))"

# 3. Tar the repo + .venv, excluding heavy / regenerable directories.
TARBALL="geoscale-offline-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar --exclude='./.git' \
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
echo "Transfer to the target, extract, and run install_offline.sh."
