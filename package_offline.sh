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

# 2. Materialize .venv/. By default, build a fresh venv with --copies (so the
#    python binary is a real file, not a symlink to the host's system python)
#    and pip-install every dep into it. If REUSE_VENV=1 and a usable .venv
#    already exists (e.g. populated by `poetry install`), reuse it as-is —
#    this lets poetry own dep resolution while package_offline only packages.
if [[ "${REUSE_VENV:-0}" == "1" && -x ./.venv/bin/python3.12 ]]; then
    # Re-copy the interpreter if it's still a symlink to the host python,
    # so the bundle stays portable to hosts that lack python3.12.
    for f in .venv/bin/python .venv/bin/python3 .venv/bin/python3.12; do
        if [[ -L "$f" ]]; then
            target="$(readlink -f "$f")"
            rm "$f"
            cp "$target" "$f"
        fi
    done
    echo "✔ reusing existing .venv at $(pwd)/.venv ($(du -sh .venv | cut -f1))"
else
    rm -rf .venv
    "$PYTHON" -m venv --copies .venv
    ./.venv/bin/pip install --upgrade pip
    ./.venv/bin/pip install -r requirements.txt
    echo "✔ venv at $(pwd)/.venv ($(du -sh .venv | cut -f1))"
fi

# 3. Tar the repo + .venv, excluding heavy / regenerable directories.
# In the benchmark/analysis config dirs, keep only template.yaml and drop the
# rest (per-run configs are regenerable). tar has no "exclude all but one", so
# build the exclude list dynamically from what's on disk.
CONFIG_EXCLUDES="$(mktemp)"
trap 'rm -f "$CONFIG_EXCLUDES"' EXIT
for cfgdir in ./scripts/analysis/config ./scripts/benchmark/v2/config; do
    [[ -d "$cfgdir" ]] && find "$cfgdir" -type f ! -name 'template.yaml' >> "$CONFIG_EXCLUDES"
done
echo "✔ keeping only template.yaml in config dirs ($(wc -l < "$CONFIG_EXCLUDES") files excluded)"

TARBALL="geoscale-offline-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar --exclude-from="$CONFIG_EXCLUDES" \
    --exclude='./.git' \
    --exclude='./datasets' \
    --exclude='./clickhouse_files' \
    --exclude='./.snakemake' \
    --exclude='./tasks' \
    --exclude='./analysis/results' \
    --exclude='./analysis/figures' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*/benchmark/v2/inputs' \
    --exclude='./logs' \
    --exclude='./scripts/*/logs' \
    --exclude='./scripts/*/outputs' \
    --exclude="./$TARBALL" \
    --exclude='./geoscale-offline-*.tar.gz' \
    -czf "$TARBALL" \
    .

echo "✔ Bundle: $TARBALL  ($(du -h "$TARBALL" | cut -f1))"
echo
echo "Transfer to the target, extract, and run install_offline.sh."
