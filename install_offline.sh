#!/usr/bin/env bash
# Verify (and, if needed, relocate) the bundled venv on the target.
#
# Run this inside the extracted bundle. The bundle already ships a fully
# pip-installed .venv/, so this script:
#   1. Confirms the venv's base-Python path in pyvenv.cfg resolves on this host.
#   2. Rewrites .venv/bin/* shebangs to point at THIS extraction's python so
#      entry-point scripts (snakemake, jupyter, …) still work after a path move.
#   3. Smoke-imports the heavy modules and runs both unit suites.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

[[ -d .venv ]] || { echo "ERROR: .venv/ missing — bundle was built without it"; exit 1; }
[[ -f .venv/pyvenv.cfg ]] || { echo "ERROR: .venv/pyvenv.cfg missing"; exit 1; }

# 1. Check pyvenv.cfg's base Python path. If it doesn't resolve on this target
#    (the common case when the build host used pyenv or a custom install path),
#    auto-detect a python3.12 on PATH and rewrite the three relevant lines.
BASE_HOME="$(awk -F= '/^home/{gsub(/^ +| +$/, "", $2); print $2}' .venv/pyvenv.cfg)"
if [[ ! -x "$BASE_HOME/python3.12" && ! -x "$BASE_HOME/python3" && ! -x "$BASE_HOME/python" ]]; then
    echo "→ pyvenv.cfg base ($BASE_HOME) not on this target; searching PATH for python3.12"
    DETECTED="$(command -v python3.12 || true)"
    if [[ -z "$DETECTED" ]]; then
        cat <<EOF
ERROR: the bundled venv references $BASE_HOME, which doesn't exist on this
       target, and python3.12 was not found on PATH either. Install
       python3.12 and re-run, or edit .venv/pyvenv.cfg manually:

           sed -i "s|^home = .*|home = /actual/path/to/python3.12/dir|" .venv/pyvenv.cfg

EOF
        exit 1
    fi
    DETECTED_DIR="$(dirname "$DETECTED")"
    sed -i \
        -e "s|^home = .*|home = $DETECTED_DIR|" \
        -e "s|^executable = .*|executable = $DETECTED|" \
        -e "s|^command = .*|command = $DETECTED -m venv --copies $REPO_ROOT/.venv|" \
        .venv/pyvenv.cfg
    BASE_HOME="$DETECTED_DIR"
    echo "✔ pyvenv.cfg rewritten — base is now $BASE_HOME"
else
    echo "✔ pyvenv.cfg base ($BASE_HOME) resolves"
fi

# 2. Rewrite shebangs in .venv/bin/* to point at this extraction's python.
NEW_PYTHON="$REPO_ROOT/.venv/bin/python"
# Only rewrite files whose first line looks like a python shebang.
while IFS= read -r -d '' f; do
    if head -c 2 "$f" 2>/dev/null | grep -q '^#!'; then
        if head -n 1 "$f" | grep -q 'python'; then
            sed -i "1s|^#!.*|#!$NEW_PYTHON|" "$f"
        fi
    fi
done < <(find .venv/bin -type f -print0)
echo "✔ entry-point shebangs point at $NEW_PYTHON"

# 3. Smoke-import + run the unit suites.
./.venv/bin/python -c "
import numpy, pandas, pyarrow, matplotlib, shapely, snakemake, PIL
print('  imports ok — numpy', numpy.__version__, '· pyarrow', pyarrow.__version__,
      '· matplotlib', matplotlib.__version__)
"
./.venv/bin/python -m unittest discover -s scripts/framework/v2/tests -t . 2>&1 | tail -3
./.venv/bin/python -m unittest discover -s scripts/benchmark/v2/tests -t . 2>&1 | tail -3

cat <<EOF

✔ Install verified.

Run things via the venv's python (no activation needed):
  ./.venv/bin/python -m scripts.benchmark.v2.cli --help
  ./.venv/bin/snakemake -s scripts/benchmark/v2/Snakefile --configfile <your-config.yaml> -j 4

EOF
