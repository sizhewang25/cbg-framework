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

# 1. Sanity-check the base Python path the venv was built against.
BASE_HOME="$(awk -F= '/^home/{gsub(/^ +| +$/, "", $2); print $2}' .venv/pyvenv.cfg)"
if [[ ! -x "$BASE_HOME/python3.12" && ! -x "$BASE_HOME/python3" && ! -x "$BASE_HOME/python" ]]; then
    cat <<EOF
ERROR: the bundled venv references a base Python at $BASE_HOME that doesn't
       exist on this target. Either install python3.12 at that path, or
       point the venv at a different base:

           sed -i "s|^home = .*|home = /actual/path/to/python3.12/dir|" .venv/pyvenv.cfg

EOF
    exit 1
fi
echo "✔ pyvenv.cfg base ($BASE_HOME) resolves"

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
