#!/usr/bin/env bash
#
# Render the MTL case viewer -- one self-contained interactive HTML per method
# -- for one or more runs.
#
#   ./create_mtl_map.sh                                # the default mesh set
#   ./create_mtl_map.sh as7018-ripe-mesh               # named runs
#   ./create_mtl_map.sh as0{1,2,3}-260728-260802-mesh  # brace expansion
#
# Output lands in
# outputs/analysis/v5/<run_id>/mtl-map/healpix-<nside>/mtl_map.<method>.html and
# opens over file:// -- no web server, unlike the v2 viewers.
#
# Env knobs, all optional:
#
#   NO_REGIONS=1    drop the MTL feasible-region layer. Seconds per run instead
#                   of minutes; what you want while iterating on anything else.
#   WORKERS=4       processes for the region replay. Unset defers to the CLI's
#                   own default (cores-1, capped at 8).
#   METHODS="a b"   render only these, instead of every combo + shortest_ping.
#   NSIDE=64        the rung to render at. Unset defers to the CLI's 128.
#   CELL_EXTENT="a b c d"   LON_MIN LON_MAX LAT_MIN LAT_MAX for the serving
#                   cells. NOT the view -- the map is pannable -- only how far
#                   the unbounded cells are drawn. Unset defers to CELL_FRAME.
#
# Unset is not the same as passing the default: an explicit flag outranks the
# CLI's own choice, so exporting WORKERS would force a value on every run
# whether or not the caller meant to. Each knob is appended only when set.
#
# WHY THIS IS A SEPARATE SCRIPT FROM create_analysis_artifacts.sh
#
#   1. Cost. Every command in that script is seconds; this one is minutes to
#      tens of minutes per run, because the MTL feasible region is never
#      serialized by the benchmark and each one is a full re-run of the planar
#      intersection (~7 s per TG on the Octant family, so ~47 min serial
#      for 399 TGs, ~7 min at -j 8). Folding it in would turn a three-run
#      artifact rebuild from minutes into an hour, and the sweep people stop
#      running is the sweep that goes stale.
#
#   2. Nothing depends on it. The map needs `build-answer-space` (v5 loads the
#      space rather than rebuilding it, so the cells it draws are the cells
#      `classify` scored), but it writes nothing any other command consumes.
#      So it sits at the end of the pipeline, and splitting it out forfeits no
#      ordering guarantee.
#
#   3. It is looked at, not computed over. The other artifacts are inputs to
#      the paper's numbers; this one answers "why did this method do THAT, on
#      THIS TG?", which is a question you ask deliberately.
#
# This script reads no config: every v5 command is addressed
# by --run-id against the benchmark output tree, so there is no configs/<run>.yaml
# branch and no "render on CLI defaults" fallback to explain.
#
# Deliberately NOT `set -e`, for create_analysis_artifacts.sh's reason: a run
# with a half-finished benchmark fails, and aborting the sweep there would hide
# the state of every later run. Failures are collected and summarized.

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO" || exit 1
export PATH="$REPO/.venv/bin:$PATH"     # the CLI is invoked as bare `python`

V5="python -m scripts.analysis.v5.cli"
BENCH_ROOT=outputs/benchmark/v2

DEFAULT_RUNS=(
  as01-260728-260802-mesh
  as02-260728-260802-mesh
  as03-260728-260802-mesh
)

if [ "$#" -gt 0 ]; then RUNS=("$@"); else RUNS=("${DEFAULT_RUNS[@]}"); fi

FAILED=()
SKIPPED=()
N_OK=0

# run <label> <command...> -- execute, keep going on failure, record which.
run() {
  local label=$1; shift
  if "$@"; then
    N_OK=$((N_OK + 1))
  else
    FAILED+=("$R :: $label")
  fi
}

# The CLI's own default method set, printed one per line.
#
# Asked of the repo rather than reconstructed by globbing fold_*/: `combo_ids`
# is the property `plot-mtl-map` itself defaults to, and SHORTEST_PING is the
# constant it appends, so this list cannot drift from the CLI's. The interpreter
# start is ~1 s per run, against minutes of replay.
methods_for() {
  python - "$1" <<'PY'
import sys

from scripts.analysis.v5.modules.map_mtl import SHORTEST_PING
from scripts.analysis.v5.modules.paths import resolve_run

print("\n".join([*resolve_run(sys.argv[1]).combo_ids, SHORTEST_PING]))
PY
}

# Knobs, resolved once: they are global to the invocation, not per run.
EXTRA=()
[ -n "$NO_REGIONS" ] && EXTRA+=(--no-regions)
[ -n "$WORKERS" ]    && EXTRA+=(--workers "$WORKERS")
[ -n "$NSIDE" ]      && EXTRA+=(--nside "$NSIDE")
# shellcheck disable=SC2206  -- word splitting is the point: four numbers.
[ -n "$CELL_EXTENT" ] && EXTRA+=(--cell-extent $CELL_EXTENT)

for R in "${RUNS[@]}"; do
  printf '\n==================== %s ====================\n' "$R"

  # A run with no benchmark output is expected state -- a run id typo, or a
  # benchmark that has not been launched -- so it is a SKIP. `resolve_run` would
  # report it too, but once per method rather than once per run.
  if [ ! -d "$BENCH_ROOT/$R" ]; then
    echo "skipping: no $BENCH_ROOT/$R"
    SKIPPED+=("$R :: no benchmark output")
    continue
  fi

  # One invocation PER METHOD, not one per run.
  #
  # `plot-mtl-map` already loops methods internally and shares the run-level
  # build (answer space, VP roster, edge table, serving cells) across them, so a
  # single call is the cheaper shape. That is given up deliberately: inside one
  # call the methods run in the order given and the first one that raises ends
  # the run, so a permanently broken combo would permanently block every method
  # after it. A few seconds of repeated setup is nothing against a replay
  # measured in minutes, and it also puts the failing METHOD in the summary
  # rather than just the run.
  METHOD_LIST=()
  if [ -n "$METHODS" ]; then
    read -r -a METHOD_LIST <<< "$METHODS"
  elif OUT=$(methods_for "$R"); then
    mapfile -t METHOD_LIST <<< "$OUT"
  else
    echo "skipping: cannot enumerate methods for $R"
    FAILED+=("$R :: enumerate methods")
    continue
  fi

  # An empty list means the run holds no combo with a targets.parquet AND the
  # baseline constant vanished -- i.e. something is wrong upstream, not here.
  if [ "${#METHOD_LIST[@]}" -eq 0 ] || [ -z "${METHOD_LIST[0]}" ]; then
    echo "skipping: no methods to render for $R"
    SKIPPED+=("$R :: no methods")
    continue
  fi

  echo "rendering ${#METHOD_LIST[@]} method(s): ${METHOD_LIST[*]}"
  for M in "${METHOD_LIST[@]}"; do
    run "plot-mtl-map[$M]" $V5 plot-mtl-map --run-id "$R" -m "$M" "${EXTRA[@]}"
  done
done

printf '\n==================== summary ====================\n'
printf 'runs: %d   maps ok: %d   failed: %d   skipped: %d\n' \
  "${#RUNS[@]}" "$N_OK" "${#FAILED[@]}" "${#SKIPPED[@]}"

if [ "${#SKIPPED[@]}" -gt 0 ]; then
  printf '\nskipped:\n'
  printf '  %s\n' "${SKIPPED[@]}"
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
  printf '\nfailed:\n'
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
