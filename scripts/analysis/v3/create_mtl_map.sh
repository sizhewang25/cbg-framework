#!/usr/bin/env bash
#
# Render the MTL case viewer -- one self-contained interactive HTML per method
# -- for one or more runs.
#
#   ./create_mtl_map.sh                           # the default set
#   ./create_mtl_map.sh as01-260728-260802-mesh   # named runs
#   ./create_mtl_map.sh as0{1,2,3}-260728-260802  # brace expansion
#
# Run from the repo root, like create_analysis_artifacts.sh: the config path is
# relative. Output lands in
# outputs/analysis/v3/<run_id>/mtl-map/<grid>-<res>/mtl_map.<method>.html and
# opens over file:// -- no web server, unlike the v2 viewers.
#
# Env knobs, all optional:
#
#   NO_REGIONS=1    drop the MTL feasible-region layer. Seconds per run instead
#                   of minutes; what you want while iterating on anything else.
#   WORKERS=4       processes for the region replay. Unset defers to the
#                   config's `plot-mtl-map.workers`, then to the CLI's own 8.
#   METHODS="a b"   render only these, instead of every combo + shortest_ping.
#   RESOLUTION=4    h3 resolution. Unset defers to the config's `common:`.
#
# Unset is not the same as passing the default: an explicit flag outranks the
# config, so exporting WORKERS would silently override every run's declared
# value. Each knob is therefore appended only when it is actually set.
#
# WHY THIS IS A SEPARATE SCRIPT FROM create_analysis_artifacts.sh
#
#   1. Cost. Every command in that script is seconds; this one is minutes to
#      tens of minutes per run, because the MTL feasible region is never
#      serialized by the benchmark and each one is a full re-run of the planar
#      intersection (~7 s per target on the Octant family, so ~47 min serial
#      for 399 targets, ~7 min at -j 8). Folding it in would turn a three-run
#      artifact rebuild from minutes into an hour, and the sweep people stop
#      running is the sweep that goes stale.
#
#   2. Nothing depends on it. The map reads no v3 artifact -- it rebuilds the
#      answer space, the seed scoring and the proximity labels in-process by
#      calling the same functions that write them -- and it writes nothing any
#      other command consumes. So it sits outside the ordered pipeline, and
#      splitting it out forfeits no ordering guarantee.
#
#   3. It is looked at, not computed over. The other artifacts are inputs to
#      the paper's numbers; this one answers "why did this method do THAT, on
#      THIS target?", which is a question you ask deliberately.
#
# A config is used when `configs/<run_id>.yaml` exists and skipped when it does
# not -- the opposite of create_analysis_artifacts.sh, which treats a missing
# config as a skip. That script's commands need the config to say what the run
# IS (source kwargs, the peer's PNI list); this one needs it only for `workers`,
# and resolves its canonical CSV off the run's own eval_source/. Demanding a
# config would make the map unrenderable for a run that has benchmark output and
# no config -- as01-260728-260802-mesh-heapfix is exactly that today.
#
# Deliberately NOT `set -e`, for create_analysis_artifacts.sh's reason: a run
# with a half-finished benchmark fails, and aborting the sweep there would hide
# the state of every later run. Failures are collected and summarized.

DEFAULT_RUNS=(
  as01-260728-260802
  as02-260728-260802
  as03-260728-260802
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

from scripts.analysis.v3.modules.map_mtl import SHORTEST_PING
from scripts.analysis.v3.modules.paths import resolve_run

print("\n".join([*resolve_run(sys.argv[1]).combo_ids, SHORTEST_PING]))
PY
}

# Knobs, resolved once: they are global to the invocation, not per run.
EXTRA=()
[ -n "$NO_REGIONS" ] && EXTRA+=(--no-regions)
[ -n "$WORKERS" ]    && EXTRA+=(--workers "$WORKERS")
[ -n "$RESOLUTION" ] && EXTRA+=(--resolution "$RESOLUTION")

for R in "${RUNS[@]}"; do
  printf '\n==================== %s ====================\n' "$R"

  # A run with no benchmark output is expected state -- a run id typo, or a
  # benchmark that has not been launched -- so it is a SKIP. `resolve_run` would
  # report it too, but once per method rather than once per run.
  if [ ! -d "outputs/benchmark/v2/$R" ]; then
    echo "skipping: no outputs/benchmark/v2/$R"
    SKIPPED+=("$R :: no benchmark output")
    continue
  fi

  CFG=configs/$R.yaml
  if [ -f "$CFG" ]; then
    V3=(python -m scripts.analysis.v3.cli --config "$CFG")
  else
    echo "note: no $CFG, rendering on CLI defaults"
    V3=(python -m scripts.analysis.v3.cli)
  fi

  # One invocation PER METHOD, not one per run.
  #
  # `plot-mtl-map` already loops methods internally and shares the run-level
  # build (answer space, proximity, VP roster, seed rings, CONUS Voronoi) across
  # them, so a single call is the cheaper shape -- measured on as01 at 5.2 s for
  # six methods against 3.2 s for one, i.e. ~2.8 s of shared setup repeated per
  # method here. That is bought deliberately: inside one call the methods run in
  # sorted order and the first one that raises ends the run, so a permanently
  # broken combo would permanently block every method after it in the alphabet.
  # 3 s per method is nothing against a replay measured in minutes, and it also
  # puts the failing METHOD in the summary rather than just the run.
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
    run "plot-mtl-map[$M]" "${V3[@]}" plot-mtl-map --run-id "$R" -m "$M" "${EXTRA[@]}"
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
