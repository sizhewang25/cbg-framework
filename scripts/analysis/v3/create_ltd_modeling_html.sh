#!/usr/bin/env bash
#
# Build the v3 LTD modeling viewers for one or more runs.
#
#   ./create_ltd_modeling_html.sh                           # the default set
#   ./create_ltd_modeling_html.sh as01-260728-260802-mesh   # named runs
#   ./create_ltd_modeling_html.sh as0{1,2,3}-260728-260802  # brace expansion
#
#   METHODS="vanilla_cbg octant_cbg_spl" ./create_ltd_modeling_html.sh   # subset
#   FOLDS="fold_4" ./create_ltd_modeling_html.sh                        # one fold
#
# Deliberately a SEPARATE script from create_analysis_artifacts.sh rather than
# another section of it, for create_mtl_map.sh's three reasons:
#
#   1. Cost. This command evaluates the fitted model over an RTT grid for every
#      VP of every fold -- ~64k predictions and ~5 MB of HTML per combo, about
#      11 s per run on as01 -- against the seconds-scale commands that sweep
#      makes up. It is the second-most expensive thing in the layer after the
#      MTL replay, and a sweep people stop running is a sweep that goes stale.
#
#   2. Nothing depends on it. It reads the fold checkpoints and the dataset
#      directly, not any v3 artifact, and writes nothing another command
#      consumes -- so it sits outside the ordered pipeline and splitting it out
#      forfeits no ordering guarantee.
#
#   3. It is looked at, not computed over. These pages answer "is the LTD fit
#      why this run moved?", which is a question you ask deliberately. They
#      change only when the LTD stage or the dataset does, never when a
#      downstream table is re-scored.
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

# Optional narrowing, passed through as repeated flags.
EXTRA=()
for m in $METHODS; do EXTRA+=(--method "$m"); done
for f in $FOLDS;   do EXTRA+=(--fold "$f");   done

for R in "${RUNS[@]}"; do
  printf '\n==================== %s ====================\n' "$R"

  # A run with no benchmark output is expected state -- a run id typo, or a
  # benchmark that has not been launched -- so it is a SKIP.
  if [ ! -d "outputs/benchmark/v2/$R" ]; then
    echo "skipping: no outputs/benchmark/v2/$R"
    SKIPPED+=("$R :: no benchmark output")
    continue
  fi

  # A missing config is a NOTE, not a skip -- create_mtl_map.sh's rule, for its
  # reason. create_analysis_artifacts.sh needs the config to say what a run IS
  # (source kwargs, the peer's PNI list); this command needs nothing from it. It
  # reads the fold checkpoints directly and finds the dataset CSV through the
  # run's own eval_source/, so demanding a config would make the viewer
  # unrenderable for a run that has benchmark output and none --
  # as01-260728-260802-mesh-heapfix is exactly that today.
  CFG=configs/$R.yaml
  if [ -f "$CFG" ]; then
    V3=(python -m scripts.analysis.v3.cli --config "$CFG")
  else
    echo "note: no $CFG, rendering on CLI defaults"
    V3=(python -m scripts.analysis.v3.cli)
  fi
  OUT=outputs/analysis/v3/$R/ltd-model

  # One interactive HTML per combo: the RTT-vs-distance fit scatter, the band the
  # fitted model returns, the 2/3 c baseline, and an optional per-target overlay.
  #
  # Needs no other v3 command -- it reads the fold checkpoints directly -- but it
  # DOES need the fit scatter, which is the one input not in the output tree. The
  # command resolves that itself, preferring a materialized
  # inputs/benchmark/v2/<source>/<run>/<setup>/<fold>/fit_samples.parquet and
  # falling back to the dataset CSV filtered by its pinned
  # `.stratification.json`. Both routes are load-bearing on today's runs:
  # as7018-ripe-mesh has only the former, as01/02/03 only the latter.
  #
  # Not guarded here on either input. Which route applies is a per-FOLD question
  # the command already answers, and a bash-side guess at it would have to
  # re-resolve the run's canonical CSV -- something only `resolve_source_csv`
  # knows how to do. So the command runs unconditionally and reports its own
  # per-combo skips; the check below is on what actually landed on disk.
  #
  # ONE invocation per run, not one per combo -- the opposite of
  # create_mtl_map.sh. That script pays ~3 s of repeated setup per method to buy
  # failure isolation, because inside one call the first method to raise ends the
  # run. Here the isolation is already inside the command: a combo whose folds
  # have no reachable fit samples is caught per combo and reported as a skip
  # while the rest still render. Splitting would also give up the per-fold
  # scatter cache that the combos of one run share.
  run plot-ltd-model "${V3[@]}" plot-ltd-model --run-id "$R" "${EXTRA[@]}"

  # A run whose folds have no reachable fit samples exits 0 having written a
  # manifest and no pages -- correct for the command (nothing failed), but it
  # would leave this script reporting an "ok" that produced nothing. An
  # uncollected dataset is expected state, so that is a SKIP, not a failure.
  if [ -z "$(ls -A "$OUT"/*.html 2>/dev/null)" ]; then
    echo "note: $R produced no viewer -- see $OUT/manifest.json for the per-combo reason"
    SKIPPED+=("$R :: no ltd-model HTML written (no reachable fit samples)")
  else
    printf 'wrote %s viewer(s):\n' "$(ls -1 "$OUT"/*.html | wc -l)"
    du -h "$OUT"/*.html | sed 's/^/  /'
  fi
done

printf '\n==================== summary ====================\n'
printf 'runs: %d   commands ok: %d   failed: %d   skipped: %d\n' \
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
