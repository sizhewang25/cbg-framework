#!/usr/bin/env bash
#
# Build the v4 analysis artifacts for the finals runs.
#
#   ./create_analysis_artifacts.sh                                  # default sets
#   ./create_analysis_artifacts.sh --mesh as01-...-mesh as02-...-mesh
#   ./create_analysis_artifacts.sh --mesh R1 R2 --weighted R3 R4
#
# Runs are grouped into two ARMS -- mesh and traffic-weighted -- because the
# three cross-dataset figures pool their inputs into one population, and a mesh
# run and its weighted subset are not one population: the weighted arm is a
# proper subset of the same targets, so pooling them would count a site twice
# and compare a dataset against itself. Each arm therefore gets its own pooled
# bars, its own Euler diagram and its own pooled error CDF, and
# `cross.cross_dir` keys the output directory on the arm as well as the dataset
# set so the second pass cannot overwrite the first.
#
# The arms are passed rather than sniffed off a `-weighted` suffix. run_finals.sh
# already holds both lists, and a caller with a differently-named arm should not
# have to discover that the naming was load-bearing.
#
# Deliberately NOT `set -e`. Per-run commands are largely independent and a run
# with a half-finished benchmark will fail several of them; aborting on the first
# failure would hide the state of every later run. Failures are collected and
# printed as a summary instead, and the script exits non-zero if any occurred.
#
# Unlike the v3 script this one reads no config: every v4 command is addressed by
# `--run-id` against the benchmark output tree. A run id with no tree under
# outputs/benchmark/v2/ is a SKIP, not a failure -- the weighted arms depend on
# traffic-annotated meshes that are enterprise-sensitive and generally not
# collected, so mesh-only is expected state rather than an error.

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO" || exit 1
export PATH="$REPO/.venv/bin:$PATH"     # the CLI is invoked as bare `python`

V4="python -m scripts.analysis.v4.cli"
BENCH_ROOT=outputs/benchmark/v2

DEFAULT_MESH=(
  as01-260728-260802-mesh
  as02-260728-260802-mesh
  as03-260728-260802-mesh
)
DEFAULT_WEIGHTED=(
  as01-260728-260802-weighted
  as02-260728-260802-weighted
  as03-260728-260802-weighted
)

# The six published variants, passed to plot-euler explicitly.
#
# `combo_ids` globs the OUTPUT TREE, not the config, so the mesh runs also carry
# `spotter_h3_cbg` -- the density MTL's preserved H3 backup, kept for the
# grid-change comparison and runnable from no config. As a seventh circle it
# saturates the layout (its label cannot clear Octant-Hull's at nside 16), so it
# is excluded here rather than by a default exclusion list inside the figure,
# which is what `guard_common_methods` exists to prevent.
#
# Only plot-euler is restricted. The outcome bars and the error CDF have room for
# the extra arm and it is a real scored method, so they keep it where it exists.
EULER_METHODS=(
  vanilla_cbg
  million_scale_cbg
  octant_cbg_hull
  octant_cbg_spl
  spotter_cbg
  shortest_ping
)

MESH=()
WEIGHTED=()
if [ "$#" -eq 0 ]; then
  MESH=("${DEFAULT_MESH[@]}")
  WEIGHTED=("${DEFAULT_WEIGHTED[@]}")
else
  ARM=""
  for a in "$@"; do
    case "$a" in
      --mesh)     ARM=mesh ;;
      --weighted) ARM=weighted ;;
      -*)
        echo "create_analysis_artifacts.sh: unknown argument: $a" >&2
        echo "usage: $0 [--mesh RUN...] [--weighted RUN...]" >&2
        exit 2 ;;
      *)
        case "$ARM" in
          mesh)     MESH+=("$a") ;;
          weighted) WEIGHTED+=("$a") ;;
          *)
            echo "create_analysis_artifacts.sh: $a came before --mesh/--weighted" >&2
            exit 2 ;;
        esac ;;
    esac
  done
fi

FAILED=()
SKIPPED=()
N_OK=0
R=_setup       # `run`'s label prefix; each section sets it to what it is building

# run <label> <command...> -- execute, keep going on failure, record which.
run() {
  local label=$1; shift
  if "$@"; then
    N_OK=$((N_OK + 1))
  else
    FAILED+=("$R :: $label")
  fi
}

# Drop run ids with no benchmark tree, recording each as a skip. Answers in the
# global KEPT rather than on stdout: `SKIPPED+=` inside a `$(...)` or a process
# substitution runs in a subshell, so every skip it recorded would be discarded
# and the summary would under-report.
KEPT=()
present() {  # $1 = arm label, rest = run ids
  local arm=$1; shift
  KEPT=()
  for r in "$@"; do
    if [ -d "$BENCH_ROOT/$r" ]; then
      KEPT+=("$r")
    else
      SKIPPED+=("$r :: no $BENCH_ROOT/$r (benchmark not run for this $arm arm)")
    fi
  done
}

present mesh "${MESH[@]}";         MESH=("${KEPT[@]}")
present weighted "${WEIGHTED[@]}"; WEIGHTED=("${KEPT[@]}")

ALL=("${MESH[@]}" "${WEIGHTED[@]}")
if [ "${#ALL[@]}" -eq 0 ]; then
  echo "nothing to do: no named run has a tree under $BENCH_ROOT/"
  printf '  %s\n' "${SKIPPED[@]}"
  exit 1
fi

# ---- per run ----------------------------------------------------------------
# Ordered by dependency, which is not the order the commands are named in:
#
#   classify            needs build-answer-space
#   plot-answer-space   needs build-bipartite, NOT build-answer-space -- one
#                       occupied target cell is exactly one class, so the
#                       bipartite artifacts already carry the answer space and
#                       the VP side with it
#   plot-error-cdf      needs classify
#
# Every cross-dataset figure below reads `classify` output, so this loop must
# finish for every run in an arm before that arm's section runs.
for R in "${ALL[@]}"; do
  printf '\n==================== %s ====================\n' "$R"

  run build-answer-space $V4 build-answer-space --run-id "$R"
  run build-bipartite    $V4 build-bipartite    --run-id "$R"

  # The grid itself: the lattice, the target cells, the VP cells, all four
  # rungs in one 2x2. Answers the question underneath the accuracy figures --
  # what was the method being asked? -- so it is built whether or not any
  # scoring succeeds.
  run plot-answer-space  $V4 plot-answer-space  --run-id "$R"

  # Score every method at every rung, ring-graded. --strict is the default and
  # is left on: exact HEALPix nesting guarantees monotone accuracy across the
  # ladder, so a violation means the grid or the metric is wrong, not the data.
  run classify           $V4 classify           --run-id "$R"

  # How far off, beside where it landed. Rung-free -- error_km is
  # prediction-to-target and identical at every nside -- so it lands in the
  # rung-free parent of the healpix-<n>/ directories.
  run plot-error-cdf     $V4 plot-error-cdf --layout per-run --run-id "$R"
done

# ---- per arm: the cross-dataset figures -------------------------------------
# One pass per arm, never one pass over both. Layouts are passed explicitly
# even where they are the default, because this is a driver and what it writes
# should be readable here rather than inferred from the CLI's defaults.
cross_arm() {
  local arm=$1; shift
  local runs=("$@")
  local args=()
  for r in "${runs[@]}"; do args+=(--run-id "$r"); done

  R="_cross[$arm]"
  printf '\n==================== cross-dataset: %s (%d run(s)) ====================\n' \
    "$arm" "${#runs[@]}"

  # Where every prediction landed, one figure per rung. `compare` gives one
  # panel per dataset, `pooled` a single count-weighted panel over all of them.
  run plot-outcome-bars $V4 plot-outcome-bars \
    --layout pooled --layout compare "${args[@]}"

  # Whether those are the SAME targets -- the question the bars cannot answer.
  # One rung (nside 128), swept over the containment tolerance instead.
  local m=()
  for x in "${EULER_METHODS[@]}"; do m+=(--method "$x"); done
  run plot-euler $V4 plot-euler "${args[@]}" "${m[@]}"

  # The pooled error distribution, beside the pooled bars.
  run plot-error-cdf-pooled $V4 plot-error-cdf --layout pooled "${args[@]}"
}

# Guarded on the arm being non-empty rather than on it having two runs: a
# one-run arm is a legitimate degenerate case (pooling one population is that
# population), and refusing it would mean a single collected weighted dataset
# produced no figures at all.
if [ "${#MESH[@]}" -gt 0 ]; then
  cross_arm mesh "${MESH[@]}"
else
  SKIPPED+=("_cross[mesh] :: no mesh run has a benchmark tree")
fi
if [ "${#WEIGHTED[@]}" -gt 0 ]; then
  cross_arm weighted "${WEIGHTED[@]}"
else
  SKIPPED+=("_cross[weighted] :: no weighted run has a benchmark tree")
fi

printf '\n==================== summary ====================\n'
printf 'runs: %d (mesh %d, weighted %d)   commands ok: %d   failed: %d   skipped: %d\n' \
  "${#ALL[@]}" "${#MESH[@]}" "${#WEIGHTED[@]}" "$N_OK" "${#FAILED[@]}" "${#SKIPPED[@]}"

if [ "${#SKIPPED[@]}" -gt 0 ]; then
  printf '\nskipped:\n'
  printf '  %s\n' "${SKIPPED[@]}"
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
  printf '\nfailed:\n'
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
