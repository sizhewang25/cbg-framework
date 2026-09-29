#!/usr/bin/env bash
#
# Build the v5 analysis artifacts for a set of runs.
#
#   ./create_analysis_artifacts.sh                          # the default group
#   ./create_analysis_artifacts.sh pro-as01-mesh pro-as02-mesh
#   ./create_analysis_artifacts.sh --group R1 R2 --group R3 R4
#
# Runs are handled in GROUPS. Every cross-dataset figure pools its inputs into
# one population, so a group is a set of runs it is meaningful to pool: three
# meshes of three different ASNs, yes; a mesh and its own traffic-weighted
# subset, no -- the weighted arm is a subset of the same targets, so pooling
# them would count a site twice and compare a dataset against itself.
#
# v4's script split the same way and called the groups `--mesh`/`--weighted`,
# because its output directory was `<datasets>@<arm>` and the arm was the only
# thing keeping the two from overwriting each other. v5 does not need that:
# `cross.cross_dir` writes `_cross/<kind>/<n>-runs-<hash>/` over the run set
# itself, so two groups cannot collide whatever they are called. The grouping
# that survives is the statistical one, which is why the flag is now just
# `--group`.
#
# Deliberately NOT `set -e`. Per-run commands are largely independent and a run
# with a half-finished benchmark will fail several of them; aborting on the
# first failure would hide the state of every later run. Failures are collected
# and printed as a summary instead, and the script exits non-zero if any
# occurred.
#
# Every command is addressed by `--run-id` against the benchmark output tree. A
# run id with no tree under outputs/benchmark/v2/ is a SKIP, not a failure.

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO" || exit 1
export PATH="$REPO/.venv/bin:$PATH"     # the CLI is invoked as bare `python`

V5="python -m scripts.analysis.v5.cli"
BENCH_ROOT=outputs/benchmark/v2

# The current finals. Named here rather than globbed: a glob would silently
# pick up a half-finished run the next time one is started.
DEFAULT_GROUP=(
  pro-as01-mesh
  pro-as02-mesh
  pro-as03-mesh
)

# No command here restricts its methods. Each discovers what to score or draw
# from the benchmark output tree -- `combo_ids` globs
# `fold_*/<combo>/targets.parquet` -- which is the single place to decide it.
# An allow-list here would have to be edited every time a combo is added, and
# until it was, the new combo would be silently unscored.

# `RUN_GROUPS`, not `GROUPS`: bash owns `GROUPS` as the caller's unix group
# ids, and assigning to it is silently ignored. Every group read back as a
# numeric gid until this was renamed.
RUN_GROUPS=()    # each entry is one group's run ids, newline-separated
current=""

# Close the group being accumulated, if it has anything in it.
flush() {
  if [ -n "$current" ]; then
    RUN_GROUPS+=("$current")
    current=""
  fi
}

if [ "$#" -eq 0 ]; then
  printf -v current '%s\n' "${DEFAULT_GROUP[@]}"; flush
else
  for a in "$@"; do
    case "$a" in
      --group)
        flush ;;
      -*)
        echo "create_analysis_artifacts.sh: unknown argument: $a" >&2
        echo "usage: $0 [RUN...] [--group RUN...]..." >&2
        exit 2 ;;
      *)
        current+="$a"$'\n' ;;
    esac
  done
  flush
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
# global KEPT rather than on stdout: `SKIPPED+=` inside a `$(...)` runs in a
# subshell, so every skip it recorded would be discarded and the summary would
# under-report.
KEPT=()
present() {
  KEPT=()
  for r in "$@"; do
    if [ -d "$BENCH_ROOT/$r" ]; then
      KEPT+=("$r")
    else
      SKIPPED+=("$r :: no $BENCH_ROOT/$r (benchmark not run)")
    fi
  done
}

ALL=()
KEPT_GROUPS=()
for g in "${RUN_GROUPS[@]}"; do
  # shellcheck disable=SC2206
  ids=($g)
  present "${ids[@]}"
  if [ "${#KEPT[@]}" -gt 0 ]; then
    printf -v joined '%s\n' "${KEPT[@]}"
    KEPT_GROUPS+=("$joined")
    ALL+=("${KEPT[@]}")
  fi
done

if [ "${#ALL[@]}" -eq 0 ]; then
  echo "nothing to do: no named run has a tree under $BENCH_ROOT/"
  [ "${#SKIPPED[@]}" -gt 0 ] && printf '  %s\n' "${SKIPPED[@]}"
  exit 1
fi

# ---- per run ----------------------------------------------------------------
# Ordered by dependency:
#
#   build-answer-space   first; everything else reads it
#   plot-answer-space    needs build-answer-space
#   classify             needs build-answer-space
#   plot-error-cdf       needs classify
#
# Every cross-dataset figure below reads `classify` output, so this loop must
# finish for every run in a group before that group's section runs.
for R in "${ALL[@]}"; do
  printf '\n==================== %s ====================\n' "$R"

  # The answer space: HEALPix grids for the target partition, and the seeds of
  # the unbounded Voronoi cell partition beside them.
  run build-answer-space $V5 build-answer-space --run-id "$R"

  # What the method was being asked. The grid lattice, the occupied target
  # grids, and the cell boundaries in one panel -- built whether or not any
  # scoring succeeds, because it is the question underneath every accuracy
  # figure. `--us-only` is the default and is passed anyway: this is a driver,
  # and what it writes should be readable here rather than inferred from the
  # CLI's defaults.
  run plot-answer-space  $V5 plot-answer-space  --run-id "$R" --us-only

  # Both labels per prediction: the uncapped grid offset and the nearest-seed
  # cell verdict.
  run classify           $V5 classify           --run-id "$R"

  # How far off, per dataset. Two variants, and they are not redundant:
  #
  #   exclude  (the default) drops the rows a method did not answer, so each
  #            curve rests on its own population. The only one that joins to
  #            accuracy.csv.
  #   sentinel parks them at 10,000 km so every curve is drawn over the same
  #            denominator and the height at the sentinel reads as the
  #            method's refusal rate -- which is the whole story for Vanilla.
  #
  # They write different filenames, so neither overwrites the other.
  run plot-error-cdf     $V5 plot-error-cdf --layout per-run --run-id "$R"
  run plot-error-cdf[sentinel] \
    $V5 plot-error-cdf --layout per-run --run-id "$R" --unanswered sentinel
done

# ---- per group: the cross-dataset figures -----------------------------------
# Layouts and modes are passed explicitly even where they are the default, for
# the reason given above: a driver should read as what it writes.
cross_group() {
  local n=$1; shift
  local args=()
  for r in "$@"; do args+=(--run-id "$r"); done

  R="_cross[group $n]"
  printf '\n==================== cross-dataset: group %s (%d run(s)) ====================\n' \
    "$n" "$#"

  # Where every prediction landed. Two layouts and two modes, so four figures:
  #
  #   compare/pooled     one panel per dataset, or every run's TGs as one
  #                      population.
  #   bounded/unbounded  each cell label broken down by ring tier, or the cell
  #                      axis alone. The pair is the section's argument: the
  #                      unbounded bars are what a nearest-seed verdict alone
  #                      credits, and the bounded ones are the same
  #                      predictions with how far out they landed visible.
  run plot-outcome-bars $V5 plot-outcome-bars \
    --layout pooled --layout compare --mode bounded --mode unbounded "${args[@]}"

  # The pooled error distribution, beside the pooled bars. Both unanswered
  # policies again, for the same reason as the per-run pass.
  run plot-error-cdf-pooled $V5 plot-error-cdf --layout pooled "${args[@]}"
  run plot-error-cdf-pooled[sentinel] \
    $V5 plot-error-cdf --layout pooled --unanswered sentinel "${args[@]}"

  # Where those predictions actually landed, on a map: one panel per
  # (method, dataset), the cells drawn under them. Both cohorts -- `correct`
  # and `wrong` -- are the default and are what the bars cannot show, which is
  # that two methods can hold the same cell share while being right about
  # different halves of the country.
  run plot-outcome-map $V5 plot-outcome-map "${args[@]}"
}

i=0
for g in "${KEPT_GROUPS[@]}"; do
  i=$((i + 1))
  # shellcheck disable=SC2206
  ids=($g)
  cross_group "$i" "${ids[@]}"
done

printf '\n==================== summary ====================\n'
printf 'runs: %d in %d group(s)   commands ok: %d   failed: %d   skipped: %d\n' \
  "${#ALL[@]}" "${#KEPT_GROUPS[@]}" "$N_OK" "${#FAILED[@]}" "${#SKIPPED[@]}"

if [ "${#SKIPPED[@]}" -gt 0 ]; then
  printf '\nskipped:\n'
  printf '  %s\n' "${SKIPPED[@]}"
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
  printf '\nfailed:\n'
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
