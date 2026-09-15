#!/usr/bin/env bash
#
# Build the v3 analysis artifacts for one or more runs.
#
#   ./create_analysis_artifacts.sh                           # the default set
#   ./create_analysis_artifacts.sh as01-260728-260802-mesh   # named runs
#   ./create_analysis_artifacts.sh as0{1,2,3}-260728-260802  # brace expansion
#
# One config per run, named after the run id (configs/<run_id>.yaml). A run
# whose config is missing is skipped rather than guessed at.
#
# Deliberately NOT `set -e`. These commands are largely independent and a run
# with a half-finished benchmark will fail several of them; aborting the whole
# sweep on the first failure would hide the state of every later run. Failures
# are collected and printed as a summary instead. The one ordered chain --
# section 5 -- guards itself internally.

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

for R in "${RUNS[@]}"; do
  CFG=configs/$R.yaml
  printf '\n==================== %s ====================\n' "$R"

  if [ ! -f "$CFG" ]; then
    echo "skipping: no $CFG"
    SKIPPED+=("$R :: no config")
    continue
  fi

  V3="python -m scripts.analysis.v3.cli --config $CFG"

  # 1. Score every method against the answer space (~2 s). Needs the answer
  #    space and eval_source — cli.sh stage 1 already produced both.
  run classify $V3 classify --run-id "$R"

  # 2. Tables
  run table-accuracy $V3 table-accuracy --run-id "$R"   # per-method top-N, fallback, error km
  run table-headline $V3 table-headline --run-id "$R"   # paper shape: dataset types x methods

  # 3. Figures
  run plot-outcome-bars        $V3 plot-outcome-bars        --run-id "$R"  # correct/wrong/fallback/error
  run plot-error-cdf           $V3 plot-error-cdf           --run-id "$R"  # error-distance CDF, log x
  run plot-error-vs-cells      $V3 plot-error-vs-cells      --run-id "$R"  # coord error vs class error
  run plot-error-vs-rank       $V3 plot-error-vs-rank       --run-id "$R"
  run plot-pareto              $V3 plot-pareto              --run-id "$R"  # accuracy vs runtime/memory
  run plot-venn                $V3 plot-venn                --run-id "$R"  # who gets which targets

  # 4. Why the numbers differ, on the targets rather than on any method
  run breakdown-accuracy       $V3 breakdown-accuracy       --run-id "$R"  # accuracy x proximity strata
  run confusion-density        $V3 confusion-density        --run-id "$R"  # what the mistakes are
  run plot-proximity-inflation $V3 plot-proximity-inflation --run-id "$R"

  # 5. PNI stack (§7.3 routing, §8.1 shortest-ping mechanism).
  #
  #    Needs a real interconnect list for THIS run's peer ASN. Only as01 has
  #    one; as02/as03 carry commented placeholders in their configs until the
  #    lists are read off the operator dashboard. Guard on the config holding a
  #    live `pni_csv:` rather than on the run id, because that is the condition
  #    that actually decides whether the stack can run.
  #
  #    Do NOT work around a skip by pointing a run at another peer's list:
  #    `peer_asn` labels the peering, it does not verify it, so that SUCCEEDS
  #    and yields a plausible-looking wrong answer.
  if ! grep -qE '^[[:space:]]*pni_csv:[[:space:]]*[^[:space:]#]' "$CFG"; then
    echo "skipping section 5: no live pni_csv in $CFG"
    SKIPPED+=("$R :: section 5, no pni_csv")
    continue
  fi

  PNI=outputs/analysis/v3/$R/pni-strategy

  # 5a. Which site-selection policy the peer's min-RTTs order by. Writes
  #     pair_split.csv, which every later step is scored on. The rest of the
  #     section is gated on this: 5b-5e would otherwise run against a stale
  #     pair_split.csv from an earlier invocation and silently score the wrong
  #     half.
  if $V3 detect-pni-strategy --run-id "$R"; then
    N_OK=$((N_OK + 1))

    # 5b. Assign each measured pair the site it most plausibly crossed.
    #     --split-csv is passed here rather than in the config because the
    #     path carries $R and would otherwise drift when the run id changes.
    run build-pni-graph $V3 build-pni-graph --run-id "$R" --split-csv "$PNI/pair_split.csv"

    # 5c. Which sites a pair COULD have crossed, by exclusion at 2/3 c. The
    #     sping-VP rows are §8.1's feasibility evidence; feasible_null.csv
    #     holds the 200-trial permutation baseline.
    run build-pni-feasibility $V3 build-pni-feasibility --run-id "$R"

    # 5d. Is min-RTT more linear in routing distance than in air distance?
    #     Scored on the held-out half only.
    run compare-pni-linearity $V3 compare-pni-linearity --run-id "$R"

    # 5e. Cross every method's correctness with distance to the nearest PNI.
    #     Unlike 5a-5d this one joins classification, so it is fold-dependent.
    run breakdown-sping-pni $V3 breakdown-sping-pni --run-id "$R"
  else
    FAILED+=("$R :: detect-pni-strategy (section 5 abandoned)")
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
