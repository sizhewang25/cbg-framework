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

  # 0. Build what every later section reads. These are v3 ANALYSIS commands,
  #    not benchmark stages: `cli.sh` runs inspect_dataset.smk and the
  #    benchmark, which produce eval_source and the fold parquets, but the
  #    answer space is built here. Omitting them made the script unusable on a
  #    clean tree -- 24 of 25 commands failed with
  #    "is not an answer space ... run `build-answer-space` first".
  #
  #    Ordered: the graph and the proximity labels are both keyed on the answer
  #    space's seeds, and proximity additionally needs the shortest-ping VP that
  #    eval_source supplies.
  run build-answer-space    $V3 build-answer-space    --run-id "$R"
  run build-bipartite-graph $V3 build-bipartite-graph --run-id "$R"
  run build-proximity       $V3 build-proximity       --run-id "$R"

  # 1. Score every method against the answer space (~2 s). Needs section 0's
  #    answer space plus eval_source, which cli.sh stage 1 produced.
  run classify $V3 classify --run-id "$R"

  # 2. Tables
  run table-accuracy $V3 table-accuracy --run-id "$R"   # per-method top-N, fallback, error km
  run table-headline $V3 table-headline --run-id "$R"   # paper shape: dataset types x methods

  # 3. Figures
  #
  #    The two dataset-geometry maps come first because they are the other
  #    consumers of section 0's answer space. Every other figure below reads
  #    *derived* artifacts (classification parquets, proximity labels), so
  #    sections 0-1 refresh them transitively; these two read the space itself,
  #    and while they were missing from this script any previously drawn map
  #    stayed stale after a class-set change.
  run plot-answer-space        $V3 plot-answer-space        --run-id "$R"  # occupied cells, targets, seeds
  run plot-bipartite-graph     $V3 plot-bipartite-graph     --run-id "$R"  # topology + flow map, distance CDF

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
  #    Wrapped in an if/else rather than guarded with `continue`: `continue`
  #    would abandon the whole run, so every section added after this one
  #    would silently stop running on any run without a PNI list.
  if grep -qE '^[[:space:]]*pni_csv:[[:space:]]*[^[:space:]#]' "$CFG"; then

    PNI=outputs/analysis/v3/$R/pni-strategy
    GRAPH=outputs/analysis/v3/$R/pni-graph

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

      # 5f. The §7.3 figure: min-RTT against VP->PNI->TG propagation delay, with
      #     the no-PNI control fitted beside it. The visual twin of 5d -- the
      #     config pins `where: is_holdout` to match its `holdout_only: true`,
      #     so both report the same fit.
      #
      #     Drawn once per ASSIGNMENT RULE, not once per run. The x axis IS the
      #     assignment -- changing the rule moves every point and every fit --
      #     so argmin alone shows how well ONE rule's paths explain the RTTs
      #     while leaving the reader no way to see whether that is the rule's
      #     doing or the site list's. All three ship as three views; which rule
      #     is actually in force stays 5a's verdict to declare, since these
      #     figures cannot separate "wrong rule" from "incomplete list" and
      #     detect-pni-strategy scores the rules against each other directly.
      #
      #     argmin reuses 5b's canonical pni-graph/. The other two are built as
      #     SIDE arms into pni-graph-<strategy>/ via --out-dir: 5c, 5d, 5e and
      #     5g all resolve pni-graph/ from the run id, so an alternative rule
      #     written there would silently reparameterize every one of them.
      #     --split-csv is repeated per arm because `where: is_holdout` needs
      #     the column, and it is the SAME split as 5b's -- the halves come from
      #     5a, so the arms stay comparable to the canonical figure.
      #
      #     Alone among these commands plot-pni-delay takes no --run-id: it
      #     reads a CSV directly, so the path is built from $R here rather than
      #     hardcoded in the config. Its input is 5b's output, which is why it
      #     sits last. Each figure is titled with its rule, because the three
      #     PNGs are otherwise identically named under different directories.
      #     The run id is left out of the title: the path already carries it,
      #     and the string is what has to fit across a 6-inch axes.
      if [ -f "$GRAPH/pni_edges.csv" ]; then
        run plot-pni-delay $V3 plot-pni-delay --csv "$GRAPH/pni_edges.csv" \
          --title "min-RTT vs two-leg delay -- strategy argmin"

        for S in tg_nearest vp_nearest; do
          ARM=$GRAPH-$S
          # Gated directly rather than through `run`, which reports a failure
          # but always returns 0: plotting an arm whose build failed would draw
          # a stale figure from a previous invocation's edge list.
          if $V3 build-pni-graph --run-id "$R" --strategy "$S" \
               --split-csv "$PNI/pair_split.csv" --out-dir "$ARM"; then
            N_OK=$((N_OK + 1))
            run "plot-pni-delay[$S]" $V3 plot-pni-delay --csv "$ARM/pni_edges.csv" \
              --title "min-RTT vs two-leg delay -- strategy $S"
          else
            FAILED+=("$R :: build-pni-graph --strategy $S (5f arm)")
          fi
        done

        # 5g. The §8.1 co-location corner, over the same 5b artifact read as a
        #     pair of legs rather than as a fit: d(sping VP, selected PNI) on x
        #     against d(selected PNI, target) on y, one point per target.
        #
        #     Fires only when $R is a TRAFFIC-WEIGHTED run. The command takes
        #     the weighted arm as --run-id and its unweighted parent as
        #     --mesh-run-id, both required, because the figure IS the
        #     subset-against-parent comparison and cannot exist before the
        #     weighted arm does. So the pairing is declared in the weighted
        #     config, as `plot-pni-colocation.mesh_run_id`, beside the two CSVs
        #     the subset was derived from -- and every mesh run simply has no
        #     5g rather than emitting half a comparison.
        #
        #     --run-id is still passed explicitly for section 5's own reason:
        #     $R is the loop's authority on which run is being built.
        #
        #     Existence-checked rather than left to fail: the parent is a
        #     separate run and needs its OWN build-pni-graph, which this loop
        #     only performs when that run is itself in "$@". An unbuilt parent
        #     is expected state, so it is a SKIP.
        #
        #     Both scales ship. They are honest views of the same points, but
        #     the legs span four orders of magnitude (0.6 km to 3,100 km on
        #     as01), so the linear panel -- the one with an origin to put the
        #     corner at -- packs the co-located mode into a few percent of its
        #     height, and only the log panel resolves it.
        MESH=$(sed -nE 's/^[[:space:]]*mesh_run_id:[[:space:]]*([^[:space:]#]+).*/\1/p' "$CFG" | head -1)
        if [ -z "$MESH" ]; then
          echo "skipping 5g: $CFG declares no plot-pni-colocation.mesh_run_id (not a weighted arm, or no parent declared)"
          SKIPPED+=("$R :: 5g plot-pni-colocation, no mesh parent declared")
        elif [ ! -f "outputs/analysis/v3/$MESH/pni-graph/pni_edges.csv" ]; then
          echo "skipping 5g: $CFG names mesh_run_id $MESH, which has no pni-graph yet"
          SKIPPED+=("$R :: 5g plot-pni-colocation, no pni-graph for $MESH")
        else
          run plot-pni-colocation     $V3 plot-pni-colocation --run-id "$R"
          run plot-pni-colocation.log $V3 plot-pni-colocation --run-id "$R" --log-axes
        fi
      else
        echo "skipping 5f/5g: 5b wrote no $GRAPH/pni_edges.csv"
        SKIPPED+=("$R :: 5f plot-pni-delay + 5g plot-pni-colocation, no pni_edges.csv")
      fi
    else
      FAILED+=("$R :: detect-pni-strategy (section 5 abandoned)")
    fi

  else
    echo "skipping section 5: no live pni_csv in $CFG"
    SKIPPED+=("$R :: section 5, no pni_csv")
  fi

  # 6. §8.1 figure 1a: min-RTT vs d(VP, target), the weighted arm stacked on
  #    its own mesh.
  #
  #    Gated on the config declaring `weighted_csv_path` -- i.e. on the run
  #    actually HAVING a weighted arm. The figure's whole content is the
  #    comparison, so a run with one dataset has nothing to draw: it would emit
  #    a single-layer scatter under a name that promises two.
  #
  #    Restricting it here also removes a duplication. `plot-distance-rtt` reads
  #    canonical CSVs, not run outputs, and a published run, its `-mesh` sibling
  #    and its `-weighted` sibling all derive from one CSV per AS -- so running
  #    this per run drew the same picture into three trees (verified
  #    byte-identical between as01-260728-260802 and as01-260728-260802-mesh).
  #
  #    Drawing a mesh layer alone is still supported and is occasionally what
  #    you want -- it is the parent population every weighted claim is relative
  #    to, and the as01/as02/as03 clouds differ materially (r 0.811 / 0.914 on
  #    as01 / as03). That is a cross-dataset question, so ask it deliberately:
  #      $V3 plot-distance-rtt --run-id <run>-mesh --out-dir <dir>
  #    rather than having every run emit its own copy.
  if grep -qE '^[[:space:]]*weighted_csv_path:[[:space:]]*[^[:space:]#]' "$CFG"; then
    # Existence-checked so an uncollected dataset is a SKIP, not a failure.
    # Every *-weighted config points at a `.tbweight` export that has not been
    # collected -- weights are enterprise-sensitive -- which is expected state.
    DECL=$(sed -nE 's/^[[:space:]]*mesh_csv_path:[[:space:]]*([^[:space:]#]+).*/\1/p' "$CFG" | head -1)
    if [ -f "$DECL" ]; then
      run plot-distance-rtt $V3 plot-distance-rtt --run-id "$R" \
        --out-dir "outputs/analysis/v3/$R/figure-distance-rtt"
    else
      echo "skipping section 6: $CFG declares $DECL, which has not been collected"
      SKIPPED+=("$R :: section 6, declared mesh CSV absent ($DECL)")
    fi
  else
    echo "skipping section 6: $CFG declares no weighted_csv_path (no weighted arm to compare)"
    SKIPPED+=("$R :: section 6, no weighted arm")
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
