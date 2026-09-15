CFG=configs/as01-260728-260802-mesh.yaml
R=as01-260728-260802-mesh
V3="python -m scripts.analysis.v3.cli --config $CFG"

# 1. Score every method against the answer space (~2 s). Needs the answer space
#    and eval_source — cli.sh stage 1 already produced both.
$V3 classify --run-id $R

# 2. Tables
$V3 table-accuracy  --run-id $R      # per-method top-N, fallback, error km
$V3 table-headline  --run-id $R      # paper shape: dataset types x methods

# 3. Figures
$V3 plot-outcome-bars        --run-id $R    # correct/wrong/fallback/error split
$V3 plot-error-cdf           --run-id $R    # error-distance CDF, log x
$V3 plot-error-vs-cells      --run-id $R    # coord error vs class error
$V3 plot-error-vs-rank       --run-id $R
$V3 plot-pareto              --run-id $R    # accuracy vs runtime/memory
$V3 plot-venn                --run-id $R    # which methods get which targets

# 4. Why the numbers differ, on the targets rather than on any method
$V3 breakdown-accuracy       --run-id $R    # accuracy x proximity strata
$V3 confusion-density        --run-id $R    # what the mistakes actually are
$V3 plot-proximity-inflation --run-id $R

# 5. PNI stack (§7.3 routing, §8.1 shortest-ping mechanism). Ordered: each step
#    reads the previous one's output, so run them in sequence. Needs a real
#    interconnect list for THIS run's peer ASN — the config pins as01's
#    14-site AS20940 list. There is no such list for as02 or as03, so this
#    block does not port to them by changing $R.
PNI=outputs/analysis/v3/$R/pni-strategy

# 5a. Which site-selection policy the peer's min-RTTs order by. Writes
#     pair_split.csv, which every later step is scored on.
$V3 detect-pni-strategy --run-id $R

# 5b. Assign each measured pair the site it most plausibly crossed.
#     --split-csv is passed here rather than in the config because the path
#     carries $R and would otherwise drift when the run id changes.
$V3 build-pni-graph --run-id $R --split-csv $PNI/pair_split.csv

# 5c. Which sites a pair COULD have crossed, by exclusion at 2/3 c. The
#     sping-VP rows are §8.1's feasibility evidence; feasible_null.csv holds
#     the 200-trial permutation baseline.
$V3 build-pni-feasibility --run-id $R

# 5d. Is min-RTT more linear in routing distance than in air distance?
#     Scored on the held-out half only.
$V3 compare-pni-linearity --run-id $R

# 5e. Cross every method's correctness with distance to the nearest PNI.
#     Unlike 5a-5d this one joins classification, so it is fold-dependent.
$V3 breakdown-sping-pni --run-id $R
