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
