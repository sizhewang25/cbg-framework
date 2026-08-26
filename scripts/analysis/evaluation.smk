# Snakemake workflow for post-benchmark analysis (scoring + metrics + visualization).
#
# Run with:
#   snakemake -s scripts/analysis/evaluation.smk \
#
# The config file must contain:
#   run_id:           benchmark run identifier
#   source:           data source name (e.g. generic_csv)
#   setup:            role assignment (e.g. anchors_to_probes)
#   v2_outputs_root:  path to benchmark outputs root
#   v2_inputs_root:   path to benchmark inputs root
#
# Optional config keys:
#   top_n_values:     list of top-N accuracy levels (default: [1])
#   skip_loo:         skip expensive LOO brittleness metric (default: false)
#   radius_km:        answer-space clustering radius (default: 50)
#
# Outputs:
#   - <v2_outputs_root>/<run_id>/<source>/<setup>/clusters/              (answer space)
#   - <v2_outputs_root>/<run_id>/<source>/<setup>/cluster_scored/        (scored predictions)
#   - <v2_outputs_root>/<run_id>/<source>/<setup>/classification_table_top{N}.parquet  (classification truth tables)
#   - scripts/analysis/outputs/<run_id>/cluster/<run_id>_classification_analysis_top{N}.csv  (per-target analysis)
#   - <v2_outputs_root>/<run_id>/bench_eval/                             (per-target metrics)
#   - visualization PNGs + CSV data files
#
# Pipeline DAG:
#   1. materialize_target_space     (answer space clusters)
#        ↓
#   2. cluster_score (parallel)     (score predictions per top-N)
#        ↓
#   3. eval_bench_results            (per-target metrics)
#        ↓
#   4. build_classification_table (parallel) (truth tables per top-N)
#        ↓
#   5. analyze_classification_table (parallel) (per-target enriched analysis per top-N)
#        ↓
#   6. plot_classification_match_bars (parallel) (visualizations per top-N)

import json
from pathlib import Path

# ---- Config resolution -------------------------------------------------------

RUN_ID = config.get("run_id")
if not RUN_ID:
    raise ValueError("config must have 'run_id' key")

SOURCE = config.get("source")
if not SOURCE:
    raise ValueError("config must have 'source' key")

SETUP = config.get("setup", "anchors_to_probes")
V2_OUTPUTS_ROOT = Path(config.get("v2_outputs_root", "scripts/benchmark/v2/outputs"))
V2_INPUTS_ROOT = Path(config.get("v2_inputs_root", "scripts/benchmark/v2/inputs"))
TOP_N_VALUES = config.get("top_n_values", [1])
SKIP_LOO = config.get("skip_loo", True)

# Normalize top-N values and create an execution chain so cluster-score does not
# write the same scored CSVs concurrently for multiple top-N jobs.
TOP_N_VALUES = sorted({int(n) for n in TOP_N_VALUES})
PREV_TOP_N = {
    TOP_N_VALUES[i]: (TOP_N_VALUES[i - 1] if i > 0 else None)
    for i in range(len(TOP_N_VALUES))
}

# Build paths
RUN_DIR = V2_OUTPUTS_ROOT / RUN_ID
SOURCE_SETUP_DIR = RUN_DIR / SOURCE / SETUP
CLUSTERS_DIR = SOURCE_SETUP_DIR / "clusters"
CLUSTER_SCORED_DIR = SOURCE_SETUP_DIR / "cluster_scored"
BENCH_EVAL_DIR = RUN_DIR / "bench_eval"

# CLI references
BENCHMARK_CLI = ".venv/bin/python -m scripts.benchmark.v2.cli"
ANALYSIS_CLI = ".venv/bin/python -m scripts.analysis.cli"
PLOT_CLI = ".venv/bin/python -m scripts.analysis.classification.plot_classification_match_bars"
CLASSIFICATION_ANALYSIS_CLI = ".venv/bin/python -m scripts.analysis.classification.classification_analysis"


# ---- Targets ----------------------------------------------------------------

rule all:
    input:
        # Answer space
        CLUSTERS_DIR / "meta.json",
        # Scored predictions (for each top-N)
        expand(
            CLUSTER_SCORED_DIR / f"_top{{top_n}}_done.txt",
            top_n=TOP_N_VALUES,
        ),
        # Classification tables (for each top-N)
        expand(
            SOURCE_SETUP_DIR / f"{RUN_ID}_classification_table_top{{top_n}}.parquet",
            top_n=TOP_N_VALUES,
        ),
        # Per-target analysis (for each top-N)
        expand(
            Path("scripts/analysis/outputs") / RUN_ID / "cluster" / f"{RUN_ID}_classification_analysis_top{{top_n}}.csv",
            top_n=TOP_N_VALUES,
        ),
        # Visualization (uses first 1-2 items from top_n_values)
        (Path("scripts/analysis/outputs") / RUN_ID / "cluster" / f"{RUN_ID}_classification_accuracy.png")
        if len(TOP_N_VALUES) == 1 and TOP_N_VALUES[0] == 1
        else (Path("scripts/analysis/outputs") / RUN_ID / "cluster" / f"{RUN_ID}_classification_accuracy_top{TOP_N_VALUES[0]}_{TOP_N_VALUES[min(1, len(TOP_N_VALUES)-1)]}.png"),


# ---- Rule 1: Materialize Answer Space ---------------------------------------

rule materialize_target_space:
    """Build the cluster answer space from evaluated targets."""
    output:
        meta = CLUSTERS_DIR / "meta.json",
        clusters = CLUSTERS_DIR / "clusters.csv",
        assignments = CLUSTERS_DIR / "assignments.csv",
    params:
        run_id = RUN_ID,
        outputs_root = str(V2_OUTPUTS_ROOT),
        inputs_root = str(V2_INPUTS_ROOT),
        radius_km = float(config.get("radius_km", 50.0)),
    shell:
        "{BENCHMARK_CLI} materialize-target-space"
        " --run-id {params.run_id}"
        " --outputs-root {params.outputs_root}"
        " --inputs-root {params.inputs_root}"
        " --radius-km {params.radius_km}"


# ---- Rule 2: Score Predictions (Per Top-N) ---------------------------------

rule cluster_score:
    """Score each combo's predictions against the cluster answer space."""
    input:
        meta = CLUSTERS_DIR / "meta.json",
        prev_done = lambda wildcards: (
            CLUSTER_SCORED_DIR / f"_top{PREV_TOP_N[int(wildcards.top_n)]}_done.txt"
            if PREV_TOP_N[int(wildcards.top_n)] is not None
            else []
        ),
    output:
        done = CLUSTER_SCORED_DIR / "_top{top_n}_done.txt",
    params:
        run_id = RUN_ID,
        source = SOURCE,
        outputs_root = str(V2_OUTPUTS_ROOT),
        inputs_root = str(V2_INPUTS_ROOT),
        clusters_dir = str(CLUSTERS_DIR),
        scored_dir = str(CLUSTER_SCORED_DIR),
        top_n = "{top_n}",
    shell:
        "{ANALYSIS_CLI} cluster-score"
        " --run-id {params.run_id}"
        " --source {params.source}"
        " --clusters-dir {params.clusters_dir}"
        " --out-dir {params.scored_dir}"
        " --outputs-root {params.outputs_root}"
        " --inputs-root {params.inputs_root}"
        " --top-n {params.top_n}"
        " && touch {output.done}"


# ---- Rule 3: Compute Per-Target Metrics ------------------------------------

# rule eval_bench_results:
#     """Compute detailed per-target metrics (MTL, LOO, etc.)."""
#     input:
#         meta = CLUSTERS_DIR / "meta.json",
#     output:
#         summary = BENCH_EVAL_DIR / "summary.parquet",
#     params:
#         run_id = RUN_ID,
#         outputs_root = str(V2_OUTPUTS_ROOT),
#         inputs_root = str(V2_INPUTS_ROOT),
#         skip_loo_flag = "--skip-loo" if SKIP_LOO else "",
#     shell:
#         "{ANALYSIS_CLI} eval-bench-results"
#         " --run-id {params.run_id}"
#         " --outputs-root {params.outputs_root}"
#         " --inputs-root {params.inputs_root}"
#         " {params.skip_loo_flag}"


# ---- Rule 4: Build Classification Table (Per Top-N) -------------------------

rule build_classification_table:
    """Build boolean classification table for top-N accuracy ranking."""
    input:
        scored_done = CLUSTER_SCORED_DIR / "_top{top_n}_done.txt",
    output:
        parquet = SOURCE_SETUP_DIR / f"{RUN_ID}_classification_table_top{{top_n}}.parquet",
        csv = SOURCE_SETUP_DIR / f"{RUN_ID}_classification_table_top{{top_n}}.csv",
    params:
        config_file = str(Path(workflow.configfiles[0])) if workflow.configfiles else "",
        run_dir = str(RUN_DIR),
        outputs_root = str(V2_OUTPUTS_ROOT),
        inputs_root = str(V2_INPUTS_ROOT),
        clusters_dir = str(CLUSTERS_DIR),
        out_dir = str(SOURCE_SETUP_DIR),
        radius_km = float(config.get("radius_km", 50.0)),
        top_n = "{top_n}",
    shell:
        ".venv/bin/python -m scripts.analysis.classification.classification_table"
        " --config {params.config_file}"
        " --run-dir {params.run_dir}"
        " --outputs-root {params.outputs_root}"
        " --inputs-root {params.inputs_root}"
        " --clusters-dir {params.clusters_dir}"
        " --radius-km {params.radius_km}"
        " --top-n {params.top_n}"
        " --out-dir {params.out_dir}"


# ---- Rule 5: Analyze Classification Table (Per Top-N) -----------------------

rule analyze_classification_table:
    """Build per-target classification analysis with geographic enrichment."""
    input:
        table = SOURCE_SETUP_DIR / f"{RUN_ID}_classification_table_top{{top_n}}.parquet",
    output:
        csv = Path("scripts/analysis/outputs") / RUN_ID / "cluster" / f"{RUN_ID}_classification_analysis_top{{top_n}}.csv",
    params:
        config_file = str(Path(workflow.configfiles[0])) if workflow.configfiles else "",
        run_dir = str(RUN_DIR),
        outputs_root = str(V2_OUTPUTS_ROOT),
        inputs_root = str(V2_INPUTS_ROOT),
        clusters_dir = str(CLUSTERS_DIR),
        radius_km = float(config.get("radius_km", 50.0)),
        top_n = "{top_n}",
    shell:
        "{CLASSIFICATION_ANALYSIS_CLI}"
        " --config {params.config_file}"
        " --run-dir {params.run_dir}"
        " --outputs-root {params.outputs_root}"
        " --inputs-root {params.inputs_root}"
        " --clusters-dir {params.clusters_dir}"
        " --radius-km {params.radius_km}"
        " --top-n {params.top_n}"
        " --table {input.table}"
        " --out {output.csv}"


# ---- Rule 6: Visualize Results (Overlay if 2+ top_n_values) ----------------

rule plot_classification_match_bars:
    """Generate classification accuracy bar charts with baseline comparison.
    
    If top_n_values has 1 item: plots solid bar for that top-N.
    If top_n_values has 2+ items: uses first two for overlay (solid + striped).
    """
    input:
        analyses = expand(
            Path("scripts/analysis/outputs") / RUN_ID / "cluster" / f"{RUN_ID}_classification_analysis_top{{top_n}}.csv",
            top_n=TOP_N_VALUES[:min(2, len(TOP_N_VALUES))],
        ),
        tables = expand(
            SOURCE_SETUP_DIR / f"{RUN_ID}_classification_table_top{{top_n}}.parquet",
            top_n=TOP_N_VALUES[:min(2, len(TOP_N_VALUES))],
        ),
    output:
        png = Path("scripts/analysis/outputs") / RUN_ID / "cluster" / (
            f"{RUN_ID}_classification_accuracy.png" 
            if len(TOP_N_VALUES) == 1 and TOP_N_VALUES[0] == 1
            else f"{RUN_ID}_classification_accuracy_top{TOP_N_VALUES[0]}_{TOP_N_VALUES[min(1, len(TOP_N_VALUES)-1)]}.png"
        ),
    params:
        config_file = str(Path(workflow.configfiles[0])) if workflow.configfiles else "",
        clusters_dir = str(CLUSTERS_DIR),
        scored_dir = str(CLUSTER_SCORED_DIR),
    shell:
        "{PLOT_CLI}"
        " --config {params.config_file}"
        " --clusters-dir {params.clusters_dir}"
        " --scored-dir {params.scored_dir}"
