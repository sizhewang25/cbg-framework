# Snakemake workflow for cross-dataset analysis.
#
# Orchestrates evaluation of multiple datasets, then generates a cross-dataset
# comparison plot showing target classification categories.
#
# Run with:
#   snakemake -s scripts/analysis/cross_dataset_evaluation.smk \
#       --configfile scripts/analysis/config/cross_datasets/all_datasets.yaml -j 4
#
# The config file must contain:
#   datasets:         list of dataset configs to evaluate (relative paths)
#   labels:           optional friendly labels for each dataset (default: config stems)
#   v2_outputs_root:  path to benchmark outputs root (inherited by each dataset)
#   v2_inputs_root:   path to benchmark inputs root (inherited by each dataset)
#
# Example config:
#   datasets:
#   labels:
#   v2_outputs_root: scripts/benchmark/v2/outputs
#   v2_inputs_root: scripts/benchmark/v2/inputs
#   output_dir: scripts/analysis/outputs/cross_dataset_comparison
#   top_n: 1
#
# Outputs:
#   - Individual evaluation outputs for each dataset
#   - Cross-dataset comparison:
#     - target_categories_comparison.png
#     - target_categories_comparison.csv

from pathlib import Path

# ---- Config resolution -------------------------------------------------------

DATASETS = config.get("datasets", [])
if not DATASETS:
    raise ValueError("config must have 'datasets' key with list of config paths")

LABELS = config.get("labels")
V2_OUTPUTS_ROOT = Path(config.get("v2_outputs_root", "scripts/benchmark/v2/outputs"))
V2_INPUTS_ROOT = Path(config.get("v2_inputs_root", "scripts/benchmark/v2/inputs"))
OUTPUT_DIR = Path(config.get("output_dir", "scripts/analysis/outputs/cross_dataset_comparison"))

# Handle top_n as either single value or list
TOP_N_CONFIG = config.get("top_n", 1)
TOP_N_VALUES = TOP_N_CONFIG if isinstance(TOP_N_CONFIG, list) else [TOP_N_CONFIG]
TOP_N_VALUES = sorted({int(n) for n in TOP_N_VALUES})

# Resolve config paths relative to config directory
CONFIG_DIR = Path(workflow.configfiles[0]).parent if workflow.configfiles else Path("scripts/analysis/config")
DATASET_CONFIGS = [CONFIG_DIR / cfg for cfg in DATASETS]

# Generate dataset identifiers from config stems
DATASET_IDS = [cfg.stem for cfg in DATASET_CONFIGS]

# Verify configs exist
for cfg in DATASET_CONFIGS:
    if not cfg.exists():
        raise ValueError(f"Dataset config not found: {cfg}")


# ---- Targets ----------------------------------------------------------------

rule all:
    input:
        # Cross-dataset comparison plots (one per top-N)
        expand(
            OUTPUT_DIR / "target_categories_comparison_top{top_n}.png",
            top_n=TOP_N_VALUES,
        ),
        expand(
            OUTPUT_DIR / "target_categories_comparison_top{top_n}.csv",
            top_n=TOP_N_VALUES,
        ),
        # Dataset characterization
        OUTPUT_DIR / "dataset_props" / "dataset_characterization_aggregated.csv",
        OUTPUT_DIR / "dataset_props" / "dataset_characterization_table.tex",
        OUTPUT_DIR / "dataset_props" / "min_inflation_boxplots_values.csv",
        OUTPUT_DIR / "dataset_props" / "proximity_stacked_bars_values.csv",
        OUTPUT_DIR / "dataset_props" / "proximity_inflation" / "min_inflation_has_used_proximity.csv",
        # Classification benchmark table
        OUTPUT_DIR / "classification" / "topk_classification_benchmark.csv",
        OUTPUT_DIR / "classification" / "topk_classification_benchmark.tex",
        # Category-filtered topology (one per top-N)
        expand(
            OUTPUT_DIR / "classification" / "top{top_n}_done.txt",
            top_n=TOP_N_VALUES,
        ),


# ---- Rule 1: Run individual dataset evaluations ----------------------------

rule evaluate_dataset:
    """Run evaluation.smk for each dataset."""
    threads: 1
    output:
        done = OUTPUT_DIR / "{dataset_id}" / "evaluation_done.txt",
    params:
        dataset_config = lambda wildcards: str(DATASET_CONFIGS[DATASET_IDS.index(wildcards.dataset_id)]),
        v2_outputs_root = str(V2_OUTPUTS_ROOT),
        v2_inputs_root = str(V2_INPUTS_ROOT),
    shell:
        ".venv/bin/snakemake"
        " -s scripts/analysis/evaluation.smk"
        " --configfile {params.dataset_config}"
        " -j 2 --latency-wait 10"
        " && mkdir -p $(dirname {output.done})"
        " && touch {output.done}"


# ---- Rule 2: Generate cross-dataset comparison plot ------------------------

rule plot_cross_dataset_categories:
    """Generate target category comparison plot across all datasets."""
    threads: 1
    input:
        done = expand(OUTPUT_DIR / "{dataset_id}" / "evaluation_done.txt", dataset_id=DATASET_IDS),
    output:
        png = OUTPUT_DIR / "target_categories_comparison_top{top_n}.png",
        csv = OUTPUT_DIR / "target_categories_comparison_top{top_n}.csv",
    params:
        dataset_configs = [str(cfg) for cfg in DATASET_CONFIGS],
        labels = LABELS if LABELS else DATASET_IDS,
        output_dir = str(OUTPUT_DIR),
        top_n = "{top_n}",
        v2_outputs_root = str(V2_OUTPUTS_ROOT),
    shell:
        ".venv/bin/python -m scripts.analysis.classification.plot_target_categories"
        " --configs {params.dataset_configs:q}"
        " --labels {params.labels:q}"
        " --out-dir {params.output_dir}"
        " --top-n {params.top_n}"


# ---- Rule 3: Characterize datasets (topology + inflation) ------------------

rule characterize_datasets:
    """Compute per-dataset topology metrics and write aggregated CSV + LaTeX table."""
    threads: 1
    input:
        done = expand(OUTPUT_DIR / "{dataset_id}" / "evaluation_done.txt", dataset_id=DATASET_IDS),
    output:
        csv    = OUTPUT_DIR / "dataset_props" / "dataset_characterization_aggregated.csv",
        tex    = OUTPUT_DIR / "dataset_props" / "dataset_characterization_table.tex",
    params:
        cross_config = str(Path(workflow.configfiles[0])) if workflow.configfiles else "",
        out_csv  = str(OUTPUT_DIR / "dataset_props" / "dataset_characterization_aggregated.csv"),
        out_tex  = str(OUTPUT_DIR / "dataset_props" / "dataset_characterization_table.tex"),
        plots_dir = str(OUTPUT_DIR / "dataset_props" / "plots"),
    shell:
        ".venv/bin/python -m scripts.analysis.dataset_props.topology.characterize_datasets"
        " --cross-config {params.cross_config}"
        " --output {params.out_csv}"
        " --latex-output {params.out_tex}"
        " --plots-out-dir {params.plots_dir}"


# ---- Rule 4: Min-inflation boxplots ----------------------------------------

rule plot_min_inflation_boxplots:
    """Plot min-RTT inflation distributions per dataset."""
    threads: 1
    input:
        csv = OUTPUT_DIR / "dataset_props" / "dataset_characterization_aggregated.csv",
    output:
        values = OUTPUT_DIR / "dataset_props" / "min_inflation_boxplots_values.csv",
    params:
        out_dir = str(OUTPUT_DIR / "dataset_props"),
    shell:
        ".venv/bin/python -m scripts.analysis.dataset_props.topology.plot_min_inflation_boxplots"
        " --csv {input.csv}"
        " --out-dir {params.out_dir}"


# ---- Rule 5: Proximity stacked bars ----------------------------------------

rule plot_proximity_stacked_bars:
    """Plot proximity breakdown stacked bars per dataset."""
    threads: 1
    input:
        csv = OUTPUT_DIR / "dataset_props" / "dataset_characterization_aggregated.csv",
    output:
        values = OUTPUT_DIR / "dataset_props" / "proximity_stacked_bars_values.csv",
    params:
        out_dir = str(OUTPUT_DIR / "dataset_props"),
    shell:
        ".venv/bin/python -m scripts.analysis.dataset_props.topology.plot_proximity_stacked_bars"
        " --csv {input.csv}"
        " --out-dir {params.out_dir}"


# ---- Rule 6: Proximity-filtered min-inflation boxplots ---------------------

rule plot_proximity_filtered_min_inflation:
    """Plot min-inflation grouped by VP-proximity category across datasets."""
    threads: 1
    input:
        done = expand(OUTPUT_DIR / "{dataset_id}" / "evaluation_done.txt", dataset_id=DATASET_IDS),
    output:
        csv = OUTPUT_DIR / "dataset_props" / "proximity_inflation" / "min_inflation_has_used_proximity.csv",
    params:
        cross_config = str(Path(workflow.configfiles[0])) if workflow.configfiles else "",
        out_dir = str(OUTPUT_DIR / "dataset_props" / "proximity_inflation"),
    shell:
        ".venv/bin/python -m scripts.analysis.dataset_props.topology.plot_proximity_filtered_min_inflation"
        " --cross-config {params.cross_config}"
        " --out-dir {params.out_dir}"


# ---- Rule 7: Classification benchmark CSV + LaTeX --------------------------

rule build_classification_benchmark:
    """Build top-K classification benchmark CSV and LaTeX table."""
    threads: 1
    input:
        done = expand(OUTPUT_DIR / "{dataset_id}" / "evaluation_done.txt", dataset_id=DATASET_IDS),
    output:
        csv = OUTPUT_DIR / "classification" / "topk_classification_benchmark.csv",
        tex = OUTPUT_DIR / "classification" / "topk_classification_benchmark.tex",
    params:
        cross_config = str(Path(workflow.configfiles[0])) if workflow.configfiles else "",
        out_csv = str(OUTPUT_DIR / "classification" / "topk_classification_benchmark.csv"),
        out_tex = str(OUTPUT_DIR / "classification" / "topk_classification_benchmark.tex"),
    shell:
        ".venv/bin/python -m scripts.analysis.benchmark.build_topk_classification_benchmark_csv"
        " --cross-config {params.cross_config}"
        " --out-csv {params.out_csv}"
        " && .venv/bin/python -m scripts.analysis.benchmark.export_benchmark_table"
        " --cross-config {params.cross_config}"
        " --out-csv {params.out_csv}"
        " --out-tex {params.out_tex}"


# ---- Rule 8: Category-filtered topology plots (per top-N) ------------------

rule plot_category_filtered_topology:
    """Plot min-inflation and proximity filtered by classification category."""
    threads: 1
    input:
        csv = OUTPUT_DIR / "dataset_props" / "dataset_characterization_aggregated.csv",
        done = expand(OUTPUT_DIR / "{dataset_id}" / "evaluation_done.txt", dataset_id=DATASET_IDS),
    output:
        done = OUTPUT_DIR / "classification" / "top{top_n}_done.txt",
    params:
        cross_config = str(Path(workflow.configfiles[0])) if workflow.configfiles else "",
        out_dir = str(OUTPUT_DIR / "classification"),
        top_n = "{top_n}",
    shell:
        ".venv/bin/python -m scripts.analysis.dataset_props.topology.plot_category_filtered_topology"
        " --cross-config {params.cross_config}"
        " --out-dir {params.out_dir}"
        " --top-n {params.top_n}"
        " && touch {output.done}"
