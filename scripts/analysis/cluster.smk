# Snakemake workflow for the cluster answer-space CBG classification eval.
#
# Run with:
#   snakemake -s scripts/analysis/cluster.smk \
#       --configfile scripts/analysis/config/<run>.yaml -j 2
#
# Materializes the cluster answer space ONCE per (source, setup) of a run, then
# plots against it — so the plots reuse one persisted clustering instead of
# re-clustering internally. Stages:
#
#   [1] cluster-eval (CLI) → <v2_outputs>/<run_id>/<source>/<setup>/
#         targets.csv, vps.csv               (merged unique entities across folds)
#         clusters/{clusters,assignments}.csv + meta.json   (global answer space)
#   [2] plot_cluster_match_bars  → <analysis>/<run_id>/cluster/<run>_cluster_accuracy.png
#   [3] plot_cluster_cdf         → <analysis>/<run_id>/cluster/cdf/   (one PNG per combo)
#   [4] plot_targets_vps         → <analysis>/<run_id>/cluster/<run>_targets_vps.png   (targets + VPs map)
#   [5] plot_ground_truth_clusters → <analysis>/<run_id>/cluster/<run>_ground_truth_clusters[_<geo>].png
#
# Config keys:
#   run_id   (required)   source (required)   setup (default probes_to_anchors)
#   radius_km            (default 50)
#   voronoi_params         (default []) — list of {level: continent|country, value: <name>};
#                        each adds a geo-filtered answer space + ground-truth cluster map
#                        (with Voronoi overlay clipped to that landmass).
#   v2_outputs_root / v2_inputs_root / analysis_root  (defaults match the repo layout)

from pathlib import Path

RUN_ID = config["run_id"]
SOURCE = config["source"]
SETUP  = config.get("setup", "probes_to_anchors")
RADIUS_KM = float(config.get("radius_km", 50))
VORONOI_PARAMS = config.get("voronoi_params", []) or []   # [{level, value}, ...]

V2_OUTPUTS_ROOT = Path(config.get("v2_outputs_root", "scripts/benchmark/v2/outputs"))
V2_INPUTS_ROOT  = Path(config.get("v2_inputs_root",  "scripts/benchmark/v2/inputs"))
ANALYSIS_ROOT   = Path(config.get("analysis_root",   "scripts/analysis/outputs"))

CLI = "python -m scripts"
RUN_DIR = V2_OUTPUTS_ROOT / RUN_ID
SETUP_DIR = RUN_DIR / SOURCE / SETUP                 # parent of all fold dirs
CLUSTERS_DIR = SETUP_DIR / "clusters"                # global answer space
ANALYSIS_CLUSTER = ANALYSIS_ROOT / RUN_ID / "cluster"
SCORED_DIR = SETUP_DIR / "cluster_scored"

# Materialized fold inputs — source-agnostic stratification.
# Auto-discover fold IDs from eval_observations.parquet files so no extra
# config key is needed; the list is empty when folds haven't been materialized.
_FOLDS_DIR = V2_INPUTS_ROOT / SOURCE / RUN_ID / SETUP
FOLD_IDS = sorted(
    p.parent.name
    for p in _FOLDS_DIR.glob("fold_*/eval_observations.parquet")
)


def _safe(value: str) -> str:
    return str(value).replace(" ", "_").replace("/", "_")


# ---- Targets ----------------------------------------------------------------
def _all_targets():
    t = [
        str(ANALYSIS_CLUSTER / f"{RUN_ID}_cluster_accuracy.png"),
        str(ANALYSIS_CLUSTER / "cdf"),
        str(ANALYSIS_CLUSTER / f"{RUN_ID}_targets_vps.png"),
        str(ANALYSIS_CLUSTER / f"{RUN_ID}_ground_truth_clusters.png"),
    ]
    for g in VORONOI_PARAMS:
        t.append(str(ANALYSIS_CLUSTER / f"{RUN_ID}_ground_truth_clusters_{_safe(g['value'])}.png"))
    if FOLD_IDS:
        t.append(str(ANALYSIS_CLUSTER / f"{RUN_ID}_stratification.png"))
    return t


rule all:
    input:
        _all_targets()


# ---- [1] cluster-eval -------------------------------------------------------
rule cluster_eval_global:
    output:
        clusters = CLUSTERS_DIR / "clusters.csv",
    params:
        run_id = RUN_ID,
        radius = RADIUS_KM,
        outputs_root = str(V2_OUTPUTS_ROOT),
        inputs_root = str(V2_INPUTS_ROOT),
    shell:
        CLI + ".benchmark.v2.cli cluster-eval"
        " --run-id {params.run_id}"
        " --outputs-root {params.outputs_root}"
        " --inputs-root {params.inputs_root}"
        " --radius-km {params.radius}"


# ---- [1b] cluster-score (pre-score predictions; reused by all plot rules) ---
rule cluster_score_global:
    input:
        clusters = CLUSTERS_DIR / "clusters.csv",
    output:
        scored_dir = directory(SCORED_DIR),
    params:
        run_id = RUN_ID,
        source = SOURCE,
        outputs_root = str(V2_OUTPUTS_ROOT),
        inputs_root = str(V2_INPUTS_ROOT),
        clusters_dir = str(CLUSTERS_DIR),
        out_dir = str(SCORED_DIR),
    shell:
        CLI + ".benchmark.v2.cli cluster-score"
        " --run-id {params.run_id}"
        " --source {params.source}"
        " --outputs-root {params.outputs_root}"
        " --inputs-root {params.inputs_root}"
        " --clusters-dir {params.clusters_dir}"
        " --out-dir {params.out_dir}"


# ---- [2] accuracy bars ------------------------------------------------------
rule cluster_bars_global:
    input:
        scored_dir = SCORED_DIR,
    output:
        png = ANALYSIS_CLUSTER / f"{RUN_ID}_cluster_accuracy.png",
    params:
        run_dir = str(RUN_DIR), source = SOURCE,
        clusters_dir = str(CLUSTERS_DIR), radius = RADIUS_KM,
        inputs_root = str(V2_INPUTS_ROOT),
        scored_dir = str(SCORED_DIR),
    shell:
        CLI + ".analysis.plot_cluster_match_bars"
        " --run-dir {params.run_dir} --source {params.source}"
        " --clusters-dir {params.clusters_dir} --radius-km {params.radius}"
        " --inputs-root {params.inputs_root}"
        " --scored-dir {params.scored_dir}"


# ---- [3] per-combo CDFs (directory of PNGs) ---------------------------------
rule cluster_cdf_global:
    input:
        scored_dir = SCORED_DIR,
    output:
        cdf_dir = directory(ANALYSIS_CLUSTER / "cdf"),
    params:
        run_dir = str(RUN_DIR), source = SOURCE,
        clusters_dir = str(CLUSTERS_DIR), radius = RADIUS_KM,
        inputs_root = str(V2_INPUTS_ROOT),
        scored_dir = str(SCORED_DIR),
    shell:
        CLI + ".analysis.plot_cluster_cdf"
        " --run-dir {params.run_dir} --source {params.source}"
        " --clusters-dir {params.clusters_dir} --radius-km {params.radius}"
        " --inputs-root {params.inputs_root}"
        " --scored-dir {params.scored_dir}"


# ---- [4] targets + VPs map --------------------------------------------------
# Reads the merged targets.csv / vps.csv that cluster_eval_global writes
# alongside clusters.csv (side outputs at SETUP_DIR); we trigger off the
# guaranteed clusters.csv and reference the two CSVs via params.
rule plot_targets_vps:
    input:
        clusters = CLUSTERS_DIR / "clusters.csv",
    output:
        png = ANALYSIS_CLUSTER / f"{RUN_ID}_targets_vps.png",
    params:
        targets = str(SETUP_DIR / "targets.csv"),
        vps = str(SETUP_DIR / "vps.csv"),
    shell:
        CLI + ".visualization.plot_targets_vps"
        " --targets {params.targets} --vps {params.vps}"
        " --out {output.png}"


# ---- [5] ground-truth cluster map -------------------------------------------
rule plot_ground_truth_clusters_global:
    input:
        clusters = CLUSTERS_DIR / "clusters.csv",
    output:
        png = ANALYSIS_CLUSTER / f"{RUN_ID}_ground_truth_clusters.png",
    params:
        clusters_dir = str(CLUSTERS_DIR),
    shell:
        CLI + ".visualization.cluster.plot_ground_truth_clusters"
        " --clusters-dir {params.clusters_dir}"
        " --out {output.png}"


rule plot_ground_truth_clusters_geo:
    input:
        clusters = CLUSTERS_DIR / "clusters.csv",
    output:
        png = ANALYSIS_CLUSTER / f"{RUN_ID}_ground_truth_clusters_{{value}}.png",
    params:
        clusters_dir = str(CLUSTERS_DIR),
        value = lambda w: w.value.replace("_", " "),
    shell:
        CLI + ".visualization.cluster.plot_ground_truth_clusters"
        " --clusters-dir {params.clusters_dir}"
        " --landmass '{params.value}'"
        " --out {output.png}"


# ---- [6] stratification diagnostic ------------------------------------------
# Reconstructed from materialized fold inputs — works for any source.
# Guarded so configs whose folds haven't been materialized yet are silently
# skipped (FOLD_IDS is empty when the fold dirs don't exist at parse time).
if FOLD_IDS:
    rule plot_stratification:
        input:
            evals = expand(
                str(_FOLDS_DIR / "{fold}" / "eval_observations.parquet"),
                fold=FOLD_IDS,
            ),
            tg_configs = str(_FOLDS_DIR / FOLD_IDS[0] / "tg_configs.parquet"),
        output:
            png = ANALYSIS_CLUSTER / f"{RUN_ID}_stratification.png",
        params:
            inputs_dir = str(_FOLDS_DIR),
        shell:
            CLI + ".analysis.plot_stratification"
            " --inputs-dir {params.inputs_dir}"
            " --out {output.png}"
