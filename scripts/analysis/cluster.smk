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
#         clusters/geo/<level>/<value>/...    (one per `cluster_geos` entry)
#   [2] plot_cluster_match_bars  → <analysis>/<run_id>/[geo/<l>/<v>/]cluster/<run>_cluster_accuracy.png
#   [3] plot_cluster_cdf         → <analysis>/<run_id>/[geo/<l>/<v>/]cluster/cdf/   (one PNG per combo)
#   [4] plot_targets_vps         → <analysis>/<run_id>/cluster/<run>_targets_vps.png   (targets + VPs map)
#
# Config keys:
#   run_id   (required)   source (required)   setup (default probes_to_anchors)
#   radius_km            (default 50)
#   cluster_geos         (default []) — list of {level: continent|country, value: <name>};
#                        each adds a per-geo answer space + a per-geo bars/cdf set.
#   v2_outputs_root / v2_inputs_root / analysis_root  (defaults match the repo layout)

from pathlib import Path

RUN_ID = config["run_id"]
SOURCE = config["source"]
SETUP  = config.get("setup", "probes_to_anchors")
RADIUS_KM = float(config.get("radius_km", 50))
CLUSTER_GEOS = config.get("cluster_geos", []) or []   # [{level, value}, ...]

V2_OUTPUTS_ROOT = Path(config.get("v2_outputs_root", "scripts/benchmark/v2/outputs"))
V2_INPUTS_ROOT  = Path(config.get("v2_inputs_root",  "scripts/benchmark/v2/inputs"))
ANALYSIS_ROOT   = Path(config.get("analysis_root",   "scripts/analysis/outputs"))

CLI = "python -m scripts"
RUN_DIR = V2_OUTPUTS_ROOT / RUN_ID
SETUP_DIR = RUN_DIR / SOURCE / SETUP                 # parent of all fold dirs
CLUSTERS_DIR = SETUP_DIR / "clusters"                # global answer space
ANALYSIS_CLUSTER = ANALYSIS_ROOT / RUN_ID / "cluster"


def _safe(value: str) -> str:
    return str(value).replace(" ", "_").replace("/", "_")


def _geo_clusters_dir(level: str, value: str) -> Path:
    return CLUSTERS_DIR / "geo" / level / _safe(value)


def _geo_analysis_dir(level: str, value: str) -> Path:
    # Mirrors _v2_io.analysis_out_dir routing: geo segment after the run_id.
    return ANALYSIS_ROOT / RUN_ID / "geo" / level / _safe(value) / "cluster"


# ---- Targets ----------------------------------------------------------------
def _all_targets():
    t = [
        str(ANALYSIS_CLUSTER / f"{RUN_ID}_cluster_accuracy.png"),
        str(ANALYSIS_CLUSTER / "cdf"),
        str(ANALYSIS_CLUSTER / f"{RUN_ID}_targets_vps.png"),
    ]
    for g in CLUSTER_GEOS:
        d = _geo_analysis_dir(g["level"], g["value"])
        t.append(str(d / f"{RUN_ID}_cluster_accuracy.png"))
        t.append(str(d / "cdf"))
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


rule cluster_eval_geo:
    output:
        clusters = CLUSTERS_DIR / "geo" / "{level}" / "{value}" / "clusters.csv",
    params:
        run_id = RUN_ID,
        radius = RADIUS_KM,
        outputs_root = str(V2_OUTPUTS_ROOT),
        inputs_root = str(V2_INPUTS_ROOT),
        # The wildcard `value` is already path-safe; cluster-eval re-applies the
        # same safe-ing, so passing the original value name back is fine here
        # because configured geo values are expected to be path-safe-equal.
        value = lambda w: w.value.replace("_", " "),
    shell:
        CLI + ".benchmark.v2.cli cluster-eval"
        " --run-id {params.run_id}"
        " --outputs-root {params.outputs_root}"
        " --inputs-root {params.inputs_root}"
        " --radius-km {params.radius}"
        " --geo-level {wildcards.level}"
        " --geo-value '{params.value}'"


# ---- [2] accuracy bars ------------------------------------------------------
rule cluster_bars_global:
    input:
        clusters = CLUSTERS_DIR / "clusters.csv",
    output:
        png = ANALYSIS_CLUSTER / f"{RUN_ID}_cluster_accuracy.png",
    params:
        run_dir = str(RUN_DIR), source = SOURCE,
        clusters_dir = str(CLUSTERS_DIR), radius = RADIUS_KM,
    shell:
        CLI + ".analysis.plot_cluster_match_bars"
        " --run-dir {params.run_dir} --source {params.source}"
        " --clusters-dir {params.clusters_dir} --radius-km {params.radius}"


rule cluster_bars_geo:
    input:
        clusters = CLUSTERS_DIR / "geo" / "{level}" / "{value}" / "clusters.csv",
    output:
        png = ANALYSIS_ROOT / RUN_ID / "geo" / "{level}" / "{value}" / "cluster" / f"{RUN_ID}_cluster_accuracy.png",
    params:
        run_dir = str(RUN_DIR), source = SOURCE,
        clusters_dir = str(CLUSTERS_DIR), radius = RADIUS_KM,
        value = lambda w: w.value.replace("_", " "),
    shell:
        CLI + ".analysis.plot_cluster_match_bars"
        " --run-dir {params.run_dir} --source {params.source}"
        " --clusters-dir {params.clusters_dir} --radius-km {params.radius}"
        " --geo-level {wildcards.level} --geo-value '{params.value}'"


# ---- [3] per-combo CDFs (directory of PNGs) ---------------------------------
rule cluster_cdf_global:
    input:
        clusters = CLUSTERS_DIR / "clusters.csv",
    output:
        cdf_dir = directory(ANALYSIS_CLUSTER / "cdf"),
    params:
        run_dir = str(RUN_DIR), source = SOURCE,
        clusters_dir = str(CLUSTERS_DIR), radius = RADIUS_KM,
    shell:
        CLI + ".analysis.plot_cluster_cdf"
        " --run-dir {params.run_dir} --source {params.source}"
        " --clusters-dir {params.clusters_dir} --radius-km {params.radius}"


rule cluster_cdf_geo:
    input:
        clusters = CLUSTERS_DIR / "geo" / "{level}" / "{value}" / "clusters.csv",
    output:
        cdf_dir = directory(ANALYSIS_ROOT / RUN_ID / "geo" / "{level}" / "{value}" / "cluster" / "cdf"),
    params:
        run_dir = str(RUN_DIR), source = SOURCE,
        clusters_dir = str(CLUSTERS_DIR), radius = RADIUS_KM,
        value = lambda w: w.value.replace("_", " "),
    shell:
        CLI + ".analysis.plot_cluster_cdf"
        " --run-dir {params.run_dir} --source {params.source}"
        " --clusters-dir {params.clusters_dir} --radius-km {params.radius}"
        " --geo-level {wildcards.level} --geo-value '{params.value}'"


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
