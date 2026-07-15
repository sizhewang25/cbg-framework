# Dataset precheck: canonical CSV -> eval_source metrics -> inspect_source
# visuals, driven by the same benchmark config yaml used by ./Snakefile
# (only source_kwargs.csv_path and an optional `precheck:` block are read;
# the combo/slice grid is ignored).
#
# Run with:
#   snakemake -s scripts/benchmark/v2/inspect_dataset.smk \
#       --configfile scripts/benchmark/v2/config/as7018_us_test01.yaml -j 1
#
# Outputs land alongside the CSV, named after its stem (tasks/
# 20260714-dataset-precheck-workflow/plan.md's Phase-0 decision — matches
# eval_source.py's and inspect_source.py's own out_dir defaults):
#   <csv_dir>/<stem>_eval_per_target.csv, _eval_clusters.csv, _eval_stats.json
#   <csv_dir>/<stem>_clusters/{clusters,assignments}.csv + meta.json
#   <csv_dir>/<stem>_vp_mesh_km.csv, _cluster_mesh_km.csv  (skipped past mesh_max_n)
#   <csv_dir>/<stem>_occurrence_cdf.png, _stats.json, _flow_map.html
#   <csv_dir>/<stem>_cluster_map.png    (Voronoi underlay only if `precheck.landmass` set)
#
# Config keys (all under an optional top-level `precheck:` block; every key
# defaults to eval_source.py's own DEFAULT_* constants, so an absent block
# reproduces a bare `eval-source --csv ...` call):
#   cluster_radius_km   (default 50)      top_n_neighbors      (default 5)
#   anycast_delta_ms    (default 10)      spearman_min_pairs   (default 8)
#   mesh_max_n          (default 2000)    landmass             (default: none — no Voronoi underlay)

from pathlib import Path

CSV_PATH = Path(config["source_kwargs"]["csv_path"])
PRECHECK = config.get("precheck", {}) or {}

CLUSTER_RADIUS_KM = float(PRECHECK.get("cluster_radius_km", 50.0))
TOP_N_NEIGHBORS = int(PRECHECK.get("top_n_neighbors", 5))
ANYCAST_DELTA_MS = float(PRECHECK.get("anycast_delta_ms", 10.0))
SPEARMAN_MIN_PAIRS = int(PRECHECK.get("spearman_min_pairs", 8))
MESH_MAX_N = int(PRECHECK.get("mesh_max_n", 2000))
LANDMASS = PRECHECK.get("landmass")

STEM = CSV_PATH.stem
OUT_DIR = CSV_PATH.parent

EVAL_CLI = "python -m scripts.benchmark.v2.cli"
INSPECT_CLI = "python -m scripts.visualization.benchmark.inspect_source"

EVAL_STATS = OUT_DIR / f"{STEM}_eval_stats.json"
CLUSTERS_DIR = OUT_DIR / f"{STEM}_clusters"
INSPECT_STATS = OUT_DIR / f"{STEM}_stats.json"
CLUSTER_MAP = OUT_DIR / f"{STEM}_cluster_map.png"


rule all:
    input:
        str(EVAL_STATS),
        str(INSPECT_STATS),
        str(CLUSTER_MAP),


# ---- [1] eval_source: canonical CSV -> per-target/per-cluster metrics -------
rule eval_source:
    input:
        csv = str(CSV_PATH),
    output:
        stats = str(EVAL_STATS),
        clusters_dir = directory(str(CLUSTERS_DIR)),
    params:
        out_dir = str(OUT_DIR),
        radius = CLUSTER_RADIUS_KM,
        top_n = TOP_N_NEIGHBORS,
        anycast_delta = ANYCAST_DELTA_MS,
        spearman_min = SPEARMAN_MIN_PAIRS,
        mesh_max_n = MESH_MAX_N,
    shell:
        EVAL_CLI + " eval-source"
        " --csv {input.csv}"
        " --out-dir {params.out_dir}"
        " --cluster-radius-km {params.radius}"
        " --top-n-neighbors {params.top_n}"
        " --anycast-delta-ms {params.anycast_delta}"
        " --spearman-min-pairs {params.spearman_min}"
        " --mesh-max-n {params.mesh_max_n}"


# ---- [2] inspect_source: CDFs + flow map + cluster/Voronoi map -------------
# The cluster map (output.cluster_map) is always produced once clusters_dir
# exists — `landmass` only adds the Voronoi underlay, it doesn't gate the map.
rule inspect_source:
    input:
        csv = str(CSV_PATH),
        clusters_dir = str(CLUSTERS_DIR),
    output:
        stats = str(INSPECT_STATS),
        cluster_map = str(CLUSTER_MAP),
    params:
        out_dir = str(OUT_DIR),
        landmass_flag = f"--landmass '{LANDMASS}'" if LANDMASS else "",
    shell:
        INSPECT_CLI +
        " --csv {input.csv}"
        " --out-dir {params.out_dir}"
        " {params.landmass_flag}"
