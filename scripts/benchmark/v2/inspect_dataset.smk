# Dataset precheck: canonical CSV -> eval_source metrics -> inspect_source
# visuals, driven by the same benchmark config yaml used by ./Snakefile
# (only source_kwargs, and an optional `precheck:` block, are read; the
# combo/slice grid is ignored).
#
# Run with:
#   snakemake -s scripts/benchmark/v2/inspect_dataset.smk \
#       --configfile scripts/benchmark/v2/config/as7018_us_test01.yaml -j 1
#
# Source-agnostic CSV resolution: `generic_csv` supplies
# `source_kwargs.csv_path` directly; `generic_presplit` instead supplies
# `source_kwargs.test_path` + `train_path` — this precheck scores `test_path`
# only. Train rows never reach a benchmark eval target (LTD-leakage guard in
# generic_presplit.py), so they carry no eval-side topology/RTT-quality
# signal for this precheck to report; train_path is otherwise ignored here.
#
# Outputs land alongside the resolved CSV, named after its stem (tasks/
# 20260714-dataset-precheck-workflow/plan.md's Phase-0 decision — matches
# eval_source.py's and inspect_source.py's own out_dir defaults):
#   <csv_dir>/<stem>_eval_per_target.csv, _eval_clusters.csv, _eval_stats.json
#   <csv_dir>/<stem>_clusters/{clusters,assignments}.csv + meta.json
#   <csv_dir>/<stem>_vp_mesh_km.csv, _cluster_mesh_km.csv  (uncapped — a huge
#     endpoint count fails loudly (MemoryError) rather than silently shrinking)
#   <csv_dir>/<stem>_occurrence_cdf.png, _stats.json, _flow_map.html
#   <csv_dir>/<stem>_cluster_map.png    (Voronoi underlay only if `precheck.landmass` set)
#
# Eval-side filters — mirrors the *actual* eval set a materialize-inputs run
# would produce, so the precheck never silently scores a wider set than the
# benchmark evaluates (see eval_source.py's `apply_eval_target_filters`):
#   source_kwargs.min_obs           (same key the benchmark run already reads)
#   top-level eval_pair_weight_min / eval_kept_traffic_fraction
#     (same top-level yaml keys ./Snakefile's `materialize` rule reads —
#     see its `eval_pair_weight_min_flag` / `eval_kept_traffic_fraction_flag`)
#
# Config keys (all under an optional top-level `precheck:` block; every key
# defaults to eval_source.py's own DEFAULT_* constants, so an absent block
# reproduces a bare `eval-source --csv ...` call):
#   cluster_radius_km   (default 50)      top_n_neighbors      (default 5)
#   anycast_delta_ms    (default 10)      spearman_min_pairs   (default 8)
#   landmass            (default: none — no Voronoi underlay)

from pathlib import Path

SRC_KWARGS = config.get("source_kwargs", {}) or {}
if "csv_path" in SRC_KWARGS:
    CSV_PATH = Path(SRC_KWARGS["csv_path"])
elif "test_path" in SRC_KWARGS:
    CSV_PATH = Path(SRC_KWARGS["test_path"])
else:
    raise ValueError(
        "inspect_dataset.smk needs source_kwargs.csv_path (generic_csv) or "
        "source_kwargs.test_path (generic_presplit) in the config yaml"
    )

PRECHECK = config.get("precheck", {}) or {}

CLUSTER_RADIUS_KM = float(PRECHECK.get("cluster_radius_km", 50.0))
TOP_N_NEIGHBORS = int(PRECHECK.get("top_n_neighbors", 5))
ANYCAST_DELTA_MS = float(PRECHECK.get("anycast_delta_ms", 10.0))
SPEARMAN_MIN_PAIRS = int(PRECHECK.get("spearman_min_pairs", 8))
LANDMASS = PRECHECK.get("landmass")

# Eval-side filters, read from the same keys the real benchmark run uses
# (not from `precheck:`) so the precheck can never drift from what
# materialize-inputs would actually evaluate.
MIN_OBS = SRC_KWARGS.get("min_obs")
EVAL_PAIR_WEIGHT_MIN = config.get("eval_pair_weight_min")
EVAL_KEPT_TRAFFIC_FRACTION = config.get("eval_kept_traffic_fraction")
if EVAL_PAIR_WEIGHT_MIN is not None and EVAL_KEPT_TRAFFIC_FRACTION is not None:
    raise ValueError(
        "config has both eval_pair_weight_min and eval_kept_traffic_fraction "
        "— the benchmark run itself only accepts one, so the precheck can't "
        "pick a side"
    )

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
        min_obs_flag = f"--min-obs {MIN_OBS}" if MIN_OBS is not None else "",
        eval_pair_weight_min_flag = (
            f"--eval-pair-weight-min {EVAL_PAIR_WEIGHT_MIN}"
            if EVAL_PAIR_WEIGHT_MIN is not None else ""
        ),
        eval_kept_traffic_fraction_flag = (
            f"--eval-kept-traffic-fraction {EVAL_KEPT_TRAFFIC_FRACTION}"
            if EVAL_KEPT_TRAFFIC_FRACTION is not None else ""
        ),
    shell:
        EVAL_CLI + " eval-source"
        " --csv {input.csv}"
        " --out-dir {params.out_dir}"
        " --cluster-radius-km {params.radius}"
        " --top-n-neighbors {params.top_n}"
        " --anycast-delta-ms {params.anycast_delta}"
        " --spearman-min-pairs {params.spearman_min}"
        " {params.min_obs_flag}"
        " {params.eval_pair_weight_min_flag}"
        " {params.eval_kept_traffic_fraction_flag}"


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
