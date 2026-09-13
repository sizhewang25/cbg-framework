"""Derive a traffic-weighted CBG dataset by pruning light (VP, target) flows.

Single stage:
  1. filter_weighted_flows – keep the smallest set of (vp_id, target_id) flows
     whose weights sum to >= KEPT_TRAFFIC_FRACTION of total mesh traffic. VPs
     and targets that lose all their flows disappear from the output; that node
     loss is reported in the summary JSON.

The output CSV is for dataset characterisation (paper §8.1 figures), NOT for the
benchmark. The benchmark consumes the WEIGHT-BEARING MESH plus a top-level
`eval_kept_traffic_fraction:` yaml key, and derives the identical threshold
itself -- see GenericCSVSource._derive_eval_weight_min_from_fraction. The
`eval_pair_weight_min` reported here is the cross-check that the two agree.

Usage (from repo root):
  snakemake -s scripts/processing/source/derive_traffic_weighted_cbg_data.smk \
      -j 1 --config mesh=datasets/final/<stem>.csv

Override defaults via --config:
  mesh                  : REQUIRED weighted mesh CSV (must carry a weight column)
  out_dir               : final output directory  (default: datasets/final)
  outputs_dir           : audit directory root    (default: scripts/processing/source/outputs)
  kept_traffic_fraction : cumulative traffic target in (0, 1] (default: 0.95)
  vp_col / target_col / weight_col : column overrides (default: vp_id/target_id/weight)
"""

from pathlib import Path

# ── configurable inputs ───────────────────────────────────────────────────────
MESH         = Path(config["mesh"])   # required: weight-bearing mesh CSV
_stem        = MESH.stem              # dataset name; drives every derived path
OUT_DIR      = Path(config.get("out_dir",     "datasets/final"))
OUTPUTS_ROOT = Path(config.get("outputs_dir", "scripts/processing/source/outputs"))
KEPT_FRAC    = float(config.get("kept_traffic_fraction", 0.95))
VP_COL       = config.get("vp_col",     "vp_id")
TARGET_COL   = config.get("target_col", "target_id")
WEIGHT_COL   = config.get("weight_col", "weight")

# ── derive intermediate / final paths ────────────────────────────────────────
OUTPUTS_DIR      = OUTPUTS_ROOT / _stem                                    # per-dataset audit dir
TRAFFIC_WEIGHTED = OUT_DIR     / f"{_stem}.traffic-weighted.csv"           # final output only
SUMMARY_JSON     = OUTPUTS_DIR / f"{_stem}.traffic-weighted.summary.json"

# ── rules ─────────────────────────────────────────────────────────────────────

rule all:
    input: TRAFFIC_WEIGHTED, SUMMARY_JSON


rule filter_weighted_flows:
    """Prune (vp_id, target_id) flows below the cumulative-traffic threshold."""
    input:
        csv = str(MESH),
    output:
        csv     = str(TRAFFIC_WEIGHTED),
        summary = str(SUMMARY_JSON),
    params:
        kept_fraction = KEPT_FRAC,
        vp_col        = VP_COL,
        target_col    = TARGET_COL,
        weight_col    = WEIGHT_COL,
        out_dir       = str(OUT_DIR),
        outputs_dir   = str(OUTPUTS_DIR),
    shell:
        """
        mkdir -p {params.out_dir}
        mkdir -p {params.outputs_dir}
        .venv/bin/python -m scripts.processing.source.filter_weighted_flows \
            --input {input.csv} \
            --output {output.csv} \
            --summary {output.summary} \
            --kept-traffic-fraction {params.kept_fraction} \
            --vp-col {params.vp_col} \
            --target-col {params.target_col} \
            --weight-col {params.weight_col}
        """
