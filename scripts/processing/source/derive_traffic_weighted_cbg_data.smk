"""Derive traffic-weighted CBG data from sanitized rows via UP-city whitelist.

Two-stage pipeline:
  1. derive_up_city_whitelist              – traffic-gated (vp_id, city) pairs
     from UP-city ``TOTAL_TB`` (default keep fraction: 0.95).
  2. filter_sanitized_by_up_city_whitelist – keep only sanitized rows whose
     (vp_id, target_norm_city) key is whitelisted.

Usage (from repo root):
  snakemake -s scripts/processing/source/derive_traffic_weighted_cbg_data.smk \
      -j 1

Override defaults via --config:
  sanitized              : sanitized CBG CSV to filter
  up_city_csv            : UP-city traffic CSV with TOTAL_TB
  out_dir                : final output directory
  outputs_dir            : intermediate/audit directory root
  kept_traffic_fraction  : cumulative traffic target in (0, 1] (default: 0.95)
"""

from pathlib import Path

# ── configurable inputs ───────────────────────────────────────────────────────
SANITIZED = Path(
    config.get(
        "sanitized",
    )
)
UP_CITY_CSV = Path(
    config.get(
        "up_city_csv",
        "datasets/up_asn_traffic/"
    )
)
OUT_DIR = Path(config.get("out_dir", "datasets/final"))
OUTPUTS_ROOT = Path(config.get("outputs_dir", "scripts/processing/source/outputs"))
KEPT_TRAFFIC_FRACTION = float(config.get("kept_traffic_fraction", 0.95))

# ── derive intermediate / final paths ────────────────────────────────────────
OUTPUTS_DIR = OUTPUTS_ROOT / _stem

WHITELIST = OUTPUTS_DIR / f"{_stem}.up-city-whitelist.csv"
WHITELIST_SUMMARY = OUTPUTS_DIR / f"{_stem}.up-city-whitelist.summary.json"

TRAFFIC_WEIGHTED = OUT_DIR / f"{_stem}.traffic-weighted.csv"
TRAFFIC_WEIGHTED_SUMMARY = OUTPUTS_DIR / f"{_stem}.traffic-weighted.summary.json"


# ── rules ─────────────────────────────────────────────────────────────────────
rule all:
    input:
        str(TRAFFIC_WEIGHTED),
        str(WHITELIST),
        str(WHITELIST_SUMMARY),
        str(TRAFFIC_WEIGHTED_SUMMARY),


rule derive_up_city_whitelist:
    """Derive (vp_id, target_norm_city) whitelist from UP-city TOTAL_TB."""
    input:
        up_city_csv=str(UP_CITY_CSV),
    output:
        csv=str(WHITELIST),
        summary=str(WHITELIST_SUMMARY),
    params:
        kept_traffic_fraction=KEPT_TRAFFIC_FRACTION,
        outputs_dir=str(OUTPUTS_DIR),
    shell:
        """
        mkdir -p {params.outputs_dir}
        .venv/bin/python -m scripts.processing.source.derive_up_city_whitelist \
            --input {input.up_city_csv} \
            --kept-traffic-fraction {params.kept_traffic_fraction} \
            --output {output.csv} \
            --summary {output.summary}
        """


rule filter_sanitized_by_whitelist:
    """Filter sanitized CBG rows to only whitelist-approved (vp_id, city) pairs."""
    input:
        sanitized_csv=str(SANITIZED),
        whitelist_csv=str(WHITELIST),
    output:
        csv=str(TRAFFIC_WEIGHTED),
        summary=str(TRAFFIC_WEIGHTED_SUMMARY),
    params:
        out_dir=str(OUT_DIR),
        outputs_dir=str(OUTPUTS_DIR),
    shell:
        """
        mkdir -p {params.out_dir}
        mkdir -p {params.outputs_dir}
        .venv/bin/python -m scripts.processing.source.filter_sanitized_by_up_city_whitelist \
            --input {input.sanitized_csv} \
            --whitelist {input.whitelist_csv} \
            --output {output.csv} \
            --summary {output.summary}
        """
