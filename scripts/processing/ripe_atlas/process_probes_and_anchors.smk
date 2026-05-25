"""Probe + anchor sanitize → per-ASN corpora → stratification pipeline.

Starts from the reproducibility datasets shipped with the repo, runs the
two sanitize steps via ClickHouse, materializes the six deployment-scenario
VP corpora, then computes the anchor-level K-fold stratification used by
the leakage-free CBG eval protocol.

Pipeline:

  reproducibility_probes.json  ─→ sanitize_probes  ─→ filtered_probes.json   ─┐
                                                                              ├─→ select_corpora ─→ asn_corpora/anchors/anchors.json
  reproducibility_anchors.json ─→ sanitize_anchors ─→ filtered_anchors.json ──┘                  + asn_corpora/probes/<continent>/probes_of_as_*.json
                                                                                                                  │
                                                                                                                  ▼
                                                                                       stratify_anchors → asn_corpora/anchors/stratification.json
                                                                                                          + asn_corpora/anchors/kfolds/anchor_fold_<0..4>.json

The first three rules hit ClickHouse:

  - sanitize_anchors.py queries `anchors_meshed_pings` (SOI sanitization).
  - sanitize_probes.py  queries `ping_10k_to_anchors` (>= N measurements).
  - select_probes_and_anchors.py queries `ping_10k_to_anchors` again for
    the city-dedup ranking (per-probe n_records + median RTT).

`stratify_anchors` is offline (no DB) — operates on the 721-anchor shared
eval set produced by `select_corpora`.

Requires CLICKHOUSE_HOST / CLICKHOUSE_PASSWORD in .env. ClickHouse can serve
the queries in parallel, so `-j 2` lets the two sanitize rules run together.

Companion to the city-geocoding pipeline at `add_city_to_probes_anchors.smk`
in this directory (Nominatim / reverse_geocoder lookups), which consumes the
outputs of this pipeline. The two are intentionally separate: this one is
fast (~minutes), the geocoding one is slow (~hours under the online Nominatim
policy).

Run from the repo root:
  snakemake -s scripts/processing/ripe_atlas/process_probes_and_anchors.smk -j 2
"""

from pathlib import Path

REPRO = Path("datasets/reproducibility_datasets/atlas")
DATASETS = Path("datasets/ripe_atlas")
ASN_CORPORA = DATASETS / "asn_corpora"
ANCHORS_DIR = ASN_CORPORA / "anchors"
PROBES_DIR = ASN_CORPORA / "probes"

# Stratification config — matches the leakage-free eval protocol (DistGeo
# K-fold, top-20 ASN buckets + others, k=5, seed=42).
STRAT_ALGO = "distgeo"
STRAT_K = 5
STRAT_SEED = 42
STRAT_TOP_N = 20


rule all:
    input:
        DATASETS / "filtered_anchors.json",
        DATASETS / "filtered_probes.json",
        ANCHORS_DIR / "anchors.json",
        ANCHORS_DIR / "stratification.json",
        *[ANCHORS_DIR / "kfolds" / f"anchor_fold_{n}.json" for n in range(STRAT_K)],


rule sanitize_anchors:
    input:
        REPRO / "reproducibility_anchors.json",
    output:
        DATASETS / "filtered_anchors.json",
        DATASETS / "removed_anchor_ips.json",
    shell:
        "python -m scripts.processing.ripe_atlas.sanitize_anchors "
        "--anchors-file {input} --output {output[0]}"


rule sanitize_probes:
    input:
        REPRO / "reproducibility_probes.json",
    output:
        DATASETS / "filtered_probes.json",
    shell:
        "python -m scripts.processing.ripe_atlas.sanitize_probes "
        "--probes-file {input} --output {output}"


rule select_corpora:
    input:
        probes=DATASETS / "filtered_probes.json",
        anchors=DATASETS / "filtered_anchors.json",
    output:
        # Sentinel outputs declared so snakemake re-triggers if any are missing.
        # The script writes additional `*_stats.json` siblings beside each one.
        ANCHORS_DIR / "anchors.json",
        ANCHORS_DIR / "anchors_stats.json",
        PROBES_DIR / "north_america" / "probes_of_as_7922.json",
        PROBES_DIR / "north_america" / "probes_of_as_7018.json",
        PROBES_DIR / "europe" / "probes_of_as_3209.json",
        PROBES_DIR / "europe" / "probes_of_as_3215.json",
        PROBES_DIR / "global" / "probes_of_as_31898.json",
        PROBES_DIR / "global" / "probes_of_as_16509.json",
    shell:
        "python -m scripts.processing.ripe_atlas.select_probes_and_anchors "
        "--probes-file {input.probes} --anchors-file {input.anchors} "
        f"--output-dir {ASN_CORPORA}"


rule stratify_anchors:
    input:
        ANCHORS_DIR / "anchors.json",
    output:
        ANCHORS_DIR / "stratification.json",
        *[ANCHORS_DIR / "kfolds" / f"anchor_fold_{n}.json" for n in range(STRAT_K)],
    shell:
        "python -m scripts.processing.ripe_atlas.stratify "
        f"--algo {STRAT_ALGO} --k {STRAT_K} --seed {STRAT_SEED} "
        f"--asn-bucket-top-n {STRAT_TOP_N} "
        "--anchors-file {input} "
        f"--output-dir {ANCHORS_DIR}"
