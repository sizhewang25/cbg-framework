"""Probe + anchor sanitize → per-ASN corpora pipeline.

Starts from the reproducibility datasets shipped with the repo, runs the
two sanitize steps via ClickHouse, then materializes the six
deployment-scenario VP corpora.

Pipeline:

  reproducibility_probes.json  ─→ sanitize_probes  ─→ filtered_probes.json   ─┐
                                                                              ├─→ select_corpora ─→ asn_corpora/...
  reproducibility_anchors.json ─→ sanitize_anchors ─→ filtered_anchors.json ──┘

All three rules hit ClickHouse:

  - sanitize_anchors.py queries `anchors_meshed_pings` (SOI sanitization).
  - sanitize_probes.py  queries `ping_10k_to_anchors` (>= N measurements).
  - select_probes_and_anchors.py queries `ping_10k_to_anchors` again for
    the city-dedup ranking (per-probe n_records + median RTT).

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


rule all:
    input:
        DATASETS / "filtered_anchors.json",
        DATASETS / "filtered_probes.json",
        ASN_CORPORA / "anchors.json",


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
        ASN_CORPORA / "anchors.json",
        ASN_CORPORA / "anchors_stats.json",
        ASN_CORPORA / "north_america" / "probes_of_as_7922.json",
        ASN_CORPORA / "north_america" / "probes_of_as_7018.json",
        ASN_CORPORA / "europe" / "probes_of_as_3209.json",
        ASN_CORPORA / "europe" / "probes_of_as_3215.json",
        ASN_CORPORA / "global" / "probes_of_as_31898.json",
        ASN_CORPORA / "global" / "probes_of_as_16509.json",
    shell:
        "python -m scripts.processing.ripe_atlas.select_probes_and_anchors "
        "--probes-file {input.probes} --anchors-file {input.anchors} "
        f"--output-dir {ASN_CORPORA}"
