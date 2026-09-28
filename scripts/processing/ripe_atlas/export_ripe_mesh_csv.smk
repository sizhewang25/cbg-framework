"""Export a per-ASN RIPE Atlas mesh from ClickHouse into the canonical raw CSV.

Stage 0 of the standardized CBG data path. It takes the corpora produced by
`process_probes_and_anchors.smk` and turns them into the one artifact the rest
of the pipeline speaks: a flat measurement CSV in the `generic_csv` schema.

  asn_corpora/probes/<region>/probes_of_as_<asn>.json   (VPs, AS-filtered)
  asn_corpora/anchors/anchors.json                      (targets, shared set)
  ClickHouse ping_10k_to_anchors                        (min RTT per pair)
            │
            ▼  export_mesh_csv.py  (src IN <corpus probes>, dst IN <anchors>)
  datasets/raw/<dataset>.csv  +  <dataset>.csv.stats.json

The AS filter is a metadata filter: the ping table has no ASN column, so the
corpus decides which probes are AS7018 and the query is scoped by their IPs.
Output is geographically unfiltered on purpose — mainland selection and SOI
sanitization belong to stage 1/2.

Run from the repo root:

  # stage 0 — this file
  snakemake -s scripts/processing/ripe_atlas/export_ripe_mesh_csv.smk -j 1

  # stages 1+2 — mainland filter, then SOI sanitization, into datasets/final/
  snakemake -s scripts/processing/source/preprocess_cbg_raw_data.smk -j 1 \
      --config raw=datasets/raw/ripe-asmix-mesh.csv

which lands datasets/final/ripe-asmix-mesh.mainland.sanitized.csv.

Override via --config:
  dataset     : output stem            (default: ripe-asmix-mesh)
  vp_asn      : VP corpus ASN          (default: 7018)
  raw_dir     : output directory       (default: datasets/raw)
  ping_table  : ClickHouse table       (default: ping_10k_to_anchors)
  max_rtt_ms  : RTT upper bound        (default: 10000)

Requires CLICKHOUSE_HOST / CLICKHOUSE_PASSWORD in .env.
"""

from pathlib import Path

DATASET     = config.get("dataset",    "ripe-asmix-mesh")
VP_ASN      = int(config.get("vp_asn", 7018))
RAW_DIR     = Path(config.get("raw_dir",    "datasets/raw"))
CORPORA     = Path(config.get("corpora_root", "datasets/ripe_atlas/asn_corpora"))
PING_TABLE  = config.get("ping_table", "ping_10k_to_anchors")
MAX_RTT_MS  = float(config.get("max_rtt_ms", 10000))

RAW_CSV   = RAW_DIR / f"{DATASET}.csv"
RAW_STATS = RAW_DIR / f"{DATASET}.csv.stats.json"

# Corpus inputs, declared so the rule re-fires when the upstream pipeline reruns.
ANCHORS = CORPORA / "anchors" / "anchors.json"
PROBES  = next(
    iter(sorted((CORPORA / "probes").glob(f"*/probes_of_as_{VP_ASN}.json"))),
    CORPORA / "probes" / "north_america" / f"probes_of_as_{VP_ASN}.json",
)


rule all:
    input: RAW_CSV, RAW_STATS


rule export_mesh_csv:
    """ClickHouse min-RTT export joined with probe / anchor / geo metadata."""
    input:
        probes  = str(PROBES),
        anchors = str(ANCHORS),
    output:
        csv   = str(RAW_CSV),
        stats = str(RAW_STATS),
    params:
        vp_asn     = VP_ASN,
        table      = PING_TABLE,
        max_rtt_ms = MAX_RTT_MS,
    shell:
        """
        .venv/bin/python -m scripts.processing.ripe_atlas.export_mesh_csv \
            --vp-asn {params.vp_asn} \
            --probes-file {input.probes} \
            --anchors-file {input.anchors} \
            --table {params.table} \
            --max-rtt-ms {params.max_rtt_ms} \
            --output {output.csv} \
            --stats-output {output.stats}
        """
