#!/usr/bin/env bash
# Serial driver: run the cluster answer-space eval (cluster.smk) for the 6
# target-geo-filter per-ASN finals. Each run is already region-scoped, so the
# global answer space IS the regional one (no cluster_geos). No ClickHouse —
# reads only the materialized parquet. Logs under logs/geo_cluster/.
# Usage:  tmux new-session -d -s cbg_cluster "bash run_geo_cluster.sh"

set -uo pipefail            # no -e: keep going if one run fails
cd "$(dirname "$0")"
export PATH="$PWD/.venv/bin:$PATH"

LOGDIR="logs/geo_cluster"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/_summary.log"
: > "$SUMMARY"

RUNS=(
  north_america_as7018_final_us
  north_america_as7922_final_us
  north_america_as7018_final_na
  north_america_as7922_final_na
  europe_as3209_final_eu
  europe_as3215_final_eu
)

for r in "${RUNS[@]}"; do
  echo "[$(date '+%F %T')] START $r" | tee -a "$SUMMARY"
  if python -m snakemake -s scripts/analysis/cluster.smk --cores all \
       --config run_id="$r" source=ripe_atlas_asn_corpora \
       >"$LOGDIR/$r.log" 2>&1; then
    echo "[$(date '+%F %T')] OK    $r" | tee -a "$SUMMARY"
  else
    echo "[$(date '+%F %T')] FAIL  $r (see $LOGDIR/$r.log)" | tee -a "$SUMMARY"
  fi
done

echo "[$(date '+%F %T')] ALL DONE" | tee -a "$SUMMARY"
