#!/usr/bin/env bash
# Serial driver: benchmark + cluster eval for the two country-target finals
# (as3215/FR, as3209/DE). For each run: cli.sh (needs ClickHouse) then
# cluster.smk. Failures are logged and the batch continues.
# Usage:  tmux new-session -d -s cbg_country "bash run_country_finals.sh"

set -uo pipefail
cd "$(dirname "$0")"
export PATH="$PWD/.venv/bin:$PATH"

LOGDIR="logs/country_runs"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/_summary.log"
: > "$SUMMARY"

RUNS=(
  europe_as3215_final_fr
  europe_as3209_final_de
)

for r in "${RUNS[@]}"; do
  echo "[$(date '+%F %T')] BENCH START $r" | tee -a "$SUMMARY"
  if ./cli.sh --configfile "scripts/benchmark/v2/config/$r.yaml" \
       >"$LOGDIR/$r.bench.log" 2>&1; then
    echo "[$(date '+%F %T')] BENCH OK    $r" | tee -a "$SUMMARY"
  else
    echo "[$(date '+%F %T')] BENCH FAIL  $r (see $LOGDIR/$r.bench.log)" | tee -a "$SUMMARY"
    continue
  fi

  echo "[$(date '+%F %T')] CLUSTER START $r" | tee -a "$SUMMARY"
  if python -m snakemake -s scripts/analysis/cluster.smk --cores all \
       --config run_id="$r" source=ripe_atlas_asn_corpora \
       >"$LOGDIR/$r.cluster.log" 2>&1; then
    echo "[$(date '+%F %T')] CLUSTER OK    $r" | tee -a "$SUMMARY"
  else
    echo "[$(date '+%F %T')] CLUSTER FAIL  $r (see $LOGDIR/$r.cluster.log)" | tee -a "$SUMMARY"
  fi
done

echo "[$(date '+%F %T')] ALL DONE" | tee -a "$SUMMARY"
