#!/usr/bin/env bash
# Serial driver: run the 6 target-geo-filter per-ASN finals via cli.sh.
# Each config runs to completion before the next; a failure is logged and the
# batch continues. Per-config logs + a rollup live under logs/geo_runs/.
# Usage:  tmux new-session -d -s cbg_geo "bash run_geo_finals.sh"

set -uo pipefail            # no -e: keep going if one config fails
cd "$(dirname "$0")"
export PATH="$PWD/.venv/bin:$PATH"   # cli.sh calls `python`; resolve to the venv

LOGDIR="logs/geo_runs"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/_summary.log"
: > "$SUMMARY"

CONFIGS=(
  north_america_as7018_final_us
  north_america_as7922_final_us
  north_america_as7018_final_na
  north_america_as7922_final_na
  europe_as3209_final_eu
  europe_as3215_final_eu
)

for c in "${CONFIGS[@]}"; do
  echo "[$(date '+%F %T')] START $c" | tee -a "$SUMMARY"
  if ./cli.sh --configfile "scripts/benchmark/v2/config/$c.yaml" \
       >"$LOGDIR/$c.log" 2>&1; then
    echo "[$(date '+%F %T')] OK    $c" | tee -a "$SUMMARY"
  else
    echo "[$(date '+%F %T')] FAIL  $c (see $LOGDIR/$c.log)" | tee -a "$SUMMARY"
  fi
done

echo "[$(date '+%F %T')] ALL DONE" | tee -a "$SUMMARY"
