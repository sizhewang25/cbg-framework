#!/usr/bin/env bash
# EU natural experiment: AS3209 (DE-central) and AS3215 (FR-western) fleets
# geolocating all Europe anchors (target_continent=Europe). Textbook-4 combos,
# into the side outputs root. Materializes inputs for the new run_ids first.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
SMK="$PY -m snakemake -s scripts/benchmark/v2/Snakefile"
CFG=scripts/analysis/partvp/cfg_textbook
LOG=scripts/analysis/partvp/eu.log
: > "$LOG"
echo "=== EU runs start $(date -Is) ===" >>"$LOG"
pids=()
for rid in europe_as3209_eu europe_as3215_eu; do
  ( $SMK --configfile "$CFG/$rid.yaml" -j 5 --rerun-incomplete >>"$LOG" 2>&1
    echo "--- $rid done rc=$? $(date -Is) ---" >>"$LOG" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "=== EU RUNS DONE $(date -Is) ===" >>"$LOG"
