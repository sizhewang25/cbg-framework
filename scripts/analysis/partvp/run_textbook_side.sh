#!/usr/bin/env bash
# Fast side reruns: textbook-4 combos only, for the 3 remaining regional configs,
# into a separate outputs root (no conflict with the canonical full rerun). The
# 4 textbook combos compute identically to the full run (same inputs, seed=42),
# so the participating-VP features are report-identical. Runs the 3 configs in
# parallel; each Snakemake uses -j 5 (one per fold).
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
SMK="$PY -m snakemake -s scripts/benchmark/v2/Snakefile"
CFG=scripts/analysis/partvp/cfg_textbook
LOG=scripts/analysis/partvp/textbook_side.log
: > "$LOG"
echo "=== textbook side runs start $(date -Is) ===" >>"$LOG"

pids=()
for rid in europe_as3215_final_fr north_america_as7018_final_us north_america_as7922_final_us; do
  ( $SMK --configfile "$CFG/$rid.yaml" -j 5 --rerun-incomplete >>"$LOG" 2>&1
    echo "--- $rid done rc=$? $(date -Is) ---" >>"$LOG" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "=== TEXTBOOK SIDE DONE $(date -Is) ===" >>"$LOG"
