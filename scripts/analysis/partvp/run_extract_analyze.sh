#!/usr/bin/env bash
# Wait for the 6 reruns to finish, extract per-target features for each, then
# run the tier-driver analysis. Idempotent: re-extracts only runs whose
# targets.parquet carry the new mtl_participants column.
set -uo pipefail
cd "$(dirname "$0")/../../.."   # repo root
PY=.venv/bin/python
LOG=scripts/analysis/partvp/extract_analyze.log
RERUN_LOG=scripts/analysis/partvp/rerun.log
: > "$LOG"

RUNS=(
  global_as16509_final global_as31898_final
  europe_as3209_final_de europe_as3215_final_fr
  north_america_as7018_final_us north_america_as7922_final_us
)

echo "=== waiting for reruns $(date -Is) ===" >>"$LOG"
for i in $(seq 1 360); do          # up to 60 min
  grep -q "ALL RERUNS DONE" "$RERUN_LOG" 2>/dev/null && { echo "reruns done" >>"$LOG"; break; }
  sleep 10
done

echo "=== extracting features $(date -Is) ===" >>"$LOG"
for rid in "${RUNS[@]}"; do
  $PY -m scripts.analysis.partvp.extract_features \
    --run-dir "scripts/benchmark/v2/outputs/$rid" \
    --out "scripts/analysis/partvp/data/$rid.parquet" >>"$LOG" 2>&1 \
    && echo "extracted $rid" >>"$LOG" || echo "FAILED extract $rid" >>"$LOG"
done

echo "=== analyzing $(date -Is) ===" >>"$LOG"
$PY -m scripts.analysis.partvp.analyze_tiers \
  --features "scripts/analysis/partvp/data/*.parquet" \
  --out-dir scripts/analysis/partvp/analysis --trees >>"$LOG" 2>&1 \
  && echo "analysis done" >>"$LOG" || echo "FAILED analysis" >>"$LOG"
echo "=== ALL EXTRACT+ANALYZE DONE $(date -Is) ===" >>"$LOG"
