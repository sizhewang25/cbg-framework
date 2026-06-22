#!/usr/bin/env bash
# Finisher: wait for ALL reruns to truly complete, re-extract the runs still
# missing a fresh feature table, then re-run the analysis over all 6.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
LOG=scripts/analysis/partvp/finish.log
RERUN_LOG=scripts/analysis/partvp/rerun.log
: > "$LOG"

RUNS=(
  global_as16509_final global_as31898_final
  europe_as3209_final_de europe_as3215_final_fr
  north_america_as7018_final_us north_america_as7922_final_us
)

echo "=== waiting for ALL RERUNS DONE $(date -Is) ===" >>"$LOG"
for i in $(seq 1 2160); do         # up to 6 h
  grep -q "ALL RERUNS DONE" "$RERUN_LOG" 2>/dev/null && { echo "reruns done $(date -Is)" >>"$LOG"; break; }
  sleep 10
done

# Extract every run whose targets carry the new column; (re)write its parquet.
for rid in "${RUNS[@]}"; do
  f="scripts/benchmark/v2/outputs/$rid/ripe_atlas_asn_corpora/probes_to_anchors/fold_0/spotter_cbg/targets.parquet"
  has=$($PY -c "import pandas as pd;print('mtl_participants' in pd.read_parquet('$f').columns)" 2>/dev/null || echo False)
  if [ "$has" != "True" ]; then echo "SKIP $rid (no fresh column yet: $has)" >>"$LOG"; continue; fi
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
echo "=== FINISH DONE $(date -Is) ===" >>"$LOG"
ls scripts/analysis/partvp/data/ >>"$LOG"
