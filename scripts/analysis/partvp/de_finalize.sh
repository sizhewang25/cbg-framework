#!/usr/bin/env bash
# Auto-finalizer: when the de side run completes, extract its features, re-run
# the full 6-run tier analysis (regenerating CSVs + box plots + trees), and
# commit. de is the 6th (confirmatory EU) run; the report's findings already
# hold on the other 5.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
LOG=scripts/analysis/partvp/de_finalize.log
: > "$LOG"

echo "=== waiting for DONE_DE $(date -Is) ===" >>"$LOG"
for i in $(seq 1 720); do          # up to 3 h
  grep -q DONE_DE scripts/analysis/partvp/textbook_de.log 2>/dev/null && { echo "de done $(date -Is)" >>"$LOG"; break; }
  sleep 15
done

$PY -m scripts.analysis.partvp.extract_features \
  --run-dir scripts/benchmark/v2/outputs_partvp/europe_as3209_final_de \
  --inputs-dir scripts/benchmark/v2/inputs/ripe_atlas_asn_corpora/europe_as3209_final_de/probes_to_anchors \
  --out scripts/analysis/partvp/data/europe_as3209_final_de.parquet >>"$LOG" 2>&1 \
  && echo "extracted de" >>"$LOG" || { echo "FAILED extract de" >>"$LOG"; exit 1; }

$PY -m scripts.analysis.partvp.analyze_tiers \
  --features "scripts/analysis/partvp/data/*.parquet" \
  --out-dir scripts/analysis/partvp/analysis --trees >>"$LOG" 2>&1 \
  && echo "analysis (6 runs) done" >>"$LOG" || { echo "FAILED analysis" >>"$LOG"; exit 1; }

git add scripts/analysis/partvp/data scripts/analysis/partvp/analysis >>"$LOG" 2>&1
git commit -q -m "data(partvp): fold in de (6th/EU confirmation) — full 6-run tier analysis

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" >>"$LOG" 2>&1 \
  && echo "committed" >>"$LOG" || echo "nothing to commit / commit skipped" >>"$LOG"
echo "=== FINALIZE DONE $(date -Is) ===" >>"$LOG"
