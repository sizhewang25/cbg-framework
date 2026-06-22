#!/usr/bin/env bash
# Regenerate targets.parquet for the 6 characterization configs with the new
# mtl_participants instrumentation. Deletes only the per-fold combo outputs and
# summary.parquet (preserves the precomputed clusters/ answer space), then
# reruns the benchmark Snakefile per config.
set -uo pipefail
cd "$(dirname "$0")/../../.."   # repo root

PY=.venv/bin/python
SMK="$PY -m snakemake -s scripts/benchmark/v2/Snakefile"
OUT=scripts/benchmark/v2/outputs
SETUP_REL=ripe_atlas_asn_corpora/probes_to_anchors
LOG=scripts/analysis/partvp/rerun.log
: > "$LOG"

CONFIGS=(
  scripts/benchmark/v2/config/global_as16509_final.yaml
  scripts/benchmark/v2/config/global_as31898_final.yaml
  scripts/benchmark/v2/config/europe_as3209_final_de.yaml
  scripts/benchmark/v2/config/europe_as3215_final_fr.yaml
  scripts/benchmark/v2/config/north_america_as7018_final_us.yaml
  scripts/benchmark/v2/config/north_america_as7922_final_us.yaml
)

echo "=== rerun start $(date -Is) ===" | tee -a "$LOG"
for cfg in "${CONFIGS[@]}"; do
  rid=$($PY -c "import yaml,sys;print(yaml.safe_load(open('$cfg'))['run_id'])")
  echo "--- $rid ($cfg) ---" | tee -a "$LOG"
  # Drop per-fold combo outputs + summary so Snakemake regenerates them;
  # keep clusters/ (answer space) intact.
  rm -rf "$OUT/$rid/$SETUP_REL"/fold_* 2>/dev/null
  rm -f  "$OUT/$rid/summary.parquet" 2>/dev/null
  $SMK --configfile "$cfg" -j 8 --rerun-incomplete >>"$LOG" 2>&1
  rc=$?
  echo "--- $rid done rc=$rc $(date -Is) ---" | tee -a "$LOG"
done
echo "=== ALL RERUNS DONE $(date -Is) ===" | tee -a "$LOG"
