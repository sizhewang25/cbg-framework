#!/usr/bin/env bash
# Serial driver for the CBG finals: 3 operator ASNs x {mesh, traffic-weighted},
# then the v4 analysis artifacts over whatever ran.
#
# Two benchmark modules, in this order:
#
#   1. weighted arms -- derive the traffic-weighted subset beside its mesh,
#      then run against it (precomputed mode)
#   2. mesh arms     -- run directly
#
# followed by scripts/analysis/v4/create_analysis_artifacts.sh over the arms
# that succeeded. Each config runs to completion before the next; a failure is
# logged and the batch continues. Per-config logs + a rollup live under
# logs/finals/.
#
# Usage:  tmux new-session -d -s cbg_finals "bash run_finals.sh"
#         bash run_finals.sh --dry-run      # preflight only, run nothing
#         bash run_finals.sh --force        # ignore the cache, recompute all
#         bash run_finals.sh --no-analysis  # benchmark only, no figures
#
# WHY --force EXISTS. Snakemake decides what to re-run from file timestamps and
# recorded params -- neither of which sees a CODE change. The combo id is the
# whole cache key (`<run>/<setup>/<fold>/<combo>/targets.parquet`), so a combo
# whose LTD/MTL/CTR implementation changed while its config did not is reported
# as up to date and silently keeps the old numbers. `--force` passes
# `--forceall` to both stages, which is what you want after touching anything
# under scripts/framework/ or scripts/libs/.
#
# The weighted arms need a mesh ANNOTATED WITH REAL TRAFFIC VOLUME -- the plain
# meshes carry no weight column. Set those paths in the *-weighted.yaml configs
# (source_kwargs.mesh_csv_path). The preflight below refuses to start until they
# exist, rather than skipping them and leaving you with a half-finished batch.
#
# Each weighted arm is two steps: derive the traffic-weighted subset beside its
# mesh, then run against it (precomputed mode), so the same CSV feeds both the
# benchmark and the §8.1 dataset figures.
#
# The analysis step groups the two arms separately and never pools them: a
# weighted run is a subset of its own mesh's targets, so one pooled population
# holding both would count a site twice and compare a dataset against itself.
# Each arm gets its own pooled bars, Euler diagram and error CDF, in its own
# `@<arm>` directory.
#
# Run ids are NEW (-mesh / -weighted suffixes). The published
# as0X-260728-260802 trees are never written to: their CSVs are reconstructions
# whose folds are pinned by a stored stratification, which re-materializing with
# (k=5, seed=42) would not reproduce.

set -uo pipefail            # no -e: keep going if one config fails
cd "$(dirname "$0")"
export PATH="$PWD/.venv/bin:$PATH"   # cli.sh calls `python`; resolve to the venv

DRY_RUN=0
RUN_ANALYSIS=1
FORCE_ARGS=()
for _arg in "$@"; do
  case "$_arg" in
    --dry-run)     DRY_RUN=1 ;;
    --force)       FORCE_ARGS=(--forceall) ;;
    --no-analysis) RUN_ANALYSIS=0 ;;
    *) echo "run_finals.sh: unknown argument: $_arg" >&2; exit 2 ;;
  esac
done

LOGDIR="logs/finals"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/_summary.log"
: > "$SUMMARY"

DERIVE_SMK="scripts/processing/source/derive_traffic_weighted_cbg_data.smk"

# Mesh arms: run directly.
MESH_CONFIGS=(
  as01-260728-260802-mesh
  as02-260728-260802-mesh
  as03-260728-260802-mesh
  # as01-materialization-test
)

# Weighted arms: derive the subset first, then run.
WEIGHTED_CONFIGS=(
  # as01-260728-260802-weighted
  # as02-260728-260802-weighted
  # as03-260728-260802-weighted
  # as01-randweight-precomputed
)

ALL_CONFIGS=("${MESH_CONFIGS[@]}" "${WEIGHTED_CONFIGS[@]}")

# Run ids of the arms that exited 0, collected as they run and handed to the
# analysis step. Tracked rather than re-derived from the config lists so a
# failed arm is not scored from whatever stale tree an earlier attempt left.
MESH_OK=()
WEIGHTED_OK=()

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }

# Read a source_kwargs path out of a config (nested under `benchmark:`).
cfg_path() {  # $1=config stem  $2=key
  python - "$1" "$2" <<'PY'
import sys, yaml
cfg, key = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(f"configs/{cfg}.yaml"))["benchmark"]["source_kwargs"]
print(d.get(key, ""))
PY
}

# A config's own run id. Every finals config currently names its file stem, but
# that is a convention rather than a rule, and the analysis step addresses runs
# by id -- so read it rather than assume it.
cfg_run_id() {  # $1=config stem
  python - "$1" <<'RUNID'
import sys, yaml
cfg = yaml.safe_load(open(f"configs/{sys.argv[1]}.yaml"))
rid = (cfg.get("run_id") or "").strip()
if not rid:
    sys.exit(f"configs/{sys.argv[1]}.yaml declares no run_id")
print(rid)
RUNID
}

# Append a config's run id to an arm's OK list, or log why it could not be.
# An empty id would reach the driver as a bare `--mesh ''`, whose benchmark-tree
# check (`-d outputs/benchmark/v2/`) passes on the empty string and hands the CLI
# a run id it cannot resolve.
record_ok() {  # $1=array name  $2=config stem
  local rid
  if ! rid=$(cfg_run_id "$2" 2>&1); then
    log "WARN  $2 ran, but its run id could not be read ($rid); not scored"
    return
  fi
  eval "$1+=(\"\$rid\")"
}

if (( ${#ALL_CONFIGS[@]} == 0 )); then
  log "ABORT: every config is commented out in MESH_CONFIGS / WEIGHTED_CONFIGS."
  exit 1
fi

# ---- preflight: nothing else may hold the Snakemake lock --------------------
# Snakemake locks the whole working directory, so a concurrent run -- or a stale
# lock left by a killed one -- makes every config here fail with a LockException
# buried in its own log. Catch it once, up front, with the fix.
if compgen -G ".snakemake/locks/*" >/dev/null; then
  # `[s]nakemake` so the pattern cannot match this script's own command line --
  # a bare -f pattern self-matches the invoking shell and always reports "a run
  # is in progress", which would hide every stale lock.
  if pgrep -f "[s]nakemake -s scripts/benchmark/v2/Snakefile" >/dev/null; then
    log "ABORT: another Snakemake run holds the lock on $PWD"
    log "       Wait for it to finish, or stop it, then re-run."
  else
    log "ABORT: a stale Snakemake lock remains in $PWD (no process is running)."
    log "       Clear it with:"
    log "         .venv/bin/snakemake -s scripts/benchmark/v2/Snakefile \\"
    log "             --configfile configs/${ALL_CONFIGS[0]}.yaml --unlock"
  fi
  exit 1
fi

# ---- preflight: every input must exist before anything runs -----------------
log "preflight: checking inputs for ${#MESH_CONFIGS[@]} mesh + ${#WEIGHTED_CONFIGS[@]} weighted runs"
MISSING=()
for c in "${MESH_CONFIGS[@]}"; do
  p=$(cfg_path "$c" csv_path)
  [[ -f "$p" ]] || MISSING+=("$c: csv_path $p")
done
for c in "${WEIGHTED_CONFIGS[@]}"; do
  p=$(cfg_path "$c" mesh_csv_path)
  # Only the weight-bearing mesh must pre-exist; the subset is derived below.
  [[ -f "$p" ]] || MISSING+=("$c: mesh_csv_path $p")
done
if (( ${#MISSING[@]} )); then
  log "ABORT: ${#MISSING[@]} required input(s) missing -- nothing was run:"
  for m in "${MISSING[@]}"; do log "         $m"; done
  log "       Weight-bearing meshes are not generated by this repo. Point the"
  log "       *-weighted.yaml configs at your traffic-annotated exports, or run"
  log "       only the mesh arms."
  exit 1
fi
if (( ${#FORCE_ARGS[@]} )); then
  log "preflight OK -- --force: every combo will be recomputed (--forceall)"
else
  log "preflight OK -- incremental; pass --force to ignore the cache"
fi

if (( DRY_RUN )); then
  log "--dry-run: preflight only, exiting before any run"
  exit 0
fi

# ---- weighted arms: derive the subset, then run -----------------------------
for c in "${WEIGHTED_CONFIGS[@]}"; do
  mesh=$(cfg_path "$c" mesh_csv_path)
  log "START $c (derive subset from $(basename "$mesh"))"
  if ! python -m snakemake -s "$DERIVE_SMK" --cores 1 \
         --config mesh="$mesh" >"$LOGDIR/$c.derive.log" 2>&1; then
    log "FAIL  $c (subset derivation; see $LOGDIR/$c.derive.log)"
    continue          # no subset -> the run would fail anyway
  fi
  if ./cli.sh --configfile "configs/$c.yaml" "${FORCE_ARGS[@]}" >"$LOGDIR/$c.log" 2>&1; then
    log "OK    $c"
    record_ok WEIGHTED_OK "$c"
  else
    log "FAIL  $c (see $LOGDIR/$c.log)"
  fi
done

# ---- mesh arms --------------------------------------------------------------
for c in "${MESH_CONFIGS[@]}"; do
  log "START $c"
  if ./cli.sh --configfile "configs/$c.yaml" "${FORCE_ARGS[@]}" >"$LOGDIR/$c.log" 2>&1; then
    log "OK    $c"
    record_ok MESH_OK "$c"
  else
    log "FAIL  $c (see $LOGDIR/$c.log)"
  fi
done

# ---- analysis: the v4 artifacts over the arms that ran ----------------------
# One invocation, both arms, because the driver is what knows the dependency
# order within a run and the two arms only ever share this call -- it pools each
# separately. Arms that produced nothing are simply not passed; the driver
# reports them as skips rather than failing, so a mesh-only batch is a clean run.
#
# Not gated on the benchmark having been fully successful: a partly-failed batch
# is exactly when you want to see the figures for the arms that did land.
if (( RUN_ANALYSIS )); then
  ANALYSIS_ARGS=()
  (( ${#MESH_OK[@]} ))     && ANALYSIS_ARGS+=(--mesh "${MESH_OK[@]}")
  (( ${#WEIGHTED_OK[@]} )) && ANALYSIS_ARGS+=(--weighted "${WEIGHTED_OK[@]}")

  if (( ${#ANALYSIS_ARGS[@]} == 0 )); then
    log "SKIP  analysis (v4): every benchmark arm failed, nothing to score"
  else
    log "START analysis (v4): mesh ${#MESH_OK[@]}, weighted ${#WEIGHTED_OK[@]}"
    if ./scripts/analysis/v4/create_analysis_artifacts.sh "${ANALYSIS_ARGS[@]}" \
         >"$LOGDIR/_analysis.v4.log" 2>&1; then
      log "OK    analysis (v4)"
    else
      log "FAIL  analysis (v4) (see $LOGDIR/_analysis.v4.log)"
    fi
    # The driver prints its own tally; surface it so the rollup is self-contained.
    sed -n '/^runs: /p' "$LOGDIR/_analysis.v4.log" | while read -r line; do
      log "      $line"
    done
  fi
else
  log "SKIP  analysis (v4): --no-analysis"
fi

log "ALL DONE"
