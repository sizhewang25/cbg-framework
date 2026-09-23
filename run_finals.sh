#!/usr/bin/env bash
# Serial driver for the CBG finals: 3 operator ASNs x {mesh, traffic-weighted},
# plus as7018_us_test01 as a mesh-only public counterpart, plus the reciprocal
# (topology-matched) public/private pairs.
#
# Each config runs to completion before the next; a failure is logged and the
# batch continues. Per-config logs + a rollup live under logs/finals/.
#
# Usage:  tmux new-session -d -s cbg_finals "bash run_finals.sh"
#         bash run_finals.sh --dry-run     # preflight only, run nothing
#         bash run_finals.sh --force       # ignore the cache, recompute all
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
# Each reciprocal pair is likewise two steps: cut BOTH datasets to the H3 res-4
# cells they both occupy (§7.3's best-effort VP-topology match), then run each
# filtered side. The pair is only comparable to itself -- each side's footprint is
# defined by the other -- so matching the same public set against a different
# operator ASN yields a different public set, and the filenames say which.
#
# Run ids are NEW (-mesh / -weighted suffixes). The published
# as0X-260728-260802 trees are never written to: their CSVs are reconstructions
# whose folds are pinned by a stored stratification, which re-materializing with
# (k=5, seed=42) would not reproduce.

set -uo pipefail            # no -e: keep going if one config fails
cd "$(dirname "$0")"
export PATH="$PWD/.venv/bin:$PATH"   # cli.sh calls `python`; resolve to the venv

DRY_RUN=0
FORCE_ARGS=()
for _arg in "$@"; do
  case "$_arg" in
    --dry-run) DRY_RUN=1 ;;
    --force)   FORCE_ARGS=(--forceall) ;;
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

# Reciprocal arms: "<public-config>:<private-config>". Both sides are filtered to
# their common footprint, then both are run. The source CSVs come from these two
# configs and the filtered CSVs from the matching `<config>-reciprocal.yaml`, so
# every path lives in a config and none is spelled out here.
RECIPROCAL_PAIRS=(
  # "as7018-ripe-mesh:as02-260728-260802-mesh"
  # "as01-materialization-test:as01-randweight-precomputed"
)

# The configs a pair runs, derived from it: two per pair, public then private.
RECIPROCAL_CONFIGS=()
for _pair in "${RECIPROCAL_PAIRS[@]}"; do
  RECIPROCAL_CONFIGS+=("${_pair%%:*}-reciprocal" "${_pair##*:}-reciprocal")
done

ALL_CONFIGS=("${MESH_CONFIGS[@]}" "${WEIGHTED_CONFIGS[@]}" "${RECIPROCAL_CONFIGS[@]}")

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
log "preflight: checking inputs for ${#MESH_CONFIGS[@]} mesh + ${#WEIGHTED_CONFIGS[@]} weighted + ${#RECIPROCAL_CONFIGS[@]} reciprocal runs"
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
for pair in "${RECIPROCAL_PAIRS[@]}"; do
  # The two UNFILTERED sources must exist; both filtered CSVs are derived below.
  # Their source configs need not be listed in MESH_CONFIGS -- a pair reads their
  # csv_path whether or not that arm is itself being run.
  for c in "${pair%%:*}" "${pair##*:}"; do
    p=$(cfg_path "$c" csv_path)
    [[ -f "$p" ]] || MISSING+=("$c (reciprocal source): csv_path $p")
  done
  # A missing -reciprocal config would otherwise surface as an empty output path
  # and a filter that writes to the repo root.
  for c in "${pair%%:*}-reciprocal" "${pair##*:}-reciprocal"; do
    [[ -f "configs/$c.yaml" ]] || MISSING+=("$c: configs/$c.yaml does not exist")
  done
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
  else
    log "FAIL  $c (see $LOGDIR/$c.log)"
  fi
done

# ---- mesh arms --------------------------------------------------------------
for c in "${MESH_CONFIGS[@]}"; do
  log "START $c"
  if ./cli.sh --configfile "configs/$c.yaml" "${FORCE_ARGS[@]}" >"$LOGDIR/$c.log" 2>&1; then
    log "OK    $c"
  else
    log "FAIL  $c (see $LOGDIR/$c.log)"
  fi
done

# ---- reciprocal arms: match both footprints, then run both sides ------------
# Called directly rather than through a Snakefile: Snakemake locks the whole
# working directory, and this step has a single input pair and no fan-out.
for pair in "${RECIPROCAL_PAIRS[@]}"; do
  pub="${pair%%:*}"; priv="${pair##*:}"
  pub_in=$(cfg_path "$pub" csv_path)
  priv_in=$(cfg_path "$priv" csv_path)
  pub_out=$(cfg_path "$pub-reciprocal" csv_path)
  priv_out=$(cfg_path "$priv-reciprocal" csv_path)

  # Audit artifact, not a dataset: the match summary goes to the same outputs/
  # tree the traffic-weighted step uses, so datasets/ holds only CSVs.
  recip_outputs="scripts/processing/source/outputs/$pub-x-$priv"
  mkdir -p "$recip_outputs"

  log "START $pub x $priv (reciprocal match)"
  if ! python -m scripts.processing.source.reciprocal_grid_filter \
         --public "$pub_in"   --out-public  "$pub_out" \
         --private "$priv_in" --out-private "$priv_out" \
         --summary "$recip_outputs/reciprocal.summary.json" \
         >"$LOGDIR/$pub-x-$priv.reciprocal.log" 2>&1; then
    log "FAIL  $pub x $priv (reciprocal match; see $LOGDIR/$pub-x-$priv.reciprocal.log)"
    continue        # no filtered CSVs -> both runs would fail anyway
  fi
  log "OK    $pub x $priv (reciprocal match)"

  for c in "$pub-reciprocal" "$priv-reciprocal"; do
    log "START $c"
    if ./cli.sh --configfile "configs/$c.yaml" "${FORCE_ARGS[@]}" >"$LOGDIR/$c.log" 2>&1; then
      log "OK    $c"
    else
      log "FAIL  $c (see $LOGDIR/$c.log)"
    fi
  done
done

log "ALL DONE"
