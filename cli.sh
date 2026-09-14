#!/usr/bin/env bash
# Run the v2 CBG benchmark, preceded by the dataset inspection.
#
# Usage: ./cli.sh --configfile configs/as01-260728-260802-mesh.yaml
#        ./cli.sh --configfile configs/... -n        # dry-run both stages
#        CBG_SKIP_INSPECT=1 ./cli.sh --configfile ... # benchmark only
#
# Prerequisite: `.venv/bin` on PATH (run_finals.sh exports it). There is no
# `.venv/bin/activate` in this checkout -- invoke interpreters by path.
#
# Stage 1 (inspect_dataset.smk) describes the *dataset*: the target space, the
# answer space, the §7.3 bipartite geometry, the §8.1 proximity strata and
# their maps. It reads only the canonical CSV, so it runs before any combo --
# if the VP geometry or the answer space is not what you expected, run-combo
# will not fix it, and it costs hours to find out afterwards.
#
# Stage 1 is SKIPPED, not failed, for sources with no canonical CSV: the
# `ripe_atlas*` sources read probe/anchor directories instead, so there is
# nothing for a CSV-only precheck to read. `resolve-edge-csv` makes that call
# and exits 3 to say so, which keeps the ~31 ripe_atlas configs working.
# Any other inspection failure is fatal by design -- that is what a gate is.

set -euo pipefail

INSPECT_SMK="scripts/benchmark/v2/inspect_dataset.smk"
BENCH_SMK="scripts/benchmark/v2/Snakefile"

# The gate needs the config by itself; the stages still get the full argv, so
# flags like -n or --unlock reach both.
CONFIGFILE=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[i]}" in
    --configfile)   CONFIGFILE="${args[i + 1]:-}" ;;
    --configfile=*) CONFIGFILE="${args[i]#*=}" ;;
  esac
done

if [[ -n "${CBG_SKIP_INSPECT:-}" ]]; then
  echo "cli.sh: CBG_SKIP_INSPECT set -- skipping the dataset inspection"
elif [[ -z "$CONFIGFILE" ]]; then
  echo "cli.sh: no --configfile given -- skipping the dataset inspection" >&2
else
  # `set +e` rather than `if ...; then`: the skip case is identified by exit
  # code 3 specifically, and conflating it with "any nonzero" would swallow a
  # genuinely broken config.
  set +e
  python -m scripts.benchmark.v2.cli resolve-edge-csv --configfile "$CONFIGFILE" >/dev/null
  rc=$?
  set -e
  case "$rc" in
    0)
      echo "cli.sh: [1/2] dataset inspection ($INSPECT_SMK)"
      python -m snakemake -s "$INSPECT_SMK" --cores 1 "$@"
      ;;
    3)
      echo "cli.sh: skipping the dataset inspection -- this config names no canonical CSV"
      ;;
    *)
      echo "cli.sh: ABORT -- could not read $CONFIGFILE (exit $rc)" >&2
      exit "$rc"
      ;;
  esac
fi

echo "cli.sh: [2/2] benchmark ($BENCH_SMK)"
python -m snakemake \
    -s "$BENCH_SMK" \
    --cores all \
    "$@"
