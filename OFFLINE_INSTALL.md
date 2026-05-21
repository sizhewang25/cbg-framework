# Offline install — ship geoscale to an air-gapped Linux machine

Two scripts, one tarball. Assumes the target has Python 3.12 installed and the
build host is a connected Linux machine with the **same** distro major
version / glibc / CPU arch as the target.

## Build (connected Linux host)

```bash
# in the repo root
./package_offline.sh
```

What it does:

1. Reads `[project.dependencies]` from `pyproject.toml` and writes
   `requirements.txt`.
2. Runs `pip wheel` to build a wheel for every dep (downloads sdists where
   no wheel is published and compiles them locally).
3. Tars the repo + `wheels/` + `requirements.txt` into
   `geoscale-offline-<timestamp>.tar.gz`, excluding heavy or build-artifact
   directories.

Typical bundle size: **300–500 MB** (most of it the numpy/scipy/matplotlib
wheels). Add a tighter `--exclude` to `package_offline.sh` if your bundle
needs to fit a particular size budget.

## Excluded from the bundle

- `datasets/` — bring data over separately (`scp` the specific files you need).
- `clickhouse_files/` and `.snakemake/` — runtime state.
- `analysis/results/`, `analysis/figures/`, every `outputs/` directory and
  `scripts/benchmark/v2/inputs/` — regenerable.
- `.git`, `__pycache__/`, `venv`, `.venv` — derived state.

## Transfer

`scp`, `rsync`, USB, signed S3 link — whatever the target environment allows.

## Install (air-gapped target)

```bash
tar -xzf geoscale-offline-*.tar.gz
cd <extracted-dir>
./install_offline.sh
```

What it does:

1. Creates `.venv/` with the target's `python3.12`.
2. `pip install --no-index --find-links wheels/ -r requirements.txt`
   (network never contacted).
3. Imports the heavy modules to surface any wheel-mismatch issues
   immediately.
4. Runs both unit suites (`scripts/framework/v2/tests`, `scripts/benchmark/v2/tests`).

If all four steps succeed you have a working install. Run things via the
venv directly — no `poetry`, no shell activation:

```bash
./.venv/bin/python -m scripts.benchmark.v2.cli --help
./.venv/bin/snakemake -s scripts/benchmark/v2/Snakefile --configfile <your-config.yaml> -j 4
```

## Customising

- **Different Python**: set `PYTHON=python3.12.5` (or any 3.12.x binary)
  before running either script. The build host's interpreter must match
  the target's interpreter at minor-version level for wheel ABIs to line up.
- **Adding a dataset**: copy the specific CSV / parquet to the right path
  under `datasets/` on the target after extraction.
- **Including your own DataSource**: it's already in the bundle if you
  committed it before running `package_offline.sh`; if not, copy it over
  manually.

## When wheel mismatches bite

If `install_offline.sh` errors on `pip install` with "no matching distribution":

1. Different glibc on target — rebuild on a host with the older glibc.
2. Different CPython minor version (3.12.3 vs 3.12.7 is fine; 3.11 vs 3.12 is not).
3. CPU arch mismatch (aarch64 wheels won't load on x86_64).

The error message names the offending package; rebuild that specific wheel
on a matching host and drop it into `wheels/` to retry.
