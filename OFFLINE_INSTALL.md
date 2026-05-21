# Offline install — ship geoscale to an air-gapped Linux machine

Two scripts, one tarball. The bundle ships a **fully pip-installed `.venv/`**
so the target needs no network and no pip resolution at all.

Constraint: build host and target must agree on **Linux distro major
version, glibc, CPU arch, and Python 3.12.x patch level**. Any of those
differing breaks the venv.

## Build (connected Linux host)

```bash
# in the repo root
./package_offline.sh
```

What it does:

1. Reads `[project.dependencies]` from `pyproject.toml` → `requirements.txt`.
2. Creates a fresh `.venv/` with `python3.12 -m venv --copies` (the
   `--copies` flag is what makes `.venv/bin/python` a real binary instead
   of a symlink to the host's `/usr/bin/python3.12`).
3. `pip install -r requirements.txt` into the venv.
4. Tars repo + `.venv/` into `geoscale-offline-<timestamp>.tar.gz`,
   excluding heavy / regenerable directories.

Typical bundle size: **~1 GB** (most of it is numpy / scipy / matplotlib /
cartopy / pillow inside `.venv/`). Use the wheels-based approach (see git
history before this commit) if you need a smaller bundle and don't mind
running `pip install` on the target.

## Excluded from the bundle

- `datasets/` — bring data over separately (`scp` the specific files you need).
- `clickhouse_files/`, `.snakemake/`, `tasks/` — runtime state.
- `analysis/results/`, `analysis/figures/`, every `outputs/` directory,
  `scripts/benchmark/v2/inputs/` — regenerable.
- `.git`, `__pycache__/`, `*.pyc` — derived state.

## Transfer

`scp`, `rsync`, USB — whatever the target environment allows. The tarball
is one file.

## Install (target)

```bash
tar -xzf geoscale-offline-*.tar.gz
cd <extracted-dir>
./install_offline.sh
```

What it does:

1. **Validates `pyvenv.cfg`** — confirms the base Python path the venv
   was built against still resolves on this host. If not, it tells you
   exactly which line to edit.
2. **Rewrites shebangs** in `.venv/bin/*` to point at this extraction's
   `python`. The entry-point scripts (`snakemake`, `pip`, `jupyter`, …)
   have the build host's absolute path baked in; this fixes them so they
   work from the new location.
3. **Smoke-imports** the heavy modules and runs both unit suites
   (22 + 24 tests). Exits non-zero on any failure.

Run things via the venv's python (no activation needed):

```bash
./.venv/bin/python -m scripts.benchmark.v2.cli --help
./.venv/bin/snakemake -s scripts/benchmark/v2/Snakefile --configfile <your-config.yaml> -j 4
```

## When things go sideways

- **`pyvenv.cfg` check fails** — `python3.12` isn't where the venv expects
  it on the target. Either install it at the right path or update
  `pyvenv.cfg`'s `home = …` line (the script prints the exact `sed` for
  you).
- **`./.venv/bin/python` segfaults / "wrong ELF class"** — build host
  arch ≠ target arch (x86_64 vs aarch64), or glibc too old / too new.
  No fix beyond rebuilding on a matching host.
- **`ImportError` on a specific module** — usually CPython minor mismatch
  (e.g. built on 3.12.7, target has 3.12.3). Match the minor.
- **`/`-style paths in shebang still wrong** — re-run `install_offline.sh`;
  it's idempotent.

## Customising

- **Different Python**: set `PYTHON=python3.12.5` (or any 3.12.x binary)
  before running `package_offline.sh`. The target's Python minor version
  must match.
- **Adding a dataset**: copy the specific CSV / parquet to the right path
  under `datasets/` on the target after extraction.
- **Including your own DataSource**: it's already in the bundle if you
  committed it before running `package_offline.sh`; if not, copy it over
  manually.
