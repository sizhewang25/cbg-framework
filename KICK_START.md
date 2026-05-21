# Kick-Start: Offline Install + Smoke Run

You just received `geoscale-offline-YYYYMMDD-HHMMSS.tar.gz`. This guide takes
you from that tarball to a fully reproduced CBG benchmark + analysis run in
about five minutes.

---

## 0. Prerequisites on the target machine

| Requirement      | Why                                                     | How to check                          |
|------------------|---------------------------------------------------------|---------------------------------------|
| Linux x86_64     | Bundle is arch-pinned                                   | `uname -m` → `x86_64`                 |
| glibc ≥ 2.39     | Bundled `python3.12` binary links host libc symbols     | `ldd --version \| head -1`             |
| `python3.12`     | The bundled venv borrows the system stdlib (not deps)   | `command -v python3.12`               |
| `tar`, GNU coreutils | Extract the tarball                                 | usually preinstalled                  |

If `python3.12` is missing on Ubuntu/Debian:

```bash
sudo apt update && sudo apt install -y python3.12
```

The patch level doesn't have to match the build host — any `3.12.x` works.

> **Older glibc (e.g. Ubuntu 22.04, glibc 2.35)?** The bundle's launcher binary
> won't start. After extracting, overwrite it with the target's interpreter:
> `cp "$(command -v python3.12)" .venv/bin/python3.12`. The `site-packages/`
> wheels stay valid (manylinux / glibc 2.17+).

---

## 1. Extract the bundle

```bash
mkdir -p ~/geoscale && cd ~/geoscale
tar -xzf /path/to/geoscale-offline-*.tar.gz
```

You should now see `install_offline.sh`, `tutorial.sh`, `cli.sh`,
`analysis.sh`, `.venv/`, `scripts/`, `pyproject.toml`, etc.

---

## 2. Verify + relocate the venv

```bash
./install_offline.sh
```

What this does:
1. Reads `.venv/pyvenv.cfg`. If its `home =` path doesn't exist on this host
   (the common case after a move), auto-detects `python3.12` on `PATH` and
   rewrites `home`, `executable`, and `command` lines so the venv knows where
   to find the stdlib.
2. Rewrites entry-point shebangs (`snakemake`, `jupyter`, …) under
   `.venv/bin/` so they point at *this* extraction's python.
3. Smoke-imports the heavy modules (`numpy`, `pandas`, `pyarrow`,
   `matplotlib`, `shapely`, `snakemake`, `PIL`) and runs the framework + v2
   benchmark unit suites.

Expected tail of the output:

```
✔ Install verified.
```

If `install_offline.sh` errors with "python3.12 was not found on PATH",
install it (see prerequisites) and re-run.

---

## 3. Activate the venv

Every script below assumes the venv is active in your shell:

```bash
source .venv/bin/activate
```

Verify with `python --version` → `Python 3.12.x` and `which python` pointing
inside `~/geoscale/.venv/bin/`.

---

## 4. Run the smoke pipeline

Three scripts, in order. Each is a thin wrapper — read them if you want to
see the exact command they invoke.

### 4a. `./tutorial.sh` — build the synthetic dataset

```bash
./tutorial.sh
```

Writes `datasets/smoke-test.csv` (750 rows, 25 VPs × 30 targets). This is a
fully synthetic dataset; no RIPE Atlas credits or network access required.

### 4b. `./cli.sh` — run the benchmark

```bash
./cli.sh --configfile scripts/benchmark/v2/config/smoke-test-01.yaml
```

Drives Snakemake through the v2 benchmark: ingest CSV → run four CBG combos
(`vanilla_cbg`, `million_scale_cbg`, `octant_cbg`, `spotter_cbg`) → summarize.

Output lands at:

```
scripts/benchmark/v2/outputs/smoke-test-01/
├── generic_csv/anchors_to_probes/all/<combo>/targets.parquet
├── generic_csv/anchors_to_probes/all/<combo>/run.json
└── summary.parquet
```

Expected runtime: a few seconds. Final line: `7 of 7 steps (100%) done`.

### 4c. `./analysis.sh` — plot the results

```bash
./analysis.sh --configfile scripts/analysis/config/smoke-test-01.yaml
```

Reads `summary.parquet` from step 4b and emits five PNGs at
`scripts/analysis/outputs/smoke-test-01/generic_csv/anchors_to_probes/all/`:

| Plot                              | What it shows                                   |
|-----------------------------------|-------------------------------------------------|
| `plot_error_cdf.png`              | Error CDF, all targets, all combos              |
| `plot_error_cdf_for_success.png`  | Error CDF restricted to successfully geolocated |
| `plot_error_diff_cdf.png`         | Error delta vs. `shortest_ping` baseline        |
| `plot_phase_runtime.png`          | Wall-time per algorithm phase                   |
| `plot_phase_memory.png`           | Peak memory per algorithm phase                 |

Final line: `6 of 6 steps (100%) done`.

---

## 5. Sanity check the outputs

```bash
python - <<'PY'
import pandas as pd
df = pd.read_parquet("scripts/benchmark/v2/outputs/smoke-test-01/summary.parquet")
print(df[["combo_id", "n_targets", "p50_error_km", "p90_error_km"]].to_string(index=False))
PY
```

You should see one row per combo with `n_targets = 25` and finite
`p50_error_km` values.

---

## 6. Bring your own data

The smoke pipeline runs against a synthetic CSV. To benchmark a real dataset,
follow the five steps below. The full contract for the CSV schema lives in
[`scripts/benchmark/v2/sources/README.md`](scripts/benchmark/v2/sources/README.md);
this section is the short version.

### 6a. Ensure the `datasets/` directory exists

```bash
mkdir -p datasets
```

`datasets/` is git-ignored and excluded from the offline bundle by design, so
on a freshly extracted bundle it may not be present yet.

### 6b. Prepare your CSV

Drop your file into `datasets/`. The schema (see
[`sources/README.md`](scripts/benchmark/v2/sources/README.md) §"Skipping the
work: GenericCSVSource") is one row per `(vp, target, RTT)` observation:

| column          | type   | required | notes                                  |
|-----------------|--------|----------|----------------------------------------|
| `vp_id`         | str    | yes      | stable VP identifier                   |
| `vp_lat`        | float  | yes      | VP latitude, degrees                   |
| `vp_lon`        | float  | yes      | VP longitude, degrees                  |
| `target_id`     | str    | yes      | stable target identifier with hard GT  |
| `target_lat`    | float  | yes      | target latitude, degrees               |
| `target_lon`    | float  | yes      | target longitude, degrees              |
| `rtt_ms`        | float  | yes      | strictly positive RTT in ms            |
| `vp_asn`        | int    | no       | auto-detected if present               |
| `vp_country`    | str    | no       | auto-detected if present               |
| `target_asn`    | int    | no       | auto-detected if present               |
| `target_country`| str    | no       | auto-detected if present               |

Rows with NaNs in any required column or with `rtt_ms <= 0` are filtered out
at load time — no need to pre-clean those.

Example:

```csv
vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,rtt_ms
probe-001,40.7128,-74.0060,anchor-A,51.5074,-0.1278,72.4
probe-001,40.7128,-74.0060,anchor-B,48.8566,2.3522,78.1
probe-002,37.7749,-122.4194,anchor-A,51.5074,-0.1278,142.0
...
```

Save it as e.g. `datasets/my-measurements.csv`.

### 6c. Point `GenericCSVSource` at your file

Edit the `DEFAULT_CSV` constant near the top of
[`scripts/benchmark/v2/sources/generic_csv.py`](scripts/benchmark/v2/sources/generic_csv.py)
(lines 51–54):

```python
DEFAULT_CSV = (
    Path(__file__).resolve().parents[4]
    / "datasets" / "my-measurements.csv"   # ← your file
)
```

Absolute paths also work if you'd rather hard-code one. To benchmark several
CSVs side-by-side, subclass `GenericCSVSource` with a distinct `name` per CSV
(see `sources/README.md` for the pattern) — that keeps each on-disk output
tree separated.

### 6d. Run the benchmark

```bash
./cli.sh --configfile scripts/benchmark/v2/config/smoke-test-01.yaml
```

The config drives one source × one slice (`all`) × four CBG combos
(`vanilla_cbg`, `million_scale_cbg`, `octant_cbg`, `spotter_cbg`). It uses
`source: generic_csv`, so it automatically picks up the CSV you wired in at
step 6c — no config edit needed for a one-off run. If you want a separate
`run_id` (recommended so outputs don't collide with the smoke run), copy the
YAML and change the top `run_id:` line:

```bash
cp scripts/benchmark/v2/config/smoke-test-01.yaml \
   scripts/benchmark/v2/config/my-run-01.yaml
# edit run_id: my-run-01
./cli.sh --configfile scripts/benchmark/v2/config/my-run-01.yaml
```

Outputs land under `scripts/benchmark/v2/outputs/<run_id>/…` with a final
`summary.parquet`.

### 6e. Plot the results

```bash
./analysis.sh --configfile scripts/analysis/config/smoke-test-01.yaml
```

The analysis config has its own `run_id` (line 11 of
[`scripts/analysis/config/smoke-test-01.yaml`](scripts/analysis/config/smoke-test-01.yaml))
— it must match the benchmark's `run_id`, since that's how the analysis step
finds the right `summary.parquet`. If you changed the benchmark `run_id` in
step 6d, copy and edit the analysis YAML the same way:

```bash
cp scripts/analysis/config/smoke-test-01.yaml \
   scripts/analysis/config/my-run-01.yaml
# edit run_id: my-run-01  (and adjust diff_pairs / group_by if desired)
./analysis.sh --configfile scripts/analysis/config/my-run-01.yaml
```

PNGs land under `scripts/analysis/outputs/<run_id>/generic_csv/<setup>/<slice>/`.

---

## 7. Where to go next

- Run without activating the venv: every wrapper works if you call
  `.venv/bin/python -m snakemake …` directly.
- See `CLAUDE.md` for the full project layout, algorithm notes, and the
  reproducibility vs. replication workflow.
- See [`scripts/benchmark/v2/sources/README.md`](scripts/benchmark/v2/sources/README.md)
  if your data doesn't fit the canonical CSV schema (ClickHouse-backed,
  multi-file landmark layouts, custom slicing, etc.).

---

## Troubleshooting

| Symptom                                                    | Fix                                                                                          |
|------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `install_offline.sh`: "python3.12 was not found on PATH"   | `sudo apt install python3.12` (or distro equivalent), re-run.                                |
| `./.venv/bin/python: cannot execute binary file`           | Arch mismatch — bundle is x86_64. Check `uname -m`.                                          |
| `version GLIBC_2.XX not found`                             | Target glibc < build host's. See "Older glibc" note above.                                   |
| `ModuleNotFoundError` after activating                     | You activated a different venv. `deactivate`, then `source ~/geoscale/.venv/bin/activate`.   |
| `cli.sh`: "Missing input files"                            | Run `./tutorial.sh` first to generate `datasets/smoke-test.csv`.                             |
| `analysis.sh`: "Missing input files: summary.parquet"      | Run `./cli.sh` first; analysis depends on its summary.                                       |
