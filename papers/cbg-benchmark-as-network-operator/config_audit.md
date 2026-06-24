# Config Audit: Data Availability, partvp Source, and Single-Source Architecture

---

## Task 1: Column availability in official final-run `targets.parquet`

### North America AS7018 final US

**vanilla_cbg** (`outputs/north_america_as7018_final_us/.../fold_0/vanilla_cbg/targets.parquet`):
```
target_id, target_lat, target_lon, n_obs, pred_lat, pred_lon, status, error, error_km,
ltd_ms, ltd_alloc_peak_bytes, ltd_rss_peak_bytes,
mtl_ms, mtl_alloc_peak_bytes, mtl_rss_peak_bytes,
ctr_ms, ctr_alloc_peak_bytes, ctr_rss_peak_bytes,
n_ltd_success, ltd_predictions, mtl_success, mtl_error, mtl_intersection_kind,
n_mtl_participants, mtl_participants,
ctr_success, ctr_error, seed
```
Shape: (16, 28). **Has `mtl_participants` and `n_mtl_participants`.** OK for failure taxonomy.

**octant_cbg** (`outputs/north_america_as7018_final_us/.../fold_0/octant_cbg/targets.parquet`):
```
target_id, target_lat, target_lon, n_obs, pred_lat, pred_lon, status, error, error_km,
ltd_ms, ltd_alloc_peak_bytes, ltd_rss_peak_bytes,
mtl_ms, mtl_alloc_peak_bytes, mtl_rss_peak_bytes,
ctr_ms, ctr_alloc_peak_bytes, ctr_rss_peak_bytes,
n_ltd_success, ltd_predictions, mtl_success, mtl_error, mtl_intersection_kind,
ctr_success, ctr_error, seed
```
Shape: (16, 26). **Missing `mtl_participants` and `n_mtl_participants`.** This is the `outputs/` (final/full) run — octant_cbg there does NOT store per-VP band data.

### Europe AS3209 final DE

**vanilla_cbg**: 28 columns, **has `mtl_participants`** (same schema as NA vanilla).  
**octant_cbg**: 28 columns, **has `mtl_participants`** — this one does store it (5 folds checked, fold_0 confirmed).

### Summary

| Run | Setup | `mtl_participants` | `n_mtl_participants` |
|-----|-------|-------------------|---------------------|
| `outputs/north_america_as7018_final_us` | vanilla_cbg | YES | YES |
| `outputs/north_america_as7018_final_us` | octant_cbg | **NO** | **NO** |
| `outputs/europe_as3209_final_de` | vanilla_cbg | YES | YES |
| `outputs/europe_as3209_final_de` | octant_cbg | YES | YES |

The NA AS7018 final US octant_cbg run in `outputs/` is the outlier. All other setups examined have `mtl_participants`. The `outputs_partvp/` runs (see Task 2) consistently store `mtl_participants` for all four textbook combos.

No polygon data (GeoJSON, WKT, or polygon column) is stored in any `targets.parquet`.

---

## Task 2: partvp config source and comparison

### What is `outputs_partvp/`?

`outputs_partvp/` is a **real directory** (not a symlink), containing 7 run directories:
```
europe_as3209_eu/
europe_as3209_final_de/
europe_as3215_eu/
europe_as3215_final_fr/
north_america_as7018_final_na/
north_america_as7018_final_us/
north_america_as7922_final_us/
```
vs. `outputs/` which has 20+ run directories (full benchmark sweep, sweep configs, etc.).

Each `outputs_partvp/` run only contains the four textbook combos: `vanilla_cbg`, `million_scale_cbg`, `octant_cbg`, `spotter_cbg`.

### What config was used to generate `outputs_partvp/north_america_as7018_final_us/`?

The `run.json` files embedded in each combo directory are the ground truth. Key findings from `fold_0`:

- **vanilla_cbg**: `ltd=low_envelope`, `mtl=spherical_circle`, `ctr=boundary_vertex_mean`  
  `mtl_kwargs: {speed_ratio: 0.6667, enable_circle_filter: true}`
- **octant_cbg**: `ltd=bounded_spline`, `mtl=planar_annulus_weighted`, `ctr=monte_carlo_medoid`
- **spotter_cbg**: `ltd=normal_dist`, `mtl=planar_annulus_weighted`, `ctr=monte_carlo_medoid`

The same `spherical_circle + boundary_vertex_mean` pattern appears in `europe_as3209_final_de` vanilla_cbg `run.json`.

**This is an older config.** The `spherical_circle` MTL predates the current `planar_circle` used in the textbook configs.

### Comparison: `outputs_partvp/` run.json vs `cfg_textbook/north_america_as7018_final_us.yaml` vs current `config/north_america_as7018_final_us.yaml`

| combo | `outputs_partvp/` run.json (actual on-disk) | `cfg_textbook/` YAML (what run_textbook_config.py would run) | `config/` YAML (current full benchmark) |
|-------|----------------------------------------------|--------------------------------------------------------------|----------------------------------------|
| vanilla_cbg | `spherical_circle` + `boundary_vertex_mean` | `planar_circle` + `geometric_centroid` | `planar_circle` + `geometric_centroid` |
| octant_cbg | `planar_annulus_weighted` + `monte_carlo_medoid` | same | same |
| spotter_cbg | `planar_annulus_weighted` + `monte_carlo_medoid` | same | same |

**Conclusion**: The `outputs_partvp/` data was generated with an **older invocation** that predates the `cfg_textbook/` YAML files. The `cfg_textbook/` YAMLs are consistent with the current `config/` benchmark configs for the four textbook combos, but the **actual on-disk data under `outputs_partvp/` for vanilla_cbg uses a different (older) setup**: `spherical_circle + boundary_vertex_mean` instead of `planar_circle + geometric_centroid`.

No separate Snakefile or log file records which script invoked the `outputs_partvp/` runs — only the `run.json` files document the actual hyperparameters used. The `cfg_textbook/` YAMLs were written *after* the fact to document the intended config; they specify `outputs_root: scripts/benchmark/v2/outputs_partvp` and would re-run with `run_textbook_config.py`.

---

## Task 3: Config single-source architecture

### How the analysis pipeline currently discovers runs

There are two distinct analysis systems:

**System A — `scripts/analysis/Snakefile` (plotting pipeline)**  
Config path: `scripts/analysis/config/<run_id>.yaml`  
Invoked as: `snakemake -s scripts/analysis/Snakefile --configfile scripts/analysis/config/<run_id>.yaml`  
Run discovery: the analysis YAML is an explicit config that reads `run_id` and `v2_outputs_root` (defaults to `scripts/benchmark/v2/outputs`). It then uses `_v2_io.discover_combos()` which globs `<run_dir>/**/targets.parquet`. Run_ids in analysis configs match benchmark config run_ids (both use `run_id` field). The `v2_outputs_root` key in the analysis config overrides the default outputs tree.

**System B — `scripts/analysis/partvp/` (failure taxonomy / characterize_failures)**  
Config path: **hardcoded** in `characterize_failures.py` as `CONFIGS` dict.  
Run discovery: fully hardcoded — each config label maps to a tuple of `(feature_parquet_path, run_dir_string)`:
```python
CONFIGS: dict[str, tuple[str, str]] = {
    "global-global": ("...data/global_as16509_final.parquet", "scripts/benchmark/v2/outputs/global_as16509_final"),
    "europe-europe": ("...data_eu/europe_as3215_eu.parquet",  "scripts/benchmark/v2/outputs_partvp/europe_as3215_eu"),
    "europe-country": ("...data/europe_as3215_final_fr.parquet", "scripts/benchmark/v2/outputs_partvp/europe_as3215_final_fr"),
    "na-na": ("...data/north_america_as7018_final_na.parquet", "scripts/benchmark/v2/outputs_partvp/north_america_as7018_final_na"),
    "na-us": ("...data/north_america_as7018_final_us.parquet", "scripts/benchmark/v2/outputs_partvp/north_america_as7018_final_us"),
}
```
`TEXTBOOK` is also hardcoded: `["vanilla_cbg", "million_scale_cbg", "octant_cbg", "spotter_cbg"]`.  
Helper scripts like `extract_features.py`, `fleet_geometry_explainability.py`, and `region_confidence.py` take `--run-dir` as a CLI argument and use `discover_combos()` for filesystem glob.

**Gap between the two systems**: The analysis Snakefile configs (`scripts/analysis/config/`) have no entries for the geo-filtered final runs (`north_america_as7018_final_us`, `europe_as3209_final_de`, etc.) — these 10 benchmark configs exist in `scripts/benchmark/v2/config/` but have no corresponding `scripts/analysis/config/` YAML. System B knows about them only through hardcoded paths.

### Proposed analysis config YAML schema

For each benchmark config `scripts/benchmark/v2/config/<name>.yaml`, a matching analysis config at `scripts/analysis/config/<name>.yaml` would add plotting/analysis params without duplicating experiment params. The benchmark config remains the single source of experiment truth; the analysis config only adds display/analysis overlays:

```yaml
# scripts/analysis/config/north_america_as7018_final_us.yaml
#
# Analysis overlay for the north_america_as7018_final_us benchmark run.
# Experiment params (combos, source_kwargs, slices) live entirely in:
#   scripts/benchmark/v2/config/north_america_as7018_final_us.yaml
# This file adds plotting + failure-analysis params only.

# Reference to the benchmark config — used by the analysis Snakefile/scripts
# to inherit run_id, source, setup, slices without duplication.
benchmark_config: scripts/benchmark/v2/config/north_america_as7018_final_us.yaml

# Override the outputs root if this run lives under outputs_partvp/ instead of outputs/.
# Omit to use the default (scripts/benchmark/v2/outputs/).
# v2_outputs_root: scripts/benchmark/v2/outputs_partvp

# Human-readable label for this config in plots and reports.
config_label: "na-us"

# Textbook combo subset for failure analysis (subset of benchmark combos).
textbook_combos:
  - vanilla_cbg
  - million_scale_cbg
  - octant_cbg
  - spotter_cbg

# Feature parquet produced by extract_features.py (input to characterize_failures).
feature_parquet: scripts/analysis/partvp/outputs/data/north_america_as7018_final_us.parquet

# Plotting knobs for the analysis Snakefile.
group_by: ltd
phase_stat: p95
include_fit: true
runtime_stat: p50
merge_folds: true

# Pairwise diff pairs for plot_error_diff_cdf.
diff_pairs:
  - [vanilla_cbg, octant_cbg]
  - [vanilla_cbg, spotter_cbg]
  - [octant_cbg, spotter_cbg]
  - [million_scale_cbg, vanilla_cbg]

# Optional: restrict CDF split to a continent (for regional corpora).
split_by_main_continent: "North America"
```

### What changes are needed to use the same run_ids as the benchmark configs

**Problem 1: System B (`characterize_failures.py` / `CONFIGS` dict) is fully hardcoded.**  
- Replace the `CONFIGS` dict with a loader that reads from a list of analysis config YAMLs.
- Each analysis config provides `config_label`, `benchmark_config` (for `run_id`), `v2_outputs_root`, `textbook_combos`, and `feature_parquet`.
- `TEXTBOOK` becomes `textbook_combos` from the config.

**Problem 2: Analysis configs for geo-filtered final runs don't exist yet.**  
Create `scripts/analysis/config/` entries for the 10 missing benchmark configs:
```
europe_as3209_final_de.yaml
europe_as3209_final_eu.yaml
europe_as3215_final_eu.yaml
europe_as3215_final_fr.yaml
north_america_as7018_final_na.yaml
north_america_as7018_final_us.yaml
north_america_as7922_final_na.yaml
north_america_as7922_final_us.yaml
```
These would include `v2_outputs_root: scripts/benchmark/v2/outputs_partvp` for the runs that currently only exist under `outputs_partvp/`.

**Problem 3: `outputs_partvp/` data for vanilla_cbg is stale (spherical_circle + boundary_vertex_mean).**  
The `cfg_textbook/` YAMLs correctly specify `planar_circle + geometric_centroid` (matching the current benchmark). Re-running `run_textbook_config.py` with the `cfg_textbook/` YAML for each affected run will overwrite the `outputs_partvp/` data with the correct config. Alternatively, the `outputs_partvp/` runs for the four textbook combos could be produced directly from the benchmark Snakefile using `outputs_root: scripts/benchmark/v2/outputs_partvp` in the analysis config's `v2_outputs_root`.

**Minimal change path**:
1. Add `benchmark_config` pointer field to analysis configs.
2. Modify the analysis Snakefile to `import yaml; benchmark_cfg = yaml.safe_load(Path(config["benchmark_config"]).read_text())` and inherit `run_id`, `source`, `setup`, `slices` from it — eliminating duplication.
3. Replace `CONFIGS` and `TEXTBOOK` in `characterize_failures.py` with a function that loads a list of analysis config YAMLs (passed via `--configs` CLI arg), reads `config_label`, `v2_outputs_root / run_id` for `run_dir`, `feature_parquet`, and `textbook_combos`.
4. Create the 10 missing analysis configs.
5. Re-run `outputs_partvp/` vanilla_cbg combos using `cfg_textbook/` YAMLs to fix the spherical_circle stale data.

---

## Key paths referenced

- `scripts/benchmark/v2/config/north_america_as7018_final_us.yaml` — current benchmark config
- `scripts/analysis/partvp/cfg_textbook/north_america_as7018_final_us.yaml` — textbook subset config (consistent with benchmark, outputs to `outputs_partvp/`, but `outputs_partvp/` data was generated with an even older pre-cfg_textbook invocation)
- `scripts/benchmark/v2/outputs_partvp/` — 7 runs, 4 textbook combos each, real directory
- `scripts/benchmark/v2/outputs/` — 20+ runs, full combo sweeps
- `scripts/analysis/_v2_io.py` — `discover_combos()` globs `**/targets.parquet`; `resolve_run_dir()` reads `run_id` from config YAML + joins onto `outputs_root`
- `scripts/analysis/partvp/characterize_failures.py` — hardcoded `CONFIGS` dict; primary consumer of `outputs_partvp/` + `mtl_participants`
