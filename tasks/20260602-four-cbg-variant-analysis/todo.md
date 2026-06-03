# Four-CBG-Variant Analysis Across VP Setups — Todo

## Phase 0: Setup & Discovery
- [x] Create workspace `scripts/paper/cbg_bench/` (package: `__init__.py`).
- [x] Confirm all 6 `*_final` runs have per-fold `targets.parquet` on disk for the 4 variants (vanilla_cbg, million_scale_cbg, octant_cbg_nofil, spotter_cbg_nofil). (90 = 18 combos × 5 folds per setup.)
- [x] Confirm matching `eval_observations.parquet` exist per fold for each setup.
- [x] Define the **config schema** (YAML): `run_id`, list of 6 setups (slug, run stem), the 4 variant combo_ids, default percentiles/thresholds. Outputs derive to `scripts/paper/cbg_bench/<run_id>/`.
- [x] Write the concrete config file (`config/four_variants.yaml`) for the current 6 `*_final` setups.
- [x] Create shared style module `scripts/paper/cbg_bench/_variant_style.py` with fixed `VARIANT_LABELS`, `VARIANT_COLORS`, `VARIANT_ORDER`, `VARIANT_PAIRS`.
- [x] Create a shared config loader + output-path helper (`_io.py`; typer `--config` option in each figure).

## Phase 1: Data assembly
- [x] Helper to merge folds per (setup, combo): concat `targets.parquet` across `fold_*` (`load_combo_targets`).
- [x] Helper to compute per-(setup, target) closest-VP distance from `eval_observations.parquet` (`min haversine(target, vp)`), keyed by `target_id` (`load_closest_vp`).
- [x] Join closest-VP distance onto the merged per-target table (`load_setup_long`).

## Phase 2: Deliverable 1 — per-setup SUCCESS-only CDFs  (`plot_error_cdf.py`)
- [x] Plot 6 figures (one per setup), 4 variant curves each, SUCCESS-only.
- [x] Compute fallback% per variant (FALLBACK / total) and show it in the legend.
- [x] Dump plotted data (sorted error arrays + CDF y, fallback%) to sibling `cdf_<slug>.json`.

## Phase 3: Deliverable 2 — per-setup summary table  (`summary_table.py`)
- [x] Compute {p5, p25, p50, p75, p95, fallback%} per variant per setup.
- [x] Rank rows by p50 ASC by default (CLI `--rank-by` / `--ascending`).
- [x] Render `table_<slug>.png` + combined `summary_table.md`; dump data to `table_<slug>.json`.

## Phase 4: Deliverable 3 — error vs closest-VP-distance, faceted by variant  (`plot_error_vs_vp_dist.py`)
- [x] Adapt `plot_per_target_sorted.py`: x-axis = closest-VP-distance ASC; facet per variant.
- [x] Keep the threshold reference lines; apply the fixed color/label mapping.
- [x] Dump plotted data to `error_vs_vpdist_<slug>.json`.

## Phase 5: Deliverable 4 — two paired scatters (one figure per pair)  (`plot_paired_scatter.py`)
- [x] Figure A: `error(vanilla_cbg)` vs `error(million_scale_cbg)`, colored by closest-VP distance, y=x line.
- [x] Figure B: `error(octant_cbg_nofil)` vs `error(spotter_cbg_nofil)`, colored by closest-VP distance, y=x line.
- [x] Each figure dumps its plotted data to a sibling `.json`.

## Phase 6: Verification
- [x] Sanity-check fallback% against `run.json` status_counts (as7018/octant_cbg_nofil: 83/713 = 11.64% — exact match).
- [x] Spot-check closest-VP distance for one target by hand (as16509 101.53.31.6: hand 1097.00 km == `_io` 1097.00 km).
- [x] Confirm JSON dumps round-trip (CDF `error_km_sorted` arrays reproduce table p5/p25/p50/p75/p95 exactly for all 4 variants).
- [x] Confirm colors/labels identical across all four deliverables (all import `_variant_style`; success counts agree 0-mismatch across cdf/table/facet JSONs).

## To discuss (parked — not finalizable yet)
- [ ] "Close but all on one side" vs "surrounded": VP azimuthal coverage / largest gap vs residual error at fixed closest-VP distance.
- [ ] Tightest-constraint RTTs: target→VP RTTs (within τ ms) for the VPs that determine the final intersection boundary.
