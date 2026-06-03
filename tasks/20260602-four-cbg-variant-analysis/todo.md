# Four-CBG-Variant Analysis Across VP Setups — Todo

## Phase 0: Setup & Discovery
- [ ] Create workspace `scripts/paper/cbg_bench/` (package: `__init__.py`).
- [ ] Confirm all 6 `*_final` runs have per-fold `targets.parquet` on disk for the 4 variants (vanilla_cbg, million_scale_cbg, octant_cbg_nofil, spotter_cbg_nofil).
- [ ] Confirm matching `eval_observations.parquet` exist per fold for each setup.
- [ ] Define the **config schema** (YAML): `run_id`, list of 6 setups (slug, run stem / outputs root, eval-inputs root), the 4 variant combo_ids, default percentiles/thresholds. Outputs derive to `scripts/paper/cbg_bench/<run_id>/`.
- [ ] Write the concrete config file for the current 6 `*_final` setups.
- [ ] Create shared style module `scripts/paper/cbg_bench/_variant_style.py` with fixed `VARIANT_LABELS` and `VARIANT_COLORS` (4 variants), plus canonical ordering.
- [ ] Create a shared config loader + output-path helper (typer `--config` option, mirrors `scripts/analysis/plot_*.py`).

## Phase 1: Data assembly
- [ ] Helper to merge folds per (setup, combo): concat `targets.parquet` across `fold_*` via `group_combos_by_id`.
- [ ] Helper to compute per-(setup, target) closest-VP distance from `eval_observations.parquet` (`min haversine(target, vp)`), keyed by `target_id`.
- [ ] Join closest-VP distance onto the merged per-target table.

## Phase 2: Deliverable 1 — per-setup SUCCESS-only CDFs
- [ ] Plot 6 figures (one per setup), 4 variant curves each, SUCCESS-only.
- [ ] Compute fallback% per variant (FALLBACK / total) and show it in the legend.
- [ ] Dump plotted data (x grid + per-variant CDF, fallback%) to sibling `.json`.

## Phase 3: Deliverable 2 — per-setup summary table
- [ ] Compute {p5, p25, p50, p75, p95, fallback%} per variant per setup.
- [ ] Rank rows by p50 ASC by default.
- [ ] Render table (md/png as fits the repo) and dump data to `.json` with same basename.

## Phase 4: Deliverable 3 — error vs closest-VP-distance, faceted by variant
- [ ] Adapt `plot_per_target_sorted.py`: x-axis = closest-VP-distance ASC; facet per variant.
- [ ] Keep the threshold reference lines; apply the fixed color/label mapping.
- [ ] Dump plotted data to `.json`.

## Phase 5: Deliverable 4 — two paired scatters (one figure per pair)
- [ ] Figure A: `error(vanilla_cbg)` vs `error(million_scale_cbg)`, colored by closest-VP distance, y=x line.
- [ ] Figure B: `error(octant_cbg_nofil)` vs `error(spotter_cbg_nofil)`, colored by closest-VP distance, y=x line.
- [ ] Each figure dumps its plotted data to a sibling `.json`.

## Phase 6: Verification
- [ ] Sanity-check fallback% against `run.json` status_counts for a couple of (setup, variant) pairs.
- [ ] Spot-check closest-VP distance for one target by hand (haversine to nearest VP).
- [ ] Confirm JSON dumps round-trip (re-plot from JSON matches the figure).
- [ ] Confirm colors/labels are identical across all four deliverables.

## To discuss (parked — not finalizable yet)
- [ ] "Close but all on one side" vs "surrounded": VP azimuthal coverage / largest gap vs residual error at fixed closest-VP distance.
- [ ] Tightest-constraint RTTs: target→VP RTTs (within τ ms) for the VPs that determine the final intersection boundary.
