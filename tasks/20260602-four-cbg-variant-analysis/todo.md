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
- [ ] Tightest-constraint RTTs: target→VP RTTs (within τ ms) for the VPs that determine the final intersection boundary. (Global-act; needs MTL internals not in outputs.)
- [ ] **MTL+CTR impact study (gates Study 2)** — discovered 2026-06-04: swapping only the geometry stack (same LTD, `_geo` = `planar_annulus_weighted`+`geometric_centroid` vs `spherical_circle`+`boundary_vertex_mean`) swings p50 by 20–55% and **flips by regime** (boundary_vertex_mean wins on far/one-sided US; geometric_centroid wins on near/surrounded Global; EU mixed). No single "best" CTR. Decision: **promote MTL+CTR to a factor** in the geometry studies rather than fixing one. Data already on disk (~18 combos: `_geo`/`_hull`/`_top`/`_c80`/`_c100`/weighted); may add 1–2 isolation combos (e.g. `spherical_circle`+`geometric_centroid`) to separate MTL from CTR (cheap post-processing, no re-measurement). Study 1 macro headline is CTR-robust; its `diagonal_split`/win-rate is CTR-dependent.
- [ ] Study 2 (one-sided vs surrounded / azimuthal gap) — carry CTR as a factor (depends on the impact study above).

## Study 1 — closest-VP distance impact (SOI baseline)  ✅ DONE
### Data
- [x] Reuse `load_setup_long` for {combo_id, target_id, closest_vp_km, error_km, status} — closest-VP already in `_io`, no new loader.
- [x] Add a binned-summary helper: `_io.binned_percentiles` — fixed-width closest-VP-distance bins → p50/p90/n per bin (min-N=5 guard → NaN).
### Figure — `plot_study1_distance.py` (config-driven typer CLI: `--config`, `--slug`)
- [x] Axis decision (revised to **log–log**): both axes log on shared `[1, 20000]` km, identical ticks on x/y and across all panels, square box (45° `y=x`), threshold lines on both axes, no binned overlay.
- [x] SOI hero panel (`study1_soi_<slug>.png`): error_km vs closest-VP distance, log–log, `y=x`, threshold gridlines both axes.
- [x] Cross-method faceted panel (`study1_facet_<slug>.png`, Vanilla / SOI / Octant / Spotter): identical log limits/ticks across all panels; per-panel fallback count in title; scatter + `y=x` + threshold gridlines.
- [x] Fixed color/label mapping from `_variant_style`; SOI hero is the dedicated baseline panel.
### Data dump
- [x] Sibling `study1_<slug>.json`: per-variant SUCCESS arrays (closest_vp_km, error_km) + n_success/n_total + `diagonal_split` (worse/on-diagonal/wins counts + pct, tol=2%) + axis settings. (`binned_percentiles` helper kept in `_io` for Study 2; `binned` block dropped from the JSON.)
### Analysis / verify
- [x] Cliff-vs-ramp: SOI binned p50 confirms **ramp + saturation** (as7018: 144→334→925→1380→1568 km through 0–1150 km, then high plateau ~5400→6800 km for the cross-Atlantic EU cluster at ~5800–6600 km). Not a cliff.
- [x] Verified: log axes render with `y=x` at 45° (square box); thresholds on both axes; n_success per variant matches deliverable-3 JSON exactly (all 6 setups share `_io`); SOI fallback as7018=0/713, as16509=3/713 (matches the original screenshot — it was the Global setup).
