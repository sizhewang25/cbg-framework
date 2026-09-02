# `outputs/benchmark/v2/` output schema reference

Reference for the v3 analysis layer. Written by `scripts/benchmark/v2/`
(`cli.py`, `runner.py`, `schema.py`, `eval_source*.py`, `eval_bench_results.py`)
plus the Snakemake cluster/classification rules.

> **Root note.** These runs live at repo-root `outputs/benchmark/v2/<run_id>/`,
> *not* at `scripts/benchmark/v2/outputs/` (which holds only `archived/`).
> `scripts/analysis/_v2_io.py::DEFAULT_OUTPUTS_ROOT` points at the old root, so
> v3 must resolve its own.

## 1. Run inventory

| run_id | dataset role | setup | slices | combos | n_vps | n_targets | n_obs | clusters (R=50km) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `as01-260728-260802` | operator | `anchors_to_probes` | `fold_0..4` | 5 | 134 | 399 | 53,262 | 18 (16 singleton) |
| `as02-260728-260802` | operator | `anchors_to_probes` | `fold_0..4` | 5 | 134 | 412 | 54,990 | 20 (18 singleton) |
| `as03-260728-260802` | operator | `anchors_to_probes` | `fold_0..4` | 5 | 134 | 458 | 61,293 | 22 (21 singleton) |
| `as7018_us_test01` | public / RIPE | `probes_to_anchors` | `fold_0..4` | 16 | 53 | 78 | 4,129 | 17 (4 singleton) |

`target_id` / `vp_id` are **opaque strings**, and their form differs per run:
`tg-fccd3e4` / `vp-2d3a875` on the operator runs (sanitized), raw IPs
(`103.196.37.98`) on as7018. Never parse them; join on them only.

## 2. Directory layout

```
outputs/benchmark/v2/<run_id>/
├── summary.parquet                       # (§4) one row per (setup,slice,combo)
├── generic_csv/<setup>/                  # (§3) inputs + per-fold method output
│   ├── vps.csv  targets.csv
│   ├── clusters/{clusters.csv,assignments.csv,meta.json}
│   ├── fold_{0..4}/<combo_id>/{targets.parquet,run.json,fit_checkpoint.pkl}
│   ├── cluster_scored/{<combo>_scored.csv,baseline.csv}      # as01-03 ONLY
│   └── <run_id>_classification_table_top{1,3}.{csv,parquet}  # as01-03 ONLY
├── eval_source/   <basename>_*           # (§5) dataset geometry, method-free
├── eval_dataset/  <basename>_*           # (§5) near-duplicate of eval_source
└── bench_eval/    <combo>_bench_per_target.csv, summary.parquet  # as7018 ONLY
```

`<basename>` is the source CSV stem, e.g.
`as01-20260728-20260802.mainland.sanitized`, `as7018-us-test01`. Glob it; do not
reconstruct it from `run_id`.

## 3. Method output — the primary artifact

### `fold_<N>/<combo_id>/targets.parquet` — one row per eval target

Richest artifact in the tree; carries the paper's (accuracy, runtime, memory)
triple *and* the phase-attribution forensics.

| group | columns |
| --- | --- |
| identity / truth | `target_id`, `target_lat`, `target_lon`, `n_obs`, `seed` |
| prediction | `pred_lat`, `pred_lon`, `status`, `error`, `error_km` |
| LTD | `ltd_ms`, `ltd_alloc_peak_bytes`, `ltd_rss_peak_bytes`, `n_ltd_success`, `ltd_predictions` |
| MTL | `mtl_ms`, `mtl_alloc_peak_bytes`, `mtl_rss_peak_bytes`, `mtl_success`, `mtl_error`, `mtl_intersection_kind`, `n_mtl_participants`, `mtl_participants` |
| CTR | `ctr_ms`, `ctr_alloc_peak_bytes`, `ctr_rss_peak_bytes`, `ctr_success`, `ctr_error` |

Nested list-of-struct columns (the per-VP evidence):

- `ltd_predictions[]`: `vp_id`, `success`, `error`, `upper_km`, `lower_km`
  — every VP's distance bound, i.e. **all** constraints LTD produced.
- `mtl_participants[]`: `vp_id`, `rtt_ms`, `echoed_upper_km`, `echoed_lower_km`,
  `vp_lat`, `vp_lon` — only the constraints MTL **admitted**.

`len(ltd_predictions)` vs `n_mtl_participants` is the direct measurement of
Phase 2b constraint filtering (as01 fold_0 vanilla: 134 → 15).

`status` ∈ `SUCCESS` | `FALLBACK` | `ERROR`. `mtl_intersection_kind` ∈
`polygon` | `none` (observed).

> **Trap — `error_km` is populated on FALLBACK rows.** On fallback, `pred_*` is
> the shortest-ping VP coordinate and `error_km` is *that* VP's error. Pooling
> `error_km` across all statuses silently floors CBG error at the baseline's,
> which is exactly the comparison the paper forbids (paper §7.2, "fallbacks
> count as failures"). **Always split by `status`** and report
> with/without-fallback separately.

### `fold_<N>/<combo_id>/run.json`

Combo config + run-level cost. Keys: `run_id`, `source`, `setup`, `slice`,
`combo_id`, `ltd`/`mtl`/`ctr` + `ltd_kwargs`/`mtl_kwargs`/`ctr_kwargs`,
`base_seed`, `pair_weight_min`, `n_targets_dropped_below_min_weight`,
`enable_fallback`, `started_at`, `n_fit_samples`, `n_targets`,
`status_counts{SUCCESS,FALLBACK,ERROR}`, `fit_success`, `fit_error`, `fit_ms`,
`fit_alloc_peak_bytes`, `fit_rss_peak_bytes`, `run_baseline_rss_bytes`,
`run_peak_rss_bytes`.

> **Trap — `(ltd, mtl, ctr)` does NOT identify a combo.** The distinguishing
> config lives in `*_kwargs`. `combo_id` is the only safe key.
>   - `octant_cbg_hull` vs `octant_cbg_spl`: identical triple; differ only by
>     `ltd_kwargs.fit_spline` (false → bounded hull, true → bounded spline).
>   - `*_top` vs base: `mtl_kwargs.highest_weight_only=true` vs
>     `weight_threshold=0.9` (Phase 2c).
>   - `*_geo`: `ctr` swapped `monte_carlo_medoid` → `geometric_centroid` (Phase 3).
>   - `spotter_cbg_c80` / `_c100`: `ltd_kwargs.target_coverage` 0.8 / 1.0.

Combo → paper variant mapping:

| combo_id | LTD | MTL | CTR | paper variant |
| --- | --- | --- | --- | --- |
| `million_scale_cbg` | `speed_of_internet` (ratio .6667) | `planar_circle` | `geometric_centroid` | SoI CBG |
| `vanilla_cbg` | `low_envelope` | `planar_circle` | `geometric_centroid` | Vanilla CBG |
| `octant_cbg_hull` | `bounded_spline` (`fit_spline=false`) | `planar_annulus_weighted` | `monte_carlo_medoid` | Octant-Hull |
| `octant_cbg_spl` / `octant_cbg` | `bounded_spline` (`fit_spline=true`) | `planar_annulus_weighted` | `monte_carlo_medoid` | Octant-Spline |
| `spotter_cbg` | `normal_dist` | `planar_annulus_weighted` | `monte_carlo_medoid` | Spotter |

as7018 adds 11 ablation combos (`_geo`, `_top`, `_c80`, `_c100`) that sweep
Phases 2c and 3 — these are RQ3 phase-attribution arms, not published variants.

### `clusters/` — the answer space

- `clusters.csv`: `cluster_id`, `centroid_lat`, `centroid_lon`, `n_members`,
  `radius_km`, `diameter_km`, `is_singleton`
- `assignments.csv`: `target_id`, `target_lat`, `target_lon`, `cluster_id`,
  `dist_to_centroid_km`
- `meta.json`: `radius_km` (50.0), `n_targets`, `n_clusters`, `n_singletons`,
  `n_unique_target_locations`

Note this is the **complete-linkage / centroid-radius-capped** answer space, not
the grid the paper draft §7.3 describes. Same role (finite metro-granular class
set seeded at target centroids); different construction. Flag when writing paper
numbers.

The v3 layer builds the grid version instead, over either of two tessellations
(H3 `res=4` by default, HEALPix `nside=128` for the paper's original setting) —
see [README.md](README.md#choosing-a-grid). Its `seeds.csv` is grid-neutral:
`grid_scheme`, `grid_resolution`, `cell_id`. The distinction that matters when
mixing numbers is that a grid has **boundaries** and this radius-capped space
does not: a facility group straddling a grid line is split into two classes at
any resolution, whereas the linkage space can guarantee grouping within its
radius.

### `cluster_scored/` + classification tables — *as01-03 only*

- `<combo>_scored.csv`: `target_id`, `status`, `success`, `match`,
  `error_to_centroid_km`, `truth_centroid_km`, `error_km`, `match_top1..3`
- `baseline.csv`: `target_id`, `vp_matches_centroid`, `vp_to_centroid_km`,
  `vp_matches_centroid_top1..3` — the **shortest-ping baseline**.
- `<run_id>_classification_table_top{1,3}.csv`: `target_id` × one bool column
  per method, including `shortest_ping`. Pivoted, ready for per-target
  agree/disagree analysis.

## 4. `summary.parquet` — aggregated cost/accuracy

One row per `(run_id, source, setup, slice, combo_id)`; 87 columns.
25 rows for as01-03 (5 folds × 5 combos), 80 for as7018 (5 × 16).

- keys: `run_id`, `source`, `setup`, `slice`, `combo_id`, `ltd`, `mtl`, `ctr`
- counts: `n_targets`, `n_success`, `n_fallback`, `n_error`
- accuracy: `error_km_{p5,p25,p50,p75,p95,mean,std}`
- per-phase cost, for each of `ltd`/`mtl`/`ctr` × `{ms, alloc_peak_bytes,
  rss_peak_bytes}` × `{p5,p25,p50,p75,p95,mean,std}`
- fit/run: `fit_ms`, `fit_alloc_peak_bytes`, `fit_rss_peak_bytes`,
  `run_baseline_rss_bytes`, `run_peak_rss_bytes`

> Per README: per-target `tracemalloc` numbers are noise-dominated for fast
> stages. Trust the aggregated p50/p95 here, and use `run_peak_rss_bytes`
> (psutil) for run-level memory.
>
> `modules/pareto.py` therefore defaults to the **`alloc`** channel aggregated to
> a percentile rather than raw per-target values: `*_rss_peak_bytes` is floored
> at one 4096-byte page for every stage faster than the 5 ms sampler, so it
> cannot rank methods at p50. It also reduces memory across stages with
> **`max`** — `instrument.py` resets tracemalloc inside each stage, so the
> columns are per-stage peaks, making `max` the pipeline high-water mark and
> `sum` the no-release upper bound. Runtime sums.
>
> `error_km_*` here pools FALLBACK rows (see §3 trap). For fallback-excluded
> accuracy, recompute from `targets.parquet`.

## 5. `eval_source/` & `eval_dataset/` — method-free dataset geometry

Computed from the source CSV + answer space alone; no variant runs. This is the
material for paper §7.3 (dataset geometry) and §8.2 (shortest-ping
characterization).

**Which side to read.** `eval_source` ⊇ `eval_dataset` in columns on the
operator runs (adds 7 kNN answer-space columns); all shared columns agree to
float rounding. `dataset_stats.json` exists **only** under `eval_dataset`. So:
per-target/cluster geometry from `eval_source`, dataset rollup from
`eval_dataset`. On as7018 both sides are identical (37 cols, no kNN columns).

### `<basename>_eval_per_target.csv` — 44 cols (operator) / 37 (as7018)

| group | columns |
| --- | --- |
| identity | `target_id`, `target_lat`, `target_lon`, `n_avail_vps` |
| closest VP (latent) | `closest_vp_km`, `closest_vp_id`, `closest_vp_lat/lon` |
| shortest-ping VP (observed) | `shortest_ping_vp_km`, `shortest_ping_vp_id`, `shortest_ping_vp_lat/lon` |
| RTT quality | `min_rtt_ms`, `min_inflation`, `rtt_weighted_dist_km`, `rtt_dist_spearman`, `closest_vp_rtt_rank`, `closest_is_shortest_ping`, `closest_to_shortest_ping_km`, `soi_violation_share` |
| answer space | `cluster_id`, `truth_centroid_lat/lon`, `cell_gap_km`, `target_distinguishable_vp_dist_km`, `closest_vp_in_same_cluster`, `shortest_ping_vp_in_same_cluster`, `closest_vp_to_centroid_km`, `shortest_ping_vp_to_centroid_km` |
| **proximity decomposition** | `n_discriminative_vps`, `has_vp_proximity`, `shortest_ping_vp_is_discriminative`, `best_discriminative_rtt_rank`, `proximity_label` |
| anycast guard | `vp_pair_disk_overlap_km`, `n_disjoint_sites`, `anycast_suspect` |
| kNN (operator runs only) | `knn_neighbor_{1,2,3}_km`, `knn_mean_km`, `knn_std_km`, `knn_gap_ratio`, `n_competitors_1_5x` |

`proximity_label` ∈ `NO_PROXIMITY` | `HAS_NOT_USED_PROXIMITY` |
`HAS_USED_PROXIMITY`. This **is** paper §8.2's failure taxonomy, precomputed:
- `NO_PROXIMITY` → structural failure (no discriminative VP exists; MTL required)
- `HAS_NOT_USED_PROXIMITY` → selection failure (reachable but unreached = the
  CBG opportunity)
- `HAS_USED_PROXIMITY` → baseline already correct (regression risk for variants)

### `<basename>_eval_stats.json`

`csv`, `n_obs`, `n_pairs`, `n_vps`, `n_targets`, then:
- `metrics.<col>.{n,min,max,mean,percentiles{p5,p25,p50,p75,p95}}` for ~17 of
  the per-target columns
- `target_clustering`: `radius_km`, `n_clusters`, `n_singletons`,
  `targets_per_cluster`, `closest_vp_within_radius_share`,
  `shortest_ping_vp_within_radius_share`, `closest_vp_in_same_cluster_share`,
  `shortest_ping_vp_in_same_cluster_share`
- `proximity`: `no_proximity_share`, `has_not_used_proximity_share`,
  `has_used_proximity_share`, `cbg_opportunity_share`,
  `opportunity_baseline_lucky_share`
- `rtt_quality`: `pair_soi_violation_share`, `anycast_suspect_share`,
  `closest_is_shortest_ping_share`

Observed contrast worth carrying into the paper: as7018 `no_proximity_share`
0.372 / `cbg_opportunity_share` 0.513 vs as01 0.0 / 0.363 — the operator VP
fleet is dense enough that *every* target has a discriminative VP.

### `<basename>_eval_clusters.csv`

`cluster_id`, `centroid_lat/lon`, `n_members`, `radius_km`, `is_singleton`,
`cell_gap_km`, `neighbor{1..5}_cluster_id`, `neighbor{1..5}_km` — the
answer-space margin structure (paper §7.4's Delaunay-neighbour role, done as
k-nearest instead).

### `<basename>_dataset_stats.json` (`eval_dataset` only)

Deep nested rollup. Top keys: `csv`, `n_obs`, `params`
(`cluster_radius_km`, `local_density_radius_km`, `top_n_asns`), `vp_nodes`,
`target_cluster_nodes`, `vp_to_cluster_edges`. Every numeric leaf is a
`{n,min,max,mean,std,percentiles{p1..p99}}` block. Covers ASN composition,
`geographic_spread` (bbox, `radius_of_gyration_km`, `convex_hull_area_km2`),
`pairwise_distance_km`, cluster counts/radii/local density, edge `density`,
and all four degree distributions. This is the §7.3 metric list, mostly
precomputed — **check here before computing any node/edge geometry.**

### `<basename>_stats.json`

Thin flow-level counts: `n_flows`, `n_unique_vps`, `n_unique_targets`,
`vp_occurrences`, `target_observations`, `has_weight`.

> `has_weight: false` on **all four** runs — no traffic weights present, so the
> paper's traffic-weighted views (§8.1) cannot be produced from these outputs.

### `<basename>_{vp,cluster}_mesh_km.csv`

Full square great-circle distance matrices. `vp_mesh_km.csv` is indexed and
headed by `vp_id`; `cluster_mesh_km.csv` by `cluster_id`.

### PNG/HTML

`_cluster_map.png`, `_occurrence_cdf.png` (all runs); `_flow_map.html`
(as7018 `eval_dataset` only). Presentational.

## 6. `bench_eval/` — MTL forensics (as7018 only)

`<combo>_bench_per_target.csv`, one row per (combo, target), 47 cols, written
by `eval_bench_results.py` as post-hoc scoring of `targets.parquet`.

| group | columns |
| --- | --- |
| identity | `combo_id`, `target_id`, `target_lat`, `target_lon`, `status`, `error_km`, `n_obs` |
| phase counts | `n_ltd_success`, `n_mtl_participants`, `mtl_success`, `mtl_intersection_kind`, `recompute_matches` |
| **region geometry** | `mtl_area_km2`, `mtl_n_components`, `truth_in_region`, `exclusion_reason` |
| participant stats | `part_{min,mean,med}_dist_km`, `part_{min,mean,med}_rtt_ms`, `part_{min,mean}_infl`, `n_part` |
| includer/excluder split | `includer_mean_{dist_km,rtt_ms}`, `excluder_mean_{dist_km,rtt_ms}`, `n_includers`, `n_excluders`, `n_outer_exclusions`, `n_inner_exclusions` |
| baseline compare | `closest_vp_id`, `closest_vp_dist_km`, `sping_vp_id`, `sping_vp_dist_km`, `closest_is_sping`, `sping_vp_is_participant`, `cell_gap_km` |
| **leave-one-out** | `loo_computed`, `loo_n_tested`, `loo_trivial`, `loo_n_flips`, `loo_any_flip`, `loo_max_error_delta_km`, `loo_critical_vp_id` |

`mtl_area_km2` → paper §8.3's equivalent radius √(A/π). `truth_in_region` +
`exclusion_reason` → the EXCLUSIVE_REGION diagnostic. `loo_*` → single-VP
brittleness (the `finding_spherical_circle_brittle` mechanism).

`bench_eval/summary.parquet` is the per-combo rollup of the above.

## 7. Cross-run asymmetries — the main constraint on v3

| artifact | as01/02/03 | as7018 |
| --- | --- | --- |
| `bench_eval/` (region area, LOO, truth_in_region) | ❌ | ✅ |
| `cluster_scored/` + classification tables | ✅ | ❌ |
| shortest-ping baseline (`baseline.csv`) | ✅ | ❌ (derive from `eval_*`) |
| kNN answer-space columns | ✅ | ❌ |
| combos | 5 (published variants) | 16 (+11 ablations) |
| setup | `anchors_to_probes` | `probes_to_anchors` |

Consequences for v3:
1. **No single run supports both the classification metric and the region
   forensics.** Cross-dataset claims must either restrict to what both have
   (`targets.parquet` + `summary.parquet` + `eval_*`) or re-run the missing
   stage. The classification metric *is* derivable for as7018 from
   `clusters/assignments.csv` + `pred_lat/lon`; region area is *not* derivable
   for as01-03 without re-running `eval_bench_results.py`.
2. **Only 5 combos are common.** Restrict cross-dataset tables to
   `{million_scale_cbg, vanilla_cbg, octant_cbg_hull, octant_cbg_spl|octant_cbg,
   spotter_cbg}` — note the Octant-Spline id differs per run.
3. Setups differ, so VP/target roles are swapped between the two families. Not
   a bug (paper §7.3 declines the head-to-head), but never pool the two.
4. `has_weight=false` everywhere → no traffic-weighted results available.
