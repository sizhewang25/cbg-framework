# v3 Analysis Iteration 2 — Plan

## Background

Iteration 1 (commit `f6d305d`) built `scripts/analysis/v3/`: a HEALPix answer
space (§7.3/§7.4), distance-to-all-seeds classification scoring (§8.1), and
Venn/UpSet overlap figures. Running it end-to-end on the four benchmark runs
under `outputs/benchmark/v2/` surfaced five concrete problems, each confirmed
against real output rather than suspected:

1. **`accuracy_topN_success_only` is dead weight.** Added to expose the
   fallback cost, but `fallback_rate` and `accuracy_topN` already carry that.
   It doubles the width of `topn_accuracy.csv` for no decision.
2. **Top-N reports N ∈ {1,2,3,5}; only 1 and 3 are wanted.** The per-target
   parquet stores the full distance vector and `truth_seed_rank`, so narrowing
   the summary costs nothing and loses nothing — any N stays derivable.
3. **Overlap artifacts silently overwrite across top-N.** `--top-n` is honoured
   in `build_membership` and appears in the figure *title*, but all five output
   filenames are fixed module constants, so a top-3 run clobbers the top-1 files.
4. **The Venn is area-proportional and cannot render the data.** matplotlib-venn
   emits "Bad circle positioning" on every run because the sets are nested.
   Verified across all four runs: `all-CBG-correct ⊆ Shortest-Ping-correct ⊆
   ≥1-CBG-correct`, with `shortest_ping_only` = 0, 1, 0, 0. No area-proportional
   layout can draw that. The figure actually wanted is the aggregate one —
   Shortest-Ping vs "at least one CBG works" — which is the rescue-vs-regression
   trade RQ2 rests on.
5. **The answer space has never been looked at.** `meta.json` reports 17 seed
   pairs within one cell pitch on `as7018_us_test01` and 4 targets whose own
   cell seed is not their nearest seed. That is the §7.3 straddle cost, and it
   is hard to believe from a number.

## Context

- Working tree: `scripts/analysis/v3/` (`cli.py` = pure wiring; all bodies in
  `modules/`). Reference: `scripts/analysis/v3/README.md`, `SCHEMA.md`.
- Files to touch: `modules/classify.py`, `modules/venn.py`, new
  `modules/map_answer_space.py`, `cli.py`, `tests/`, `README.md`.
- Benchmark runs: `as01/as02/as03-260728-260802` (operator, 5 combos each) and
  `as7018_us_test01` (RIPE, 16 combos). Analysis outputs land in
  `outputs/analysis/v3/<run_id>/` which is **gitignored**.
- **Reuse targets found in the repo** (do not reimplement):
  - `scripts/visualization/cluster/plot_ground_truth_clusters.py` —
    `plot_targets_only(df, out_path, *, extent, target_size)` is the closest
    static-map starting point; also holds `_plot_voronoi_underlay`.
  - `scripts/visualization/cluster/plot_targets_vps.py` — target/VP dot styling
    and `US_MAINLAND_EXTENT = (-125.0, -66.0, 24.0, 50.0)`.
  - The cartopy "house style" basemap block, identical in 4 modules:
    `ccrs.PlateCarree()` + OCEAN `#eaf2f8` / LAND `#f6f4ef` / COASTLINE 0.4
    `#999` / BORDERS 0.25 `#ccc`. Map convention: `dpi=140`,
    `bbox_inches="tight"`, `plt.close(fig)`.
  - `scripts/analysis/plot_stratification.py` — lazy cartopy import pattern.
  - `scripts/visualization/cluster/voronoi.py` — `build_landmass_voronoi`,
    `resolve_landmass`, `LandmassVoronoi.focus_extent` if Voronoi is wanted later.
- **API facts verified this session**: `matplotlib_venn.venn2_unweighted` /
  `venn3_unweighted` exist in 1.1.2 and accept `subset_label_formatter`.
  `astropy_healpix.HEALPix(nside, order="nested").boundaries_lonlat(pix, step)`
  returns the 4 cell corners at `step=1` — this is what draws the grid cells.

## Goals

1. `topn_accuracy.csv` carries exactly: `method, n_targets, n_solved,
   n_fallback, n_error, fallback_rate, accuracy_top1, accuracy_top3,
   error_km_p50, error_km_p90`.
2. Overlap artifacts are top-N-suffixed, so top-1 and top-3 coexist on disk.
3. The default Venn is an unweighted 2-set Shortest-Ping vs ≥1 CBG figure with
   counts and percentages in the regions, and no positioning warning.
4. A static single-panel map of the answer space exists per run, showing
   occupied cells, per-seed fill colour, target dots and seed markers.
5. The fallback-as-failure invariant is preserved: `as01` Vanilla still reads
   `accuracy_top1 = 0.411` (164/399).

## Approach

**1 · Trim `topn_accuracy.csv`** (`modules/classify.py`)
`DEFAULT_TOPN` → `(1, 3)`. Drop the `accuracy_top{n}_success_only` column.
Rename `error_km_p{50,90}_success_only` → `error_km_p{50,90}` but **keep the
`solved` mask behind them** — a FALLBACK row's `error_km` is the shortest-ping
VP's error, not a CBG error, so pooling it would corrupt the error distribution
the way crediting it corrupts accuracy. Say so in the docstring so the shorter
name is not read as "over all rows". Update the `fallback_policy` string in
`manifest.json`, which names the removed column.

**2 · Top-N-suffixed filenames** (`modules/venn.py`)
Replace the five filename constants with one `_artifact(stem, ext, top_n)`
helper returning `f"{stem}.top{top_n}.{ext}"`. Route every write in
`render_overlap` through it. No CLI change — `--top-n` already exists.

**3 · Unweighted 2-set Venn** (`modules/venn.py`)
New `plot_sp_vs_cbg_venn(...)`: collapse the membership matrix to
`Shortest-Ping` and `≥1 CBG` (`.drop(columns=[SHORTEST_PING]).any(axis=1)`),
draw with `venn2_unweighted`, label regions `n\n(pct%)` via
`subset_label_formatter`. Subtitle **must** state the CBG pool size. Wiring:
`overlap_venn.top<N>.png` always (the 2-set figure);
`overlap_upset.top<N>.png` when ≥3 methods; `overlap_venn_methods.top<N>.png`
only when `--venn-method` is passed. Delete the auto-chosen-triple heuristic —
given the nesting it picks an uninformative triple. Switch the retained
`plot_venn` to the `*_unweighted` draws for the same reason.

**4 · Static answer-space map** (new `modules/map_answer_space.py`, command
`plot-answer-space`, registered in `cli.py::_COMMAND_MODULES`)
Single panel, occupied cells only, layered bottom-up: house-style basemap →
occupied HEALPix cells as filled polygons from `boundaries_lonlat`, one colour
per seed cycling `tab20` → target dots → seed centroid markers. Extent
precedence: `--extent` › `--us-only` (`US_MAINLAND_EXTENT`) › target bbox
padded ~8%. Output
`outputs/analysis/v3/<run_id>/target-answer-space/answer_space_map.png`.
Input via `load_answer_space`; path via `RunPaths.answer_space_dir`.

**5 · Tests and docs** — rewrite the two `test_classify.py` assertions that name
the removed column; add tests for the 2-set collapse, the missing-baseline
error, and `_artifact` name distinctness; add a minimal map smoke test; update
`README.md`.

## Caveats

- **The map must answer "how is the clustering determined?" in the figure.**
  There is no clustering algorithm: occupied cell ↔ seed is a **bijection**,
  two targets share a class iff `ang2pix` returns the same cell. Put `nside`,
  cell pitch, K and n_targets in the caption so the figure cannot be mistaken
  for the older agglomerative `clusters/` space (radius-capped complete linkage).
- **The CBG pool size changes the answer on `as7018_us_test01`**, whose default
  method set is 16 combos including 11 ablation arms: `cbg_only` reads 40 over
  all combos vs 31 over the published 5. Without the subtitle the Venn is
  ambiguous. `--method` is how a caller restricts the pool.
- At `nside=128` a cell is ~51 km ≈ 0.5° — small at continental scale. Use a
  saturated fill and a dark edge so occupied cells read. `tab20` only has 20
  colours and `as7018` has K=27, so fill colour is an identity cue, not a scale.
- **Not drawing the Voronoi partition.** It is a different shape from the cells
  (a seed's region extends well beyond its 51 km cell) and would muddle a figure
  whose subject is the quantizer.
- Stale un-suffixed `overlap_*.png` from iteration 1 will linger unless
  `outputs/analysis/v3` is removed before regenerating.
- Do not disturb the fallback-as-failure invariant while editing `topn_summary`;
  `solved` still gates `n_solved` and the error percentiles.
