# v3 Analysis Iteration 2 — Todo

## Phase 0: Setup & Discovery
- [x] Re-read `modules/classify.py::topn_summary` and `modules/venn.py::render_overlap` to confirm every `_success_only` and filename-constant reference (`grep -rn "success_only\|_CSV\|_PNG" scripts/analysis/v3/`)
- [x] Confirm the baseline invariant before touching anything: `as01` Vanilla `accuracy_top1 == 0.411`, 106 FALLBACK rows all with `truth_seed_rank == 0`
- [x] Read `scripts/visualization/cluster/plot_ground_truth_clusters.py::plot_targets_only` and `plot_targets_vps.py` for the basemap block, dot styling and `US_MAINLAND_EXTENT`

## Phase 1: Trim the summary
- [x] `DEFAULT_TOPN = (1, 3)` in `modules/classify.py`
- [x] Drop `accuracy_top{n}_success_only` from `topn_summary`; keep the `solved` mask
- [x] Rename `error_km_p{50,90}_success_only` → `error_km_p{50,90}`, documenting in the docstring that they are computed over SUCCESS rows only and why
- [x] Update the `fallback_policy` string in the `manifest.json` written by `classify_cmd`

## Phase 2: Top-N-suffixed overlap artifacts
- [x] Add `_artifact(stem, ext, top_n)` to `modules/venn.py`; delete the five fixed filename constants
- [x] Route every write in `render_overlap` through it (membership, intersections, pairwise, venn, upset)

## Phase 3: Unweighted SP-vs-CBG Venn
- [x] Add `plot_sp_vs_cbg_venn(...)` using `venn2_unweighted` with a `subset_label_formatter` emitting `n\n(pct%)`
- [x] Raise a clear error when the `shortest_ping` column is absent from the membership matrix
- [x] Put the CBG pool size in the subtitle (e.g. "≥1 of 5 CBG variants")
- [x] Make it the default `overlap_venn.top<N>.png`; move the explicit per-method figure to `overlap_venn_methods.top<N>.png` behind `--venn-method`
- [x] Delete the auto-chosen-triple heuristic; switch retained `plot_venn` to `venn2_unweighted` / `venn3_unweighted`

## Phase 4: Static answer-space map
- [x] New `modules/map_answer_space.py` with `register(app)` exposing `plot-answer-space`; add to `cli.py::_COMMAND_MODULES`
- [x] Draw occupied cells from `HEALPix.boundaries_lonlat(pix, step)`, one `tab20` colour per seed
- [x] Overlay target dots and seed centroid markers; use the repo's cartopy house-style basemap (lazy import)
- [x] Extent precedence: `--extent` › `--us-only` › target bbox padded ~8%
- [x] Caption states nside, cell pitch, K, n_targets, and that occupied cell ↔ seed is a bijection (no clustering algorithm)
- [x] Write to `<run>/target-answer-space/answer_space_map.png` at `dpi=140`

## Phase 5: Tests & docs
- [x] Rewrite the two `test_classify.py` assertions naming `accuracy_top1_success_only`, keeping the fallback-never-correct guarantee under test
- [x] Add tests: 2-set collapse partitions to 4 regions summing to n; collapse raises without the baseline column; `_artifact` names differ across top-N
- [x] Add a minimal `test_map_answer_space.py` smoke test (PNG written, non-empty)
- [x] Update `README.md`: policy paragraphs, output tree with top-N suffixes and `answer_space_map.png`, `plot-answer-space` row in the module table

## Phase 6: Verification
- [x] `pytest scripts/analysis/v3/tests/ -q` green
- [x] `rm -rf outputs/analysis/v3` then regenerate: build-answer-space, classify, plot-venn at `--top-n 1` and `--top-n 3`, plot-answer-space (all `--all-runs`)
- [x] `topn_accuracy.csv` has exactly the 10 target columns, no `_success_only`
- [x] `as01` Vanilla still `accuracy_top1 = 0.411` — invariant preserved
- [x] `overlap_upset.top1.png` and `overlap_upset.top3.png` coexist; top-3 membership has strictly more `True` cells
- [x] `as01` Venn shows `0 / 254 / 75 / 70` with no "Bad circle positioning" warning
- [x] `as7018` Venn subtitle names the pool; restricting `--method` to the published 5 moves `≥1 CBG only` from 40 to 31
- [x] Eyeball `as7018` vs `as01` answer-space maps for the straddle contrast (K=27 with 11 singletons vs K=18)
