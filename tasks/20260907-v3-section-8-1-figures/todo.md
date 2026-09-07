# §8.1 Figures (v3) — Todo

## Phase 0: Shared style groundwork
- [ ] Add `ORDINAL_STEPS`, `DIVERGING_BLUE`, `DIVERGING_RED`, `DIVERGING_MID` to `diagram/common/palette.py`, each with its validator verdict in the docstring the way `_VARIANT_HUES` records its audit
- [ ] Hoist `_C_AXIS` from `pareto.py:392` into `palette.py` and have `pareto.py` import it, removing one of the duplicated ink blocks
- [ ] Add `_annotate_multipanel(fig, *, title, subtitle, note)` to `diagram/common/draw.py` — the existing `_annotate_figure` is axes-bound and cannot title a 4-panel figure
- [ ] Pin the ramps in tests: monotone lightness, single hue, `_C_AXIS` resolves to one value, the six variant hues unchanged

## Phase 0b: §8.1 headline table (done 2026-09-07)
- [x] Move `short_label` from `pareto.py` into `diagram/common/labels.py` and re-export it, so a table command does not import matplotlib
- [x] New `modules/headline_table.py` + `table-headline`: datasets down the rows, methods across the columns
- [x] Best-in-row mark uses a tie rule (within 1 SE of the row's best), not an argmax
- [x] Reserved `<DATASET> WEIGHTED` row per dataset, printing `—`, with the pending reason in the caption
- [x] Explicit `--weighted-run-id <dataset>=<run_id>` pairing — `short_dataset` cannot infer it
- [x] Fallback rate in-cell and only where non-zero
- [x] Body table at top-1, appendix table at top-3
- [x] `test_headline_table.py` — 32 tests
- [x] `table-headline: {}` in `configs/cross-as01-as03.yaml`; config and explicit flags byte-identical
- [x] README: pipeline step 3e, the `_cross/` layout, and a "paper's own layout" section
- [ ] Insert the rendered top-1 table into draft §8.1 (lines 342-348) and move the long table to the appendix

## Phase 0c: error-distance CDF (done 2026-09-07)
- [x] Extract `io.solved_mask` out of `classify.topn_summary` — the CDF is its second caller, and it carries the BASELINE case a `status == "SUCCESS"` filter drops
- [x] Hoist `_C_AXIS` from `pareto.py` into `palette.py`; `pareto` imports it
- [x] One shared `PUBLISHED_METHODS` in `labels.py`, used by both `table-headline` and `plot-error-cdf`
- [x] New `modules/figure_error_cdf.py` + `plot-error-cdf`: per-method CDF of `error_to_target_km`, log x, solved rows only, Shortest-Ping grey dashed
- [x] Reuse `confusion.load_scored` rather than adding a third parquet reader
- [x] Log floor 0.1 km, not 1 km — the v2 floor clamped 8-20 rows/method
- [x] Percentiles use numpy's default, matching `classify` — `nearest` disagreed with `topn_accuracy.csv` by 0.7-1.3 km
- [x] Guide verticals in neutral ink, not the v2 plotter's green/orange/red (= Octant-Hull/Vanilla/Spotter hues)
- [x] `%g` tick formatter — `ScalarFormatter` rendered the 0.1 km decade as `0`
- [x] `test_figure_error_cdf.py` — 16 tests
- [x] `plot-error-cdf: {}` in the three per-run configs
- [x] README: pipeline step 3f and an "error half" section
- [ ] Insert the CDF into the paper draft (§8.3's `TODO: error distance`)

## Phase 0d: error vs class-error scatter (done 2026-09-07)
- [x] New `modules/figure_error_scatter.py` + `plot-error-vs-cells`: x = error distance (log), y = class boundaries crossed (0 = correct cell), one panel per method
- [x] Recompute crossings for **all** solved rows — `confusion_pairs.csv` keeps only wrong rows, so its `y == 0` column is missing entirely
- [x] Verified against `confusion_pairs.csv` on shared rows: crossings delta 0, error delta 0
- [x] Margin reference read per run from `seeds.csv` (151/172/160 km), not hardcoded
- [x] Two disagreement regions counted per method, disjoint by construction (`>` far, `<=` near)
- [x] Seeded jitter so two renders of one dataset do not look like two datasets
- [x] Share the error axis, row filter and distance column with `plot-error-cdf`
- [x] `test_figure_error_scatter.py` — 16 tests
- [x] `plot-error-vs-cells: {}` in the three per-run configs; README step 3g + a section
- [ ] Insert into the paper draft (§2.4(a)'s disagreement claim, and §8.1's dense-region subsection)

## Phase 0e: rework the error-vs-class figures (2026-09-07)
- [x] Replace jitter+bins with rug bands: one thin line per point in the method's hue, alpha accumulation for density
- [x] Switch the denominator to `n_targets` so band 0 == `accuracy_top1`; label the shortfall as the fallback rate
- [x] Row-share % on each band's right edge
- [x] Annotate each band's median error above its rule
- [x] Drop the margin rule, both disagreement annotations, the footnote line and the two orphaned summary columns
- [x] Add the `tg_seed_rank` y mode in the same module, as its own CLI command
- [x] Rename the y axes: "Cells away from the true class" / "Top cell index of the true class"
- [x] Make the rank axis 1-indexed with four bands (1/2/3/4+), so `index <= N` is top-N
- [x] Pin band-0 share == `accuracy_top1` and cumulative rank <= 2 == `accuracy_top3` on all runs
- [x] Update `test_figure_error_scatter.py`, README and the three per-run configs

## Phase 0f: band layout, per review (2026-09-07)
- [x] Lay the bands out as contiguous grid rows — `ylim (0, n_bands)`, separators between rows, bottom row resting on the x axis
- [x] Bare-integer band ticks; what band 1 means lives in the axis label and footnote
- [x] Tick at the band's centre, not the row's, since it labels the data and not the gutter
- [x] Median rule exactly the band's height, so it reads as one of the targets rather than an annotation layer
- [x] Gutter above each band only — `BAND_BOTTOM = 0.0`, so a band rests on its row's floor
- [x] Compact y: `ROW_INCHES` 0.78 → 0.50 and `PANEL_CHROME_INCHES` 2.0 → 1.7, for 24% less figure height
- [x] Derive every y coordinate from `band_span` / `band_centre` so the row invariants are testable
- [x] Pin the four row invariants plus the gutter's physical floor (≥ 0.13 in for the 7 pt readout)

## Phase 1: Figure D's data step (build first — most load-bearing)
- [ ] Add `COVARIATES = ("tg_seed_nearest_vp_km", "min_inflation")` and `breakdown_by_covariate(membership, labels, *, weights=None)` to `breakdown.py`
- [ ] Reuse `confusion.density_bins` for the quantile binning rather than re-deriving edges
- [ ] Emit `accuracy_by_covariate.csv` (`method, method_label, covariate, bin, lo, hi, n_targets, n_correct, accuracy, top_n, covariate_p50`) from `breakdown-accuracy`
- [ ] Add `--covariate-bins` (default 5) and make it degrade to fewer bins when a run cannot support them
- [ ] Test: bins sum to the target count per (method, covariate, N); a single-valued covariate collapses to one bin and reports it; `weights=None` reproduces unweighted counts exactly

## Phase 2: The figure modules
- [ ] `modules/figure_covariate.py` → `plot-covariate`; log x for VP distance, linear for inflation, per-bin `n`, direct end-labels, SoI dashed over Shortest-Ping solid, crossover annotated
- [ ] `modules/figure_accuracy.py` → `plot-accuracy`; reads `accuracy_table.build()`, takes `--run-id` repeatably, four panels shared x, as7018 ruled off and tagged `probes_to_anchors`
- [ ] `modules/figure_strata.py` → `plot-strata`; two files (`strata_accuracy.png`, `flag_effect.png`)
- [ ] `modules/figure_confusion.py` → `plot-confusion`; four ordinal classes on the left, symlog `boundary_margin_km` CDF on the right
- [ ] Register all four in `cli.py:_COMMAND_MODULES` after `accuracy_table`, before `venn`
- [ ] Route every filename through `labels.artifact_name` so the top-N suffix is never dropped
- [ ] Emit a CSV twin beside every PNG (this is what discharges the contrast WARN's table-view obligation)

## Phase 3: Degenerate-case and grid guards
- [ ] `n_targets == 0` stratum renders as a gap with an `n=0` tick, never a 0.0 point
- [ ] `zero_variance` φ cell renders as a hatched "no variance" cell, never the diverging midpoint
- [ ] `is_tautological` cell gets a heavy border and a `†` footnote
- [ ] `seeds_crossed == 0` segment set apart by a surface gap plus hatch
- [ ] Figures B/C/D refuse a grid whose inputs are absent, naming the missing file
- [ ] as7018 defaults to the six published variants; `--method` opts the 11 arms back in
- [ ] One test file per figure module pinning each of the above

## Phase 4: Configs and verification
- [ ] Add `plot-strata` / `plot-confusion` / `plot-covariate` sub-blocks to the four per-run configs, and `plot-accuracy` to `configs/cross-as01-as03.yaml`
- [ ] Run the full pipeline on as02 (the run with contrast on every axis)
- [ ] Run on as01 / as03 / as7018 — the degenerate runs must render, not crash
- [ ] `python -m pytest scripts/analysis/v3/tests/ -q` (442 passing before this work)
- [ ] Re-run `validate_palette.js` on both ramps against `#ffffff` and record the verdict
- [ ] **Open every PNG and look at it** — the validator checks colour, not layout: label collisions, overplotted Shortest-Ping/SoI, and whether the `n=0` / `no variance` marks read as "no data" rather than "zero"

## Phase 5: Paper draft
- [ ] §8.1 L342-348: replace `【NEED table】` with Figure A; move the table to the appendix
- [ ] §8.1 L349-359: mark the traffic-weighted sub-sections as explicit TODO (no data — U3)
- [ ] §8.1 L360-367: rewrite to the v3 vocabulary, and drop the inherited gloss that `geometry_only` is unwinnable
- [ ] §8.1 L369-372: point at Figure C, naming the `0` bin as the fallback case
- [ ] §8.1 L375-379: point at Figure A's fourth panel and state the three-way confound
