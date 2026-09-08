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

## Phase 0g: cross-dataset band grid (2026-09-07)
- [x] Extract `_draw_panel` from `plot_bands`, so the per-run and cross figures cannot drift in band geometry, alpha, median rule or share denominator
- [x] Verify the extraction is byte-identical: all six per-run PNGs and band CSVs re-render to the same md5
- [x] `load_cross_points` — each run through the same `load_points`/`band_table`, tagged `dataset` + `run_id`, nothing pooled
- [x] `plot_cross_bands` — methods down the rows, datasets across the columns; dataset headers in ink on the top row, method names as row labels in their hue
- [x] Keep the 4.9 in panel width so the log x axis is the same length as the per-run figure's
- [x] `CROSS_ROW_CHROME_INCHES = 0.42` against `PANEL_CHROME_INCHES = 1.7` — a shared x axis and a row label need padding and nothing else
- [x] `footnote_text` / `panel_header` hoisted, so both figures make the same promises about the same marks
- [x] Two or more `--run-id` switches to the grid (`plot-venn`'s convention); `--all-runs` stays per-run; mixed setups refused via `cross.guard_one_setup`
- [x] `--out-dir` on a single run refused by name rather than silently ignored
- [x] Verified: all 18 correct-band shares equal `accuracy_top1` and all 18 cumulative-through-band-3 equal `accuracy_top3`, delta 0.000000
- [x] 13 tests: the per-dataset denominator, the dataset key, column sort, concat integrity, grid orientation, band-height parity, render, output path, panel header, footnote clause
- [x] `configs/cross-as01-as03.yaml` gains `plot-error-vs-rank` / `plot-error-vs-cells`; README pipeline step 3g, `_cross/` tree and section prose updated

## Phase 0h: pooled layout, per review (2026-09-07)
- [x] `--layout pooled|compare` (repeatable), default `pooled`; `pooled` merges the runs' targets and reuses the six-panel `plot_bands` unchanged
- [x] `pool_counts` sums each method's row counts over the runs that carry it, so a missing variant is scored on the targets it ran on
- [x] `guard_disjoint_targets` refuses to pool runs sharing a `target_id` — it would count them once per run in the denominator and draw them twice
- [x] `weighting_check` records the pooled micro-average against the datasets' macro mean per method: max delta 0.0064, so the target weighting is measured not waved away
- [x] "shares are target-weighted" goes in the pooled subtitle, not only the manifest
- [x] `layout_stem` keeps both figures on disk: `error_vs_rank.<grid>.png` pooled, `error_vs_rank_by_dataset.<grid>.png` compare
- [x] One load serves both layouts — pinned by counting `load_points` calls
- [x] Verified: pooled correct-band shares equal the target-weighted mean of the three runs' `accuracy_top1` to 5e-5 (the band CSV's rounding)
- [x] 9 more tests; `configs/cross-as01-as03.yaml` asks for both layouts; README and module docstring rewritten around the two questions

## Phase 0i: dataset-type headline table + outcome bars (2026-09-07)
- [x] Row model grows a scope level: kind-major, each group led by its aggregate; rows carry a run *set*, so `pending` stays one rule (`n_runs == 0`) across both scopes
- [x] Aggregate is a target-count micro-average; breakdown rows print `accuracy_topN` **verbatim** — recomputing moves 3 of 36 cells a digit (as01 Spotter 0.389→0.388) and breaks the "cannot disagree with table-accuracy" invariant
- [x] Counts reconstructed as `round(acc * n)`, pinned as the *unique* integer for that rate, and cross-checked against `overlap_membership.top{1,3}.csv` — delta 0 on all 36 cells
- [x] `†` on a pooled winner that does not lead every dataset it pools (top-3 Octant-Hull leads 1 of 3); `‡` on a cell pooled over fewer datasets than its row
- [x] `render_markdown` groups on `row_index` — `groupby(["dataset", ...])` drops null keys, i.e. every aggregate row, silently
- [x] Single dataset suppresses the aggregate, and its breakdown rows carry the kind themselves (`AS01 MESH`) since an indent needs something to indent under
- [x] `guard_disjoint_targets` hoisted `figure_error_scatter` → `cross.py` with an `ids_by_run` + caller-supplied `remedy` signature; a table command must not import matplotlib
- [x] `guard_one_setup` now applies here — the aggregate voids `table-accuracy`'s "no averaging can happen" exemption; `--allow-mixed-setups` overrides
- [x] Target ids read from `*_seed_distances.parquet` (a `classify` output in the same dir), not `target_labels.csv` (a `build-proximity` one) — no new command dependency
- [x] Two defects fixed: `cross_dir` leaked the dataset key and forked output on a weighted run; `row_plan` silently dropped a second run for one dataset; `n_reserved_rows` counted every aggregate
- [x] `accuracy_rows(include_counts=True)` — opt-in, so `table-accuracy`'s published column set is unchanged
- [x] New `modules/figure_outcome_bars.py` + `plot-outcome-bars`: correct/wrong/fallback/error stacks, hue = method, hatch = dataset type, ghost outlines for the uncollected weighted half
- [x] Counts and pooling imported from `headline_table`, so figure CSV == table CSV by construction (verified delta 0)
- [x] `--layout pooled|compare`; `guard_partition` asserts the four segments sum to `n_targets` per bar
- [x] `test_cross.py` (8, first direct test of the guard), `test_figure_outcome_bars.py` (13), +15 in `test_headline_table.py` — suite 544 → 595
- [x] `configs/cross-as01-as03.yaml` gains `plot-outcome-bars`; README pipeline 3e/3e2, module table, `_cross/` tree and "paper's own layout" rewritten
- [x] §8.1 body gets the rendered top-1 table and the figure callout
- [x] Per review: colour is the **outcome** (green correct / red wrong / grey failed), not the variant; `n_error` folds into `failed` (both are failures to answer) while the CSV keeps the split
- [x] Fills picked against checks not by eye — monotone lightness L* 34/54/73, worst deuteran/protan dE 16.8, dE 11.8 from the nearest variant hue; label ink chosen per fill by luminance
- [x] Every segment carries its own percentage to one decimal, y axis switched to percent to match; `MIN_LABEL_SHARE` guards a sliver too thin to hold one
- [x] Tick labels left neutral — hueing them would put "Octant-Hull" in green and "Spotter" in red directly under green and red segments
- [x] Bars ordered best-first by pooled correct rate, one order shared across `compare` panels so an x position means the same method everywhere
- [x] Prose removed (subtitle + footnote); two horizontal untitled legends tight under the title, placed by measurement; pending kinds get a dashed swatch matching their bars
- [x] `PROVISIONAL_WEIGHTED` — hard-coded top-1 rates from an earlier run so the mesh-vs-weighted layout can be reviewed now; gated on the row having no run, so a real `--weighted-run-id` supersedes it with no flag
- [x] No denominator invented (n stays NaN → never marked best) and no top-3 invented (appendix weighted row stays reserved)
- [x] Table marks it `§` with a footnote; figure CSV carries a `provisional` column; manifest gains `provisional_rows`. The PNG itself does **not** mark it — captions must
- [x] Bars ordered best-first by pooled correct rate, one order shared across `compare` panels so an x position means the same method everywhere
- [x] Prose removed (subtitle + footnote); two titled legends instead, one per encoding channel; the uncollected campaign names itself in its legend entry

## Phase 0j: geometry-vs-routing scatter (2026-09-07)
- [x] New `modules/figure_proximity_inflation.py` + `plot-proximity-inflation`: x = `sping_vp_to_tg_seed_km` (log), y = `min_inflation`, one dot per target
- [x] Colour is the dataset type — mesh grey, traffic-weighted red — and no variant hue appears, since no method runs in this figure
- [x] Both columns read from `build-proximity`, neither recomputed: `min_inflation` is `eval_source`'s constant, `sping_vp_to_tg_seed_km` is the VP `classify` scores the baseline on
- [x] `share_proximate_sping_vp` in the summary CSV == the headline table's Shortest-Ping column (0.637 / 0.369 / 0.432, 0.476 pooled) — the seam is a number, not an assertion
- [x] Reference marks: the `y = 1` speed-of-internet floor, and the seed margin as a p25-p75 band rather than a line, since the threshold is per target
- [x] Marginal histograms on both axes, density-normalised so a subset series is comparable to the set it came from
- [x] `legend_kinds` keeps the uncollected traffic-weighted series in the key, drawn as an outlined marker
- [x] Guards: an all-NaN `min_inflation` refused by run id naming `--source-csv`; a VP on its seed clamped to a 0.1 km floor and counted, never `log(0)`
- [x] `_spearman` without a scipy import, verified against `scipy.stats.spearmanr` to 12 dp on all four series
- [x] `--layout pooled|compare`; `test_figure_proximity_inflation.py` (23) — suite 612 -> 635
- [x] `configs/cross-as01-as03.yaml` gains `plot-proximity-inflation`; README module table, pipeline 3e3, `_cross/` tree and a "Why the dataset types differ" section
- [x] §8.1 gets the four definitions (VP proximity · discrimination power · shortest-ping VP · min-RTT inflation), the formation paragraph, a figure slot and four numbered readouts
- [ ] Flagged, not fixed: §8.2 uses "discriminative" for the argmin rule, where the code and the new §8.1 text use it for the half-margin guarantee
- [x] Second pass: `--x-metric closest|sping` (repeatable, default `closest`) — the shortest-ping VP's distance is chosen *by* RTT, so it carries the y axis; against the nearest measured VP ρ falls from 0.54 to -0.02
- [x] One points CSV and one summary carry both metrics, so `spearman_rho_closest` / `spearman_rho_sping` compare over the same targets and the same y
- [x] `share_inside_margin_<metric>`, all four diamond flags, and `sping_is_closest_vp` (6.5% by id against 47.6% by class) in the summary
- [x] A row missing either x metric is dropped from both, so the two axes never carry two denominators
- [x] §8.1 readouts rewritten: opportunity is saturated (96.8% ceiling), selection is what fails (47.6%), the two axes are independent. The earlier "co-occur" claim is withdrawn as an axis artifact

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
