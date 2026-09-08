# §8.1 Figures (v3) — Report

**Status**: In Progress
**Created**: 2026-09-07
**Last Updated**: 2026-09-07

## Summary

Three artifacts built and verified: the §8.1 **headline table**
(`table-headline`), the **error-distance CDF** (`plot-error-cdf`) and the
**error-vs-class-error scatter** (`plot-error-vs-cells`). The four originally
planned figures are still planned only. The design is recorded in [plan.md](plan.md)
and was settled against the data rather than in the abstract — two pilots were
run before any figure was specified, and one of them reordered the figure set.

### Verification — `table-headline`

| check | result |
| --- | --- |
| `pytest scripts/analysis/v3/tests/` | **485 passed** (453 before, +32 new) |
| 18 shared cells vs `table-accuracy` at top-1 and top-3 | **max abs delta 0.0** on accuracy and fallback rate |
| `--config` vs explicit flags | **byte-identical** on both `.md` renders |
| `--weighted-run-id` without `=` | refused, naming the expected form |
| reserved rows | `accuracy` NaN, `pending` true, `is_best` false; render `—`, never `0.000` |

## Findings

### The rendering layer is the only gap

The predecessor task left every §8.1 table on disk and 442 tests passing.
`breakdown-accuracy`, `confusion-density` and `table-accuracy` emit CSV and
markdown only; there is no plotting code for any of them. The overlap
(Venn/UpSet/ring/Euler), Pareto and cartopy-map families are the only figures
in the tree.

### Figure D was ranked fourth and is actually first

Piloted on as02 before speccing anything. Top-1 accuracy by
`tg_seed_nearest_vp_km` quintile (~80-88 targets/bin):

| VP→seed km | Sh.-Ping | SoI | Vanilla | Oct-Hull | Oct-Spline | Spotter |
| --- | --: | --: | --: | --: | --: | --: |
| 4-11 | **0.659** | 0.659 | 0.375 | 0.534 | 0.432 | 0.261 |
| 17-24 | 0.481 | 0.481 | 0.753 | 0.558 | 0.481 | 0.143 |
| 26-42 | 0.414 | 0.414 | 0.195 | **0.874** | 0.655 | 0.586 |
| 43-92 | 0.250 | 0.250 | 0.125 | 0.750 | **0.875** | 0.450 |
| 115-165 | 0.012 | 0.000 | 0.062 | **0.525** | 0.425 | 0.612 |

Three findings in one panel: a **crossover** at ~25-40 km (below it the
baseline wins, above it Octant-Hull wins by 2-40×) which is §9's
when-to-use-what measured rather than asserted; the baseline's **monotone total
collapse** 0.659 → 0.012 against Octant-Hull never dropping below 0.525; and
**SoI tracking Shortest-Ping to three decimals in every bin but the last**,
which makes the "baseline in disguise" finding visible instead of tabulated.

`min_inflation` separates too, despite spanning only 1.35-2.04, and
non-monotonically: Shortest-Ping 0.511 → 0.821 → 0.427 → 0.077 → 0.014.

### Figure C's stack has exactly four classes, and one of them is not a mistake

Piloted `seeds_crossed` across all four runs at top-1. No run needs a fifth
class beyond `{0, 1, 2, 3+}`, which matches the 4-step ordinal ramp exactly.

The `0` bin exists **only for Vanilla CBG** and is large — 47.5% / 26.0% /
29.3% / 24.0% of its wrong rows on as01 / as02 / as03 / as7018. It is not a
near miss: it is a fallback whose class was right. On as01 all 106 of Vanilla's
fallbacks have `tg_seed_rank == 0`, so the variant scores 0.411 where crediting
them would give 0.677.

### Colour: the repo had no ordinal or diverging palette

Three of the four figures need encodings `palette.py` does not carry. Validated
with the dataviz skill's `validate_palette.js` against the repo's **real**
surface (`#ffffff`, not the skill's default `#fcfcfb`):

| role | steps | verdict |
| --- | --- | --- |
| ordinal | `#86b6ef` `#3987e5` `#256abf` `#104281` | ALL PASS (light end 2.11:1, hue spread 3°) |
| diverging red arm | `#e87b7a` `#d94d4c` `#b53433` `#7d2221` | ALL PASS (light end 2.78:1, hue spread 4°) |

Re-validating the existing six variant hues produced a constraint that shaped
the layouts: they pass the *adjacent* pairlist clean (ΔE 9.1) but only **WARN**
under `--pairs all` (ΔE 6.9 deutan, Spotter ↔ SoI). So Figure D — the only
panel with all six series in one coordinate space — needs direct end-labels,
not a legend alone. A first-pass diverging spec starting each arm at the
neutral midpoint **failed** the light-end contrast check; the arms start one
step in from gray instead.

### Two claims from the exploration that did not survive checking

- **`octant_cbg` on as7018 is not a different algorithm from the operator runs'
  `octant_cbg_spl`.** An exploration pass reported a "label collision" needing
  a footnote. Reading both runs' `run.json` shows identical configuration
  (`bounded_spline` / `fit_spline=True` / `planar_annulus_weighted` /
  `monte_carlo_medoid`), so `labels.py`'s aliasing is correct and no footnote
  is warranted. All six variants appear on all four runs.
- **`SCHEMA.md`'s `seeds.csv` column list is stale.** It documents
  `delaunay_degree`; the code now emits `class_adjacency_degree`, and
  `seeds_crossed == 1` is this layer's class adjacency by construction.

### Verification — `plot-error-cdf`

| check | result |
| --- | --- |
| `pytest scripts/analysis/v3/tests/` | **501 passed** |
| `n_plotted` vs `topn_accuracy.csv`'s `n_solved`, all 3 runs | **delta 0** |
| `error_km_p50` vs `topn_accuracy.csv`, all 3 runs | **delta 0.000000 km** |
| `topn_accuracy.csv` after extracting `io.solved_mask` | **byte-identical** |
| rows clamped to the log floor | **none** (observed min 0.135 km vs 0.1 km floor) |

### Four things the CDF surfaced

**The v2 plotter's 1 km log floor was clipping real data.** 8 to 20
sub-kilometre errors per method across as01/02/03 — up to 4.4% of a run's
targets, observed minimum 0.135 km. At a 1 km floor the left tail of the best
curves flattens and their p5 reads as exactly the clamp. Floor is 0.1 km, the
manifest records the would-be-clamped count, and the clamp never touches the
percentile CSV.

**Two files in one directory disagreed about `error_km_p50`.** This module
started with `method="nearest"` (the v2 plotter's choice, so a quoted
percentile named a real target and matched the world-map viewer's bookmarks)
and came out 0.7-1.3 km from `topn_accuracy.csv` on as02/as03. Since that file
is what the paper's accuracy table reads, the CDF now uses numpy's default and
the two agree exactly.

**The solved-row predicate had to be shared, not copied.** `classify` computed
it inline. Extracted to `io.solved_mask`, so the CDF's denominator equals
`n_solved` by construction — and so the BASELINE case is encoded once.
Shortest-Ping's rows are all `BASELINE`, never `SUCCESS`; a hand-written
`status == "SUCCESS"` filter would have silently dropped the baseline curve.

**The v2 plotter's threshold colours are this paper's variant hues.** Green,
orange and red guides at 100/500/1,000 km would read as Octant-Hull, Vanilla
and Spotter. Guides are neutral ink here.

### Verification — the two band figures

| check | result |
| --- | --- |
| `pytest scripts/analysis/v3/tests/` | **533 passed** |
| correct-band share vs `accuracy_top1`, both modes, all 3 runs | **delta 0.000000** |
| rank mode's cumulative through band 3 vs `accuracy_top3`, all 3 runs | **delta 0.000000** |
| band shares vs `1 - fallback_rate` | **delta 1e-4** (4-decimal rounding across bands) |
| crossings vs `confusion_pairs.csv` on shared rows | **delta 0** (874 / 1,297 / 1,477 rows) |
| level-0 rows these figures add | **1,414 / 1,100 / 1,178** — absent from `confusion_pairs.csv` |

### Layout, settled over four review rounds

The figures went through jitter → bins → rug bands → grid rows. What stuck:

- **Rows are contiguous grid cells** (`ylim (0, n_bands)`, separators between
  them), so the bottom row rests on the x axis and each panel reads as a stack
  of tracks rather than marks floating at ticks.
- **The gutter is above the band only.** It exists because contiguous rows leave
  the median readout nowhere else to go — below, it just left every band
  hovering above its own separator.
- **The tick sits at the band's centre, not the row's**, since it labels the
  data and not the cell that also holds the gutter.
- **The median rule is exactly the band's height.** It was taller on both
  sides, which made it read as an annotation layer over the band instead of as
  one of the targets in it.
- **Ticks are bare integers.** What band 1 means is in the axis label and the
  footnote, so the scale is not one annotated value beside three plain ones.
- **Compaction is physical, not fractional** — see the lesson; `ROW_INCHES` is
  the knob, cut 0.78 → 0.50 for 24% less figure height (2,298 → 1,737 px).

Every y coordinate derives from `band_span` / `band_centre`, which is what makes
the row invariants testable rather than implied: rows start at the axis, bands
never touch, the readout stays inside its own row, the tick lands on the band's
centre, and the gutter keeps ≥ 0.13 in of physical room for the 7 pt number.

### The denominator was the load-bearing decision

Band 0 only equals top-1 accuracy against the right denominator.
`seeds_crossed == 0` and `tg_seed_rank == 0` are the same event (the crossing
matrix is >= 1 off the diagonal) and both are top-1 correctness, but
`topn_accuracy.csv` divides by *every* target and counts fallbacks as failures.
Dividing by solved rows instead would have made band 0 disagree with the
accuracy the paper reports on the one method that falls back — as02 Vanilla
reads 0.298 over 412 targets and 0.365 over its 337 solved rows.

Consequence and a gain: bands sum to `1 - fallback_rate`, so Vanilla's four
bands total 81.8% and the panel header names the missing 18.2% as fallbacks.

### What the pair shows

The two axes correlate (median error rises monotonically with class error on
every method), so the readable result is the *shape* of each band stack, and
the three mechanisms separate cleanly:

- **Shortest-Ping / SoI** — 37% in the correct band, as a few tight discrete
  stripes under 60 km. The answer is a VP coordinate, so errors repeat exactly.
- **Octant-Hull** — 65% correct but smeared from 0.3 km to 600 km.
- **Spotter** — 41% correct at a 195 km median, the worst of the six.

### §2.4(a)'s claim, counted (first design, superseded)

The section asserts accuracy and error distance can disagree and that the
disagreement is a finding. On as02, against a 172 km margin:

- **288 points** are right-class-but-beyond-margin — the answer space absorbing
  a coordinate error, which is what a bounded metro-granular criterion is for.
- **111 points** are wrong-class-but-within-margin — nearest-seed snapping,
  the artifact §8.1's dense-region subsection asks about.

The axes do correlate (median error rises monotonically with crossings on every
method: 25 → 530 → 757 → 1,448 km for Shortest-Ping), which is what makes the
off-diagonal points worth naming.

**Each method has a signature in those two regions.** Shortest-Ping and SoI
score **zero** in the first: when they get the cell right the coordinate is
always inside the margin, because the answer *is* a VP coordinate and a
VP-proximate target is genuinely close. Spotter is the mirror image — 126 in
the first region, zero in the second, since it never has a good coordinate to
lose to a boundary.

### The table's two non-obvious decisions

**The best-in-row mark needed a tie rule.** A bare argmax over-claims where the
sample cannot separate the methods: as03's gap from Octant-Hull (0.502) to
Spotter (0.474) is 0.028 against a standard error of 0.023 on 458 targets, and
as02's baseline leads SoI by 0.003. The mark is now "within one standard error
of the row's best", one threshold per row. It earns its keep immediately — on
as03 at top-3, Shortest-Ping and SoI both reach 0.926 and *beat* Octant-Hull,
so the row shows two marks.

**Top-1 alone would have hidden that flip.** Octant-Hull wins all three top-1
rows, so the body table states one conclusion three times; top-3 has three
different winners including the baseline. Hence the appendix table.

### A design flaw the tests caught

The first version paired a weighted run to its mesh twin via `short_dataset`,
which everywhere else in the layer reduces `as01-260728-260802` to `as01`. It
strips only an *all-numeric* trailing tail, so no plausible weighted run name
reduces to its dataset: `as01-weighted-260728` and
`as01-260728-260802-weighted` both come back unchanged, and
`as01w-260728-260802` reduces to `as01w`. The pairing is now explicit
(`--weighted-run-id as01=<run_id>`), with a test pinning that `short_dataset`
cannot do the job — so nobody re-introduces the inference.

## The cross-dataset band grid

`plot-error-vs-rank` / `plot-error-vs-cells` now take two or more `--run-id`
and render one grid instead of one figure per run: methods down the rows,
datasets across the columns, into `_cross/error-vs-class/as01+as02+as03/`.

**What it shows that the per-run figures cannot.** The per-run figure
establishes that each method has a signature in the accuracy/error pair; the
grid asks whether the signature survives a change of dataset, and the answer is
that it degrades in a specific way rather than dissolving. Octant-Hull's
correct band runs 72.9% / 65.0% / 50.2% across as01/02/03 while its
one-cell-out band runs 21.6% / 28.6% / 40.2% — the targets it loses slide one
band up, they do not scatter. Vanilla's two failure modes move independently:
its fallback share is worst on as01 (26.6% / 18.2% / 20.3%) while its correct
band is *best* there (44.1% / 29.8% / 30.8%), so giving up and being wrong are
not the same difficulty.

**Orientation is a choice, not a default.** Methods own the rows because the
cross-dataset comparison is a within-method read, so a method's three datasets
sit side by side. The transpose puts the same 18 panels on the page and turns
that comparison into a vertical scan across two intervening rows. Three columns
also keep the 4.9 in panel width, and therefore the length of the log x axis,
identical to the per-run figure — which is the whole basis for reading an x
position across the two figures. Six method columns would need ~3 in panels and
could not label five decades without collisions.

**In the comparison layout nothing is pooled.** as01/02/03 have disjoint target
sets of different sizes (399 / 412 / 458), so a pooled band share is
target-weighted — which the pooled layout below reports and quantifies rather
than avoiding. Each comparison panel runs through the same
`load_points` / `band_table` pair the per-run figure uses and prints its own
`n`. Verified: all 18 correct-band shares equal that run's `accuracy_top1` and
all 18 cumulative-through-band-3 values equal its `accuracy_top3`, delta
0.000000.

**The refactor is provably inert.** `_draw_panel`, `panel_header` and
`footnote_text` came out of `plot_bands` so the two figures share one panel
renderer. All six per-run PNGs and both band CSVs per run re-render to
identical md5s afterwards, so the extraction moved code and changed no output.

## The pooled layout

`--layout` now chooses between two cross-dataset figures, default `pooled`:

* **`pooled`** merges the runs' targets into one 1,269-target population and
  reuses the per-run six-panel renderer unchanged. Legitimate because the
  operator runs' target sets are disjoint — 399 + 412 + 458 distinct ids, and
  `guard_disjoint_targets` checks it rather than trusting it, since a shared id
  would be counted once per run in the denominator and drawn twice in the band.
  It is the same pooling `plot-venn` already performs over these three runs.
* **`compare`** is the per-dataset grid, now suffixed `_by_dataset`.

**Pooling is where the density encoding starts working.** At 399 targets a band
is a few dozen stripes; at 1,269 Octant-Hull's correct band is a continuous
smear over 789 targets — 0.135 km to 466 km, 33 km median, 4-135 km IQR. The
figure stops being a list of marks and becomes a distribution.

**The weighting caveat is measured, not asserted.** A pooled share is a
micro-average, so as03 carries 36% of the denominator and it is *not* the mean
of the three datasets' accuracies. `weighting_check` puts both in the manifest
per method; the largest gap across the six is 0.0064 (Octant-Spline: 0.5524
pooled against 0.5588 macro), and Spotter is the only method where pooling
helps rather than hurts. So the two averages agree here, which is worth knowing
precisely because it could easily not have been true — Spotter's per-dataset
accuracy rises with dataset size (0.389 / 0.413 / 0.474) while everyone else's
falls.

Verified: pooled correct-band shares equal the target-weighted mean of the
three runs' `accuracy_top1` to 5e-5, the band CSV's rounding.

Suite: 553 passing, up 22.

## Conclusions

To be filled when the figures land.


## The dataset-type headline table, and the bars beside it

§8.1 hands off to a subsection about *dataset types* ("Traffic-weighted targets
vs MESH targets"), and the table it handed off from was per-AS. Regrouped so the
type owns the row, led by a target-count micro-average with the breakdown
beneath, plus `plot-outcome-bars` drawing the same numbers as correct / wrong /
fallback compositions.

### Verification

| check | result |
| --- | --- |
| `pytest scripts/analysis/v3/tests/` | **595 passed** (544 before, +51) |
| pooled counts vs `overlap_membership.top{1,3}.csv`, 36 cells | **delta 0** |
| aggregate `n_correct` vs sum of its breakdown rows | **delta 0**, both top-Ns |
| 18 breakdown cells vs `table-accuracy` | **delta 0** on accuracy and fallback |
| figure CSV vs table long CSV, 6 methods | **delta 0** on share, count and fallback |
| micro vs macro, max over methods | **0.0064** (top-1), matching the band figures |
| weighted cells / bars | all `—` / all ghost outlines |

### The pooled row says what §8.1 needs it to

Top-1 micro over 1,269 targets: Octant-Hull **0.622**, Octant-Spline 0.552, SoI
0.485, Shortest-Ping 0.476, Spotter 0.427, Vanilla 0.347 (fb 0.216). That
ranking — Octant-Hull, Octant-Spline, SoI — is exactly the order the draft's
story line asserts for the mesh campaign, and the 0.136 spread across the top
three is the quantity it asks to "show". The same order holds at top-3.

### Three things the review caught before they shipped

**Recomputing the breakdown rows would have broken the module's premise.** The
obvious simplification — one evaluation path, counts everywhere — moves three of
the thirty-six printed cells by a digit (as01 Spotter 0.389 → 0.388, as02 SoI
0.366 → 0.367, as02 Vanilla 0.298 → 0.299), so `table-accuracy` and
`table-headline` would print different numbers for the same measurement. Only
the pooled row, which has no published rate, recomputes.

**`groupby(["dataset", "kind"])` would have deleted every pooled row.** Aggregate
rows have `dataset = None` and pandas drops null group keys by default — no
exception, no warning, just the old table back. The render key is `row_index`.

**The pooled top-3 winner leads one dataset in three.** Octant-Hull takes the
pooled row at 0.891 having led only as02; as01 goes to Octant-Spline (0.957) and
as03 to Shortest-Ping and SoI (0.926). A bare bold there is a target-weighting
artifact stated as a fleet-wide result, so it carries a `†` and a footnote
naming what it lost. Top-1 is unanimous, which is why this would have shipped
unnoticed on the body table alone.

### The aggregate row changed which guards apply

`accuracy_table` documents why it skips `guard_one_setup`: "this table puts each
run on its own row where no such averaging can happen." A pooled row *is* that
averaging, so the exemption lapsed — `--run-id as01 --run-id as7018_us_test01`
would have micro-averaged a 134-VP fleet with a 53-VP one, and the target-id
guard would not have caught it, because those target sets are disjoint. Both
guards now run, and both are skipped when the aggregate is suppressed.

### Encoding budget, and what it cost the figure

Hue is the method and hatch is the dataset type, which spends both free
channels. So no stack segment may carry a texture, and the segments separate by
lightness in neutral ink instead — which also means the figure needs no palette
entry that does not already exist, and does not block on Phase 0's ramp work.
Two collisions found by rendering it: a filled `mesh` legend swatch is exactly
the `fallback` swatch (both `_C_MUTED`), and the `error` swatch is the lightest
ink step against white, which is a near-invisible box beside a word. The kind
entries are outline-only and never-occurring segments are dropped from the
legend, with the footnote still naming the full partition.
