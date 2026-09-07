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

### Verification — `plot-error-vs-cells`

| check | result |
| --- | --- |
| `pytest scripts/analysis/v3/tests/` | **517 passed** |
| crossings vs `confusion_pairs.csv` on shared rows | **delta 0** (874 / 1,297 / 1,477 rows) |
| error vs `confusion_pairs.csv` on shared rows | **delta 0.000000 km** |
| level-0 rows this figure adds | **1,414 / 1,100 / 1,178** — absent from `confusion_pairs.csv` |
| the two disagreement regions | disjoint by construction (`>` far, `<=` near), pinned by test |

### §2.4(a)'s claim, counted

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

## Conclusions

To be filled when the figures land.
