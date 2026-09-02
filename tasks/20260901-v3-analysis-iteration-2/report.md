# v3 Analysis Iteration 2 — Report

**Status**: Complete
**Created**: 2026-09-01
**Last Updated**: 2026-09-01

## Summary

Second iteration on `scripts/analysis/v3/`, driven by five problems found while
running iteration 1 (commit `f6d305d`) across all four benchmark runs. Scope:
narrow `topn_accuracy.csv`, stop overlap artifacts overwriting across top-N,
replace the area-proportional Venn with an unweighted Shortest-Ping vs ≥1 CBG
figure, and add a static map of the answer space.

## Findings

Established during planning, from real output rather than inspection:

- **The correctness sets are fully nested on every run.**
  `all-CBG-correct ⊆ Shortest-Ping-correct ⊆ ≥1-CBG-correct`, with
  `shortest_ping_only` = 0, 1, 0, 0 across as01/as02/as03/as7018. This is why
  matplotlib-venn warns on every render — no area-proportional layout can draw
  nested sets — and it is why the unweighted variant is the fix rather than a
  cosmetic preference.
- **Vanilla CBG's fallbacks are not bad answers, they are the baseline's good
  answers.** On as01, 106 of 399 rows are FALLBACK and **all 106** have
  `truth_seed_rank == 0`. Counted as failures (correct, per §7.2) Vanilla scores
  164/399 = 0.411; credited it would score 270/399 = 0.677 and jump from near
  the bottom to near the top of the table.
- **CBG rescue counts are large and the regression count is ~zero.** `cbg_only`
  = 75 / 238 / 178 / 31–40 across the runs while `shortest_ping_only` ≈ 0. That
  asymmetry is the RQ2 headline the 2-set Venn is meant to carry.
- **The method pool changes the as7018 answer.** Over all 16 combos `cbg_only`
  = 40; over the published 5 it is 31. The Venn must state its pool size.
- **The answer space differs structurally between datasets.** as01: 399 targets
  at only 20 unique coordinates → K=18, 0 singletons, median margin 145 km,
  3 seed pairs within one cell pitch. as7018: 77 unique coordinates among 78
  targets → K=27, 11 singletons, median margin 17 km, 17 seed pairs within one
  pitch, and 4 targets whose cell seed is not their nearest seed. The §7.3
  straddle cost is negligible on one dataset and material on the other.

## Conclusions

All five goals met; 55 tests pass; all four runs regenerated from scratch.

**Delivered**

1. `topn_accuracy.csv` narrowed to 10 columns — `method, n_targets, n_solved,
   n_fallback, n_error, fallback_rate, accuracy_top1, accuracy_top3,
   error_km_p50, error_km_p90`. `DEFAULT_TOPN = (1, 3)`. The `_success_only`
   family is gone; the error columns keep the solved-only denominator behind an
   unqualified name, documented in the docstring.
2. Every overlap artifact is top-N suffixed via `venn.artifact_name(...)`.
   Verified top-1 and top-3 coexist and that top-3 membership is strictly larger
   (as01 1369 → 1909 True cells; as7018 387 → 844).
3. `overlap_venn.top<N>.png` is now an unweighted 2-set Shortest-Ping vs ≥1 CBG
   figure with counts, percentages, a pool-size subtitle and a "none correct"
   annotation. as01 reads 0 / 254 / 75 / 70 as predicted, with no positioning
   warning on any run.
4. `plot-answer-space` renders occupied cells, targets and seeds on one cartopy
   panel for all four runs.
5. The fallback-as-failure invariant held throughout: as01 Vanilla still
   `accuracy_top1 = 0.411` (164/399, 106 fallbacks).

**Verification results** — all six Phase 6 checks passed. Column sets match on
all four runs; the as01 invariant holds; top-N artifacts coexist and are
monotone; the as01 Venn regions are exact; restricting `--method` to the five
published variants moves as7018's CBG-only region from 40 to 31, confirming the
pool subtitle is load-bearing; the as7018 map shows the expected straddle pairs.

**Two problems found and fixed during implementation**

- `matplotlib_venn.venn2_unweighted` / `venn3_unweighted` are deprecated *and
  broken* in 1.1.2: they forward `normalize_to` into a layout algorithm that
  rejects it, raising `ValueError`. The plan named these functions directly.
  Worked around by driving `venn2`/`venn3` with an explicit
  `DefaultLayoutAlgorithm(fixed_subset_sizes=...)`.
- The first map render triggered Natural Earth 10m downloads in the test suite,
  because a single-seed auto-extent is small enough that cartopy switches to
  high-resolution features. Fixed by passing an explicit extent in that test —
  the suite is now network-free and runs in 2.9s instead of 13s.

**Deviation from plan**: the plan did not specify a "none correct" annotation.
Added it because a Venn cannot show the region outside its circles, and that
region is 70/399 on as01 and 30/78 on as7018 — too large to leave to
subtraction.

**Not done / deliberately out of scope**: no Voronoi overlay on the map (it is a
different shape from the cells and would muddle the quantizer story).

---

## Follow-up (same session, beyond the original plan)

Three further changes requested after review, plus one correction I made to the
first of them.

**nside sweep with grouped outputs.** `build-answer-space`, `classify`,
`plot-venn` and `plot-answer-space` all take a repeatable `--nside` and a
`--sweep` shorthand for the 128/64/32/16 hierarchy. Both output trees are now
grouped by `nside-<x>/`. The grouping of `target-cls-accuracy/` was *not* asked
for and is the correction: without it, sweeping four nsides would overwrite one
`topn_accuracy.csv` four times and leave no record of which grid produced the
survivor — the same bug class the `.top<N>` suffix fixes for the top-N axis.
`classify` reads the nside from the answer space itself, so an explicit
`--answer-space` still lands in the matching directory. `nside_sweep.csv`
compares the partitions side by side.

**Finding — coarsening cannot fix straddling.** The request was motivated by
EWR/JFK and SFO/SJC being split at nside 128. Both pairs are *closer* than the
50.9 km pitch (33.9 km and 48.9 km), so resolution was never the problem;
boundary alignment was. And because NESTED cells nest, a boundary at nside 16 is
also a boundary at 128 — coarsening removes only the finer lines, so a pair
split at a coarse level can never be merged by any nside. Measured on as7018:
the NY metro's 10 targets occupy 3 classes at nside 128, 2 at nside 64, and
still 2 at nside 16 (407 km cells). The Bay Area goes 4 -> 2 and stops. A grid
cannot guarantee facility grouping at any resolution; a radius-capped linkage
(the benchmark's existing `clusters/`) can.

**Finding — accuracy is strongly nside-sensitive.** as7018 best top-1 runs
0.397 / 0.474 / 0.603 / 0.628 across 128/64/32/16. Any reported accuracy has to
name its nside.

**UpSet: fixed row order, CBG labels, bars replaced by numbers.** Rows pinned to
`PREFERRED_ORDER` via `sort_categories_by=None`, reversed on input because
upsetplot draws the first category at the bottom. `label_for` now suffixes
`CBG` onto anything that is not `shortest_ping`, which covers the 11 as7018
ablation arms that were rendering as bare combo ids. Both bar charts suppressed
(`intersection_plot_elements=0`, `totals_plot_elements=0`) with
`Intersections (%)` annotated above each column and `True (%)` right of each
row — two headers because the quantities differ: columns are disjoint and sum
to 100%, rows are overlapping set totals and do not. Columns ranked by
intersection size, largest first (changed from degree on request); the trade is
that column position is data-dependent, so cross-run comparison must match dots
rather than positions.

Two implementation notes: a figure-level `suptitle` lands on the rotated column
numbers, so the title moved onto the matrix axis with `pad=44`; and a method
with zero correct targets drops out of `upset.totals` entirely, so that lookup
is defensive.

**Verification**: 60 tests pass; all four runs rebuilt across the full sweep at
both top-1 and top-3; as01 Vanilla still reads `accuracy_top1 = 0.411` at
nside 128, the invariant from iteration 2.
