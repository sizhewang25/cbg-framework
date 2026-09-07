# §8.1 Figures (v3) — Plan

## Background

`papers/cbg-benchmark-as-network-operator/paper-flow-draft-v2.md` §8.1 answers
RQ1 — *what accuracy is achievable by latency-based geolocation under
operator-realistic conditions*. It is currently four prose sub-sections, each
naming a figure it does not have, plus one `【NEED table】` at line 344.

The analysis is **already complete**. Its predecessor task,
[20260906-v3-section-8-1-evaluation](../20260906-v3-section-8-1-evaluation/),
closed with 442 tests passing and every §8.1 table on disk. What is missing is
the rendering layer: `breakdown-accuracy`, `confusion-density` and
`table-accuracy` emit CSV and markdown only. The only §8.1-adjacent figures in
the tree are the Venn/UpSet/ring/Euler overlap family, the cost/accuracy Pareto
and the two cartopy maps.

So three findings the predecessor's report already identified are sitting in
CSVs with no figure to carry them:

- **SoI CBG is the Shortest-Ping baseline in disguise** (φ = 0.947 / 0.995 /
  0.987 against a baseline-correctness indicator on as01 / as02 / as03).
- **Vanilla misses narrowly; the baseline misses by a region** (median
  `boundary_margin_km` over wrong top-1 rows: 31 / 70 / 87 km vs 670 / 606 /
  281 km).
- **`geometry_only` is where Octant-Hull earns its keep** — on as02 that
  stratum's 40 targets get 0/40 from Shortest-Ping and Vanilla, and **39/40**
  from Octant-Hull.

## Context

### Inputs (all on disk, `h3-4`)

| input | path |
| --- | --- |
| headline accuracy | `accuracy_table.build()` in `modules/accuracy_table.py`, from each run's `topn_accuracy.csv` + `target-proximity/*/meta.json` |
| strata | `<run>/target-cls-accuracy/h3-4/accuracy_by_taxonomy.csv` |
| flag effects | ` … /accuracy_by_flag.csv` |
| mistakes | ` … /confusion_pairs.csv`, `confusion_by_density.csv` |
| covariates | `<run>/target-proximity/h3-4/target_labels.csv` × `overlap_membership.top<N>.csv` |

### Runs and methods in scope

`as01-260728-260802`, `as02-260728-260802`, `as03-260728-260802` (operator,
134 VPs, 399/412/458 targets) and `as7018_us_test01` (public RIPE, 53 VPs, 78
targets). Grid **h3 res 4 only** — `healpix-128` carries parquets and
`topn_accuracy.csv` but *no* breakdown or confusion outputs.

The six published variants (`shortest_ping`, `million_scale_cbg`,
`vanilla_cbg`, `octant_cbg_hull`, `octant_cbg_spl` / `octant_cbg`,
`spotter_cbg`). as7018's 11 ablation arms are RQ3 material and are excluded by
default.

### Decisions taken with the user (2026-09-07)

| # | Decision |
| --- | --- |
| U1 | Build **all four** figures: A headline accuracy, B proximity strata, C near-miss anatomy, D continuous covariates. |
| U2 | **PNG only**, house style — 200 dpi, `_SURFACE` white, palette and labels imported from `diagram/common/`. No PDF path. |
| U3 | Traffic-weighted views are **placeholder only**. `has_weight` is false on all four runs; plumb an unused weight parameter so a weighted run needs no signature change, produce nothing weighted, and leave §8.1's second and third sub-sections as explicit TODOs. |

## Goals

1. **Two validated ramps in `palette.py`** — an ordinal one-hue ramp and a
   blue↔red diverging scale, plus `_C_AXIS` hoisted out of `pareto.py`, and a
   figure-level `_annotate_multipanel` in `draw.py`.
2. **`accuracy_by_covariate.csv`** — a third output of `breakdown-accuracy`,
   the only new table, binning correctness on `tg_seed_nearest_vp_km` and
   `min_inflation`.
3. **Four `plot-*` commands** emitting five PNGs plus CSV twins.
4. **§8.1 rewritten against the figures**, accuracy table demoted to the
   appendix, and the weighted sub-sections marked TODO rather than written.

## Approach

### Colour: hue is already taken

The repo's six variant hues are pinned by identity and colour-blindness
validated. Three of the four figures need encodings the repo has **no** palette
for. Validated with the dataviz skill's `validate_palette.js` against the
repo's real surface (`#ffffff`, not the skill's default `#fcfcfb`):

| role | steps | verdict |
| --- | --- | --- |
| ordinal, one hue light→dark | `#86b6ef` `#3987e5` `#256abf` `#104281` | `--ordinal`: ALL PASS (monotone L, ΔL ≥ 0.06, light end 2.11:1, hue spread 3°) |
| diverging blue arm | same four steps | as above |
| diverging red arm | `#e87b7a` `#d94d4c` `#b53433` `#7d2221` | `--ordinal`: ALL PASS (light end 2.78:1, hue spread 4°) |
| diverging midpoint | `#f0efec` | must read as "nothing" — no hue here |

Re-validating the **existing** six variant hues on `#ffffff` constrains the
figures two ways. On the *adjacent* pairlist they pass clean (worst ΔE 9.1
protan); under `--pairs all` they only **WARN** (worst ΔE 6.9 deutan, Spotter ↔
SoI), which is legal *only with secondary encoding*. So any panel putting all
six series in one shared coordinate space — Figure D's line panels — needs
direct end-labels, not a legend alone. Panels giving each method its own row
(A, B, C) satisfy this by construction. The contrast WARN on `#1baf7a` (2.82:1)
and `#eda100` (2.17:1) obliges a table view; every figure ships a CSV twin.

**Rule adopted:** where an ordinal/diverging encoding shares a *panel* with
variant hues, render it in the house neutral ink steps (`_C_GRID` → `_C_MUTED`
→ `_C_INK_2`), following `pareto.py`'s precedent ("neutral, because hue is
taken"). The blue ordinal ramp and the diverging scale appear only in panels
where no variant hue does.

### The four figures

**A · `plot-accuracy`** — four panels in one row, shared x. Method on y in
`PREFERRED_ORDER`; a dumbbell from top-1 (filled dot) to top-3 (open dot) in
the method's hue, so the connector length *is* the gap §8.1 argues about.
Fallback rate as a hatched segment anchored at the top-1 dot — non-zero only
for Vanilla, so it reads as a per-variant defect. Dataset context in the panel
subtitle (`n VPs · n targets · K seeds · geometry_only x%`). **as7018's panel
gets a rule and a `probes_to_anchors` tag**, because §7.3 declines the
head-to-head and it differs in setup, VP count and target count at once.

Reads `accuracy_table.build()` directly rather than the emitted CSVs, so it
takes `--run-id` repeatably and renders all four runs in one figure instead of
stitching two `_cross` directories.

**B · `plot-strata`, two files** — split because (i)+(ii) encode by variant hue
and (iii) by a diverging scale, and one figure may not carry two colour systems.

- `strata_accuracy.png`: a thin neutral-ink composition bar over
  `geometry_only / selection_miss / selection_hit`, and below it a slope chart
  of per-method top-1 accuracy across those three strata (an ordered ladder of
  baseline favourability), with direct end-labels.
- `flag_effect.png`: φ heatmap, methods × the four diamond flags, on the
  diverging scale, φ printed in ink.

**C · `plot-confusion`** — two panels. Left: stacked composition of wrong top-1
rows on the ordinal ramp, **exactly four classes** confirmed against all four
runs — `0` (a fallback whose class was right), `1` (adjacent), `2`, `3+`.
Right: CDF of `boundary_margin_km` over wrong rows, one line per method, x on
**symlog** (0 to ~1400 km on as02/as03, past 3800 on as01, and near-zero
matters because it means the flip was a tie-break).

`confusion_by_density.csv`'s tertiles do **not** get a panel — three bins is
too coarse and Figure D's five-bin curves supersede them. They stay in the
appendix with the note that the effect is real but not universal
(Shortest-Ping 0.33→0.71→0.82 on as01, 0.16→0.42→0.51 on as02, but
**0.67→0.25→0.44** on as03).

**D · `plot-covariate`** — x = quantile bins at their `p50`, y = top-1
accuracy, one line per method with ringed markers and direct end-labels. Log x
for `tg_seed_nearest_vp_km`, linear for `min_inflation`. Per-bin `n` under each
tick. Annotate the crossover where Octant-Hull overtakes the baseline.

### Figure D is the most load-bearing, so build it first

Piloted on as02 (quintiles, ~80-88 targets/bin):

| VP→seed km | Sh.-Ping | SoI | Vanilla | Oct-Hull | Oct-Spline | Spotter |
| --- | --: | --: | --: | --: | --: | --: |
| 4-11 | **0.659** | 0.659 | 0.375 | 0.534 | 0.432 | 0.261 |
| 17-24 | 0.481 | 0.481 | 0.753 | 0.558 | 0.481 | 0.143 |
| 26-42 | 0.414 | 0.414 | 0.195 | **0.874** | 0.655 | 0.586 |
| 43-92 | 0.250 | 0.250 | 0.125 | 0.750 | **0.875** | 0.450 |
| 115-165 | 0.012 | 0.000 | 0.062 | **0.525** | 0.425 | 0.612 |

It carries three things no other §8.1 figure does: a **crossover** (below
~25 km the baseline wins, above it Octant-Hull wins by 2-40×) which is §9's
when-to-use-what measured rather than asserted; the baseline's **monotone total
collapse** (0.659 → 0.012) against Octant-Hull never dropping below 0.525; and
**SoI tracking Shortest-Ping to three decimals in every bin but the last**,
making the "baseline in disguise" finding visible rather than tabulated.

`min_inflation` also separates despite its narrow range (1.35-2.04):
Shortest-Ping 0.511 → 0.821 → 0.427 → 0.077 → 0.014, Octant-Hull 0.875 → 0.526
→ 0.829 → 0.648 → 0.315. Non-monotone, and worth showing as such.

### Where the new table belongs

`accuracy_by_covariate.csv` goes in **`breakdown.py`** as a third output of
`breakdown-accuracy`, because that module already performs this exact join for
the boolean flags (`build_membership` → boolean target×method matrix, inheriting
the fallbacks-are-failures policy and the common-denominator check, joined to
`load_proximity`'s labels on `target_id`, `breakdown.py:120-141`). A new module
would duplicate the join and be free to drift from the flag table beside it.

The binner exists too: `confusion.density_bins(series, *, n_bins)` is generic in
everything but its name and carries the hard-won "`edges` always has at least
two entries" contract (`confusion.py:98-129`). Reuse it.

## Error-vs-class figures, redesigned 2026-09-07

First cut used a jittered scatter with a margin reference. Reworked after
review into two band figures sharing one renderer.

**Rug bands, not bins and not jitter.** Each point is a thin vertical line
spanning its band, in the method's hue at low alpha; coincident values darken
by overplotting. Density becomes a property of the data rather than of a bin
width or a random offset. The known limit is that alpha accumulation saturates
at roughly 1/α coincident lines (~7 at α=0.15), and Shortest-Ping and SoI have
exact repeated x values because many targets share a VP coordinate — so their
darkest stripes stop distinguishing 7 coincident points from 30. The row-%
label carries the count, so this costs within-row shading detail and no number.

Per-method line colour also means no sequential ramp and no colorbar: the
palette is used exactly as the rest of the paper uses it.

**The denominator is `n_targets`, not `n_solved`.** Band 0 must equal top-1
accuracy, and it only does against the total. `level == 0` ⟺ `tg_seed_rank ==
0` ⟺ top-1 correct (the crossing matrix is ≥1 off-diagonal, so band 0 is
exactly "predicted the true cell"), but `topn_accuracy.csv` divides by every
target and counts fallbacks as failures. Verified on as02 Vanilla: 123 level-0
rows, 123/412 = 0.298 = its top-1 accuracy, while 123/337 solved = 0.365 and
matches nothing.

Consequence, and a gain: bands sum to `1 - fallback_rate`, so Vanilla's four
bands total 81.8% and the missing 18.2% *is* its fallback rate, visible in the
figure for free and labelled as such.

**Two y modes, one module, two commands.** `seeds_crossed` (cells away from the
true class, 0-indexed because it counts boundaries) and the **nearest-cell
index** of the true class (`tg_seed_rank + 1`, 1-indexed because it is an index
into the cells ordered by distance from the estimate). The 1-indexing is what
makes the second axis read straight off the reported metric — `index <= N` *is*
top-N, so band 1 is top-1 accuracy and bands 1-3 sum to top-3, where
`tg_seed_rank < N` needed translating.

SCHEMA warns the two quantities get confused and that they order methods
differently, so they live in one module with the distinction documented once;
the CLI exposes them separately so each figure has its own command. Both draw
four bands with the top one a bucket: cells 0/1/2/3+, index 1/2/3/4+.

**The margin reference is removed** — dashed rule, both disagreement
annotations, the footnote line and the two orphaned summary columns. The figure
now makes one point: error distance and classification accuracy are different
metrics, and each method has its own pattern in both.

**Assertions pinned rather than eyeballed:** the bottom band's share equals
`accuracy_top1` for every method on every run, and the rank mode's cumulative
share through band 3 equals `accuracy_top3`.

## Caveats

- **The tables already encode every degenerate case correctly**, so the figure
  layer's whole obligation is *never render a NaN as zero*. `geometry_only` on
  as01/as03 emits `n_targets = 0` with `accuracy = NaN`; a zero-variance flag
  emits `phi = NaN`, `rate_difference = NaN`, `zero_variance = True`. A NaN
  accuracy is a gap with an `n=0` tick; a NaN φ is a hatched "no variance" cell,
  **never** the diverging midpoint, which would read as φ = 0 = "no effect".
- **`octant_cbg` (as7018) and `octant_cbg_spl` (operator) are the same
  configuration** — both `bounded_spline` / `fit_spline=True` /
  `planar_annulus_weighted` / `monte_carlo_medoid`, verified from each run's
  `run.json`. `labels.py`'s aliasing to one label is correct; **do not add a
  footnote claiming they differ.** All six variants appear on all four runs.
- **`seeds_crossed == 0` is not a near miss — it is a fallback whose class was
  right.** On as01 all 106 of Vanilla's fallbacks have `tg_seed_rank == 0`:
  scored 0.411, credited 0.677. It must be visually set apart from genuine
  misses. Shares of Vanilla's wrong rows: 47.5% / 26.0% / 29.3% / 24.0% on
  as01 / as02 / as03 / as7018.
- **`seeds_crossed == 1` *is* this layer's class adjacency** and agrees with
  `seeds.csv`'s `class_adjacency_degree` by construction. The retired
  `delaunay_degree` column is gone; **`SCHEMA.md`'s `seeds.csv` column list is
  stale** on this point — read columns off disk.
- **as7018 needs its own bin counts.** 78 targets over 5 bins is ~16 each; use
  3 and print `n`. Any per-stratum rate there rests on tens of targets.
- **as7018 carries 17 methods** in `confusion_pairs.csv` and the parquet set.
  Restrict to `PREFERRED_ORDER` by default; `--method` opts the arms back in.
- **Only `h3-4` has the inputs.** Figures B/C/D must refuse a grid whose inputs
  are absent, naming the missing file, rather than half-render.
- **A missing variant is skipped, never interpolated** —
  `pareto.py::dataset_lines` already documents and tests this; reuse the rule.
- `tests/test_config.py` validates every checked-in `configs/*.yaml` against
  the *live* CLI, so a command added without its sub-block will surface there.

## Out of scope (recorded, not forgotten)

- **Anything weighted** (U3). Plumb the parameter, produce no weighted artifact.
- **The VP&TG topology-paired dataset** — §8.1's fourth table row needs a
  `scripts/benchmark/v2/` re-run with co-curated VP and target sets, not an
  analysis-layer script (predecessor task D4).
- **§8.2 and §8.3 figures.** `accuracy_by_taxonomy.csv` and the resolvability
  Venn family are their inputs, and Figure B1's strata panel is reused there,
  but their own figures are a separate task.
- **A PDF output path** (U2). One line per module if camera-ready needs it.
- **`healpix-128` figures.** The grid stays supported; its numbers are not
  reported.
