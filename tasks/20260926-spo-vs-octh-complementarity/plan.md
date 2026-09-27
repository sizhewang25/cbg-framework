# SPO vs OCT-H Complementarity — Plan

## Background

The unbounded cell metric ranks Spotter first pooled (63.7% against Octant-Hull's
63.1%), while error distance ranks it last (p50 242.6 km against 131.1 km). An
exploratory session traced that inversion to the per-site level and produced a
set of observations that now need reproducible backing scripts and paper-ready
figures.

Three mechanisms were proposed during that exploration and **falsified**. They
are recorded here so nobody spends time on them again:

- **Unequal cell areas / hull-seed cells absorb the damage.** Dead: interior
  cells carry the *longer* offset tail on as02 and as03 (max 8 and 12 against
  the exterior cells' 6 and 6).
- **Far-correct predictions escape in unbounded directions.** Dead: the
  recession-cone test (`d·(p_j − p_s) ≤ 0 ∀j`) finds **0%** of Spotter's
  far-correct predictions in an unbounded direction on any mesh. The Arctic
  case travels north-*east* of Seattle, and everything east is another seed, so
  that ray is bounded — at 2,385 km.
- **Cell shape (elongation) predicts which method wins.** Dead: minimum-rotated-
  rectangle ratios overlap heavily between the two win categories, and the
  measure is frame-dependent for unbounded cells anyway.

What survived is below.

## Context

**Substrate.** `as01/as02/as03-260728-260802-mesh`, nside 128, 65 sites over
1,269 targets (~20 IP replicas per site, byte-identical coordinates). Existing
artifacts under `outputs/analysis/v5/{<run>/classify/healpix-128/*_tgs.parquet,
<run>/answer-space/healpix-128/seeds.csv}`.

**Columns that matter.** `cell_label` ∈ {correct, wrong, unanswered};
`pred_dist_to_tg_grid` (uncapped grid offset, added in `09958a7`);
`pred_dist_to_tg_km`; `tg_seed_id`; `site_id`; `status`.

**Two traps, both hit during exploration.**
1. `solved_mask` must be applied. Without it Vanilla reads 270 correct on as01
   instead of 163 — all 107 of its FALLBACK rows carry the baseline coordinate
   and all 107 are cell-correct, because Vanilla refuses precisely on the
   VP-adjacent targets where returning the VP is trivially right.
2. A per-site **majority** rule (`correct_frac >= 0.5`) silently reads a 9/20 as
   a loss. It misclassified Los Angeles as an exclusive Spotter win when
   Octant-Hull solved 9 of 20 replicas there *and* was the more accurate method
   (median offset 2 against 3). Categories must compare counts directly.

**Prototype.** `/tmp/.../scratchpad/xor_spo_vs_octh.png` and the two outcome
maps are the visual precedent; `modules/figure_outcome_map.py` (committed
`7644092`) is the structural precedent for a v5 figure module.

## Goals

Four paper-ready figures plus the stats behind them, each backing one claim.
Figures carry **title and figure only** — no footnotes, no caption block, no
stats box. Every number a figure rests on goes in the CSV twin and the manifest.

**C1 — The three meshes rank the two methods differently.**
as01 SPO 8–1 (McNemar exact **p = 0.039**, the only resolved mesh), as02 OCT-H
10–5 (p = 0.302), as03 8–9 (p = 1.000). Pooled 21–20, flat.

**C2 — Spotter's wins are exclusively peripheral; Octant-Hull's span the whole
range.** All 21 SPO-win sites lie ≥ **1,020 km** from the seed-cloud centroid,
with p25 **1,634 km** — above OCT-H's median of 1,459. OCT-H wins from **91 km**
out.

Distance is great-circle to the seeds' **spherical** centroid
(`geodesy.spherical_centroid`, the same routine `seeds` places a seed with),
not planar in EPSG:5070. The plan originally said EPSG:5070; that was wrong.
The partition is defined by great-circle nearest seed and `classify` uses no
projection, so the plane is a drawing concern. The earlier planar figures
(1,018 / 1,627 / 98) differ by a median 5 km and rank the 65 sites identically
to Spearman 0.9995 — nothing here turns on it, which is why consistency wins.

**C3 — Spotter is far more stable per site.** All-or-nothing on **61 of 65
sites (93.8%)** against Octant-Hull's **39 of 65 (60.0%)**; OCT-H splits 26
sites. This is the one finding that is a property of the *estimator* rather
than of the metric.

**C4 — Spotter never outperforms on error.** Measured on the **565 targets
both methods place in the correct cell** (34 `(run, site_id)` pairs):
`diff = offset_SPO - offset_OCT-H` is positive — Octant-Hull is closer — on
**86.5%** of them, against **4.2%** where Spotter is closer and 9.2% equal.
Median diff **+2 grids** (~100 km), p95 **+8**, max +19. Stable across meshes:
86.9% / 85.7% / 86.8%. Per site, Octant-Hull has the closer median on **29 of
34**, Spotter on 2.

Companion statistic, on the 21 sites Spotter wins: offset p5/p50/p95 of
1/3/**35** against 0/4/**12**, and Octant-Hull is the more accurate method on
**11 of the 21 sites Spotter "wins"**. The both-correct cohort is the headline
— 565 observations, consistent across meshes — and the 21-site figure is the
per-site companion.

## Approach

**Four atomic modules**, one per claim — not one module with a `--figure`
selector. Each emits PNG + CSV twin + manifest into `_cross/<kind>/<combo>/`,
is registered on the CLI, and is tested independently.

Shared plumbing goes in one small helper the four import: the per-site join of
two methods' correct counts, the four-way category (`tied` / `SPO wins` /
`OCT-H wins` / `neither`, **by direct count comparison, never a threshold**),
and the centroid-distance computation.

McNemar moves out of the figure and into the manifest and CSV.

**Sign convention, fixed:** `diff = offset_SPO - offset_OCT-H` throughout, so
positive means Spotter is further from the target. The C4 CDF therefore piles
up to the *right* of zero. That is the finding, not a plotting accident -- an
earlier expectation of left-skew was the opposite of what the data shows.

**Site key is `(run_id, site_id)`.** `site_id` is assigned per run and collides
across meshes; keying on it alone silently merges the three. This already
produced a wrong site count once during exploration (20 instead of 34).

## Caveats

- **Effective n is 65 sites, not 1,269 targets.** Every statistic is per site.
  Quote distinct-site counts beside every count.
- **Only as01 is statistically resolved.** as02 and as03 are directional. The
  mesh × method interaction itself has never been tested — "the meshes rank
  them differently" is an observation, not a tested effect.
- **The three meshes share facilities** (65 sites at 43 distinct coordinates),
  so pooled statistics are not fully independent and pooled p-values are
  optimistic.
- **Centroid distance is nearly collinear with "coastal"** on a CONUS answer
  space; do not claim to have separated the two.
- A win can be one replica wide (as03 has a 20–19). McNemar counts direction
  only, never margin.
- Everything here compares two methods on a metric the section argues is
  broken. The figures illustrate *what the cell metric rewards*; they are not
  evidence that Spotter is good.
