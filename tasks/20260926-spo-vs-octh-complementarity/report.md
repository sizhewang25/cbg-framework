# SPO vs OCT-H Complementarity — Report

**Status**: In Progress
**Created**: 2026-09-26
**Last Updated**: 2026-09-26

## Summary

Backing scripts and paper-ready figures for four claims about where Spotter and
Octant-Hull disagree on the cell axis. Derived from an exploratory session whose
figures live only in a session scratchpad; this task makes them reproducible.

## Findings

### Established, ready to back (see `plan.md` for the numbers)

- **C1** three meshes rank the two methods differently; only as01 resolves
  (p = 0.039).
- **C2** all 21 SPO-win sites are >= 1,018 km from the seed-cloud centroid;
  OCT-H wins from 98 km out.
- **C3** Spotter is all-or-nothing on 61/65 sites (93.8%) against OCT-H's
  39/65 (60.0%).
- **C4** on the 565 targets both methods place correctly, OCT-H is closer on
  86.5% against Spotter's 4.2% (median diff +2 grids, ~100 km), consistent
  across all three meshes. Companion: on the 21 sites Spotter wins, OCT-H is
  more accurate on 11, and Spotter's offset p95 is 35 against 12.

### Falsified during exploration — do not re-propose

- Unequal cell areas / hull-seed cells as the mechanism.
- Escape in unbounded directions (recession cone: 0% on every mesh).
- Cell elongation as a predictor of which method wins.

### Corrected mid-exploration

- A per-site majority rule misclassified Los Angeles; categories must compare
  counts directly.
- `solved_mask` omission inflated Vanilla's correct count 163 -> 270.
- "13 only-SPO sites / 1,164 km" was superseded by "21 sites / 1,018 km" once
  the majority rule was dropped.
- C4 was expected to skew left (Spotter closer). It skews right, decisively:
  OCT-H closer on 86.5% of both-correct targets. The expectation was checked
  before any module was written.

### Backed by a script, 2026-09-26

- **C1 is built and reproduces `plan.md` exactly**, with no adjustment to any
  number: as01 8–1 (p = 0.039), as02 5–10 (p = 0.302), as03 8–9 (p = 1.000),
  pooled 21–20 over 65 sites and 1,269 targets. `modules/contest.py` +
  `modules/figure_contest_map.py`, CLI `plot-contest-map`, 42 tests.

- **C2 is built**, re-derived from the CSV twin rather than a notebook: every
  one of the 21 SPO-win sites is at least **1,020 km** from the seed-cloud
  centroid (p25 1,634, median 1,928), while Octant-Hull wins from **91 km** out
  with a median of 1,459. SPO's lower quartile is above OCT-H's median.
  `modules/figure_peripherality.py`, CLI `plot-peripherality`, 15 tests.

- **C3 is built** and reproduces `plan.md` exactly from the CSV twin: Spotter
  is all-or-nothing on **61 of 65** sites (93.8%) against Octant-Hull's **39**
  (60.0%, 26 splits), and its per-site spread of grid error distance has median
  **0.31** grids against **1.22**, with 21 zero-spread sites against 5. Each
  box is 65 pooled sites (20 + 22 + 23) over 1,269 targets.
  `modules/figure_stability.py`, CLI `plot-stability`, 11 tests.
- **C3 cuts both ways.** Spotter is perfectly consistent on more sites *and*
  worse on its worst: its per-site spread runs to **15.8** grids where
  Octant-Hull's stops at **6.5**. With whiskers at p5/p95 and no outliers that
  tail is off the page, so the prose has to carry it.
- **The spread metric was compared against an alternative and kept.** It is the
  standard deviation of `pred_dist_to_tg_grid` — the spread of the error
  *magnitude*, under which two replicas five grids out in opposite directions
  read as perfect agreement. The alternative, the spread of the prediction
  cloud itself (RMS grid distance to the site's prediction centroid), ranks the
  65 sites at **Spearman 0.90**, reaches the same conclusion, and differs on
  **11 sites** that read as perfectly consistent under the simpler measure
  while their predictions were up to two grids apart. Decision: keep the
  simpler statistic, document the limitation, pin it with a test.
- **Consistency is not accuracy and this figure cannot tell them apart.** as01
  has a site whose twenty replicas all land in one grid **38 grids from the
  truth** — a spread of 0, perfectly stable and consistently ~1,900 km wrong.

- **C4 is built** and reproduces `plan.md` exactly from the CSV twin: on the
  **565** targets both methods place correctly, Octant-Hull is nearer on
  **86.5%** against Spotter's **4.2%**, level on 9.2%; median difference **+2**
  grids, p95 **+8**, max **+19**; stable across meshes at 86.9 / 85.7 / 86.8;
  and per site Octant-Hull has the nearer median on **29 of 34**.
  `modules/figure_error_diff.py`, CLI `plot-error-diff`, 16 tests.

- **C5 is built**, backing "Spotter can win with very large errors at the
  periphery, but that is a small portion". On the 243 targets only Spotter
  places correctly its offset runs to **35 grids** (~1,780 km) with p90 **11**;
  on Octant-Hull's 236 it stops at **8**, with 31 landing in the target's own
  grid where Spotter has none. The far tail is **13.6%** of Spotter's cohort
  and **2.6%** of all targets. `modules/figure_exclusive_error.py`, CLI
  `plot-exclusive-error`, 14 tests.

### Found while building C1

- **The 1x3 map cannot be printed with its per-site labels.** A label is eight
  characters at a fixed point size; three panels at a 7 in `\textwidth` give
  each panel 2.3 in, where one label spans 11 deg of a 65 deg frame. Not a
  tuning problem — twenty of them do not fit. `--no-labels` (counts to the CSV)
  and `--ncols 1 --panel-width 7` (a 7 x 14 in figure) are both rendered; the
  choice is the paper's. C2–C4 are not maps and are not affected.
- **The success-ratio figure was a violin twice and is now a CDF.** Over 65
  sites piled on 0 and 1, Scott's bandwidth spread Spotter's 4 split sites into
  a waist as wide as Octant-Hull's 26 — the figure denied the very claim it was
  drawn for — and the KDE put density above 1.0 on a variable that is a share.
  Narrowing the bandwidth and bounding by reflection fixed the arithmetic but
  left a shape that still did not read as a violin. A share of 65 sites
  concentrated on two values has no shape a smoother can be trusted with, so it
  is now an **empirical CDF**: exact, no bins, no bandwidth. The claim reads off
  the geometry — the jump at 0 plus the jump at 1 *is* the unanimity rate
  (0.354 + 0.585 = 0.938 for SPO, 0.185 + 0.415 = 0.600 for OCT-H), and between
  them SPO is flat where OCT-H climbs.
- **The spread figure no longer shows its tail, by request.** Whiskers are p5
  and p95 with no outliers, so Spotter's worst site at 15.8 grids is off the
  page. It is the half of C3 that runs against Spotter, and it now survives
  only as `spread_max` in the twin and the manifest — worth stating in the
  prose, because the figure alone reads as "strictly more stable" and that is
  not what the data says.
- **A committed C1 test was asserting against the wrong site.** `sites.site_ids`
  numbers sites in sorted coordinate order, so `PLACES[3]` (Miami) is `site_id`
  0; `test_the_fallback_site_is_a_win_not_a_tie` named the site by its `PLACES`
  index, landed on Seattle, and passed for an unrelated reason. Caught by a C3
  test that happened to need the same site. There is now a `place_to_site`
  fixture and the constant is named `BETA_FALLBACK_PLACE`.
- **Both ends of the peripherality distance are seeds.** It was site-to-seed-
  centroid; it is now seed-to-seed-centroid, so the measurement lives entirely
  in the answer space and sites sharing a seed share a value (61 distinct
  distances over 65 sites). It moved 8 sites by at most 12 km and changed no
  quantile — the point is coherence, not the numbers.
- **The seed's distance is not its cell's, and it does not matter here.**
  Checked because a peripheral seed can own a large cell reaching back toward
  the centre: as01's Seattle seed is 2,349 km out and its cell begins at 1,241.
  The two rank the 65 sites at Spearman 0.87, so it is a real difference. But
  the inward reach is the same size in both categories (median 404 km where
  SPO wins, 333 where OCT-H does), and the claim holds under either — by cell
  reach, SPO never wins inside 770 km while OCT-H wins a site whose cell
  *contains* the centre. Measured with a projection-free ray scan in
  `classify`'s own nearest-seed rule (agrees with the planar Voronoi to a
  median 7 km); the code was not kept, since the axis stays on the seed.
- **The centroid is spherical, not planar.** The plan specified EPSG:5070 and
  that was wrong: the cell partition is defined by great-circle nearest seed
  and `classify` uses no projection at all, so measuring peripherality in the
  drawing plane imports a rendering concern into a number. `seeds` already
  places every seed with `geodesy.spherical_centroid`; the centre of the seeds
  is now found the same way. The two answers differ by a median 5 km (max 22)
  and rank the 65 sites identically to Spearman 0.9995, so no claim moved —
  1,018/1,627/98 became 1,020/1,634/91.
- **The loading layer moved into `contest.py`.** `ContestData`, `load` and
  `contest_table` are substrate, not drawing, and C2–C4 all need them.
  `contest_table` now carries `centroid_km` per site, computed against each
  run's own seed cloud.
- **The label relaxation has to run in units of a label box.** A site label is
  four times wider than it is tall, so a single isotropic separation in degrees
  is wrong on one axis whichever value it takes. Scaling the space first fixes
  it, and the manifest now reports the separations each panel *achieved*
  (`label_crowding`), not the ones it asked for — currently 0 overlapping pairs
  on all three.

## Conclusions

<Pending.>
