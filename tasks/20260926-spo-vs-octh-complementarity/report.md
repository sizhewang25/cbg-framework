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

### Found while building C1

- **The 1x3 map cannot be printed with its per-site labels.** A label is eight
  characters at a fixed point size; three panels at a 7 in `\textwidth` give
  each panel 2.3 in, where one label spans 11 deg of a 65 deg frame. Not a
  tuning problem — twenty of them do not fit. `--no-labels` (counts to the CSV)
  and `--ncols 1 --panel-width 7` (a 7 x 14 in figure) are both rendered; the
  choice is the paper's. C2–C4 are not maps and are not affected.
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
