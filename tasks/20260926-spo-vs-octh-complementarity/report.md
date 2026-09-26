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

## Conclusions

<Pending.>
