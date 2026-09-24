# v5 Evaluation Backing Scripts — Report

**Status**: In Progress
**Created**: 2026-09-24
**Last Updated**: 2026-09-24

## Summary

Hardening the §1b best-case cohort analysis into saved modules. v4 delivered
the two VP distances and the per-method bounds; this task covers the layer
above — which targets each method finds easy, how those sets overlap, and the
per-target mechanism behind the far-RTT cases.

## Findings

### Already file-backed (v4, `vp_proximity.{p5,p25}.csv`)
Per-method geo/RTT bounds, violin quantiles, tie shares, distinct-value counts,
gap ratios, population VP distances.

### Not yet file-backed — the scope of this task
Cohort union and degree histogram; overlap against Shortest-Ping; the
shared / answered-not-best / refused decomposition with error percentiles;
the baseline-margin share; far-tail counts; the eight-target mechanism table.

### Claims rejected during drafting — do not re-propose
- **Overlap bar chart / Euler / UpSet figure.** Rejected for the two failure
  modes recorded in §1b. A 6-set Euler is also undrawable here: 154 singletons
  and a maximum 4-way intersection at p5.
- **"Multilateration is stable."** Spotter out-reaches both Octants on far-VP
  targets and is the worst method; reach alone is not evidence.
- **"All methods benefit from VP proximity except Spotter."** Spotter's p5
  targets do have close VPs (2.2–48.6 km, 70% inside the S-P bound); it fails
  to use them. The real split is baselines vs CBG variants.
- **"Vanilla relies 100% on multilateration and achieves good results."**
  Backwards. Its 0% overlap is 57% refusal plus 85 km median error on the rest.
- **"Indirect routing" as the single mechanism for the far-RTT cases.** Seven
  of the eight are short-range unrankability (0.22 ms of propagation under a
  3.1 ms RTT); only `tg-ac636bd` is indirect routing.
- **p5 violin as a shipped figure.** 2–11 distinct values per cohort.

## Conclusions

<Pending.>
