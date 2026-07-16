# eval_bench_results.py — Report

**Status**: Complete
**Created**: 2026-07-15
**Last Updated**: 2026-07-15

## Summary

Built `scripts/benchmark/v2/eval_bench_results.py`, the output-side twin of
`eval_source.py`: a per-(combo, target) postcheck over a completed benchmark
run, covering participant-VP stats, MTL region area/n_components, truth
inclusion against the actual recomputed region (with outer-vs-inner exclusion
direction), includer/excluder VP stats, the closest-VP/shortest-ping-VP
bridge, answer-space cell gap, and leave-one-participant-out (LOO)
brittleness. Wired as `cli.py eval-bench-results`. Verified end-to-end against
the existing `as7018_us_test01` run for `vanilla_cbg` (planar_circle) and
`octant_cbg` (planar_annulus_weighted, stochastic Monte Carlo CTR) — no
re-benchmark needed since that run was already on disk.

## Findings

- **`recompute_matches_share = 1.0`** for both combos: reconstructing
  `LTDResult`s from `ltd_predictions` + `eval_observations.parquet` and
  re-running the *actual* registered MTL class reproduces the original run's
  success flag and participating-VP set exactly, for every target in both
  combos. This validates the whole approach — LOO's counterfactual reruns are
  operating on the same inputs the original pipeline saw, not a drifted
  approximation.
- **Brittleness is real but rare for `vanilla_cbg`** (rigid disk MTL):
  1/64 SUCCESS targets flips (`37.10.126.72`, 5 participants, dropping VP
  `70.236.176.160` empties the intersection). `octant_cbg`
  (weighted-annulus, aggregates many faces) showed zero flips across all 78
  — consistent with weighted aggregation being structurally more robust to
  any single VP than rigid disk intersection, and a nice confirmation that
  the metric is discriminating between MTL families rather than firing
  uniformly.
- **Brittleness also shows up as silent error blow-up, not just hard
  flips**: a 2-participant `vanilla_cbg` target had `loo_any_flip=False` but
  `loo_max_error_delta_km≈856` — dropping either of its two disks still
  "succeeds" (a single disk is trivially a valid, if huge, region) but the
  error explodes. Tracking both `loo_n_flips` and `loo_max_error_delta_km`
  separately was the right call; a flip-only signal would have missed this.
- **The inner/outer exclusion split works as designed**: `vanilla_cbg`
  (disk-only, `lower_km=0` always) produced zero `INNER` exclusion labels —
  correct, since a disk has no inner bound to violate. `octant_cbg`
  (annulus) produced `INNER` (20) and `BOTH` (15) alongside `OUTER` (25),
  the first time this codebase distinguishes *which* bound direction drove
  a truth-exclusion rather than reporting one aggregate "blocker" fraction.
- **`truth_in_region` diverges sharply from `status`**: `octant_cbg` is
  SUCCESS on all 78 targets but its recomputed region literally contains
  the truth point only 18% of the time; `vanilla_cbg` succeeds on 64/78 and
  contains truth 67% of the time among those. This is the "tolerance
  dividend" / EXCLUSIVE-but-correct phenomenon from prior memory
  ([[finding_exclusive_region_verified]]) made directly measurable per
  target rather than inferred.

## Conclusions

All 7 originally-discussed metric groups plus the 4 agreed additions (MTL
area/n_components, truth-inclusion with exclusion direction, includer/
excluder stats, LOO brittleness) are implemented and produce non-trivial,
sensible, cross-validated signal on real data. The design decision to
reconstruct and reuse the actual registered `MTLMethod`/`CTRMethod` classes
(rather than reimplementing geometry) is what made LOO both tractable and
verifiably faithful (`recompute_matches`). The `spherical_circle` vertex-list
code path wasn't exercised by the test config's 16 combos (none use it) but
was unit-checked in isolation and behaves correctly.
