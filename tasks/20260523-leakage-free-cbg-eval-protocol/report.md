# Leakage-Free CBG Evaluation Protocol — Report

**Status**: Design discussion in progress (not finalized)
**Created**: 2026-05-23
**Last Updated**: 2026-05-23

## Summary

Captures an in-progress proposal to restructure the v2 benchmark so that LTD fitting and pipeline evaluation never share the same target anchor. Currently `fit_samples.parquet` and `eval_observations.parquet` contain the same (probe → anchor) observations, which introduces optimism bias in any fitted LTD (Octant spline, bounded_spline, NormalDist, Spotter). SoI LTD is unaffected.

The proposal converges on K=5 anchor-level cross-validation, applied uniformly across all LTD variants, with a diversified VP corpus (greedy 1K + ASN-balance cap) to neutralize geographic and institutional skew.

No code has been written. The proposal is still being debated; see "Open Questions" below.

## Findings

### Settled (or close to settled)

- **Leakage is real for fitted LTDs.** Per-VP fits include the eval anchor's (RTT, distance) as a training point, causing answer-memorization to varying degrees per variant.
- **LOO is not the right answer.** It conflates leakage-freedom with per-query refitting, which no real CBG deployment would do. Compute cost is also prohibitive (~24h/combo for naive LOO with splines).
- **K=5 anchor-level cross-validation is the practical sweet spot.** Train each LTD on 4 folds' anchors, eval on the held-out fold. ~10 min/combo. Matches "fit once, deploy to many" semantics.
- **Anchors must be the eval targets.** Probes have noisy GT (last-mile delay, user-reported coords). Anchors give us the only target-side RTT clean enough to trust.
- **VP diversification matters.** The IMC 2023 greedy 1K selection mostly handles geographic clustering. ASN-balance cap (cap probes per ASN) further neutralizes single-CDN dominance.
- **Same protocol across all variants** — including the no-split "paper-faithful" mode kept as a comparison baseline, runnable as a separate `holdout_policy` combo.

### Open questions (for the next discussion round)

1. **Fold granularity.** K=5 is the default proposal, but K=10 gives smaller eval batches per fold (~72 anchors) which could be too noisy; single 80/20 split is cheapest but gives one point estimate, not a distribution.
2. **Closed-form LOO for NormalDist + Spotter.** Their LOO updates are O(1) per held-out target — should we run real LOO for these variants and K-fold for spline-based variants? Asymmetric but methodologically cleaner. Or stick with K=5 uniformly.
3. **VP-corpus default.** Greedy 1K only, vs greedy 1K + ASN cap, vs sweep both as separate combo configurations.
4. **`ANCHORS_TO_PROBES` disposition.** Earlier decision was "keep both setups." User's latest take treats it as not scientifically meaningful (noisy probe-target GT) — should we keep it as a stress test, drop it entirely, or run only the SoI variant on it (where leakage doesn't apply)?
5. **Spotter-style global μ vs per-VP curves.** The LTD interface needs to distinguish; design proposal is a flag on the base class but unresolved.
6. **Octant institutional-exclusion rule on top of K-fold.** Probably unnecessary given diversified VP corpus — but should be measured before deciding.
7. **Materialize output shape.** K fit parquets + K eval parquets, vs one parquet with a `fold` column. Latter is cleaner but adds filter logic at read time.

### Reference material gathered

- Octant explicitly uses LOO-by-target plus institutional exclusion: *"In our PlanetLab dataset, nodes serve both as landmarks and targets... of course, the node's own position information is not utilized when it is serving as a target."* and *"No two hosts in our evaluation reside in the same institution."*
- Spotter argues per-landmark calibration is statistically unreliable, recommending global μ + per-VP σ: *"Due to the small sample size it is technically infeasible to infer reliable landmark specific delay models."*
- IMC 2023 (Darwich et al.) does not split fit/eval. Our paper-faithful baseline mirrors them but carries the optimism caveat.
- CBG (Gueye et al. 2006) uses SoI with no fitting — leakage-moot for that variant.

## Conclusions

To be written when the proposal is locked in. Currently this is a thinking document, not a decision document.
