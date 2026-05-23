# Leakage-Free CBG Evaluation Protocol — Plan

> **Status: NOT FINALIZED** — this captures the in-progress proposal. Discussion is ongoing; expect the methodology to keep evolving before we lock it in.

## Background

The current v2 benchmark (`scripts/benchmark/v2/`) materializes `fit_samples.parquet` and `eval_observations.parquet` from the *same* set of (probe → anchor) RTT observations. Every anchor that appears as an eval target also appears in the training corpus that each VP's LTD fits on. For any *fitted* LTD (Octant spline, normal-distribution, Spotter, bounded-spline), this means the curve was trained on the exact (RTT, distance) point it's later asked to predict — i.e., partial answer-memorization.

Per-VP optimism bias from this leakage is real for any fitted LTD. SoI LTD is unaffected (no fit). This skews cross-variant comparisons in favor of more expressive fitted models.

The proposed protocol restructures the benchmark to fit-and-evaluate on disjoint subsets of anchors, applied uniformly across all LTD variants for fair comparison.

## Context

### Data inventory

- `anchors_meshed_pings` — anchor↔anchor RTT mesh (clean GT, well-connected endpoints)
- `ping_10k_to_anchors` — probe→anchor RTT (current primary corpus, 12K probes × 723 anchors)
- No probe↔probe table exists (would need 12K² ≈ 144M new measurements)

### Available reference work

Audited papers (in `papers/cbg-variant-benchmark-proposal/refs/`):
- **Octant (Wong et al.)**: Uses LOO-by-target. Quote: *"In our PlanetLab dataset, nodes serve both as landmarks and targets... of course, the node's own position information is not utilized when it is serving as a target."* Also imposes institutional diversity: *"No two hosts in our evaluation reside in the same institution."*
- **Spotter (Laki et al. 2011)**: Argues per-landmark calibration is unreliable due to small per-VP sample size: *"To infer landmark specific delay models the overall calibration set has to be divided into significantly smaller landmark specific chunks. These smaller data sets contain only a few hundred points... Due to the small sample size it is technically infeasible to infer reliable landmark specific delay models."* Their fix: global μ, per-VP σ.
- **CBG (Gueye et al. 2006)**: SoI-style, no fitting (leakage-moot).
- **IMC 2023 (Darwich et al.)**: Does not split fit/eval — what we're currently replicating, with the optimism caveat.

### Existing helpers in the codebase

- `datasets/reproducibility_datasets/generated/reproducibility_greedy_probes.json` — IMC 2023's greedy geographic-diversity selection (~1000 probes). Reusable as canonical VP corpus.
- `_compute_soi_removed_ips` in `RipeAtlasSource` — already filters ~54 worst-GT probes; the remainder is the leakage-free starting point.

## Goals

1. **Eliminate fit/eval target overlap** in the benchmark, for every fitted LTD variant.
2. **Apply a uniform protocol** across all CBG variants — same fold structure, same VP corpus — so comparisons reflect technique differences, not eval conditions.
3. **Preserve deployment realism**: train once per fold, apply to all eval targets in that fold (no LOO per-query refit).
4. **Mitigate institutional/geographic skew** in VP selection, so eval accuracy isn't inflated by VP-density near specific anchor locations.
5. **Keep a "paper-faithful" mode** (no split) available alongside the new leakage-free mode, so IMC 2023 comparability is preserved.

## Approach (current proposal — open for revision)

### Primary methodology

| Axis | Choice |
|---|---|
| Setup | `PROBES_TO_ANCHORS` |
| VP corpus | Greedy ~1000 probes (IMC 2023 selection) + ASN-balance cap (~5–10 per ASN) |
| Eval targets | 723 anchors split into 5 folds, deterministic by anchor_id |
| Holdout | K=5 cross-validation; fit LTD on 4 folds' anchors, eval on the held-out fold |
| Uniformity | Same fold IDs + VP corpus across all LTD variants |
| `ANCHORS_TO_PROBES` | Kept as secondary (noisier eval, sanity check only) |

### Why not LOO

LOO conflates "leakage-free" with "per-query model refit." A real deployed CBG system fits *once* on a fixed landmark corpus and applies the model to many queries. LOO is also infeasible at scale: ~12K VPs × ~720 anchors × ~10ms per spline fit ≈ 24 hours per combo for naive LOO. K=5 → ~10 minutes per combo.

For LTDs with closed-form LOO (NormalDist: μ_{−X} = (N·μ − x_X)/(N−1); Spotter similar), we could optionally do real LOO at no extra cost. Worth checking if the marginal honesty justifies the asymmetric protocol.

### Why anchor-only targets

Probes have noisy GT (user-reported coords, residential last-mile delay). Using probes as eval targets makes RTT-distance fitting noisy *on the target side*, which per-VP calibration cannot absorb. Anchors have professionally-set GT in well-connected datacenters → target-side contribution to RTT is minimal, and per-VP calibration absorbs VP-side last-mile bias.

This logic effectively demotes `ANCHORS_TO_PROBES`. Probes-as-targets give a "more targets" stress test but the eval numbers aren't trustworthy as a leaderboard.

### VP selection diversification

The greedy 1K-VP set neutralizes geographic clustering. An additional ASN-balance ceiling (cap per ASN) prevents single-CDN-dominance (Cloudflare, OVH, HE). Combined with anchor holdout, this *probably* makes Octant's "exclude landmarks within K km of target" rule unnecessary — but we should measure before adding more knobs.

### Spotter-style global vs per-VP fits

Spotter's complaint about per-VP sample size matters: per-VP curves see at most ~720 anchors → ~580 in a K=5 training fold. The LTD interface should support both:
- **Per-VP curves** (Octant, bounded_spline): fit independently per VP using its rows in the training fold.
- **Globally-pooled fits** (Spotter, optionally NormalDist): fit μ on the union of all training rows, σ per VP. Same train/test split; just consumed differently.

## Caveats

- **The proposal is not finalized.** Open design questions listed in `report.md`. Implementation has not begun.
- **The current LTD interface (`scripts/framework/v2/ltd/base.py`) takes a single `fit(samples)` call.** A fold-aware protocol needs either (a) the runner to fit-then-predict per fold (heavier orchestration), or (b) LTDs to expose `fit_with_holdout(samples, eval_ids)` (cleaner interface). API change is non-trivial.
- **The materialize step writes one fit + one eval parquet today.** Folded materialize either writes K fit parquets + K eval parquets, or one big parquet with a `fold` column and the runner filters at read time. The latter is probably cleaner.
- **`ComboSpec` does not currently have a holdout axis.** Adding `holdout_policy` lets us sweep paper-faithful vs leakage-free as combo variants instead of forking the codebase.
- **Compute cost of K=5 across all combos is ~5× the current run cost.** For ~10 combos × paper-faithful + leakage-free = ~100 combos × ~10 minutes ≈ ~17 hours. Acceptable but worth budgeting.
- **VP corpus diversification interacts with already-deployed combos.** Switching from "all probes" to "greedy 1K + ASN-balance" changes the baseline numbers we report; we'd want to run both for transparency.
- **Reproducibility tradeoff.** The IMC 2023 paper does not split; adopting K-fold means our headline numbers are not directly comparable to theirs (a feature, not a bug, but worth documenting).
