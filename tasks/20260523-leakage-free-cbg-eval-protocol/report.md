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

---

## 2026-05-24 — Implementation complete

**Status**: Phase 0–2 complete; Phase 3 partially complete (unit + smoke tests done; leakage measurement pending)
**Last Updated**: 2026-05-24

### Design decisions locked in (Phase 0)

| Decision | Choice |
|---|---|
| Fold count | K=5 (uniform across all LTD variants) |
| Split axis | Anchor-level (not probe-level) |
| Algorithm | Sechidis et al. 2011 iterative multi-label stratification |
| Spatial blocking | Roberts et al. 2017 k-means pre-clustering (`spatial_clusters=30`) |
| ASN bucketing | Top-20 ASNs get own bucket; rest → `"other_AS"` |
| Materialize shape | Separate `slice_id()` per fold (no changes to materialize/runner) |
| ANCHORS_TO_PROBES | Holdout stripped at construction with WARNING; treated as secondary pressure test |
| VP corpus | Deferred; K-fold protocol is VP-selection-independent |

### New files

- **`scripts/benchmark/v2/sources/holdout.py`** — self-contained algorithm module
  - `HoldoutPolicy(frozen dataclass)`: k=5, fold_index, seed=42, labels=("country","asn_bucket"), asn_bucket_top_n=20, spatial_clusters=30
  - `AnchorInfo(NamedTuple)`: ip, lat, lon, country, asn
  - `compute_fold_assignments(anchors, policy) → dict[str, int]`: public entry point
  - Uses `scipy.cluster.vq.kmeans2` on 3D unit-vector projection (lat/lon → (x,y,z)); handles antimeridian wrap-around
  - Sechidis: unit-first ordering (by rarest label), sum-of-needs scoring → country + ASN-bucket max-min ≤ 2

- **`scripts/benchmark/v2/tests/test_holdout.py`** — 17 unit tests
  - Covers: determinism, input-order invariance, disjointness, balanced sizes, country balance, singleton ASN, ASN bucketing (None/top-N/other), spatial cluster atomicity, clamp behavior, HoldoutPolicy validation, slice_suffix format

### Modified files

- **`scripts/benchmark/v2/sources/ripe_atlas.py`**
  - `holdout: Optional[HoldoutPolicy] = None` constructor param
  - `slice_id()` returns `"{base}__{fold{i}of{k}_seed{s}}"` when holdout set
  - `_apply_holdout()` called after `_apply_slice()`; populates `_train_anchors` / `_test_anchors`
  - `iter_fit_samples()` skips anchors not in `_train_anchors` when holdout set
  - `iter_eval_targets()` skips anchors not in `_test_anchors` when holdout set
  - `iter_vp_configs()`, `iter_tg_configs()` unchanged

- **`scripts/benchmark/v2/tests/test_sources.py`** — added `TestRipeAtlasSourceHoldout` (6 tests)
  - fit vs eval anchor IDs disjoint
  - sweep 0..K-1: union of eval sets = full corpus; pairwise fold disjointness
  - slice_id carries fold suffix
  - vp_configs count unchanged
  - ANCHORS_TO_PROBES strips holdout + no suffix + logs WARNING

- **`scripts/benchmark/v2/sources/RIPE_ATLAS_DATA.md`**
  - `holdout` row added to constructor knobs table
  - Full "Holdout (leakage-free fit / eval split)" section added with algorithm, HoldoutPolicy params, slice_id directory layout, per-setup behavior, spatial vs label balance tradeoff

### Test results

```
54 tests total (test_holdout.py × 17 + test_sources.py × 37)
PASS: 54 / FAIL: 0 / ERROR: 0
```

### Algorithm notes

**Initial label-first approach caused country max-min=3** (C2 distribution: [3,2,2,0,3]).
Root cause: processing labels one at a time locked placements for rare labels that then forced a bad fold for common labels.

**Fix**: switch to unit-first with sum-of-needs scoring:
- Sort units by rarity of their rarest label (rarest unit first)
- For each unit, score each fold by **sum** of remaining_need across all the unit's labels
- Sum (not max) means a unit carrying multiple under-served labels prefers folds starved on multiple axes
- After fix: country max-min ≤ 2, ASN-bucket balance maintained jointly

**Spatial clustering tradeoff** (Roberts et al. 2017):
- `spatial_clusters=30` (default): spatial atomicity preserved; US, DE may show per-fold country imbalance of ~10–15%
- `spatial_clusters=None`: per-country max-min ≤ 2 across all folds; spatial autocorrelation leakage path not blocked
- Roberts et al. recommendation: report both and treat the gap as the "autocorrelation premium"

### Pending (Phase 3)

- [ ] Run bounded_spline in paper-faithful vs K=5 mode; measure optimism delta
- [ ] Re-run all CBG variants under new protocol; tabulate vs current numbers
- [ ] Document in `papers/cbg-variant-benchmark-proposal/`

---

## 2026-05-24 — Methodology decision: anchor-level K-fold with stratification

After re-deriving the protocol from how CBG actually consumes train/eval data
(per-VP RTT-distance fitting, then multi-VP centroid prediction), the
methodology is locked to **anchor-level K-fold cross-validation with
stratification**.

### What's locked in

- **K-fold granularity: anchor-level (not pair-level).** Per-pair K-fold
  leaks because CBG aggregates RTT→distance predictions across many VPs at
  prediction time. If anchor X's pairs are split across folds, eval on X
  still uses VPs whose LTDs were fit on `(VP_i, X)` — those VPs have
  memorized X's exact RTT-distance point. Holding X out as a unit, across
  all VPs simultaneously, is the only way to make the multi-VP centroid
  prediction leakage-free.

- **Stratification axes (per fold):**
  1. **ASN diversity** — each fold's eval set must span ASN buckets so the
     accuracy headline isn't dominated by one CDN's geometry (Cloudflare,
     Vultr, Google).
  2. **Spatial spread** — each fold's eval set must include both regional
     and global distances from any given VP, so the benchmark tests
     short-range and long-range geolocation under equal weight.

### What's rejected

| Alternative | Why rejected |
|---|---|
| **Per-pair K-fold** (split `(VP, anchor)` rows) | Leaks via multi-VP centroid aggregation — anchor X's LTD point exists in at least one VP's training set as long as one of X's pairs is in train. Anchor is the atomic leakage unit, not the pair. |
| **Random K-fold (no stratification)** | Mean accuracy is unbiased, but per-fold variance is high on the long tail (a 5-anchor country has ~33% chance of zero coverage in some fold), making reported numbers noisier than necessary. |
| **Spatial k-means as headline mode** | Answers a different question ("novel-metro generalization") at the cost of ASN/country balance. Kept as the robustness-check companion, not the leaderboard. |

### Stratification algorithm — open

Two candidates implementing the locked-in protocol:

1. **Sechidis multi-label stratification** (already implemented in
   `scripts/benchmark/v2/sources/holdout.py`) — balances country + ASN-bucket
   via iterative greedy assignment.

2. **DistGeoKFoldPolicy** (proposed; see Phase 4 in `todo.md`) — per-ASN-
   bucket greedy-Prim ordering + round-robin into K folds. Reuses
   `select_vps(strategy="dist_geo")` from `scripts/vp_selection/strategies.py`.
   Each fold gets ~1/K of each ASN bucket with explicit intra-bucket spatial
   spread.

Plan: implement DistGeoKFoldPolicy alongside Sechidis, compare fold
composition stats (country/ASN balance, intra-fold pairwise-distance
distribution) and downstream CBG accuracy under both policies. Pick whichever
gives cleaner per-fold spatial diversity without sacrificing ASN balance.

---

## 2026-05-24 — DistGeoKFoldPolicy: implementation complete (Phase 4)

### Shipped

- **`DistGeoKFoldPolicy`** (frozen dataclass, `kind="dist_geo_kfold"`) in
  [holdout.py](../../scripts/benchmark/v2/sources/holdout.py). Parameters:
  k=5, fold_index, seed=42, asn_bucket_top_n=20.

- **`compute_dist_geo_fold_assignments(anchors, policy)`**: algorithm per
  the approved plan —
  1. ASN bucket via existing `_bucket_asns` (top-N + "other_AS").
  2. For each bucket: pairwise haversine → `select_vps(strategy="dist_geo")`
     (reused from [scripts/vp_selection/strategies.py](../../scripts/vp_selection/strategies.py))
     → greedy-Prim ordered list.
  3. Balanced round-robin: preferred fold = `i % k`; shifts to smallest fold
     when current fold is already 2+ ahead of the global min (absorbs
     singleton-bucket placements across the corpus).

- **Method-based dispatch**: both `HoldoutPolicy` and `DistGeoKFoldPolicy`
  now expose `compute_fold_assignments(self, anchors)` and `slice_suffix()`.
  `RipeAtlasSource._apply_holdout()` calls `self._holdout.compute_fold_assignments(...)`
  with no isinstance/kind checks — adding a third policy needs zero source
  changes.

### Test results

```
72 tests total (test_holdout × 17 + test_dist_geo_holdout × 18 + test_sources × 37)
PASS: 72 / FAIL: 0 / ERROR: 0
```

`TestRipeAtlasSourceHoldout` was parametrized via `subTest` so the 4
parametrizable tests run under both policies (Sechidis spatial=k, DistGeo
default). Two policy-agnostic tests (`test_holdout_none_preserves_full_eval_set`,
`test_anchors_to_probes_setup_strips_holdout_with_warning`) stayed
single-policy.

### Open question that surfaced during implementation

dist_geo's seed enters only via `select_vps._max_edge_start` (random pick of
one endpoint of the max-distance edge). On symmetric synthetic corpora both
endpoints can produce equivalent orderings → different seeds give identical
outputs. The Sechidis-style "different seed differs" assertion does not hold
for DistGeo. Captured in `lesson.md`; the test was rewritten to assert "seed
is accepted and produces a valid mapping" rather than "different seeds
differ."

### Still pending (Phase 4 verification)

- [ ] Downstream accuracy comparison: bounded_spline + other LTDs under
  both policies, median accuracy + leaderboard stability check

### Real-data fold composition comparison — 2026-05-24

End-to-end smoke test against live ClickHouse (`ping_10k_to_anchors`,
`anchors_meshed_pings`) + the bundled `reproducibility_probes_and_anchors.json`.
752 anchors after sanitization, 9683 VPs.

| Metric | HoldoutPolicy (Sechidis, spatial_clusters=30, seed=42) | DistGeoKFoldPolicy (seed=42, asn_bucket_top_n=20) |
|---|---|---|
| Fold sizes (anchors) | 91 / 90 / 212 / 114 / 245 | 151 / 151 / 150 / 150 / 150 |
| Country max-min spread | **68** (DE: 0/0/68/2/32) | **6** (GB/NL/RU) |
| Country mean spread | 6.31 | 1.94 |
| ASN-bucket max-min spread | **135** (other_AS: 62/66/166/91/197) | **3** (other_AS: 116/115/118/118/115) |
| ASN-bucket mean spread | 9.14 | 1.14 |
| Intra-fold mean pairwise km | 4357–8088 (mean 6231) | 5260–6464 (mean 6004) |

Sechidis with spatial blocking gives the expected spatial-atomicity profile
— DE lands almost entirely in fold 2, NL+GB in fold 4. The 30 k-means
clusters are uneven in size and that cascades into uneven fold sizes
(91…245). That's the autocorrelation-protection cost.

DistGeo gives near-perfect balance: ~150 anchors/fold, country spread ≤ 6,
ASN-bucket spread ≤ 3, intra-fold pairwise-distance distribution flat
across folds. Exactly the "balanced + diverse" profile we designed for —
at the cost of not blocking metro overlap between train and test.

These are the Roberts et al. extremes; the gap between them is the
"autocorrelation premium" that the downstream LTD accuracy comparison
should quantify.
