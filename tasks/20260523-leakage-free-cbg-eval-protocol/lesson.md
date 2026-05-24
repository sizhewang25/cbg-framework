# Leakage-Free CBG Evaluation Protocol — Lessons

## 2026-05-23

Lessons will be captured here as corrections and discoveries occur during the task.

Initial framing (will likely be revisited):

- **LOO is intuitive but wrong for benchmarking deployable systems.** It implies per-query refitting, which is not what any production CBG service does. The right question is "given a model fit on a fixed landmark corpus, how well does it generalize to novel IPs?" — that's K-fold, not LOO.
- **Octant's institutional-exclusion rule was specific to their PlanetLab setup**, where universities had multiple nodes in the same building. RIPE Atlas anchors are clustered differently (a few CDNs hold many anchors in many metros), so blindly copying the rule may not be the right adaptation — the underlying concern (trivially-near landmarks) is different.
- **Probes as eval targets sound appealing for "12K targets" scale**, but probe GT noise contaminates the eval signal in a way per-VP calibration can't absorb. Anchors-only-as-targets is the right call even if it caps eval size at 723.

## 2026-05-24 — Implementation session

### Algorithm design

- **Label-first Sechidis breaks joint label balance.** The original Sechidis algorithm processes labels one at a time (rarest first) and assigns units greedily to that label's need. When there are multiple balance axes (country + ASN-bucket), locking placements for the rare-label pass can trap the common-label pass into unavoidable imbalance. Example: C2 ended up [3,2,2,0,3] across 5 folds. Fix: switch to **unit-first ordering** — process units in rarest-label order, score each fold by **sum** of remaining_need across **all** the unit's labels. Sum (not max) ensures a unit carrying multiple under-served labels prefers folds starved on multiple axes simultaneously, giving joint balance ≤ 2 for both axes.

- **Spatial blocking vs label balance is a fundamental tradeoff, not a bug.** Roberts et al. 2017 is explicit: spatial k-means atomicity deliberately trades label balance for spatial isolation. The recommended workflow is to run both (`spatial_clusters=30` and `spatial_clusters=None`) and treat the performance gap as the "autocorrelation premium" — i.e., the degree to which naive label-stratified CV inflates the benchmark score due to spatial autocorrelation. Don't try to fix the label imbalance away; quantify it.

### Dependency management

- **sklearn/scikit-learn is not in the project's pyproject.toml** despite being used in other parts of the codebase (indirect transitive dep). Do not rely on it in new modules. **scipy IS an explicit dependency** — use `scipy.cluster.vq.kmeans2(data, k, seed=seed, minit="++", missing="warn")` for k-means. The `seed` parameter sets the RNG; `minit="++"` gives k-means++ initialization, which is the scipy equivalent of sklearn's default.

### Source design

- **Strip holdout at construction for setups where it doesn't apply, not at iterator level.** ANCHORS_TO_PROBES has noisy probe GT; applying holdout there would produce confusingly-named directories (`fold0of5_seed42`) with no actual filtering. Stripping at `__init__` with a logged WARNING means the slice_id carries no suffix, logs are clean, and the caller sees the correct behavior without surprises later. Iterator-level silently-ignoring holdout is worse because it creates a mismatch between what the slice_id advertises and what the iterators emit.

- **Encoding fold identity in `slice_id()` (not in a separate field) was the right call.** It means materialize, inputs.py, and the runner need zero changes — they just see another source with a different string ID. Each fold appears as a separate output directory with no orchestration changes. The cost is that the user must invoke materialize K times (or write a trivial loop), which is acceptable for K=5 and eliminates complexity at the framework level.

## 2026-05-24 — DistGeoKFoldPolicy implementation

### Methodology

- **Per-pair K-fold leaks for CBG; anchor-level is the correct granularity.**
  Worked through this with the user. CBG aggregates RTT→distance predictions
  across multiple VPs at prediction time. If anchor X's `(VP_i, X)` pairs are
  split across folds, eval on X still uses VPs whose LTDs were trained on
  `(VP_i, X)` — those VPs have memorized X's RTT-distance point exactly. The
  leakage unit is the anchor, not the pair. Captured in `report.md` lock-in.

- **"Balanced + diverse folds" and "spatial blocking" pull in opposite
  directions.** Sechidis-style stratification and dist_geo round-robin both
  give each fold a *spread* set of anchors — so train and test share metros
  freely. Spatial k-means blocking does the opposite: each fold is one
  geographic cluster. These answer different scientific questions
  (novel-anchor in known metro vs novel metro). Don't conflate them — the
  Roberts et al. workflow is to run both and report the gap.

### Implementation

- **`select_vps`'s `dist_geo` mode is seed-light by design.** The seed only
  picks one endpoint of the max-distance edge in `_max_edge_start`; the
  greedy-Prim continuation is deterministic. On symmetric corpora, both
  endpoints lead to equivalent orderings → the seed parameter can be a
  no-op. Don't write a "different seed differs" test for DistGeoKFoldPolicy
  — assert that the seed is accepted and the result is valid instead.

- **Polymorphic dispatch via method (`compute_fold_assignments`) on each
  policy class is cleaner than isinstance/kind dispatch in the source.**
  Adding a third policy now requires zero changes to `RipeAtlasSource` — it
  only knows `policy.compute_fold_assignments(...)` and
  `policy.slice_suffix()`. Backward compat preserved by adding the same
  method to existing `HoldoutPolicy` (wraps the module-level function).

- **Balanced round-robin (smallest-fold tiebreak), not pure `i mod K`.**
  Singleton ASN buckets land first (per-bucket loop) and disrupt the
  modular cadence across buckets. Tracking `fold_sizes` and shifting to the
  smallest fold when the modular preference is ahead by 2+ keeps the
  global fold-size spread ≤ 1.

- **`scripts.vp_selection.strategies.VpMeta` is generic enough to reuse
  for anchors** — it carries lat/lon/asn/city/country, which is exactly the
  metadata dist_geo needs. No new "AnchorMeta" type required. Lazy-import
  from inside the algorithm function avoids pulling vp_selection at module
  load time.

- **Decouple the split decision from the source.** Originally `RipeAtlasSource`
  accepted any of three holdout policies and ran the algorithm at
  `_apply_holdout` time. Replaced with a single `PartitionPolicy` that reads
  a precomputed JSON written by `scripts/processing/ripe_atlas/partition.py`.
  Two wins: (a) the split is a reviewable artifact you can diff / version /
  visualize without re-running the source; (b) the source's active corpus
  (post-sanitize, post-RTT-filter) and the partition can disagree — the
  intersect-and-warn rule makes the mismatch visible instead of silently
  recomputing a different split. The algorithm classes (`HoldoutPolicy`,
  `DistGeoKFoldPolicy`) stay in `holdout.py` because `partition.py` still
  uses them. Lesson: when an artifact is both the input to a pipeline *and*
  the output of a deterministic computation, persist it explicitly rather
  than recomputing — the persisted form forces alignment and enables review.

- **For anchor-only SOI sanitization, only phase 1 (anchor-mesh) is safe.**
  The IMC 2023 pipeline runs a second phase against `ping_10k_to_anchors`
  to catch more violators. That phase greedily removes the IP with the
  most SOI violations on each iteration — fine when the goal is the union
  of bad anchors + bad probes, but unsafe for an *anchor-only* filter
  because the bad-GT side could be the probe. Phase 1 has both endpoints
  curated (anchors with paper-quality GT), so any flagged IP is
  unambiguously a bad anchor. Phase 1 alone on the canonical 723-anchor
  file flags 0 — confirming the canonical set is pre-curated, not
  revealing a bug.
