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
