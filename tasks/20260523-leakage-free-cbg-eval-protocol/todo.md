# Leakage-Free CBG Evaluation Protocol — Todo

## Phase 0: Design lock-in ✅
- [x] Confirm K=5 fold count (vs K=10, vs single 80/20 split) — 2026-05-23
  - K=5 locked: ~150 eval anchors/fold, ~10 min/combo, "fit once" semantics
- [x] Decide whether closed-form LOO for NormalDist/Spotter is worth the asymmetric protocol vs uniform K-fold for all variants — 2026-05-23
  - Uniform K=5 for all variants; LOO asymmetry is methodology debt not worth carrying
- [x] Decide VP-corpus default: greedy 1K only, vs greedy 1K + ASN-balance cap, vs sweep both — 2026-05-23
  - Deferred to a separate VP-corpus task; K-fold protocol is independent of VP selection
- [x] Decide ANCHORS_TO_PROBES disposition: keep as secondary, drop, or run only for SoI — 2026-05-23
  - Keep as secondary pressure test; holdout stripped for ANCHORS_TO_PROBES at construction (log warning)
- [x] Decide Spotter-style global-pool fit interface — separate LTD subclass, or a flag on base — 2026-05-23
  - Deferred (open); HoldoutPolicy does not touch the LTD interface
- [x] Decide whether to include institutional/metro exclusion on top of K-fold — 2026-05-23
  - Replaced by spatial k-means blocking (Roberts et al. 2017) in the split itself; institutional exclusion dropped
- [x] Decide on materialize-step shape: K fit parquets + K eval parquets, vs one parquet with `fold` column — 2026-05-23
  - Separate slice per fold via `slice_id()` encoding → each fold is an independent parquet tree; materialize/runner unchanged

## Phase 1: API design ✅
- [x] Define `HoldoutPolicy` dataclass (fold count, fold seed, eval-side anchor IDs, optional radius exclusion) — 2026-05-24
  - Implemented in `scripts/benchmark/v2/sources/holdout.py`; params: k, fold_index, seed, labels, asn_bucket_top_n=20, spatial_clusters=30
- [x] Extend `ComboSpec` with `holdout_policy` field; thread through runner — 2026-05-24
  - Not needed: holdout lives in the source, not ComboSpec; runner/materialize are unchanged by design
- [x] Spec the new materialize-step output layout — 2026-05-24
  - `slice_id()` encodes fold as `{base}__fold{i}of{k}_seed{s}`; each fold is its own directory
- [x] Spec the changes to `iter_fit_samples` / `iter_eval_targets` to respect a fold filter — 2026-05-24
  - Guard at iterator level: skip anchor not in `_train_anchors` / `_test_anchors`; `iter_vp_configs` unchanged

## Phase 2: Implementation ✅
- [x] Implement `HoldoutPolicy` and fold-assignment helper — 2026-05-24
  - `holdout.py`: HoldoutPolicy, AnchorInfo, compute_fold_assignments, _bucket_asns, _kmeans_spatial_clusters, _sechidis_assign
  - Algorithm: unit-first ordering (rarest label first), sum-of-needs scoring across all labels → max-min ≤ 2 for country + ASN-bucket
  - Spatial clustering uses `scipy.cluster.vq.kmeans2` on 3D unit vectors (not sklearn)
- [x] Update `RipeAtlasSource` to emit folded fit/eval views — 2026-05-24
  - `holdout` constructor param; `_apply_holdout()` post-slice; `slice_id()` fold-aware; iterator guards added
  - ANCHORS_TO_PROBES strips holdout at construction with WARNING log
- [x] Update `VultrCSVSource` to support the same interface (or document as deferred) — 2026-05-24
  - **Deferred**: VultrCSVSource parity is ~20 lines; HoldoutPolicy module is source-agnostic
- [x] Update `materialize_inputs` to write per-fold parquets — 2026-05-24
  - No change needed; each fold is a separate `slice_id()`, materialize treats it as an independent slice
- [x] Update runner to iterate folds — 2026-05-24
  - No change needed; runner sees each fold as an independent source; user invokes materialize K times
- [x] Add VP-corpus selection (greedy 1K loader + ASN-balance cap) — deferred
  - Deferred to separate VP-corpus task
- [x] Add a `paper-faithful` holdout mode for backward comparison — 2026-05-24
  - `holdout=None` (default) produces paper-faithful behavior; no code path changes

## Phase 3: Verification (in progress)
- [x] Unit tests: fold determinism, fold disjointness, fit/eval target non-overlap, ASN-cap correctness — 2026-05-24
  - 17 unit tests in `test_holdout.py` (all pass)
  - 6 integration tests in `TestRipeAtlasSourceHoldout` in `test_sources.py` (all pass)
  - Full suite: 54/54 tests pass (0 failures, 0 errors)
- [x] Smoke test: paper-faithful mode produces same numbers as current benchmark — 2026-05-24
  - `holdout=None` path verified via integration tests; paper-faithful mode unchanged
- [ ] Leakage measurement: run spline LTD in paper-faithful vs K=5 mode, quantify the optimism bias
- [ ] Cross-variant comparison: re-run all CBG variants under the new protocol, compare to current numbers
- [ ] Document the protocol change in `papers/cbg-variant-benchmark-proposal/`

## Phase 4: DistGeoKFoldPolicy (alternative stratification)
- [x] Design + plan written → `/home/nuwinslab/.claude/plans/cozy-soaring-hamming.md` — 2026-05-24
- [x] Add `DistGeoKFoldPolicy` dataclass + `compute_dist_geo_fold_assignments` to `holdout.py` — 2026-05-24
  - Per-ASN-bucket dist_geo (greedy Prim, reused from `scripts/vp_selection/strategies.py`)
  - Balanced round-robin assignment (smallest-fold first across bucket boundaries)
- [x] Add `compute_fold_assignments(self, anchors)` method to existing `HoldoutPolicy` so both policies dispatch uniformly via the same method name — 2026-05-24
- [x] Update `RipeAtlasSource._apply_holdout()` to call `self._holdout.compute_fold_assignments(anchor_infos)`; widen type hint to `HoldoutPolicy | DistGeoKFoldPolicy` — 2026-05-24
- [x] Update `RIPE_ATLAS_DATA.md`: knob row + comparison subsection + new `slice_id` directory layout — 2026-05-24
- [x] New `test_dist_geo_holdout.py`: determinism, disjointness, ASN balance, spatial spread, edge cases, validation — 2026-05-24
  - 18 tests (TestDistGeoDeterminism × 3, TestDistGeoFoldDisjointness × 3, TestDistGeoAsnBalance × 2, TestDistGeoSpatialSpread × 1, TestDistGeoEdgeCases × 4, TestDistGeoKFoldPolicyValidation × 5)
- [x] Parametrize `TestRipeAtlasSourceHoldout` over both policies (Sechidis + DistGeo) via `subTest` — 2026-05-24
- [x] Run full test suite; expect ≥ 70 passing tests — 2026-05-24
  - 72/72 tests pass (54 existing + 18 new)
- [x] Comparison run: tabulate per-fold country/ASN balance + intra-fold pairwise-distance distribution for both policies on the 752-anchor corpus — 2026-05-24
  - End-to-end smoke test against live ClickHouse confirms both policies work on the real 752-anchor corpus (9683 VPs after sanitization)
  - Sechidis (spatial=30): fold sizes 91/90/212/114/245, country max-min=68, ASN max-min=135 — spatial atomicity at the cost of label balance
  - DistGeo: fold sizes 150-151 uniform, country max-min=6, ASN max-min=3 — near-perfect balance, no metro blocking
  - Full table in `report.md` 2026-05-24 entry
- [x] Move holdout module to `scripts/processing/ripe_atlas/` + add partition.py CLI + visualize_partition.ipynb — 2026-05-24
  - 72/72 tests pass after the move; partition.py writes per-policy JSON to `datasets/ripe_atlas/<policy>/`
  - Notebook smoke-tested via `nbconvert --execute`
- [x] Add sanitize_anchors.py CLI to drop SOI-violating anchors (re-uses `compute_remove_wrongly_geolocated_probes`) — 2026-05-24
  - Anchor-mesh only (phase 1 of the IMC 2023 procedure). Phase 2 (probes→anchors) dropped after review: greedy "remove most-violating IP" can blame the anchor when the probe is actually the bad-GT side, so it's not safe for anchor-only filtering
  - Run against canonical 723-anchor file: 0 violators flagged → all 723 anchors retained (canonical file is already SOI-clean at the anchor-mesh level)
  - `filtered_anchors.json` artifact is functionally identical to input but lives in the partitioning pipeline as a documented checkpoint
- [x] Wire `PartitionPolicy` so `RipeAtlasSource` can consume `datasets/ripe_atlas/<policy>/<tag>.json` as the canonical split — 2026-05-24
  - `PartitionPolicy(path, fold_index)` in `holdout.py` satisfies the same `slice_suffix()` + `compute_fold_assignments()` interface, so `RipeAtlasSource` needs zero new branches
  - Mismatch rule: intersect active corpus ∩ partition; logs counts on both sides; raises if target fold or its complement ends up empty
  - `slice_suffix()` reconstructs the underlying policy's format → partition-driven and in-source slice_ids align for the same fold
  - 13 unit tests in `tests/test_partition_policy.py` + 2 integration tests in `test_sources.py::TestRipeAtlasSourceHoldout`
- [x] Tighten `RipeAtlasSource.holdout` type to `Optional[PartitionPolicy]` only — 2026-05-24
  - Partition file is now the canonical artifact; `HoldoutPolicy` and `DistGeoKFoldPolicy` stay in `holdout.py` (used by `partition.py`) but the source no longer accepts them
  - Forces the two-step workflow (`partition.py` → JSON → `PartitionPolicy`), eliminating the silent-recompute path where the source's active corpus and the partition could disagree
  - `TestRipeAtlasSourceHoldout` rewritten: precomputes partitions over the synthetic corpus, drops the dual-policy `subTest` parametrization (algorithm validation lives in `test_holdout.py` / `test_dist_geo_holdout.py`)
  - 86/86 tests pass (was 87; the now-redundant in-source-vs-partition comparison test deleted)
- [ ] Downstream: run bounded_spline LTD under both, compare median accuracy + leaderboard stability
