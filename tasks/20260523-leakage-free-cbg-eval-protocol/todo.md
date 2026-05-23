# Leakage-Free CBG Evaluation Protocol — Todo

> Proposal is not finalized. Phase 0 (design lock-in) must complete before implementation phases begin.

## Phase 0: Design lock-in (in progress — needs more discussion)
- [ ] Confirm K=5 fold count (vs K=10, vs single 80/20 split)
- [ ] Decide whether closed-form LOO for NormalDist/Spotter is worth the asymmetric protocol vs uniform K-fold for all variants
- [ ] Decide VP-corpus default: greedy 1K only, vs greedy 1K + ASN-balance cap, vs sweep both
- [ ] Decide ANCHORS_TO_PROBES disposition: keep as secondary, drop, or run only for SoI (leakage-moot there)
- [ ] Decide Spotter-style global-pool fit interface — separate LTD subclass, or a flag on the base
- [ ] Decide whether to include institutional/metro exclusion on top of K-fold (Octant rule), or measure-first and only add if needed
- [ ] Decide on materialize-step shape: K fit parquets + K eval parquets, vs one parquet with `fold` column

## Phase 1: API design (not started)
- [ ] Define `HoldoutPolicy` dataclass (fold count, fold seed, eval-side anchor IDs, optional radius exclusion)
- [ ] Extend `ComboSpec` with `holdout_policy` field; thread through runner
- [ ] Define LTD interface change for fold-aware fitting — propose `fit(samples)` stays, runner orchestrates fold filtering, OR add `fit_per_fold(samples, fold_assignments)` to base
- [ ] Spec the new materialize-step output layout
- [ ] Spec the changes to `iter_fit_samples` / `iter_eval_targets` / `iter_tg_configs` on `DataSource` to respect a fold filter

## Phase 2: Implementation (blocked on Phase 0 + 1)
- [ ] Implement `HoldoutPolicy` and fold-assignment helper
- [ ] Update `RipeAtlasSource` to emit folded fit/eval views
- [ ] Update `VultrCSVSource` to support the same interface (or document that it's deferred)
- [ ] Update `materialize_inputs` to write per-fold or fold-tagged parquets
- [ ] Update runner to iterate folds, fit per fold, evaluate the held-out fold
- [ ] Add VP-corpus selection (greedy 1K loader + ASN-balance cap)
- [ ] Add a `paper-faithful` holdout mode (no split) for backward comparison

## Phase 3: Verification (blocked on Phase 2)
- [ ] Unit tests: fold determinism, fold disjointness, fit/eval target non-overlap, ASN-cap correctness
- [ ] Smoke test: paper-faithful mode produces ~same numbers as current benchmark (sanity check)
- [ ] Leakage measurement: run spline LTD in paper-faithful vs K=5 mode, quantify the optimism bias
- [ ] Cross-variant comparison: re-run all CBG variants under the new protocol, compare to current numbers
- [ ] Document the protocol change in `papers/cbg-variant-benchmark-proposal/` so reviewers can see the methodology
