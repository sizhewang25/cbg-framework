# VP Selection — Agreement-Methodology Replication — Todo

> Depends on [../20260523-leakage-free-cbg-eval-protocol/](../20260523-leakage-free-cbg-eval-protocol/). Phase 4 (ComboSpec wiring) is blocked on parent's Phase 1 (API design).

## Phase 0: Open design decisions (must resolve before Phase 2)
- [ ] Choose which CBG variant computes the full-pool verdict for the agreement metric (one reference variant, or one verdict per variant)
- [ ] Decide ε threshold for "agreement" (default 40 km from IMC 2023; verify on data)
- [ ] Decide whether to include `h1_country` / `h1_continent` (Cho found they don't beat random — likely skip)
- [ ] Decide K sweep points (default proposal: 50, 100, 200, 400, 800, 1600, 3200 on log scale)
- [ ] Decide seed count for stochastic strategies (`random`, `h2_as`) — default 5 seeds

## Phase 1: Setup & discovery
- [ ] Confirm `anchors_meshed_pings` schema in ClickHouse exposes the columns needed (target_id, source_id, min_rtt, timestamp)
- [ ] Verify the SOI-filtered anchor cohort produced by `RipeAtlasSource._compute_soi_removed_ips` is accessible / re-usable for calibration
- [ ] Confirm `LowEnvelopeLTD._fit` accepts the per-anchor `FitSample` shape we'll construct (`vp_coord` = this anchor, `probe_coord` = other anchor, `latency` = min-RTT)
- [ ] Read upstream selection logic in `analyze_air.py` and identify what changes for our pool size
- [ ] Verify upstream files compile/import after dropping `pycountry_convert` (or substitute with a static continent mapping)
- [ ] Sketch the `VpMeta` dataclass + the `select_vps()` signature

## Phase 2: Implementation
- [ ] **First deliverable.** Implement `scripts/vp_selection/calibrate_speed.py` — anchor-mesh post-SOI + per-anchor `LowEnvelopeLTD._fit` + fastest-envelope extraction ($S = 2 / \min_i \text{slope}_i$) + hourly stability check + JSON/PNG outputs
- [ ] Implement `scripts/vp_selection/strategies.py` — lift `_select_prim` + `select_prim`; expose `select_vps(pool, pair_distances, strategy, seed)` returning `{k: [vp_ids]}`
- [ ] Implement geodesic pair-distance generator (`scripts/vp_selection/pair_distances.py`) with parquet cache
- [ ] Implement RTT pair-distance generator (anchor-pool only, sourced from `anchors_meshed_pings`)
- [ ] Implement `scripts/vp_selection/agreement.py` — full-pool + subset benchmark runner; consumes calibrated $S$ from Step 1; outputs one parquet per (variant, strategy) with agreement-vs-K and accuracy-vs-K
- [ ] Add a thin CLI wrapper so each piece is runnable standalone for debugging

## Phase 3: Verification
- [ ] Calibration sanity: confirm intra-window stability of $S$ (≤1% drift across hourly recomputations); if not, document why
- [ ] Calibration cross-check: per-anchor speed distribution looks reasonable; flag any anchors with extreme slopes that warrant investigation
- [ ] Calibration cross-check: our $S$ within ~10% of Cho's 153 km/ms; if not, investigate before adopting
- [ ] Unit tests: strategy determinism (same seed → same selection), monotonicity (`select_vps(k=K)` ⊃ `select_vps(k=K−1)` where applicable), cluster-balance correctness for `h1_*`
- [ ] Sanity check: run `dist_geo` on Cho's 780-anchor input — compare against their published `anchorSelectionAll.csv` results to confirm we get the same selection sequence
- [ ] Spot-check the agreement metric: 100% selection should give 100% agreement and matching accuracy

## Phase 4: Integration (blocked on parent task Phase 1)
- [ ] Add `vp_corpus_strategy` axis to `ComboSpec`
- [ ] Wire the agreement-harness output into the leakage-free benchmark's `holdout_policy` × `vp_corpus_strategy` sweep
- [ ] Pick default VP corpus based on sweep results; update parent task's Q3 decision in `report.md`
- [ ] Document the chosen default and the runner-up in `papers/cbg-variant-benchmark-proposal/`
