# VP Selection — Agreement-Methodology Replication — Todo

> Depends on [../20260523-leakage-free-cbg-eval-protocol/](../20260523-leakage-free-cbg-eval-protocol/). Phase 4 (ComboSpec wiring) is blocked on parent's Phase 1 (API design).

## Phase 0: Open design decisions (must resolve before Phase 2)
- [ ] Choose which CBG variant computes the full-pool verdict for the agreement metric (one reference variant, or one verdict per variant)
- [ ] Decide ε threshold for "agreement" (default 40 km from IMC 2023; verify on data)
- [ ] Decide whether to include `h1_country` / `h1_continent` (Cho found they don't beat random — likely skip)
- [ ] Decide K sweep points (default proposal: 50, 100, 200, 400, 800, 1600, 3200 on log scale)
- [ ] Decide seed count for stochastic strategies (`random`, `h2_as`) — default 5 seeds

## Phase 1: Setup & discovery
- [x] Confirm `anchors_meshed_pings` schema in ClickHouse exposes the columns needed (target_id, source_id, min_rtt, timestamp) — 714K rows accessible
- [x] Verify the SOI-filtered anchor cohort produced by `RipeAtlasSource._compute_soi_removed_ips` is accessible / re-usable for calibration — Phase 1 removes 9 IPs; Phase 2 redundant on same table
- [x] Confirm `LowEnvelopeLTD._fit` accepts the per-anchor `FitSample` shape we'll construct — yes, but its LP baseline (`THEORETICAL_SLOPE = 0.01 ms/km`) pegs calibration at 200 km/ms; have to drop down to `RTTDistanceModel.fit(baseline_slope=2/c)` directly
- [ ] Read upstream selection logic in `analyze_air.py` and identify what changes for our pool size
- [ ] Verify upstream files compile/import after dropping `pycountry_convert` (or substitute with a static continent mapping)
- [ ] Sketch the `VpMeta` dataclass + the `select_vps()` signature

## Phase 2: Implementation
- [x] **First deliverable.** Implement `scripts/vp_selection/calibrate_speed.py` — anchor-mesh post-SOI + `n_measurements ≥ 100` filter + per-anchor LP fit via `RTTDistanceModel.fit()` (production `baseline_slope = THEORETICAL_SLOPE = 0.01 ms/km`) + pegged-anchor detection + p99-as-headline + JSON/PNG outputs. **Result: S = 168.62 km/ms (+10.2% vs Cho)**; 548 fitted, 212 skipped low-n, 1 pegged (Tel Aviv anchor with high n=292 but still pegs — real GT/clock anomaly).
- [ ] Add hourly-window stability check to `calibrate_speed.py` (Cho's Fig. 2 analog) — requires a per-timestamp ClickHouse query, not currently exposed by `compute_rtts_per_dst_src`
- [ ] Implement `scripts/vp_selection/strategies.py` — lift `_select_prim` + `select_prim`; expose `select_vps(pool, pair_distances, strategy, seed)` returning `{k: [vp_ids]}`
- [ ] Implement geodesic pair-distance generator (`scripts/vp_selection/pair_distances.py`) with parquet cache
- [ ] Implement RTT pair-distance generator (anchor-pool only, sourced from `anchors_meshed_pings`)
- [ ] Implement `scripts/vp_selection/agreement.py` — full-pool + subset benchmark runner; consumes calibrated $S$ from Step 1; outputs one parquet per (variant, strategy) with agreement-vs-K and accuracy-vs-K
- [ ] Add a thin CLI wrapper so each piece is runnable standalone for debugging

## Phase 3: Verification
- [ ] Calibration sanity: confirm intra-window stability of $S$ (≤1% drift across hourly recomputations); if not, document why — blocked on hourly-window query support
- [x] Calibration cross-check: per-anchor speed distribution looks reasonable — median 128.2 km/ms ≈ Katz-Bassett 133 km/ms; p99 168.6; max 186.0 (clean tail after low-n filter)
- [x] Calibration cross-check: our $S$ within reasonable range of Cho's 153 km/ms — p99 is +10.2%; explained by 2-3 yr network evolution + different anchor pool
- [ ] Unit tests: strategy determinism (same seed → same selection), monotonicity (`select_vps(k=K)` ⊃ `select_vps(k=K−1)` where applicable), cluster-balance correctness for `h1_*`
- [ ] Sanity check: run `dist_geo` on Cho's 780-anchor input — compare against their published `anchorSelectionAll.csv` results to confirm we get the same selection sequence
- [ ] Spot-check the agreement metric: 100% selection should give 100% agreement and matching accuracy

## Phase 4: Integration (blocked on parent task Phase 1)
- [ ] Add `vp_corpus_strategy` axis to `ComboSpec`
- [ ] Wire the agreement-harness output into the leakage-free benchmark's `holdout_policy` × `vp_corpus_strategy` sweep
- [ ] Pick default VP corpus based on sweep results; update parent task's Q3 decision in `report.md`
- [ ] Document the chosen default and the runner-up in `papers/cbg-variant-benchmark-proposal/`
