# VP Selection — Agreement-Methodology Replication — Report

**Status**: Step 1 (speed-limit calibration) complete; Step 2+ design in progress
**Created**: 2026-05-23
**Last Updated**: 2026-05-23

## Summary

Captures an in-progress plan to replicate Cho et al. (TMA 2024)'s VP-selection agreement methodology on our 723-anchor / ~12K-probe pool, generalized to a hard-GT setting. The output of this task is the measurement that resolves Phase 0 Q3 of the parent leakage-free CBG eval protocol task.

Upstream code from `https://github.com/grace71/tma24-vp-ls` is checked into [scripts/vp_selection/upstream_py/](../../scripts/vp_selection/upstream_py/) and [scripts/vp_selection/upstream_csv/](../../scripts/vp_selection/upstream_csv/) — pristine, with `UPSTREAM_LICENSE` preserved. Our adapted versions live as `strategies.py`, `agreement.py`, `calibrate_speed.py`, `pair_distances.py` at the `scripts/vp_selection/` top level.

**Step 1 done**: [scripts/vp_selection/calibrate_speed.py](../../scripts/vp_selection/calibrate_speed.py) implemented and run. **S = 168.62 km/ms (p99)** over 548 fitted anchors (`n_measurements ≥ 100`, post-SOI). 212 anchors skipped for low sample count, 1 anchor pegged at the LP floor (`146.185.219.73` — n=292 but slope still hits 0.01, real GT/clock anomaly). Detailed results in [outputs/speed_calibration.json](../../scripts/vp_selection/outputs/speed_calibration.json) and [outputs/speed_calibration.png](../../scripts/vp_selection/outputs/speed_calibration.png).

## Findings

### Step 1 result: calibrated speed limit S = 168.62 km/ms (p99)

- Anchor-mesh post-SOI, **filter `n_measurements ≥ 100`**, per-anchor LP best-line via `RTTDistanceModel.fit()` with its production default `baseline_slope = THEORETICAL_SLOPE = 0.01 ms/km` (= 2/3·c cap = SOI). p99 over fitted slopes.
- 761 anchors total in mesh; **212 skipped** for low sample count; 548 fitted; **1 pegged** at LP floor (`146.185.219.73`, Tel Aviv, n=292 — high-n but still pegs at slope=0.01, real GT-or-clock anomaly, excluded).
- Distribution (km/ms, excluding pegged): min 88.7, p25 119.0, **p50 128.2** (≈ Katz-Bassett 133), p75 139.3, p95 165.0, **p99 168.6**, max 186.0.
- p99 chosen as headline instead of max — see lesson on outlier-sensitivity. **+10.2% over Cho 2024's 153 km/ms** (down from +21.3% before the n-filter); the previous gap was inflated by 5 sparse-data anchors whose LP slopes were noisy artifacts, not network signal.

### Settled (or close to settled)

- **Speed-limit calibration locked in.** Anchor-mesh (post-SOI filter) + per-anchor LP best-line via `RTTDistanceModel.fit()` with its production `baseline_slope = THEORETICAL_SLOPE = 0.01 ms/km` + p99 over fitted anchors. Probes are unsuitable for this step despite matching the eventual benchmark data — VP-side last-mile + GT noise survives the SOI filter and would inflate $S$. Only the scalar $S$ extrapolates downstream to probe data; per-anchor envelope curves stay anchor-only.
- **SOI and the LP measure different things.** SOI's per-pair predicate `2·d/rtt > 200 km/ms` treats *all* of RTT as propagation; the LP fits `rtt = slope·d + intercept` where intercept absorbs per-pair setup delay, so `2/slope` is the *marginal* propagation speed. Post-SOI data can therefore still have anchors where the LP pegs at the SOI floor (6 of 757 in our run). SOI is necessary, not sufficient — LP-floor pegging is the additional check that catches the residual cases.
- **Two metrics, both required.** "Agreement with full-pool verdict" (Cho's metric — preserves their question) AND "accuracy vs hard GT" (our generalization — answers what we actually care about). Reporting only one is misleading: a high-agreement subset can be efficient *and bad*, and a low-agreement subset can be *better than full-pool* if the full pool is noisy.
- **Strategies to include in sweep**: `random`, `dist_geo`, `dist_rtt` (anchor-pool only), `h1_as`, `h1_city`, `h2_as`. Skip `h1_country` / `h1_continent` (Cho's data shows they don't beat random).
- **Fold-invariant selection.** Pick the VP corpus *once globally* per strategy, reuse across all K=5 anchor folds. Otherwise selection-noise contaminates fold-variance. (Aligns with parent task's protocol.)
- **`dist_rtt` is anchor-pool-only.** No probe↔probe RTT mesh exists; running it would require 12K² ≈ 144M new pings. Anchor-subset `dist_rtt` is the most we can do.
- **Reuse Cho's intra-day stability check.** Their Fig. 2 (≥99.5% RTT correlation hour-to-hour) is reusable as a sanity check on our calibration window.

### Open questions

1. **Full-pool verdict — one reference variant, or one per variant?** Computing agreement *per variant* against a per-variant full-pool verdict is methodologically clean (each variant judged against its own best signal). But it conflates two questions: "is the subset good?" and "is the variant good?". Computing agreement against a single reference variant's verdict (e.g., SoI) lets us compare strategies cleanly across variants.
2. **ε threshold for agreement.** 40 km matches IMC 2023's primary threshold; 100 km matches CBG's typical city-level. Probably report multiple ε's.
3. **K sweep points.** Default proposal: 50, 100, 200, 400, 800, 1600, 3200. Open: is 3200 enough headroom, or should we go further toward the full ~10K pool?
4. **Seed count for stochastic strategies.** 5 is the default; raise if variance is large.
5. **What anchors as the "pool" for `dist_rtt`?** All 723, or the per-fold training subset (~580 anchors)? Per-fold means the RTT graph changes per fold, which breaks fold-invariance — almost certainly use all 723.
6. **`pycountry_convert` substitution.** Upstream uses this library for continent lookup. We probably want a static continent mapping JSON to avoid a dependency. Open.

### Reference material on hand

- Cho et al. (TMA 2024) — *Selection of Landmarks for Efficient Active Geolocation*. Audit notes in [papers/cbg-variant-benchmark-proposal/refs/Cho_2024_Landmark_Selection.md](../../papers/cbg-variant-benchmark-proposal/refs/Cho_2024_Landmark_Selection.md).
- Upstream code: 4 scripts in `scripts/vp_selection/upstream_py/`, 2 CSVs in `upstream_csv/`.
- Key reusable functions: `analyze_air.py::_select_prim` (lines 253–321), `analyze_air.py::select_prim` (lines 323–404), `analyze_air.py::analyze_distance` (lines 216–243, geodesic pair-distance).

## Conclusions

To be written once the sweep produces measurements. Currently a planning document, not a decision document.
