# VP Selection — Agreement-Methodology Replication — Plan

> **Status: NOT FINALIZED** — depends on the leakage-free CBG eval protocol task ([../20260523-leakage-free-cbg-eval-protocol/](../20260523-leakage-free-cbg-eval-protocol/)). Resolves its Phase 0 Q3 (VP-corpus default).

## Background

The leakage-free CBG eval protocol design is stuck on Phase 0 Q3: which VP-corpus strategy (random / greedy-geo / +ASN-cap / H1-AS / H2-AS / DIST-RTT) should be the default. We have a strong hint from Cho et al. (TMA 2024, *Selection of Landmarks for Efficient Active Geolocation*) — H2-AS reaches 100% accept/reject agreement with 32% of their 780-anchor pool. But two things prevent us from just adopting their default:

1. **Pool size mismatch.** Cho selects from 780 RIPE anchors; we select from ~12K RIPE probes. The 32%/280 number won't transfer linearly.
2. **Metric mismatch.** Cho's "agreement" is country-level accept/reject — they had no hard GT (commercial VPN endpoints with only operator-claimed country). We have hard anchor coordinates as GT, so we can score selection strategies on actual point-estimate error, not just agreement.

This task replicates Cho's agreement methodology *on our setup*, generalized to hard-GT point-error, and produces the sweep that picks our VP-corpus default.

## Context

### Upstream

Cloned `https://github.com/grace71/tma24-vp-ls` into [scripts/vp_selection/](../../scripts/vp_selection/):
- [upstream_py/analyze_air.py](../../scripts/vp_selection/upstream_py/analyze_air.py) — geodesic Prim + cluster/H1/H2 variants + random-100 init. Reference impl of every selection strategy in the paper.
- [upstream_py/analyze_topo.py](../../scripts/vp_selection/upstream_py/analyze_topo.py) — RTT Prim (DIST-RTT).
- [upstream_py/retrieve_topo.py](../../scripts/vp_selection/upstream_py/retrieve_topo.py) — mesh-ping retrieval. We don't need this — `anchors_meshed_pings` in ClickHouse covers it.
- [upstream_py/send-ping.py](../../scripts/vp_selection/upstream_py/send-ping.py) — VPN ping orchestration. Not relevant for us.
- [upstream_csv/anchorSelectionAll.csv](../../scripts/vp_selection/upstream_csv/anchorSelectionAll.csv) — 780-anchor metadata (pid, lat, lon, city, country, asn). Reference shape for our adapted loader.
- [upstream_csv/iso3166.csv](../../scripts/vp_selection/upstream_csv/iso3166.csv) — ISO-3 ↔ ISO-2 mapping.

### What upstream doesn't ship (we substitute)

- `csv/final_result.csv` — VPN agreement results, withheld for anonymity. We replace with our own *full-pool verdict* computed per CBG variant on our 723 anchors.
- `pickle/lm_dist.pickle` — pairwise geodesic-distance cache. Trivial to regenerate (haversine O(N²)).
- `pickle/mesh_pings_*.pickle` — replaced by ClickHouse `anchors_meshed_pings`.
- VPN-targets CSV — replaced by our anchor targets (`tg_configs.parquet`).

### Calibrated constant worth lifting

Cho's calibrated speed limit is **0.51 c = 153 km/ms** (their Fig. 2), tighter than the standard 2/3 c we use (`default.SPEED_OF_INTERNET`). They calibrated against anchor pairwise distances. Worth re-calibrating on our 723-anchor mesh before adopting.

## Goals

1. **Replicate Cho's selection strategies** as a clean reusable module: `random`, `dist_geo`, `dist_rtt`, `h1_as`, `h1_city`, `h2_as`. Each takes a candidate VP pool + metadata and emits `{k: [vp_ids]}` for k = 1..N.
2. **Build the agreement harness** adapted for hard GT: for each (CBG variant, strategy, k) compute both
   - *Agreement* with the full-pool verdict (Cho's framing — preserved-decision check).
   - *Accuracy* against ground-truth coordinates (our framing — actual quality).
3. **Recalibrate speed limit** on our 723-anchor mesh (mirror Cho's Fig. 2 protocol) and verify intra-day RTT stability ≥99.5%.
4. **Pick the default VP corpus** based on the strategy that achieves high agreement with the smallest K *and* shows acceptable accuracy. Document the tradeoff curve for transparency.
5. **Resolve the parent task's Phase 0 Q3** with measurement, not opinion.

## Approach

### Step 1 — Speed-limit calibration in [scripts/vp_selection/calibrate_speed.py](../../scripts/vp_selection/calibrate_speed.py) ✓ done

**First deliverable — complete.** Result: **S = 185.56 km/ms (p99, +21.3% vs Cho)**.

1. **Source**: `anchors_meshed_pings` (anchor↔anchor min-RTT), restricted to the SOI-survived anchor set from `RipeAtlasSource._compute_soi_removed_ips`. Both endpoints are hard-GT + datacenter-grade — the cleanest input we have for this kind of calibration.
2. **Per-anchor envelope fit**: for each anchor treated as a "VP", construct `FitSample`s from its outgoing mesh pings (`vp_coord` = this anchor, `probe_coord` = other anchor, `latency` = min-RTT). Call `RTTDistanceModel.fit()` directly (the LP under `LowEnvelopeLTD`) with its production default `baseline_slope = THEORETICAL_SLOPE = 0.01 ms/km` (= 2/3·c speed cap = SOI). Same LP constraint that benchmark runs use — no methodological asymmetry.
3. **Per-anchor implied one-way speed**: $v_i = 2 / \text{slope}_i$ km/ms (factor 2 because RTT is round-trip; asymptotic at large $d$ when intercept becomes negligible).
4. **Detect pegged anchors**: any with `slope_i ≤ THEORETICAL_SLOPE + ε` (= speed ≥ 200 km/ms = SOI cap) are degenerate fits — their data implies marginal propagation faster than 2/3·c, so the LP pinned them at the floor. Causes: GT slippage, clock skew, sparse-data overfit. Exclude from the headline calibration; keep in the JSON diagnostics. **Our run: 6 of 757 anchors pegged.**
5. **Headline S = p99** over non-pegged anchors. Raw max can be outlier-sensitive (one anchor's noisy fit can drive max); p99 captures the fast-network tail without a single bad fit dominating. Max, p95, p50, etc. retained as diagnostics.
6. **Outputs**: `outputs/speed_calibration.json` (S, full distribution, per-anchor records flagged with `pegged_at_baseline`, list of fastest non-pegged + pegged anchors) and `outputs/speed_calibration.png` (family of per-anchor envelopes, p99 line, Cho reference, SOI/LP-floor reference, pegged anchors highlighted).
7. **Cross-check**: our $S$ is +21.3% over Cho's 153 km/ms — plausibly explained by 2-3 yr network evolution + different anchor pool. p50 = 131.7 km/ms matches Katz-Bassett's 133 km/ms cited by Cho.

Notes:
- Only the scalar $S$ extrapolates downstream (used to constrain probe RTT triples in the agreement verifier). The per-anchor envelope curves themselves stay anchor-only — they are GT-specific to each anchor.
- **SOI ≠ LP-floor pegging.** SOI's per-pair check `2·d/rtt > 200 km/ms` treats *all* of RTT as propagation; the LP fit absorbs per-pair setup delay into an intercept, so `2/slope` is *marginal* speed at large $d$. The two checks catch different things; both are necessary.
- **Stability check deferred**: requires a per-timestamp ClickHouse query, not exposed by `compute_rtts_per_dst_src` today. Tracked in todo as a Phase 2 follow-up.

### Step 2 — Lift selection algorithms into [scripts/vp_selection/strategies.py](../../scripts/vp_selection/strategies.py)

Distill `analyze_air.py::_select_prim` + `select_prim` into a single function:

```python
def select_vps(
    pool: dict[int, VpMeta],           # {vp_id: VpMeta(lat, lon, asn, city, country)}
    pair_distances: dict[tuple, float], # geodesic OR rtt graph edge weights
    strategy: Literal["random", "dist_geo", "dist_rtt", "h1_as", "h1_city", "h2_as"],
    seed: int,
) -> dict[int, list[int]]:              # {k: [vp_ids in selection order]}
```

Fold-invariant: one corpus per (strategy, seed), reused across all K=5 anchor folds and all CBG variants. Keeps selection-variance out of fold-variance.

### Step 3 — Pair-distance generators

- **Geodesic graph**: `O(N²)` haversine over the candidate VP pool. Cache to `outputs/pair_distances_geo.parquet`.
- **RTT graph**: per-VP min-RTT among the VP↔VP measurements. Sourced from `anchors_meshed_pings` (anchor pool only); for probes we don't have probe↔probe RTTs (would require ~144M new measurements per the leakage-free task plan). So `dist_rtt` is only runnable with anchors-as-pool, not probes-as-pool — document this constraint.

### Step 4 — Full-pool verdict + agreement harness in [scripts/vp_selection/agreement.py](../../scripts/vp_selection/agreement.py)

For each CBG variant (Octant spline, bounded_spline, NormalDist, Spotter, SoI):

1. Run benchmark with full VP pool → store $\hat{p}_\text{full}(t)$ per target $t$.
2. For each strategy × $k \in \{50, 100, 200, 400, 800, 1600, 3200\}$ (log scale):
   - Subset = strategy's first $k$ VPs.
   - Run benchmark with subset → $\hat{p}_\text{sub}(t)$.
   - Compute *agreement* = fraction of targets where $\lVert \hat{p}_\text{full}(t) - \hat{p}_\text{sub}(t) \rVert < \varepsilon$. Default $\varepsilon = 40$ km (the IMC 2023 paper's primary threshold). Also report at $\varepsilon \in \{0, 100, 500\}$.
   - Compute *accuracy* = median, p25, p75 of $\lVert \hat{p}_\text{sub}(t) - p_\text{GT}(t) \rVert$.
   - The calibrated $S$ from Step 1 is the speed-limit constraint used by the verifier when classifying RTT-implied violations.
3. Output one parquet per (variant, strategy) with columns `k, agreement_40km, agreement_100km, accuracy_median, accuracy_p75, ...`.

### Step 5 — ComboSpec integration (deferred to parent task)

Once strategies + agreement harness work standalone, wire `vp_corpus_strategy` as a `ComboSpec` axis so the leakage-free benchmark can sweep it alongside `holdout_policy`. This is where this task hands off back to the parent.

## Caveats

- **`dist_rtt` is anchor-pool-only.** We don't have probe↔probe RTT measurements at scale (12K² ≈ 144M pings). Either run `dist_rtt` on the 723-anchor subset only and accept the small-pool comparison, or skip it for the probe-pool sweep.
- **Pool-size headroom unknown.** Cho's 32% number is for 780-anchor pool; with ~12K probes we may either (a) plateau much sooner (large pool has more redundancy) or (b) need a relatively larger K because the diversity tail is fatter. We'll know from the sweep.
- **"Agreement" with full-pool can be misleading** if the full-pool point estimate is itself bad. Always report agreement *alongside* accuracy — never as a standalone signal.
- **Strategies have stochastic components.** `random`, `h2_as` (random-100 init) need multiple seeds. Default: 5 seeds, report mean±std at each K.
- **Cho excluded probes by design.** Their argument: probes have noisy GT and bandwidth limits. We have to include probes (they're the bulk of our VP pool). The IMC 2023 greedy filter already drops the worst-GT probes; this task layers structural diversity on top, doesn't try to redo GT cleanup.
- **Pure refactors of upstream code stay in `upstream_py/`.** Our adapted/lifted versions go in `strategies.py`, `agreement.py`, `calibrate_speed.py` at the `scripts/vp_selection/` top level. Keep the line clean so it's obvious what's pristine third-party vs ours.
- **Compute cost.** Per-variant full sweep ≈ 7 strategies × ~7 K-values × 5 seeds × 1 benchmark run. With ~10 minutes/run, that's ~40 hours per variant. Budget accordingly; the harness should checkpoint so we can resume.
- **Not finalized.** Open: (a) which CBG variant to use for the full-pool verdict computation — do we sweep this too, or fix one as the reference? (b) ε threshold for "agreement" — 40 km matches IMC 2023 but may be wrong for our setup; needs to be measured. (c) whether to include `h1_country` / `h1_continent` (Cho found they don't beat random — probably skip).
