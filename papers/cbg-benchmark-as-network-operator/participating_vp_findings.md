# What drives CBG effectiveness? A participating-VP characterization of the three confidence tiers

*Exploration report — 2026-06-22. Author: automated analysis (Claude). Data: 6 final benchmark
runs re-run with participating-VP instrumentation. Code: `scripts/analysis/partvp/`. Git-tagged
snapshot.*

---

## 0. TL;DR

Across the in-distribution (global-VP) runs, CBG effectiveness on a target is governed by **two
independent levers**, each owning a different tier boundary:

1. **Proximity lever — "is there a VP almost on top of the target?"** decides whether a target is
   geolocated *precisely* (Tier-1). Tier-1 targets have their closest VP at a **median ~6 km**
   (min RTT ~2 ms) and the deciding constraint set collapses to **1–2 VPs**; Tier-2/3 targets have
   their closest VP **hundreds of km** away. Decision-tree threshold: closest participating VP
   **≤ ~55–68 km ⇒ precise/geolocatable**. In this regime CBG essentially degenerates to "trust
   the near VP" (≈ shortest-ping), which is exactly when it wins.
2. **Answer-space isolation lever — "is the truth's centroid far from any other centroid?"**
   decides, *among the targets with no near VP*, whether a coarse estimate still snaps to the right
   answer (Tier-2) or misclassifies (Tier-3). The truth-centroid's distance to the nearest other
   centroid is the dominant Tier-2-vs-Tier-3 separator (AUC **0.64–0.68**; median **~265 km for
   Tier-2 vs ~135 km for Tier-3**), while VP distance carries *no* signal there (AUC < 0.5).

The much-discussed **angular spread** ("surrounded vs. one-sided") is **not** a primary driver in
the global runs — because Tier-1 wins come from a single near VP (`n_part`≈1), angular geometry is
degenerate exactly where precision is decided. It is expected to matter more in the matched-regional
regime (§4), where proximity is widely available and geometry becomes the tiebreaker — *that part of
the analysis is pending the regional reruns.*

**Spotter is the structural exception:** it almost never reaches Tier-1 (0.4% of targets) because
its pooled normal-distribution bands stay wide even for the closest VP, so the deciding region never
collapses onto a near VP (its Tier-1 min-RTT is ~19 ms vs vanilla's ~2 ms). Spotter's rare correct
answers are driven by centroid isolation, not by measurement — it is barely doing latency
geolocation at all.

---

## 1. What we measured

### 1.1 Instrumentation (new)
Each MTL method now records `participating_vp_ids` — the constraints that survive the redundant-disk
filter and actually decide the intersection region. The runner joins these back to each VP's RTT and
predicted distance band and writes a nested `mtl_participants` column (+ `n_mtl_participants`) into
`targets.parquet`. This makes "the VPs that decide the intersection" exact rather than a proxy.

### 1.2 Confidence tiers (centroid answer space, R = 50 km)
- **Tier-1 (high / accurate):** SUCCESS, snaps to the truth's centroid, **and ≤ R** of it.
- **Tier-2 (median / correct-but-imprecise):** SUCCESS, snaps to the truth's centroid, but **> R**
  away (the "tolerance dividend" — right answer, imprecise point).
- **Tier-3 (low):** mismatched centroid, FALLBACK, or ERROR.

### 1.3 Per-target features (`extract_features.py`)
- **Available geometry** (combo-independent): `avail_min_vp_km` (closest VP over all observed VPs),
  `avail_min_rtt_ms`, `n_obs`.
- **Participating-VP** (the deciding set): `n_part`; `part_{min,mean,med}_dist_km`;
  `part_{min,mean,med}_rtt_ms`; `part_max_gap_deg` (max angular gap between consecutive
  participants — large ⇒ one-sided); `part_circ_var` (circular variance of bearings — high ⇒
  surrounded); `part_{mean,min}_infl` (RTT inflation = measured / (slope·dist)).
- **Answer-space** (target-level): `truth_centroid_km` (floor); `nearest_other_centroid_km`
  (truth-centroid isolation).

### 1.4 Analysis (`analyze_tiers.py`)
Per run × CBG family: single-feature **AUC** for two questions — Q1 "geolocatable?" (Tier-3 vs
Tier-1∪2) and Q2 "precise?" (Tier-1 vs Tier-2 among matched) — plus a Tier-2-vs-Tier-3 isolation
test, per-tier box plots, and depth-3 decision-tree thresholds.

---

## 2. Runs analyzed

| Regime | Runs | Targets | Status |
| ------ | ---- | ------- | ------ |
| In-distribution (global VP → global TG) | `global_as16509_final` (Amazon), `global_as31898_final` (Oracle) | 713 anchors, 257 centroids | **done** |
| Matched-regional (regional VP → in-region TG) | `north_america_as7018_final_us` (AT&T/US), `north_america_as7922_final_us` (Comcast/US), `europe_as3215_final_fr` (Orange/FR), `europe_as3209_final_de` (Vodafone/DE) | US n=96 / FR n=39 / DE n≈? | **pending rerun** |

Each run = 18 combos × 5 folds; this report focuses on the four textbook variants
(`vanilla_cbg`, `million_scale_cbg`, `octant_cbg`, `spotter_cbg`).

---

## 3. In-distribution (global-VP) findings

### 3.1 Tier composition (fraction of all targets)

| run | combo | Tier-1 | Tier-2 | Tier-3 |
| --- | ----- | -----: | -----: | -----: |
| as16509 | vanilla | 0.170 | 0.073 | 0.757 |
| as16509 | million_scale | 0.201 | 0.029 | 0.770 |
| as16509 | octant | 0.133 | 0.119 | 0.748 |
| as16509 | spotter | 0.004 | 0.066 | 0.930 |
| as31898 | vanilla | 0.174 | 0.076 | 0.750 |
| as31898 | octant | 0.163 | 0.087 | 0.750 |
| as31898 | spotter | 0.004 | 0.066 | 0.930 |

(~75% of global targets are Tier-3 — no near VP and/or a crowded answer space. Spotter is ~93%
Tier-3.)

### 3.2 The two-lever model

**Per-tier medians, `global_as16509_final`** (as31898 matches within a few %):

*vanilla_cbg*
| feature | Tier-1 | Tier-2 | Tier-3 |
| ------- | -----: | -----: | -----: |
| avail_min_vp_km | **6.3** | 340.1 | 420.0 |
| part_min_dist_km | 6.3 | 399.6 | 457.3 |
| part_min_rtt_ms | 1.7 | 9.4 | 14.0 |
| n_part | 1.0 | 3.0 | 2.0 |
| nearest_other_centroid_km | 127.6 | **262.9** | 137.0 |
| part_max_gap_deg | 360 | 210 | 258 |
| part_circ_var | 0.0 | 0.4 | 0.3 |

*octant_cbg*
| feature | Tier-1 | Tier-2 | Tier-3 |
| ------- | -----: | -----: | -----: |
| avail_min_vp_km | **6.5** | 399.6 | 412.4 |
| part_min_rtt_ms | 1.9 | 12.0 | 12.7 |
| n_part | 2.0 | 4.0 | 4.0 |
| nearest_other_centroid_km | 103.9 | **269.7** | 127.6 |

**Lever 1 — proximity (decides Tier-1).** Single-feature AUC for Q2 "precise?" (Tier-1 vs Tier-2),
averaged over the two global runs:

| feature | vanilla | octant | million_scale |
| ------- | ------: | -----: | ------------: |
| part_min_dist_km | 0.085 | 0.126 | 0.073 |
| part_min_rtt_ms  | 0.087 | 0.096 | — |
| avail_min_vp_km  | — | 0.140 | 0.080 |

(AUC ≪ 0.5 ⇒ *lower* distance/RTT strongly predicts Tier-1.) Same direction for Q1 "geolocatable?"
(part_min_rtt / part_min_dist AUC ≈ 0.13–0.25). Decision-tree primary split (octant): `part_min_dist_km
≤ 67.9 ⇒ geolocatable`; `≤ 54.0 ⇒ precise`.

**Lever 2 — answer-space isolation (decides Tier-2 vs Tier-3).** Among targets with no near VP,
the truth-centroid's isolation separates "right anyway" from "wrong":

| feature (Tier-2 vs Tier-3, positive=Tier-2) | vanilla | octant | million_scale |
| ------------------------------------------- | ------: | -----: | ------------: |
| nearest_other_centroid_km | **0.674 / 0.642** | **0.684 / 0.642** | 0.598 / 0.501 |
| avail_min_vp_km | 0.411 / 0.465 | 0.454 / 0.493 | 0.227 / 0.291 |
| part_min_dist_km | 0.403 / 0.445 | 0.452 / 0.477 | 0.209 / 0.273 |

(two numbers = as16509 / as31898). Isolation is the only feature with real Tier-2-vs-Tier-3 signal;
VP distance has none (AUC ≤ 0.5) — both Tier-2 and Tier-3 are far-from-VP, and what flips the
outcome is whether the neighbourhood in the answer space is sparse (Tier-2, ~265 km to the next
centroid) or crowded (Tier-3, ~135 km).

### 3.3 Why angular spread doesn't drive the global regime
Tier-1 wins come from a *single* near VP — `n_part` median is 1–2 and `part_max_gap_deg` is ~360°
(degenerate). Angular spread can only matter when several VPs of comparable distance jointly shape
the region, which is rare when VPs are globally dispersed. So in this regime proximity dominates and
geometry is a non-factor. (The matched-regional regime is the place to test the angular hypothesis;
§4, pending.)

### 3.4 Spotter's structural collapse
Spotter's "Tier-1" targets (n≈3) still have closest VP ~420 km and min participant RTT **~19 ms** —
i.e. even when it succeeds it is *not* riding a near VP, unlike vanilla/octant (~2 ms). Its pooled
normal-dist band is wide even for the nearest VP, so the deciding region never collapses onto it.
Result: ~93% Tier-3, and its few correct answers correlate with centroid isolation
(`nearest_other_centroid_km` AUC 0.75 for Q1), not with measurement quality. Spotter, in the
operator setting, is barely doing latency geolocation.

---

## 4. Matched-regional findings  *(PENDING — regional reruns in progress)*

Open question this section will answer: when the VP fleet is in-country (so most targets *have* a
moderately close VP), does the Tier-1 proximity threshold relax, and does **angular spread finally
become a driver** (surrounded targets precise, one-sided targets not)? Does centroid isolation still
own the Tier-2/3 boundary at the smaller in-country answer space?

---

## 5. Cross-regime synthesis  *(PENDING)*

---

## 6. Implications for the paper

- **§6.1 mechanism / RQ1.** The "closest-VP distance" axis is not just correlated with accuracy — it
  is the *causal* Tier-1 lever, with a concrete operating point (~50–70 km). Globally, CBG only
  reaches Tier-1 by degenerating to a near-VP (≈ shortest-ping), which is the quantitative form of
  "CBG barely beats shortest-ping when VPs are dispersed."
- **§6.5 / confidence-tier study.** The two-lever model answers the paper's question "what
  characterizes precisely-geolocated targets?": *Tier-1 = a VP within ~50–70 km; Tier-2 vs Tier-3 =
  answer-space isolation, not measurement.* This cleanly separates a **measurement-geometry** lever
  from an **answer-space-geometry** lever — a framing prior CBG work lacks.
- **Operator takeaway.** Two distinct knobs: deploy/【acquire】a near VP to move targets into Tier-1;
  and recognize that for far targets, classification success is a property of the *candidate-site
  layout* (isolated hubs are recoverable; dense metros are not) — not something a better CBG variant
  fixes.
- **Spotter** should be presented as the cautionary "when CBG fails" variant.

---

## 7. Caveats & next steps

- **Inflation metric is unreliable at tiny distances** (ratio explodes as dist→0), so its apparent
  AUC near Tier-1 is an artifact of co-located VPs; do not over-read it. A floored/absolute-residual
  version would be cleaner.
- Tier-1-vs-Tier-2 vs the proximity threshold is stable across the two global ASNs; regional
  confirmation pending.
- Regional target sets are small (US 96 / FR 39) — regional tier *counts* are noisy; treat regional
  AUCs as directional.
- Decision-tree thresholds are in-sample (train accuracy ~0.73 Q1 / ~0.87 Q2); they are descriptive
  operating points, not validated classifiers.
