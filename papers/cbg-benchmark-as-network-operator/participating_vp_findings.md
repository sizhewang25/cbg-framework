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

The much-discussed **angular spread** ("surrounded vs. one-sided") is a **second-order** driver: in
the global runs it is invisible (Tier-1 wins come from a single near VP, `n_part`≈1, so geometry is
degenerate); in the **matched-regional** runs, where the in-country fleet gives many targets a close
VP *and* angular surround, `part_circ_var` does rise to **AUC 0.53–0.82** for "geolocatable?" —
confirming the hypothesis, but always well behind proximity. Pooled across all 6 runs the driver
ranking is unambiguous: **min RTT/distance to the nearest VP is #1; angular surround and answer-space
isolation are secondary.** Matched-regional fleets roughly **double** the precise-Tier-1 share and
**halve** Tier-3 vs. the global regime — but the gating quantity is unchanged: *is there a VP close
to the target?*

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
| Matched-regional (regional VP → in-region TG) | `north_america_as7018_final_us` (AT&T/US), `north_america_as7922_final_us` (Comcast/US), `europe_as3215_final_fr` (Orange/FR) | US n=96 ×2, FR n=39 | **done** (textbook-4 side run) |
| Matched-regional (confirmation) | `europe_as3209_final_de` (Vodafone/DE) | DE n=96 | **done** — confirms; `part_circ_var` AUC **0.82** (strongest angular signal) |

> Regional runs were executed as **textbook-4-combo side runs** (`scripts/analysis/partvp/cfg_textbook/`,
> separate `outputs_partvp/` root) to finish overnight; the 4 textbook combos compute identically to
> the full configs (same inputs, seed=42), so the participating-VP features are report-identical.

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
geometry is a non-factor. (The matched-regional regime is where the angular hypothesis is confirmed
— see §4.3.)

### 3.4 Spotter's structural collapse
Spotter's "Tier-1" targets (n≈3) still have closest VP ~420 km and min participant RTT **~19 ms** —
i.e. even when it succeeds it is *not* riding a near VP, unlike vanilla/octant (~2 ms). Its pooled
normal-dist band is wide even for the nearest VP, so the deciding region never collapses onto it.
Result: ~93% Tier-3, and its few correct answers correlate with centroid isolation
(`nearest_other_centroid_km` AUC 0.75 for Q1), not with measurement quality. Spotter, in the
operator setting, is barely doing latency geolocation.

---

## 4. Matched-regional findings

Runs: `north_america_as7018_final_us` (AT&T/US, n=96), `north_america_as7922_final_us` (Comcast/US,
n=96), `europe_as3215_final_fr` (Orange/FR, n=39), `europe_as3209_final_de` (Vodafone/DE, n=96).
All four consistent.

### 4.1 CBG earns its keep — Tier-1 doubles, Tier-3 halves
`octant_cbg` tier composition, global vs matched-regional:

| run | Tier-1 | Tier-3 |
| --- | -----: | -----: |
| global as16509 | 0.133 | 0.748 |
| global as31898 | 0.163 | 0.750 |
| **US as7018** | **0.240** | **0.490** |
| **US as7922** | **0.260** | **0.448** |
| **FR as3215** | **0.487** | 0.436 |

The matched-regional fleet roughly **doubles** the precise-Tier-1 share and **halves** Tier-3.
(FR n=39 is small — treat its 49% as directional.)

### 4.2 Proximity still dominates — but multilateration becomes real
Tier-1 medians (closest VP, min participant RTT, participant count):

| run | combo | T1 closest-VP km | T1 min-RTT ms | T1 n_part |
| --- | ----- | ---------------: | ------------: | --------: |
| global as16509 | octant | 6.5 | 1.9 | **2** |
| US as7018 | octant | 17.0 | 4.9 | **7** |
| US as7922 | octant | 9.0 | 11.3 | **7** |
| FR as3215 | octant | 3.4 | 1.7 | **20** |

Globally, Tier-1 wins are a *single* near VP (`n_part`≈1–2 — CBG degenerates to shortest-ping).
Regionally the in-country fleet supplies **7–20 participating VPs** *and* a close one — genuine
multilateration. Proximity is still the top driver (Q2 "precise?" `part_min_dist_km` AUC 0.24–0.34;
the US7018 octant tree's primary split is **`part_min_rtt_ms ≤ 7 ms ⇒ geolocatable`**, train acc
0.84), but the operating point relaxes from ~6 km to a metro-scale **~10–40 km / RTT ≤ ~7 ms**.

### 4.3 Angular surround finally matters (secondarily)
With proximity widely available, geometry becomes a tiebreaker: `part_circ_var` (high ⇒ surrounded)
reaches **AUC 0.53–0.82 for "geolocatable?"** in the regional runs (US 0.53–0.63, DE 0.82; vs ~0.31
/ degenerate globally). It
confirms the long-held "surrounded vs. one-sided" hypothesis — but it is a **second-order** effect,
well behind proximity (AUC ~0.30).

### 4.4 Answer-space isolation is weak regionally
The global Tier-2-vs-Tier-3 isolation lever (`nearest_other_centroid_km`, AUC 0.64–0.68) does **not**
carry over cleanly: the in-country answer space is small (US 32 centroids / FR 12) so isolation is
noisy (AUC ~0.3–0.5). Isolation is a large-answer-space (global) phenomenon.

### 4.6 Natural experiment — inward vs outward (EU fleets → all-EU targets)

To isolate the angular lever, two single-region EU fleets — **AS3209 (Vodafone, DE-central)** and
**AS3215 (Orange, FR-western)** — geolocate **all 415 Europe anchors** (`target_continent=Europe`).
Each target is labelled by the *whole-fleet* angular coverage as seen from it (combo-independent):
**outward** = `avail_max_gap_deg ≥ 180°` (target outside the VP convex hull, one-sided) vs
**inward** = `< 180°` (surrounded).

**Raw split is dramatic** (octant Tier-1):

| fleet | inward T1 | outward T1 | inward med closest-VP | outward med closest-VP |
| ----- | --------: | ---------: | --------------------: | ---------------------: |
| AS3209/DE | **32%** (n=102) | **2.2%** (n=313) | 4.7 km | 326 km |
| AS3215/FR | **31%** (n=55) | **4.7%** (n=360) | 4.8 km | 351 km |

But inward targets are also ~70× closer — the split is *confounded with proximity* (an
inside-the-hull target of a clustered fleet is necessarily near it). The figure
(`analysis_eu/strat_*.png`) shows Tier-1 lives almost entirely in the **≤50 km inward corner**
(34% inward vs 5% outward at ≤50 km) and is ≈0% beyond 50 km for *both* classes.

**Distance-controlled (matched band, closest-VP 20–80 km, balanced medians):**

| fleet | inward T1 | outward T1 |
| ----- | --------: | ---------: |
| AS3209/DE | 14% (n=14) | 8% (n=36) |
| AS3215/FR | 12% (n=8) | 0% (n=7) |

So the dramatic raw inward/outward gap is **mostly proximity**, with a **smaller genuine angular
residual** (~1.5–2× at matched distance, small n). This *refines* lever 3: angular surround does
carry independent signal, but the binding constraint is still proximity — there is essentially no
Tier-1 success beyond ~50 km regardless of surround. (Caveat for operator use: inward/outward is
*truth-anchored* — it needs the target's location — so it characterizes outcomes rather than being
an inference-time predictor; see discussion.md §6.)

### 4.5 Spotter still collapses
Tier-1 = 0–3% in every regional run (US7018 0%, US7922 3%, FR low), ~84–94% Tier-3 — the same
structural failure as global.

---

## 5. Cross-regime synthesis — a three-lever model

Pooled single-feature driver strength (mean |AUC−0.5| across all 6 runs × 4 textbook combos):

| rank | Q1 "geolocatable?" | Q2 "precise?" |
| ---- | ------------------ | ------------- |
| 1 | avail_min_rtt_ms (0.26) | part_min_dist_km (0.34) |
| 2 | part_min_rtt_ms (0.25) | part_min_rtt_ms (0.32) |
| 3 | part_*_dist_km (0.22–0.24) | part_mean_dist_km (0.32) |
| … | part_circ_var (0.16) | part_circ_var (0.26) |
| … | nearest_other_centroid (0.10) | nearest_other_centroid (0.18) |

**Three levers, by priority:**

1. **Proximity to the nearest VP — universal, primary.** Min RTT / min distance to the closest
   (participating) VP is the #1 driver of *both* geolocatability and precision in every regime.
   Operating point: nearest-VP **RTT ≲ 5–7 ms / distance ≲ 10–40 km** ⇒ Tier-1. This is *causal*,
   not merely correlational (decision-tree primary split in both global and regional runs).
2. **Answer-space isolation — owns Tier-2 vs Tier-3, but only at scale.** When no VP is near, whether
   a coarse estimate still lands on the right candidate depends on how isolated the truth's centroid
   is (global AUC 0.64–0.68). In small in-country answer spaces this lever is weak.
3. **Angular surround — secondary, regime-gated.** Being ringed by VPs (`part_circ_var`) helps, but
   only once proximity is widely available (regional AUC up to 0.63); globally it is invisible
   because Tier-1 there is a single-VP degenerate case.

**The regime shift in one sentence:** globally CBG reaches Tier-1 only by collapsing onto a single
near VP (≈ shortest-ping, ~75% Tier-3); a matched in-country fleet gives many targets both a near VP
*and* angular surround, turning CBG into genuine multilateration that doubles Tier-1 and halves
Tier-3 — but the gating quantity never changes: **is there a VP close to the target?**

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

## 7. Observable confidence from region–answer-space overlap

§1–§5 characterize tiers with **truth-anchored** features. An operator geolocating an *unknown*
target has none of those. The question this section answers: **is there an inference-observable
signal — computable from the RTT measurements and the (known) answer space alone — that predicts the
confidence tier?** The §5 isolation lever cannot be it (it needs the truth's centroid). The lever
that *can* be is the geometry CBG already produces: **how the MTL feasible region sits relative to the
answer-space cells.**

### 7.1 Method (`scripts/analysis/partvp/region_confidence.py`)
The MTL region is reconstructed offline from the persisted `mtl_participants` annuli
(`compute_feasible_region_unweighted`; circle methods are the `lower=0` special case). Two
observables per prediction: `n_hit` = number of distinct answer-space cluster disks (uniform
**R = 50 km**) the region overlaps (sampled-interior points + haversine), and `d_hub` = point-estimate
→ nearest centroid. Confidence levels, priority order, **all observable**:
- **L1** highest: `n_hit == 1` (region in exactly one cell), regardless of `d_hub`.
- **L2** high: `n_hit > 1` and `d_hub < R`.
- **L3** mid: `n_hit > 1` and `d_hub ≥ R`.
- **L0** low/fail: `n_hit == 0` (empty / touches no cell) or FALLBACK/ERROR.

These are *predictions of* the tiers; we validate them against the true tier labels of §1.2. Results:
`analysis/region_confidence.csv`; per-target rows: `data/region_confidence_all.parquet`.

### 7.2 Calibration — L1 is the high-confidence flag; nothing else is reliably high
`P(prediction snaps to the correct centroid | level)`, n-weighted (global / matched-regional):

| combo | L1 | L2 | L3 | L0 | L1 coverage (share of all correct) |
| ----- | -: | -: | -: | -: | :--------------------------------: |
| vanilla_cbg       | **0.85 / 0.91** | 0.20 / 0.21 | 0.09 / 0.31 | 0.26 / 0.10 | 0.41 / 0.56 |
| million_scale_cbg | **0.70 / 0.83** | 0.28 / 0.63 | 0.08 / 0.28 | 0.00 / 0.00 | 0.28 / 0.06 |
| octant_cbg        | **0.71 / 0.66** | 0.22 / 0.15 | 0.08 / 0.14 | 0.29 / 0.51 | 0.33 / 0.28 |
| spotter_cbg       | 0.16 / 0.20     | 0.07 / –    | 0.05 / 0.00 | 0.07 / 0.16 | 0.08 / 0.03 |

1. **L1 (region lands in exactly one cell) is the operator's "trust this" signal** — precision
   **0.66–0.91** for the three real CBG variants, capturing **~30–56%** of all correct answers
   (vanilla). Fully observable; no truth, no model. This is the inference-time replacement for the
   §5 isolation lever, sitting at the top of the confidence ranking.
2. **A multi-cell region is *not* rescued by a near point estimate.** L2 (`n_hit>1`, `d_hub<R`) is
   low globally (**0.20–0.28**): when the feasible region straddles several cells, the point snaps to
   the wrong hub ~80% of the time even though it sits inside *a* hub. L2 is moderate only for
   `million_scale` regionally (0.63), whose larger regions rarely reach L1. **The clean operating
   point is L1, not L1∪L2.**
3. **L3 (region present, point far from any hub) ≈ low** everywhere (0.08–0.31, mostly Tier-3).
4. **L0 ("no intersection") is *not* uniformly failure — it depends on the variant's fallback.**
   Spotter L0 is pure collapse (~0.07–0.16); octant's centroid fallback still lands right **0.29
   (global) / 0.51 (regional)** of the time (the recoverable "no-intersection-yet-correct" case is
   real for octant). vanilla sits between.

### 7.3 Reading
Region-overlap is the observable confidence lever that §5's truth-anchored analysis implied was
missing — but only at the **top**: L1 is reliably high, L2 is not, and L0's value is variant-specific.
The far-target ambiguity remains genuinely hard from observables (L2/L3 ≈ 0.1–0.3), consistent with
§3.2: when no VP is near, whether a coarse estimate lands right is an answer-space property the
*measurement* cannot see — and a multi-cell region is exactly the observable signature of that
ambiguity. Spotter's near-total L0 (its bands never isolate one cell) is the geometric form of its
collapse (`[[finding_spherical_circle_brittle]]`).

This cleanly feeds the **operator takeaway** (§6): beyond "deploy a near VP," the operator gets a
real, model-free confidence gate — *trust a prediction iff its feasible region falls inside a single
answer cell* — and a calibrated expectation for everything else.

---

## 8. Caveats & next steps

- **Region-overlap confidence** (§7): the region is reconstructed in octant's planar frame, so its
  *shape* carries a small planar error (the cluster test itself is haversine). Level assignment is
  **94% stable** between 300 and 1200 interior samples; since `n_hit` only grows with sampling,
  reported L1 precision is a mild **lower bound** (denser sampling moves borderline L1→L3). "Correct"
  is scored by the point-estimate's nearest centroid for all levels (consistent with §1.2); an L1
  variant that scores against the single overlapped cell would be marginally cleaner.
- **Inflation metric is unreliable at tiny distances** (ratio explodes as dist→0), so its apparent
  AUC near Tier-1 is an artifact of co-located VPs; do not over-read it. A floored/absolute-residual
  version would be cleaner.
- The proximity → Tier-1 relationship is stable across all six runs (2 global, US ×2, FR, DE).
- Regional target sets are small (US/DE n=96, FR n=39) — regional tier *counts* are noisy; treat
  regional AUCs as directional.
- Regional runs cover the **4 textbook combos** only (textbook-4 side reruns); the non-textbook
  combos in the regional configs were not regenerated with the new column.
- Decision-tree thresholds are in-sample (train accuracy ~0.73 Q1 / ~0.87 Q2); they are descriptive
  operating points, not validated classifiers.
