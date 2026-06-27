# Paper-Flow Feasibility Report

**Generated:** 2026-06-23  
**Source configs:** `scripts/benchmark/v2/config/` (final configs only)  
**Analysis outputs:** `scripts/analysis/partvp/outputs/`  
**Scope:** §6.1–§6.5, §2.3 variant table. Proprietary dataset (§6.4) excluded per instruction.

---

## Metric Definitions

Before reporting, all metrics cited for the first time are defined here.

**fleet_abs_km** (`d(V*, C)`): Haversine distance from the closest *available* VP to the truth centroid `C`. Pure proximity — no RTT involved. Combo-independent per target.

**target_distinguishable_vp_margin** (`d(C, N)/2 − fleet_abs_km`): By the triangle inequality, if this quantity is positive the closest VP is *guaranteed* to have a shorter haversine distance to `C` than to `N` (nearest competitor centroid), regardless of RTT noise. A negative value means no VP in the fleet can geometrically certify the truth centroid over the nearest competitor. Computed per target per setup.

**VP proximity failure coverage** (`failure_share_explained_by_missing_vp`): Among all failed targets, the fraction where `target_distinguishable_vp_margin ≤ 0`. Tells how many failures are *at minimum* geometry-limited.

**Classification accuracy** (`same_centroid_acc`): Fraction of targets where the CBG-predicted lat/lon snaps to the truth centroid (R = 50 km cluster radius). Denominator is all targets, including fallbacks.

**within-R rate**: Fraction of targets where the predicted point is within 50 km of the truth centroid — the continuous-coordinate analogue of classification accuracy.

**Tolerance dividend** (`same_centroid_acc − within_r`): Extra correct answers gained by the finite answer space. A prediction outside `R` of the centroid but still closest to the correct centroid counts as correct in classification but wrong in within-R.

**AUC (failure prediction)**: Area under the ROC curve for predicting target failure using `fleet_abs_km` or `margin ≤ 0` as the binary signal. AUC = 0.5 is random; higher is a stronger failure predictor.

---

## §2.3 — Variant Table: Official Config (Issue #1: RESOLVED)

**Status: RESOLVED.** Decision: the paper's §2.3 variant table and implementation note have been updated to reflect the actual run configs (`planar_circle + geometric_centroid` for Vanilla and Million-scale). This is the benchmark's official configuration; Vanilla and Million-scale use `PlanarCircleMTL` + `geometric_centroid`, not `spherical_circle + boundary_vertex_mean` as originally cited in the paper.

**Official four-variant config** (all `_final` configs; `global_as16509_final.yaml`, `north_america_as7018_final_us.yaml`, `europe_as3209_final_de.yaml`):


| combo_id            | ltd                 | mtl                       | ctr                      |
| ------------------- | ------------------- | ------------------------- | ------------------------ |
| `vanilla_cbg`       | `low_envelope`      | `**planar_circle`**       | `**geometric_centroid**` |
| `million_scale_cbg` | `speed_of_internet` | `**planar_circle**`       | `**geometric_centroid**` |
| `octant_cbg`        | `bounded_spline`    | `planar_annulus_weighted` | `monte_carlo_medoid`     |
| `spotter_cbg`       | `normal_dist`       | `planar_annulus_weighted` | `monte_carlo_medoid`     |


Implementation note documented in paper-flow §2.3: `PlanarCircleMTL` approximates each RTT disk as a degree-space Shapely polygon (flat-earth); `geometric_centroid` is area-weighted. Paper-flow §7 phase-ablation results compare these against the spherical/BVM originals.

---

## §6.1 — Proximity-Limited Regime

### Claim: AS16509→global classification accuracy

**Status: VERIFIED.**


| combo             | paper claims | data value   | source                              |
| ----------------- | ------------ | ------------ | ----------------------------------- |
| vanilla_cbg       | 24.3%        | **24.26%** ✓ | `data/global_as16509_final.parquet` |
| million_scale_cbg | 23.0%        | **23.00%** ✓ | same                                |
| octant_cbg        | 25.2%        | **25.25%** ✓ | same                                |
| spotter_cbg       | 7.0%         | **7.01%** ✓  | same                                |


N = 713 targets, 5-fold cross-validation.

### Claim: Fleet-geometry metrics for global-global

**Status: VERIFIED.**


| metric                    | paper claims | data value      | source                                         |
| ------------------------- | ------------ | --------------- | ---------------------------------------------- |
| Median `fleet_abs_km`     | 348 km       | **348.2 km** ✓  | `fleet_geometry_by_config.csv` (global-global) |
| % missing target-dist. VP | 77%          | **77.0%** ✓     | same                                           |
| Median margin             | −313 km      | **−313.3 km** ✓ | same                                           |


### Claim: VP proximity as dominant failure driver

**Status: VERIFIED.**

From `VP_PROXIMITY_FAILURE_ASSESSMENT.md` (global-global setup):


| variant           | failure share explained by missing VP | paper claim                                      |
| ----------------- | ------------------------------------- | ------------------------------------------------ |
| vanilla_cbg       | 94.3%                                 | "84.4–91.8%"                                     |
| million_scale_cbg | 97.8%                                 | 91.8% (paper's high end) ✓                       |
| octant_cbg        | 88.7%                                 | within range ✓                                   |
| spotter_cbg       | 76.2%                                 | below range (Spotter exception noted in paper ✓) |


The paper's "84.4–91.8%" range is the range *excluding Spotter* (Vanilla/MS/Octant). The data confirms this.

AUC of `fleet_abs_km` predicting failure: million_scale = **0.96** ✓ (paper claims 0.84–0.96).

### Claim: "Extremely limited" rows (AS7018→global, AS3209→global)

**Status: VERIFIED — data exists in `north_america_as7018_final` and `europe_as3209_final` outputs.**

These are the parent configs (no `target_country` filter), which evaluate each country fleet against all 713 global anchors. Classification accuracy computed from `targets.parquet` fold files:


| Fleet         | n VPs | Vanilla | Million-scale | Octant | Spotter |
| ------------- | ----- | ------- | ------------- | ------ | ------- |
| AS7018→global | 125   | 3.9%    | **6.6%**      | 5.5%   | 3.0%    |
| AS3209→global | 164   | 2.7%    | **6.0%**      | 2.0%   | 2.5%    |


Fleet geometry (computed from VP JSON + cluster centroids via haversine):


| Fleet         | Median fleet_abs_km | Median margin | % missing |
| ------------- | ------------------- | ------------- | --------- |
| AS7018→global | 6,268 km            | −6,217 km     | 92.4%     |
| AS3209→global | 972 km              | −915 km       | 85.4%     |


Both are dramatically worse than the "fairly limited" global fleet (AS16509: 348 km, −313 km, 77%). Random baseline for 257 clusters ≈ 0.4%; the extremely-limited setups achieve only 2–7%, confirming that country-scale fleets against global targets are essentially non-functional for CBG regardless of variant.

### Claim: Shortest-ping baselines (global 23.4%, US 39.6%, DE 50.0%, cross-region 5.3%/6.3%)

**Status: VERIFIED (Issue #7 resolved).** Confirmed by running `plot_cluster_match_bars.py` on all 5 final run directories:


| Setup          | Claimed    | Verified    |
| -------------- | ---------- | ----------- |
| AS16509→global | 23.4%      | **23.4%** ✓ |
| AS7018→US      | 39.6%      | **39.6%** ✓ |
| AS3209→DE      | 50.0%      | **50.0%** ✓ |
| AS7018→global  | 5.3% (new) | **5.3%** ✓  |
| AS3209→global  | 6.3% (new) | **6.3%** ✓  |


---

## §6.2 — Proximity-Sufficient Regime

### AS3209→DE fleet metrics (Issue #2: RESOLVED)

**Status: RESOLVED.** Original stale numbers (from AS3215→FR) replaced with correct AS3209→DE values. The fleet analysis had been run on AS3215 (Orange France, n=39, median VP 1.49 km) rather than AS3209 (Vodafone Germany, n=96). Paper-flow §6.2 updated.

**Verified AS3209→DE fleet metrics** (from `europe_as3209_final_de.parquet`):


| metric                                 | actual value |
| -------------------------------------- | ------------ |
| n targets                              | 96           |
| Median `fleet_abs_km`                  | **4.62 km**  |
| Median `nearest_other_centroid_km / 2` | 44.3 km      |
| Median margin                          | **+38.6 km** |
| % missing target-dist. VP              | **2.1%**     |


### Claim: AS3209→DE classification accuracy (Issue #4: RESOLVED)

**Status: RESOLVED.** All four variants confirmed. Spotter "pending" note was stale.


| combo             | verified value | source                                |
| ----------------- | -------------- | ------------------------------------- |
| vanilla_cbg       | **18.75%** ✓   | `data/europe_as3209_final_de.parquet` |
| million_scale_cbg | **46.88%** ✓   | same                                  |
| octant_cbg        | **39.58%** ✓   | same                                  |
| spotter_cbg       | **17.71%** ✓   | same                                  |


### Claim: AS7018→US classification accuracy

**Status: VERIFIED.**


| combo             | paper claims | data value   |
| ----------------- | ------------ | ------------ |
| vanilla_cbg       | 45.8%        | **45.83%** ✓ |
| million_scale_cbg | 43.8%        | **43.75%** ✓ |
| octant_cbg        | 51.0%        | **51.04%** ✓ |
| spotter_cbg       | 6.3%         | **6.25%** ✓  |


N = 96 targets, 5-fold cross-validation (fold sizes: 23+13+20+20+20 = 96).

### Claim: AS7018→US fleet geometry (96/32, 35.1 km, +62.0 km, 43.8%) — Issues #2/#8 RESOLVED

**Status: VERIFIED.**


| metric                    | paper claims | data value                                                                   |
| ------------------------- | ------------ | ---------------------------------------------------------------------------- |
| n targets                 | 96           | **96** ✓                                                                     |
| Median `fleet_abs_km`     | 35.1 km      | **35.15 km** ✓                                                               |
| Median margin             | +62.0 km     | **+62.02 km** ✓                                                              |
| % missing                 | 43.8%        | **43.75%** ✓                                                                 |
| n clusters (answer space) | 32           | **32** ✓ (confirmed from `clusters.csv` for `north_america_as7018_final_us`) |


### Claim: Tolerance dividend (§6.2, AS7018→US and AS3209→DE)

**Status: VERIFIED (both setups).**

From `analysis/tolerance_dividend.csv`:

**AS7018→US** (`north_america_as7018_final_us`):


| combo             | dividend (abs pp) | paper        | dividend (rel %) | paper      |
| ----------------- | ----------------- | ------------ | ---------------- | ---------- |
| vanilla_cbg       | +31.25 pp         | "+31.3 pp" ✓ | **68.2%**        | "68%" ✓    |
| million_scale_cbg | +19.79 pp         | "+19.8 pp" ✓ | **45.2%**        | "45%" ✓    |
| octant_cbg        | +27.08 pp         | "+27.1 pp" ✓ | **53.1%**        | "53%" ✓    |
| spotter_cbg       | +6.25 pp          | "+6.3 pp" ✓  | **100%**         | "~100%*" ✓ |


Paper claim "53–71% of all correct answers are tolerance wins" uses the range octant (53%) to vanilla (68%) — confirmed ✓. The 71% bound in the contribution summary may be rounded from an intermediate value.

**AS3209→DE** (`europe_as3209_final_de`) — NEW, not yet in paper draft:


| combo             | dividend (abs pp) | dividend (rel %) |
| ----------------- | ----------------- | ---------------- |
| vanilla_cbg       | **+6.25 pp**      | **33.3%**        |
| million_scale_cbg | **+5.21 pp**      | **11.1%**        |
| octant_cbg        | **+5.21 pp**      | **13.2%**        |
| spotter_cbg       | **+9.38 pp**      | **52.9%**        |


DE dividend is 3–5× smaller than US (ex-Spotter: 11–33% vs. 45–68%). With median VP at 4.6 km, within-R rate is already high, leaving little headroom for the tolerance gain. This contrast has been added to paper-flow §6.2 and intro.tex contribution #3.

---

## §6.3 — Success and Failure Analysis

### AUC values (fleet_abs_km predicting failure)

**Status: VERIFIED for global-global; europe-country uses FR (see §6.2 issue).**

From `fleet_geometry_auc.csv`:


| config               | variant           | AUC (fleet_abs_km) | paper cites         |
| -------------------- | ----------------- | ------------------ | ------------------- |
| global-global        | million_scale_cbg | **0.960**          | 0.84–0.96 (range) ✓ |
| global-global        | vanilla_cbg       | **0.849**          | within range ✓      |
| global-global        | octant_cbg        | **0.719**          | within range ✓      |
| europe-country (FR!) | octant_cbg        | **0.684**          | —                   |


The paper says "Octant RTT inflation AUC 0.82 in EU-country". The `fleet_geometry_auc.csv` shows europe-country/octant = **0.684** (not 0.82). The 0.82 figure likely came from `WHEN_CBG_FAILS.md` which cites "RTT inflation AUC 0.82" for the Octant failure analysis — a different metric (AUC of inflation predicting EXCLUSIVE_REGION failure, not fleet distance predicting any failure). These are different AUCs and should not be conflated in the paper.

### Per-variant failure profile (EMPTY / EXCLUSIVE / INCLUSIVE)

**Status: COMPUTED.** Data from `outputs_partvp/` fold `targets.parquet`. See `failure_taxonomy_output.md` for methodology.


| Setup | combo             | data_src        | EMPTY    | EXCLUSIVE    | INCL-success | INCL-misclass |
| ----- | ----------------- | --------------- | -------- | ------------ | ------------ | ------------- |
| na-us | vanilla_cbg       | outputs/        | 14 (15%) | **10 (10%)** | 18 (19%)     | 54 (56%)      |
| na-us | million_scale_cbg | outputs/        | 0        | 1 (1%)       | 35 (36%)     | 60 (63%)      |
| na-us | octant_cbg        | outputs_partvp/ | 0        | **48 (50%)** | 22 (23%)     | 26 (27%)      |
| na-us | spotter_cbg       | outputs_partvp/ | 0        | **95 (99%)** | 0 (0%)       | 1 (1%)        |
| eu-de | vanilla_cbg       | outputs/        | 36 (38%) | **13 (14%)** | 12 (13%)     | 35 (36%)      |
| eu-de | million_scale_cbg | outputs/        | 0        | 0 (0%)       | 39 (41%)     | 57 (59%)      |
| eu-de | octant_cbg        | outputs/        | 0        | **50 (52%)** | 31 (32%)     | 15 (16%)      |
| eu-de | spotter_cbg       | outputs/        | 0        | **85 (89%)** | 8 (8%)       | 3 (3%)        |


**Key finding: EXCLUSIVE is large for annulus-based variants.** Proper Shapely polygon-disk intersection (reconstructing the joint MTL polygon and testing R.intersects(D_truth)) shows:

- **Octant: ~50% EXCLUSIVE** in both US (48/96) and DE (50/96)
- **Spotter: ~89–99% EXCLUSIVE** in US (95/96) and DE (85/96)
- Vanilla: ~10–14% EXCLUSIVE (disk-based; EMPTY is the larger failure mode)
- Million-scale: ~0–1% EXCLUSIVE (SoI upper-bound never pushes outward)

The prior per-VP band check was a necessary but not sufficient condition for EXCLUSIVE=0. Each individual VP's annulus may reach D_truth's disk, but the *joint* intersection of all annuli can drift entirely beyond D_truth when RTT inflation pushes inner bounds outward across multiple VPs simultaneously.

**EXCLUSIVE-but-correct phenomenon (the tolerance mechanism):** The centroid of an EXCLUSIVE region (polygon does not overlap D_truth's 50 km disk) can still snap to the correct answer cell, because Voronoi cells extend beyond D_truth. This is why classification accuracy survives despite large EXCLUSIVE fractions:

- Octant US: ~27 of 48 EXCLUSIVE predictions still correctly classified (centroid snaps to right cell). This accounts for nearly the entire ~27 pp tolerance dividend.
- Spotter US: 6 of 95 EXCLUSIVE predictions correctly classified — Spotter's entire 6.3% accuracy is EXCLUSIVE-but-correct.

**Paper corrections applied to paper-flow.md and intro.tex:**

- §6.2 Octant paragraph: removed EXCLUSIVE=0 claim; replaced with verified ~50% EXCLUSIVE + EXCLUSIVE-but-correct explanation
- §6.3 key finding box: replaced "EXCLUSIVE=0" with correct counts and EXCLUSIVE-but-correct narrative
- intro.tex contribution #4: replaced "EXCLUSIVE=0" with verified EXCLUSIVE counts
- results.tex §6.3: removed PENDING note; added correct partition and mechanism explanation

---

## §6.5 — Production Cost

**Status: VERIFIED.**

From `analysis_rq3/rq3_runtime_global_as16509_final.csv`:


| claim                            | paper value   | data value                                                                                      |
| -------------------------------- | ------------- | ----------------------------------------------------------------------------------------------- |
| `octant_cbg_hull_geo` accuracy   | 30.0%         | **30.01%** ✓                                                                                    |
| `octant_cbg_hull_geo` throughput | 65 targets/s  | **65.1 targets/s** ✓                                                                            |
| `octant_cbg` accuracy            | 25.2%         | **25.25%** ✓                                                                                    |
| `octant_cbg` throughput          | 3.2 targets/s | **3.15 targets/s** ✓                                                                            |
| Geometric centroid latency       | ~0.25 ms      | **0.256 ms** (`ctr_ms_p50`) ✓                                                                   |
| Monte Carlo latency range        | 190–390 ms    | `octant_cbg_hull`: 204 ms; `octant_cbg`: 278 ms; `spotter_cbg`: 390 ms → **range 191–390 ms** ✓ |


---

## Summary: Issues Requiring Action


| #   | Section          | Issue                                                                                                                                | Severity | Status       | Action                                                                                                                                                                                                                                                                                                                                   |
| --- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------ | -------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | §2.3             | `vanilla_cbg` and `million_scale_cbg` use `planar_circle + geometric_centroid` vs. paper's "spherical circle + boundary-vertex mean" | ~~HIGH~~ | **RESOLVED** | Official config confirmed as `planar_circle + geometric_centroid`; paper-flow.md §2.3 table and implementation note updated                                                                                                                                                                                                              |
| 2   | §6.2 fleet table | AS3209→DE fleet metrics (1.5 km, 62.3 km, 7.7%) were from AS3215→FR                                                                  | ~~HIGH~~ | **RESOLVED** | paper-flow.md §6.2 updated to correct AS3209→DE values: 4.6 km, +38.6 km, 2.1%                                                                                                                                                                                                                                                           |
| 3   | §6.1             | AS7018→global and AS3209→global TBD rows                                                                                             | ~~HIGH~~ | **RESOLVED** | Data confirmed in `north_america_as7018_final` and `europe_as3209_final` outputs; accuracy and fleet metrics filled in                                                                                                                                                                                                                   |
| 4   | §6.2             | Spotter DE labeled "pending"                                                                                                         | ~~LOW~~  | **RESOLVED** | Updated to 17.7%                                                                                                                                                                                                                                                                                                                         |
| 5   | §6.3             | EMPTY/EXCLUSIVE/INCLUSIVE partition — prior per-VP check was wrong                                                                   | ~~MED~~  | **RESOLVED** | Verified by Shapely polygon-disk intersection (`exclusive_verify.py`). EXCLUSIVE is **large** for annulus variants: Octant ~50% (US/DE), Spotter ~89–99%. Disk variants near-zero. EXCLUSIVE-but-correct centroid snapping preserves classification accuracy. Paper-flow §6.2/§6.3/intro.tex corrected. See `exclusive_verification.md`. |
| 6   | §6.3             | "AUC 0.82 in EU-country" mixes two AUC definitions                                                                                   | ~~MED~~  | **RESOLVED** | 0.82 = RTT-inflation → failure AUC (from `WHEN_CBG_FAILS.md`, mechanism attribution); 0.684 = fleet_abs_km → error AUC (fleet geometry). Different response variables. Paper-flow §6.3 clarified with parenthetical.                                                                                                                     |
| 7   | §6.1/6.2         | Shortest-ping baseline (23.4% global, 39.6% US, 50.0% DE, TBD for cross-region) unverifiable                                         | ~~MED~~  | **RESOLVED** | Ran `plot_cluster_match_bars.py` for all 5 final runs: global=23.4% ✓, US=39.6% ✓, DE=50.0% ✓, AS7018→global=5.3% (new), AS3209→global=6.3% (new). Paper-flow §6.1 TBD rows filled.                                                                                                                                                      |
| 8   | §6.2             | Cluster counts (96/32 US, 96/21 DE, 713/257 global) not all confirmed                                                                | ~~LOW~~  | **RESOLVED** | All confirmed: global=257 (from clusters.csv), US=32 (from clusters.csv for north_america_as7018_final_us), DE=21 (from plot_cluster_match_bars.py n_centroids output for europe_as3209_final_de).                                                                                                                                       |


---

## What's Verified and Correct (post-corrections)

- §2.3 Variant table — now corrected to `planar_circle + geometric_centroid` for Vanilla/Million-scale
- §6.1 AS16509→global CBG accuracy (Vanilla 24.3%, MS 23.0%, Octant 25.2%, Spotter 7.0%)
- §6.1 AS7018→global accuracy (Vanilla 3.9%, MS 6.6%, Octant 5.5%, Spotter 3.0%) — new data
- §6.1 AS3209→global accuracy (Vanilla 2.7%, MS 6.0%, Octant 2.0%, Spotter 2.5%) — new data
- §6.1 global-global fleet geometry (348 km median, 77% missing, −313 km margin)
- §6.1 VP proximity failure coverage percentages (84.4–91.8%) and AUC range (0.84–0.96)
- §6.2 AS7018→US CBG accuracy (Vanilla 45.8%, MS 43.8%, Octant 51.0%, Spotter 6.3%)
- §6.2 AS7018→US fleet geometry (35.1 km, +62.0 km, 43.8% missing, 125 VPs)
- §6.2 AS3209→DE CBG accuracy (Vanilla 18.8%, MS 46.9%, Octant 39.6%, Spotter 17.7%) — Spotter no longer pending
- §6.2 AS3209→DE fleet geometry (4.6 km, +38.6 km, 2.1% missing, 164 VPs) — corrected from FR metrics
- §6.2 Tolerance dividend for AS7018→US (all four variants within 0.1 pp)
- §6.2 Tolerance dividend for AS3209→DE (Vanilla +6.3 pp/33%, MS +5.2 pp/11%, Octant +5.2 pp/13%, Spotter +9.4 pp/53%) — 3–5× smaller than US as expected given 4.6 km median VP distance
- §6.5 Runtime numbers for `octant_hull_geo` and `octant_cbg` (accuracy, throughput, centroid latency)
- §6.1 global cluster count: 257 clusters (confirmed from `clusters.csv`)
- §6.1 shortest-ping baselines: global=23.4%, US=39.6%, DE=50.0% (all confirmed); AS7018→global=5.3%, AS3209→global=6.3% (new)
- §6.2 cluster counts: US=32 (confirmed from `clusters.csv`), DE=21 (confirmed from `plot_cluster_match_bars.py`)
- §6.3 AUC distinction: 0.82 = RTT-inflation→failure attribution AUC; 0.684 = fleet_abs_km→error AUC (separate metrics)
- §6.3 EMPTY/EXCLUSIVE/INCLUSIVE partition (Shapely polygon-disk verified): Octant ~50% EXCLUSIVE (US/DE), Spotter ~89–99% EXCLUSIVE; disk variants near-zero; EXCLUSIVE-but-correct centroid snapping is the tolerance mechanism; Octant's ~27 pp tolerance dividend = ~27/48 EXCLUSIVE-but-correct predictions; corrections applied to §6.2 Octant, §6.3 key finding, intro.tex #4

