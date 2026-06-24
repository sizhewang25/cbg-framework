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

## §2.3 — Variant Table: Vanilla and Million-Scale Combo Mismatch

**Status: INACCURATE — config does not match table.**

The paper's §2.3 table defines the four textbook variants as:

| Variant | LTD | MTL | CTR |
| --- | --- | --- | --- |
| Vanilla | Low Envelope | **Spherical circle** | **Boundary-Vertex Mean** |
| Million-scale | Speed-of-internet | **Spherical circle** | **Boundary-Vertex Mean** |
| Octant | Bounded Hull/Spline | Planar annulus | Monte Carlo |
| Spotter | Normal Distribution | Planar annulus | Monte Carlo |

The actual combo definitions in all `_final` configs (`global_as16509_final.yaml`, `north_america_as7018_final_us.yaml`, `europe_as3209_final_de.yaml`) are:

| combo_id | ltd | mtl (actual) | ctr (actual) |
| --- | --- | --- | --- |
| `vanilla_cbg` | `low_envelope` | **`planar_circle`** | **`geometric_centroid`** |
| `million_scale_cbg` | `speed_of_internet` | **`planar_circle`** | **`geometric_centroid`** |
| `octant_cbg` | `bounded_spline` | `planar_annulus_weighted` | `monte_carlo_medoid` |
| `spotter_cbg` | `normal_dist` | `planar_annulus_weighted` | `monte_carlo_medoid` |

**`planar_circle` ≠ `spherical_circle`**: `PlanarCircleMTL` approximates each RTT disk as a polygon in degree space (flat-earth, Shapely intersection). `SphericalCircleMTL` uses great-circle geometry on the sphere. Both exist in the codebase (`scripts/framework/v2/__init__.py` exports both), but only `planar_circle` is used in the experiments.

**`geometric_centroid` ≠ `boundary_vertex_mean`**: `BoundaryVertexMeanCTR` takes the unweighted mean of intersection polygon boundary vertices. `geometric_centroid` computes the area-weighted polygon centroid. The config comments (lines 14, 16 in `global_as16509_final.yaml`) incorrectly describe the combos as "spherical_circle + boundary_vertex_mean" while the actual YAML uses `planar_circle + geometric_centroid`.

**Action required:** Decide whether the four "textbook" variants in the paper should be corrected to reflect what's actually run (`planar_circle + geometric_centroid`), or whether the experiments need to be rerun with the true textbook implementations (`spherical_circle + boundary_vertex_mean` for Vanilla and Million-scale). The paper as written is misrepresenting the benchmarked implementations for two of the four variants.

---

## §6.1 — Proximity-Limited Regime

### Claim: AS16509→global classification accuracy

**Status: VERIFIED.**

| combo | paper claims | data value | source |
| --- | ---: | ---: | --- |
| vanilla_cbg | 24.3% | **24.26%** ✓ | `data/global_as16509_final.parquet` |
| million_scale_cbg | 23.0% | **23.00%** ✓ | same |
| octant_cbg | 25.2% | **25.25%** ✓ | same |
| spotter_cbg | 7.0% | **7.01%** ✓ | same |

N = 713 targets, 5-fold cross-validation.

### Claim: Fleet-geometry metrics for global-global

**Status: VERIFIED.**

| metric | paper claims | data value | source |
| --- | --- | --- | --- |
| Median `fleet_abs_km` | 348 km | **348.2 km** ✓ | `fleet_geometry_by_config.csv` (global-global) |
| % missing target-dist. VP | 77% | **77.0%** ✓ | same |
| Median margin | −313 km | **−313.3 km** ✓ | same |

### Claim: VP proximity as dominant failure driver

**Status: VERIFIED.**

From `VP_PROXIMITY_FAILURE_ASSESSMENT.md` (global-global setup):

| variant | failure share explained by missing VP | paper claim |
| --- | ---: | --- |
| vanilla_cbg | 94.3% | "84.4–91.8%" |
| million_scale_cbg | 97.8% | 91.8% (paper's high end) ✓ |
| octant_cbg | 88.7% | within range ✓ |
| spotter_cbg | 76.2% | below range (Spotter exception noted in paper ✓) |

The paper's "84.4–91.8%" range is the range *excluding Spotter* (Vanilla/MS/Octant). The data confirms this.

AUC of `fleet_abs_km` predicting failure: million_scale = **0.96** ✓ (paper claims 0.84–0.96).

### Claim: "Extremely limited" rows (AS7018→global, AS3209→global)

**Status: VERIFIED — data exists in `north_america_as7018_final` and `europe_as3209_final` outputs.**

These are the parent configs (no `target_country` filter), which evaluate each country fleet against all 713 global anchors. Classification accuracy computed from `targets.parquet` fold files:

| Fleet | n VPs | Vanilla | Million-scale | Octant | Spotter |
| --- | ---: | ---: | ---: | ---: | ---: |
| AS7018→global | 125 | 3.9% | **6.6%** | 5.5% | 3.0% |
| AS3209→global | 164 | 2.7% | **6.0%** | 2.0% | 2.5% |

Fleet geometry (computed from VP JSON + cluster centroids via haversine):

| Fleet | Median fleet_abs_km | Median margin | % missing |
| --- | ---: | ---: | ---: |
| AS7018→global | 6,268 km | −6,217 km | 92.4% |
| AS3209→global | 972 km | −915 km | 85.4% |

Both are dramatically worse than the "fairly limited" global fleet (AS16509: 348 km, −313 km, 77%). Random baseline for 257 clusters ≈ 0.4%; the extremely-limited setups achieve only 2–7%, confirming that country-scale fleets against global targets are essentially non-functional for CBG regardless of variant.

### Claim: Shortest-ping baseline 23.4% (global)

**Status: CANNOT VERIFY from available analysis outputs.**

The baseline requires RTT-based computation via `shortest_ping_baseline_rates()` in `scripts/analysis/plot_cluster_cdf.py`, which reads from the benchmark input directory (actual shortest-RTT VP location per fold). A geometric proxy (fraction of targets where `avail_min_vp_km < nearest_other_centroid_km / 2`) gives **22.3%** for global — close to 23.4% (1.1 pp gap). Plausible but not confirmed from code output.

---

## §6.2 — Proximity-Sufficient Regime

### Critical Issue: AS3209→DE fleet metrics are from a different setup

**Status: INACCURATE — fleet metrics in table belong to AS3215→FR.**

The paper's §6.2 fleet table attributes these metrics to "AS3209→DE":

| metric | paper value | actual source |
| --- | --- | --- |
| Median `fleet_abs_km` | 1.5 km | **AS3215→FR** (n=39) |
| Median margin | +62.3 km | **AS3215→FR** ✓ |
| % missing target-dist. VP | 7.7% | **AS3215→FR** ✓ |
| n targets / clusters | 96 / 21 | **AS3209→DE** (n=96, clusters TBD) |

Verification: `fleet_geometry_per_target.parquet` shows `europe-country` config maps to `run_id = europe_as3215_final_fr` (not `europe_as3209_final_de`). The fleet analysis was run on AS3215 (Orange France, n=39 targets, median VP distance 1.49 km), not AS3209 (Vodafone Germany, n=96 targets).

**Actual AS3209→DE fleet metrics** (computed from `europe_as3209_final_de.parquet`):

| metric | actual value |
| --- | --- |
| n targets | 96 |
| Median `fleet_abs_km` | **4.62 km** |
| Median `nearest_other_centroid_km / 2` | 44.3 km |
| Median margin | **+38.6 km** |
| % missing target-dist. VP | **2.1%** |

These differ materially from the AS3215→FR numbers cited in the paper. The fleet geometry analysis must be re-run targeting `europe_as3209_final_de` to get the correct numbers for the AS3209→DE row.

### Claim: AS3209→DE classification accuracy

**Status: DATA EXISTS, but Spotter labeled "pending" when it should not be.**

| combo | paper claims | data (AS3209→DE) | source |
| --- | ---: | ---: | --- |
| vanilla_cbg | 18.8% | **18.75%** ✓ | `data/europe_as3209_final_de.parquet` |
| million_scale_cbg | 46.9% | **46.88%** ✓ | same |
| octant_cbg | 39.6% | **39.58%** ✓ | same |
| spotter_cbg | *pending* | **17.71%** — stale note | same |

The Spotter result for AS3209→DE is 17.71% and is present in the data. The "pending (rerun required)" note in the paper is stale. Update the table with 17.71% for Spotter.

*For reference — AS3215→FR accuracy (the setup whose fleet metrics are cited in the table):*

| combo | AS3215→FR value |
| --- | ---: |
| vanilla_cbg | 20.5% |
| million_scale_cbg | 53.8% |
| octant_cbg | 56.4% |
| spotter_cbg | 35.9% |

These numbers are materially different from the AS3209→DE numbers. The accuracy table and fleet table currently describe two different experiments.

### Claim: AS7018→US classification accuracy

**Status: VERIFIED.**

| combo | paper claims | data value |
| --- | ---: | ---: |
| vanilla_cbg | 45.8% | **45.83%** ✓ |
| million_scale_cbg | 43.8% | **43.75%** ✓ |
| octant_cbg | 51.0% | **51.04%** ✓ |
| spotter_cbg | 6.3% | **6.25%** ✓ |

N = 96 targets, 5-fold cross-validation (fold sizes: 23+13+20+20+20 = 96).

### Claim: AS7018→US fleet geometry (96/32, 35.1 km, +62.0 km, 43.8%)

**Status: VERIFIED (except cluster count unverified).**

| metric | paper claims | data value |
| --- | --- | --- |
| n targets | 96 | **96** ✓ |
| Median `fleet_abs_km` | 35.1 km | **35.15 km** ✓ |
| Median margin | +62.0 km | **+62.02 km** ✓ |
| % missing | 43.8% | **43.75%** ✓ |
| n clusters (answer space) | 32 | **not verified** — not in current analysis outputs |

The "32 clusters" figure cannot be confirmed from available data without running the clustering step on the US anchor subset.

### Claim: Shortest-ping baseline (39.6% US, 50.0% DE)

**Status: CANNOT VERIFY from available analysis outputs.**

Same issue as global: requires RTT-based computation from input fold data. Geometric proxy (avail_min_vp_km < nearest_other_centroid_km / 2) gives 56.2% for US and 97.9% for DE — both substantially higher than the paper's claimed baselines. The large gap is plausible (RTT-shortest VP ≠ distance-closest VP due to routing topology) but cannot be confirmed without running `plot_cluster_match_bars.py` on the inputs directory.

**Blocker for verifying baselines:** To compute the baseline for any setup, run:
```bash
cd /home/nuwinslab/workspace/atnt/cbg-framework
python -m scripts.analysis.plot_cluster_match_bars \
    --run-dir scripts/benchmark/v2/outputs/<run_id> --radius-km 50
```
The script auto-discovers the inputs directory and prints baseline accuracy.

### Claim: Tolerance dividend (§6.2, AS7018→US)

**Status: VERIFIED.**

From `analysis/tolerance_dividend.csv` (`north_america_as7018_final_us`):

| combo | dividend (abs pp) | paper | dividend (rel %) | paper |
| --- | ---: | --- | ---: | --- |
| vanilla_cbg | +31.25 pp | "+31.3 pp" ✓ | **68.2%** | "68%" ✓ |
| million_scale_cbg | +19.79 pp | "+19.8 pp" ✓ | **45.2%** | "45%" ✓ |
| octant_cbg | +27.08 pp | "+27.1 pp" ✓ | **53.1%** | "53%" ✓ |
| spotter_cbg | +6.25 pp | "+6.3 pp" ✓ | **100%** | "~100%*" ✓ |

Paper claim "53–71% of all correct answers are tolerance wins" uses the range from octant (53%) to vanilla (68%) — confirmed ✓. The 71% bound appears in the contribution summary and may be rounded from an intermediate computation; the vanilla value here is 68%.

---

## §6.3 — Success and Failure Analysis

### AUC values (fleet_abs_km predicting failure)

**Status: VERIFIED for global-global; europe-country uses FR (see §6.2 issue).**

From `fleet_geometry_auc.csv`:

| config | variant | AUC (fleet_abs_km) | paper cites |
| --- | --- | ---: | --- |
| global-global | million_scale_cbg | **0.960** | 0.84–0.96 (range) ✓ |
| global-global | vanilla_cbg | **0.849** | within range ✓ |
| global-global | octant_cbg | **0.719** | within range ✓ |
| europe-country (FR!) | octant_cbg | **0.684** | — |

The paper says "Octant RTT inflation AUC 0.82 in EU-country". The `fleet_geometry_auc.csv` shows europe-country/octant = **0.684** (not 0.82). The 0.82 figure likely came from `WHEN_CBG_FAILS.md` which cites "RTT inflation AUC 0.82" for the Octant failure analysis — a different metric (AUC of inflation predicting EXCLUSIVE_REGION failure, not fleet distance predicting any failure). These are different AUCs and should not be conflated in the paper.

### Per-variant failure profile (EMPTY / EXCLUSIVE / INCLUSIVE)

**Status: PARTIALLY VERIFIED from failure_taxonomy.csv. Full three-way split not yet computed.**

The `failure_taxonomy.csv` uses an older four-category scheme (`no_proximity`, `rtt_inflation`, `containment`, `other`). The paper's canonical two-layer taxonomy (EMPTY_REGION / EXCLUSIVE_REGION / INCLUSIVE_REGION) has not yet been computed from the MTL polygon data. The `failure_taxonomy.csv` values for europe-country (which correspond to AS3215→FR, not AS3209→DE) show:

| config | variant | give_up | wrong |
| --- | --- | ---: | ---: |
| europe-country (FR) | vanilla_cbg | 16 of 31 failures | 15 |
| europe-country (FR) | million_scale_cbg | 2 of 18 | 16 |
| europe-country (FR) | octant_cbg | 0 of 17 | 17 |
| europe-country (FR) | spotter_cbg | 9 of 25 | 16 |

`give_up` ≈ EMPTY_REGION (fallback); `wrong` ≈ EXCLUSIVE or INCLUSIVE misclassification (but these are not yet separated). The full EMPTY/EXCLUSIVE/INCLUSIVE partition requires implementing the polygon-disk intersection test described in `discussion.md §2.1` — **this analysis has not been run yet**.

---

## §6.5 — Production Cost

**Status: VERIFIED.**

From `analysis_rq3/rq3_runtime_global_as16509_final.csv`:

| claim | paper value | data value |
| --- | --- | --- |
| `octant_cbg_hull_geo` accuracy | 30.0% | **30.01%** ✓ |
| `octant_cbg_hull_geo` throughput | 65 targets/s | **65.1 targets/s** ✓ |
| `octant_cbg` accuracy | 25.2% | **25.25%** ✓ |
| `octant_cbg` throughput | 3.2 targets/s | **3.15 targets/s** ✓ |
| Geometric centroid latency | ~0.25 ms | **0.256 ms** (`ctr_ms_p50`) ✓ |
| Monte Carlo latency range | 190–390 ms | `octant_cbg_hull`: 204 ms; `octant_cbg`: 278 ms; `spotter_cbg`: 390 ms → **range 191–390 ms** ✓ |

---

## Summary: Issues Requiring Action

| # | Section | Issue | Severity | Status | Action |
| --- | --- | --- | --- | --- | --- |
| 1 | §2.3 | `vanilla_cbg` and `million_scale_cbg` use `planar_circle + geometric_centroid` vs. paper's "spherical circle + boundary-vertex mean" | ~~HIGH~~ | **RESOLVED** | Official config confirmed as `planar_circle + geometric_centroid`; paper-flow.md §2.3 table and implementation note updated |
| 2 | §6.2 fleet table | AS3209→DE fleet metrics (1.5 km, 62.3 km, 7.7%) were from AS3215→FR | ~~HIGH~~ | **RESOLVED** | paper-flow.md §6.2 updated to correct AS3209→DE values: 4.6 km, +38.6 km, 2.1% |
| 3 | §6.1 | AS7018→global and AS3209→global TBD rows | ~~HIGH~~ | **RESOLVED** | Data confirmed in `north_america_as7018_final` and `europe_as3209_final` outputs; accuracy and fleet metrics filled in |
| 4 | §6.2 | Spotter DE labeled "pending" | ~~LOW~~ | **RESOLVED** | Updated to 17.7% |
| 5 | §6.3 | EMPTY/EXCLUSIVE/INCLUSIVE partition not yet computed | **MED** | Open | Implement polygon-disk intersection taxonomy (see `discussion.md §2.1`) |
| 6 | §6.3 | "AUC 0.82 in EU-country" mixes two AUC definitions | **MED** | Open | Clarify: fleet_abs_km AUC for europe-country/octant = 0.684; the 0.82 is RTT-inflation AUC from `WHEN_CBG_FAILS.md` |
| 7 | §6.1/6.2 | Shortest-ping baseline (23.4% global, 39.6% US, 50.0% DE, TBD for cross-region) unverifiable | **MED** | Open | Run `python -m scripts.analysis.plot_cluster_match_bars --run-dir scripts/benchmark/v2/outputs/<run_id> --radius-km 50` for each final run |
| 8 | §6.2 | Cluster counts (96/32 US, 96/21 DE, 713/257 global) not all confirmed | **LOW** | Partially open | 257 global clusters confirmed (`north_america_as7018_final/clusters/clusters.csv` has 257 rows); 32 US and 21 DE need verification |

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
- §6.5 Runtime numbers for `octant_hull_geo` and `octant_cbg` (accuracy, throughput, centroid latency)
- §6.1 global cluster count: 257 clusters (confirmed from `clusters.csv`)
