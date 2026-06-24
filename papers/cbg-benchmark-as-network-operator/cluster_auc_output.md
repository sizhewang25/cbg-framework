# Cluster Counts and AUC Clarification

## Cluster counts

Row counts are data rows (header excluded). The `europe_as3209_final_de` output directory does not yet exist — that run has not been executed.

| run_id | n_clusters | target_country_filter |
| --- | ---: | --- |
| global_as16509_final | 257 | (none — global fleet, all targets) |
| north_america_as7018_final | 257 | (none — NA fleet, all targets) |
| europe_as3209_final | 257 | (none — EU fleet, all targets) |
| north_america_as7018_final_us | 32 | US |
| europe_as3209_final_de | — (not run yet) | DE |

Note: all three unrestricted runs (`global_as16509_final`, `north_america_as7018_final`, `europe_as3209_final`) share an identical cluster file (same 257-row CSV, same centroid on row 0). The cluster assignment is over the global anchor set (R=50 km, complete-linkage agglomerative), which is independent of the VP fleet, so this is expected.

## AUC 0.82 vs 0.684 — source and meaning

These two AUC values measure **different things** for **different predictors** and are not directly comparable.

### AUC 0.82 — from `WHEN_CBG_FAILS.md`

**File:** `scripts/analysis/partvp/outputs/analysis_fail/WHEN_CBG_FAILS.md`

**Config / variant:** `europe-country` / `octant_cbg`

**What it measures:** AUROC of the feature `part_min_infl` (RTT inflation — the ratio of measured min-RTT to the free-space RTT implied by VP–target distance) for predicting whether a given target **fails** (i.e., is either WRONG or GIVE_UP in the cluster-accuracy sense). This is a **failure-attribution AUC**: does higher RTT inflation correlate with failure?

**Result:** 0.82 — meaning that among the 39 europe-country targets, higher per-target RTT inflation is a strong predictor of octant's failure, confirming that RTT inflation is the dominant mechanism in the country-scale regime.

### AUC 0.684 — from `fleet_geometry_auc.csv`

**File:** `scripts/analysis/partvp/outputs/analysis_fleet/fleet_geometry_auc.csv`

**Config / variant:** `europe-country` / `octant_cbg`

**What it measures:** AUROC of `avail_min_vp_km` (the distance from the target to its nearest available VP in the fleet) for predicting whether the target's **absolute error** is large (i.e., error above a threshold, the "bad" half of the distribution) vs. small. This is a **fleet-geometry AUC**: does having a closer fleet VP correlate with a smaller absolute geolocation error?

**Result:** 0.684 — a moderate positive signal, meaning closer VPs modestly improve absolute accuracy, but the relationship is weaker at country scale (because proximity is largely solved and RTT inflation/containment dominate).

### Are they comparable?

No. The two AUCs differ in three ways:

1. **Response variable:** WHEN_CBG_FAILS uses binary cluster-accuracy failure (WRONG/GIVE_UP vs. correct); fleet_geometry uses large-vs-small absolute-error split.
2. **Predictor feature:** WHEN_CBG_FAILS uses RTT inflation (`part_min_infl`); fleet_geometry uses fleet VP proximity (`avail_min_vp_km`).
3. **Purpose:** WHEN_CBG_FAILS attributes failures to mechanisms (proximity / inflation / containment); fleet_geometry assesses how fleet geometry drives raw accuracy.

### Recommended paper language

Distinguish the two explicitly:

> "RTT inflation is the dominant failure mechanism for Octant at country scale (AUC = 0.82, feature: per-target min-RTT inflation, response: cluster-accuracy failure). Separately, fleet VP proximity has only a moderate effect on absolute error at country scale (AUC = 0.68, feature: nearest-VP distance, response: large-vs-small absolute error), consistent with proximity being largely resolved in the country-restricted regime."

Do not conflate them or use "AUC 0.82" without specifying that it is the **inflation→failure** AUC from the mechanism-attribution analysis, not the fleet-geometry AUC.
