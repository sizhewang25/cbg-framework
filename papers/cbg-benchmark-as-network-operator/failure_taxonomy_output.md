# EMPTY / EXCLUSIVE / INCLUSIVE Partition

## Data availability

Data source: `scripts/benchmark/v2/outputs_partvp/{run_id}/` (the filtered, country-scoped
benchmark runs that feed the `scripts/analysis/partvp/outputs/data/*.parquet` pre-extracted
features). Each combo's `fold_*/targets.parquet` supplies `status`, `pred_lat/lon`,
`target_lat/lon`, `mtl_intersection_kind`, and `mtl_participants` (per-VP band data).

**EMPTY_REGION** (status = FALLBACK):
Confirmed directly from `status == 'FALLBACK'`. Verified against
`mtl_intersection_kind == 'none'` — the two fields agree perfectly for all combos
(0 anomalies across both setups and all 4 textbook combos). No FALLBACK row ever
has a non-`'none'` intersection kind, and no SUCCESS row has `'none'`.
Vanilla and million_scale use `vertex_list`; octant and spotter use `polygon` /
`multipolygon` for their non-empty intersections.

**INCLUSIVE-success** (SUCCESS + correct centroid):
`status == 'SUCCESS'` and the prediction snaps to the truth's cluster centroid
(`pred` nearest centroid == `target` nearest centroid in the R=50 km answer space).
Verified: counts match the `match` column in the pre-extracted partvp parquets exactly
for all 8 combo × setup combinations.

**EXCLUSIVE vs INCLUSIVE-misclass** (SUCCESS + wrong centroid):

For **disk MTL** (vanilla\_cbg, million\_scale\_cbg): each VP contributes a disk
`[0, up_km]` around itself. The 50 km D_truth disk around the truth centroid is
reachable from VP's disk iff `up_km >= d(VP, truth_centroid) - 50`. If any VP fails
this, truth cannot be in R → EXCLUSIVE. If all VPs can reach D_truth → INCLUSIVE-misclass.

For **annulus MTL** (octant\_cbg, spotter\_cbg): VP bands are `[lo_km, up_km]`. The
feasibility check is `lo <= d + 50 AND up >= max(0, d - 50)`. A blocking band does
**not** empty the region for annulus MTL (weighting absorbs it), so the band check
is only used for match=False cases. match=True cases are forced to INCLUSIVE-success
(the prediction is the truth's cluster centroid, which by definition is inside or at the
centroid of R).

**Key finding**: EXCLUSIVE count is **0** for all 8 combos in both proximity-sufficient
setups. For disk MTL this means all participating VP disks are large enough to reach D_truth
even when the prediction is wrong (proximity-sufficient VP fleet keeps VP disks wide and
overlapping). For annulus MTL, the same holds: all VP annular bands can geometrically
reach the 50 km disk around the truth centroid.

The entire "wrong" mass is therefore **INCLUSIVE-misclass**: R is non-empty and
geometrically overlaps D_truth, but the centroid of R falls in a different cluster.

---

## na-us (AS7018 → US, proximity-sufficient, 96 targets × 4 combos)

| combo | EMPTY | EXCLUSIVE | INCLUSIVE-success | INCLUSIVE-misclass | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| vanilla\_cbg | 15 (16%) | 0 (0%) | 44 (46%) | 37 (39%) | 96 |
| million\_scale\_cbg | 0 (0%) | 0 (0%) | 42 (44%) | 54 (56%) | 96 |
| octant\_cbg | 0 (0%) | 0 (0%) | 49 (51%) | 47 (49%) | 96 |
| spotter\_cbg | 0 (0%) | 0 (0%) | 6 (6%) | 90 (94%) | 96 |

---

## europe\_as3209\_final\_de (AS3209 → DE, proximity-sufficient, 96 targets × 4 combos)

| combo | EMPTY | EXCLUSIVE | INCLUSIVE-success | INCLUSIVE-misclass | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| vanilla\_cbg | 36 (38%) | 0 (0%) | 18 (19%) | 42 (44%) | 96 |
| million\_scale\_cbg | 0 (0%) | 0 (0%) | 45 (47%) | 51 (53%) | 96 |
| octant\_cbg | 0 (0%) | 0 (0%) | 38 (40%) | 58 (60%) | 96 |
| spotter\_cbg | 0 (0%) | 0 (0%) | 17 (18%) | 79 (82%) | 96 |

---

## Consistency with failure\_taxonomy.csv

`failure_taxonomy.csv` (from `characterize_failures.py`) reports `match`, `wrong`, and
`give_up` counts. The mapping is:

- `give_up` = EMPTY (status == FALLBACK)
- `match` = INCLUSIVE-success
- `wrong` = EXCLUSIVE + INCLUSIVE-misclass

All 8 combos × 2 setups match exactly against the pre-extracted partvp parquets.
Note: `failure_taxonomy.csv` uses `europe-country` which maps to `europe_as3215_final_fr`
(a different setup from `europe_as3209_final_de`), so direct comparison for EU-DE is
not available there. EU-DE counts are instead verified against
`scripts/analysis/partvp/outputs/data/europe_as3209_final_de.parquet`.

---

## Notes and caveats

**Data source matters**: the `outputs_partvp/` directory (not the main `outputs/`) holds
the geo-filtered runs (target\_country='US' for na-us, target\_country='DE' for EU-DE)
that the partvp feature extraction is based on. Using `outputs/` instead gives different
(unfiltered) target sets and inconsistent match counts.

**EXCLUSIVE = 0 everywhere**: In these proximity-sufficient setups the VP fleet is dense
enough that every VP's contribution (disk or annulus) geometrically reaches the 50 km
disk around the truth centroid, even when the CBG prediction lands on the wrong cluster.
EXCLUSIVE failures require VP bands that are *too narrow* to span from the VP's location
to within 50 km of the truth — a regime that only arises when VPs are very close and
their RTT-derived radii are tight relative to the target-to-truth-centroid distance.
This is not the case in these proximity-sufficient setups.

**All wrong predictions are INCLUSIVE-misclass**: the MTL feasible region R is non-empty
(SUCCESS) and geometrically overlaps D_truth, but the centroid of R snaps to a different
cluster than the ground truth. This is a **centroid estimation failure within a valid
feasible region** — a subtler failure mode than either a collapsed region (EMPTY) or
a region that structurally excludes the truth (EXCLUSIVE).

**Annulus MTL (octant, spotter) caveat**: for match=False cases, the band-based
EXCLUSIVE/INCLUSIVE split is approximate (weighting absorbs blockers; the actual polygon
is not reconstructed). However, since all band checks pass (no VP annulus is geometrically
too far from D_truth), the INCLUSIVE-misclass classification is well-supported.

**vanilla\_cbg EMPTY is large in EU-DE (38%)**: the rigid disk MTL (low-envelope
`echoed_upper_km`) is tightly calibrated and frequently empties when the calibrated radius
is too small for the true target-to-VP distance. This matches the `f_containment` share
in `failure_taxonomy.csv` for erroneous-containment attribution.

**Classification pipeline**: cluster index built with `build_answer_space` at R=50 km,
using the precomputed `clusters/clusters.csv` for NA-US and re-derived from targets for
EU-DE (no precomputed cluster dir). Both setups produce 32 (NA) and 21 (EU) centroids
over 96 targets each.
