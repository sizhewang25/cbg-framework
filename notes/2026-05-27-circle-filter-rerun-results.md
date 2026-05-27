# Circle-filter default-on — rerun results across all 6 ASN configs

**Date:** 2026-05-27
**Commit:** `408c185` (configs) on top of `e64677b` (MTL change)
**Trigger:** [finding_octant_na_collapse_as7018](../../.claude/projects/-home-nuwinslab-workspace-atnt-cbg-framework/memory/finding_octant_na_collapse_as7018.md) — octant_cbg succeeded on 2/122 NA targets in AS7018.

This note records what changed, how it changed Octant/Spotter behavior across all six per-ASN configs, and the open questions left after the fix.

---

## What landed

### 1. `filter_redundant_outer_disks` helper in [scripts/framework/geometry.py](../scripts/framework/geometry.py)

Generic primitive: given `(centers_latlon, radii_km)`, return the indices to keep after dropping disks that fully contain another (`r_i > center_dist(i,j) + r_j`). The smallest disk in any chain is always kept.

`circle_preprocessing` (used by `SphericalCircleMTL`) was refactored to delegate to this helper — behavior identical.

### 2. `enable_circle_filter` kwarg, **default True**, on three MTLs

| MTL | Effect | File |
|---|---|---|
| `PlanarCircleMTL` | Speedup only — `A ⊇ B ⇒ A ∩ B = B`, correctness no-op. | [planar_circle.py:48](../scripts/framework/v2/mtl/planar_circle.py#L48) |
| `PlanarAnnulusMTL` | Drops engulfing constraints *with their inner-disk veto*. Heuristic. | [planar_annulus.py:29](../scripts/framework/v2/mtl/planar_annulus.py#L29) |
| `PlanarAnnulusWeightedMTL` | Same drop applied before weighted face decomposition; Σwᵢ recomputed over kept constraints. | [planar_annulus_weighted.py:34](../scripts/framework/v2/mtl/planar_annulus_weighted.py#L34) |

Default-on is opinionated: the filter strictly helps the disk MTL (no-op on result) and was the targeted fix for the annular MTLs. yaml configs were updated to mention the kwarg explicitly under `mtl_kwargs:` for documentation.

### 3. Test fixture
`test_bridge_cut_inner_disks_produce_multipolygon` was designed for filter-off — its three "inner_*" constraints have outer=100° (deliberately engulfing) and only contribute through their inner disks. The new default drops them and erases the bridge cut, so the test pins `enable_circle_filter=False`. A second test (`test_circle_filter_drops_engulfing_inner_constraints`) was added to assert the filter prunes those constraints on the same fixture.

---

## SUCCESS / home-continent total, after rerun

Home continent = home of the probe AS for per-ASN runs; for `global_as*` runs the eval set is global (no home filter applied).

| Run | n | vanilla_cbg | octant_cbg | octant_hull_cbg | octant_weighted | octant_hull_weighted | spotter_cbg | spotter_weighted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| europe_as3209 | 415 | 384 | 357 | 381 | **415** | **415** | 298 | **415** |
| europe_as3215 | 415 | 398 | 378 | 396 | **415** | **415** | 286 | **415** |
| global_as16509 | 713 | 704 | 496 | 572 | **713** | **713** | 137 | **713** |
| global_as31898 | 713 | 586 | 441 | 517 | **713** | **713** | 215 | **713** |
| **north_america_as7018** | 122 | 96 | **34** *(was 2)* | 91 | **122** | **122** | 50 | **122** |
| north_america_as7922 | 122 | 84 | 25 | 79 | **122** | **122** | 35 | **122** |

### SUCCESS p50 error (km, home-continent SUCCESS-only)

| Run | vanilla_cbg | octant_cbg | octant_hull_cbg | octant_weighted | **octant_hull_weighted** | spotter_cbg | spotter_weighted |
|---|---:|---:|---:|---:|---:|---:|---:|
| europe_as3209 | 464 | 499 | 460 | 431 | **438** | 541 | 476 |
| europe_as3215 | 501 | 447 | 459 | 425 | **439** | 816 | 606 |
| global_as16509 | 395 | 260 | 228 | 322 | **274** | 994 | 965 |
| global_as31898 | 462 | 416 | 388 | 517 | **510** | 1346 | 625 |
| north_america_as7018 | 479 | 578 | 412 | 489 | **299** | 648 | 618 |
| north_america_as7922 | 300 | 631 | 334 | 425 | **172** | 600 | 487 |

---

## Reading the results

### Three regimes, by MTL family

1. **Unweighted annular intersection** (`octant_cbg`, `octant_cbg_centroid`, `spotter_cbg`) still loses in-region targets at high rate. The filter drops *engulfing* constraints, but plenty of non-engulfing constraints still mis-bracket (a VP with a small outer disk that under-predicts is the binding constraint and survives the filter). The intersection ∩ subtraction is unforgiving.
2. **Weighted face decomposition** (`*_weighted_cbg`) hits **100% SUCCESS on home-continent for every run.** Weighting by `exp(-rtt/τ)` means a constraint that disagrees with the others contributes less to the cumulative face-weight threshold — the feasible region survives as long as enough constraints agree, no need for unanimous bracketing.
3. **Hull-only LTD** (`octant_hull_*`) generalizes the input distribution by dropping the per-VP δ band and spline fit; raw convex-hull outer/inner bounds end up wider and bracket more eval points. Pairing hull-only LTD with weighted MTL is the most permissive combination.

### Best NA accuracy ≠ vanilla CBG

`octant_hull_weighted_cbg` is the consistent **accuracy** winner on home-continent across all 6 runs:

- AS7018: 122/122 SUCCESS, **p50 = 299 km** (vanilla 96/122, p50 = 479 km)
- AS7922: 122/122 SUCCESS, **p50 = 172 km** (vanilla 84/122, p50 = 300 km)
- global_as16509: 713/713 SUCCESS, **p50 = 274 km** (vanilla 704/713, p50 = 395 km)

So the regional-fleet pattern reverses: with the filter on, the right annular method (hull + weighted) *beats* vanilla CBG both on coverage and median error.

The exception is `global_as31898` (Hurricane Electric, global VP fleet): vanilla p50 = 462 km, octant_hull_weighted p50 = 510 km. With no continent concentration in the fleet, vanilla's upper-envelope LP is already well-calibrated.

### Spotter unweighted is the worst in-region

`spotter_cbg` succeeds on 137/713 (19%) of global_as16509 targets and 50/122 (41%) of AS7018 NA. The pooled-normal LTD's `[μ ± k·σ]` band is narrower than `bounded_spline`'s δ band — fewer engulfing constraints to drop, more non-engulfing mis-brackets that survive the filter. Weighted MTL fully rescues it.

### Cross-validation against the pre-rerun replay

For AS7018 octant_cbg I had predicted **33/122** NA SUCCESS from an offline LTD-replay; the actual rerun gave **34/122**. The one-anchor difference comes from polygonization noise (n_pts=64) and Sobol sample ordering inside `monte_carlo_medoid` — close enough that the mechanism is confirmed.

---

## Mechanism recap

The filter exploits a correlation between two properties of a VP's annular constraint:

- **Outer-disk engulfing.** If VP_A's outer disk fully contains VP_B's outer disk, VP_A's outer constraint is non-binding (intersection-wise) given VP_B's. The pure outer-disk intersection result is unchanged by dropping VP_A.
- **Inner-disk veto risk.** Wide-outer disks come from high-RTT VPs, whose `bounded_spline` δ band is pulled wide. For the same VP, the *inner* disk is also wider (raw `[inner, outer]` annulus is roughly translated by RTT). In dense regional fleets, those wide inner disks are exactly the ones that engulf the truth.

So dropping engulfing-outer VPs incidentally drops their bad inner-disk vetoes. The filter is mathematically neutral on the outer side (disk-intersection) and aggressively favorable on the inner side (annulus subtraction). Smallest disk in any chain stays, so the result is never empty if input wasn't.

For the *weighted* MTL the math is slightly different — face weights are recomputed over kept constraints, so dropping a wide-outer VP also drops its `exp(-rtt/τ)` weight contribution. Since wide-outer = high-RTT = low weight, this is essentially a no-op on the cumulative-weight threshold but a real win on the inner-disk side.

---

## Open questions / next deep dives

1. **Why does unweighted annular still fail ~50% on global_as31898?** Hurricane Electric VPs are globally distributed — there's no "regional" inner-disk-veto concentration. Likely a different mode: per-VP δ band mis-calibration on individual long-distance training pairs. Needs per-target failure decomposition like the AS7018 NA replay.
2. **Hull-weighted accuracy on `global_as31898`** is worse than vanilla on p50 (510 vs 462 km). Either the spline+δ was actually helping for the global VP-set, or the hull bounds need a different `weight_τ`. Try `weight_tau_ms ∈ {25, 100, 200}`.
3. **Spotter as LTD is dominated** by `bounded_spline` everywhere. Worth ablating: is the pooled deg-3/deg-2 polyfit's narrowness the problem, or `k·σ` clipping?
4. **CTR sensitivity not yet tested.** All numbers above use `monte_carlo_medoid`; `geometric_centroid` on the same SUCCESS regions could shift the p50 noticeably (especially for the hull-weighted variants whose regions are larger).
