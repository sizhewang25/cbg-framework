# VP Proximity Metrics for Failure Reassessment

## Context

We are reassessing CBG failures with a narrower focus on fleet geometry. The
original `characterize_failures.py` mixed several mechanisms:

- fleet proximity / resolvability
- RTT inflation
- erroneous containment / blockers
- residual centroid geometry

For the next pass, we want VP proximity to be explicit, centroid-consistent, and
not dependent on a globally learned km threshold across different VP-target
setups.

## Decision

Use the centroid answer space directly.

Definitions for one target:

- `C`: truth target centroid
- `N`: nearest competing centroid
- `L = d(C, N)`: local centroid gap
- `R`: cluster radius, currently 50 km
- `V*`: closest available VP to `C`
- `A = d(V*, C)`: absolute closest-VP-to-truth-centroid distance

We will keep two complementary VP-proximity metrics:

1. Absolute proximity:

   ```text
   fleet_abs_km = d(V*, C)
   ```

2. Target-distinguishable VP margin:

   ```text
   target_distinguishable_vp_distance_km = d(C, N) / 2
   target_distinguishable_vp_margin_km =
       target_distinguishable_vp_distance_km - fleet_abs_km
   ```

Interpretation:

```text
target_distinguishable_vp_margin_km > 0
```

means the closest available VP is inside `d(C,N)/2`. By the triangle
inequality, that VP is guaranteed to be closer to the truth centroid than to the
nearest competing centroid. This is a loose VP-proximity certificate.

We use the margin because it keeps the target-distinguishable decision boundary
while preserving physical km units. It tells us how many km of slack the fleet
has, or how many km the closest VP misses the loose target-distinguishable bound
by.

We intentionally choose `d(C,N)/2` instead of `d(C,N)/2 - R` because this pass
only asks whether a VP can favor the truth centroid. It does not require
worst-case separation between the full radius-`R` cluster regions.

## What Not To Do

Do not pick the absolute km threshold by cross-validating across VP-target
setups and treating the setups as exchangeable. The setups have different target
geographies, VP fleets, and answer-space densities, so cross-setup threshold
selection is not the right conceptual object.

If we report a global descriptive absolute threshold, it should be clearly
labeled descriptive only. The primary proximity rule should be the
target-specific margin derived from `d(C,N)/2 - fleet_abs_km`.

## Reassessment Plan

The next VP-proximity correlation pass should answer:

1. How strongly does `fleet_abs_km` correlate with failure?
2. How strongly does `target_distinguishable_vp_margin_km` correlate with
   failure?
3. Does `target_distinguishable_vp_margin_km <= 0` explain failures better than
   a fixed km cut?
4. Are failures with `target_distinguishable_vp_margin_km > 0` mostly due to non-proximity
   mechanisms such as containment, RTT inflation, or residual centroid geometry?

The current preparation script is:

```text
scripts/analysis/partvp/fleet_geometry_explainability.py
```

The setup x variant failure assessment script is:

```text
scripts/analysis/partvp/assess_vp_proximity_failures.py
```

It writes the focused artifacts under:

```text
scripts/analysis/partvp/outputs/analysis_fleet/
```
