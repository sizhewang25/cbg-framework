Plan: Replace planar_circle with spherical_circle intersection in Calibrated CBG [ABANDONED]
Context
Currently Calibrated CBG uses planar_circle polygon intersection (find_circles_intersection()) while Vanilla CBG uses spherical_circle intersection (circle_intersections()). This conflates two variables: intersection geometry AND fallback strategy. We want to isolate just the fallback difference by making Calibrated CBG also use spherical_circle intersection.

After this change:

Vanilla CBG: LP + spherical_circle intersection + closest-VP fallback
Calibrated CBG: LP + spherical_circle intersection + inverse-radius weighted average fallback
Million-Scale CBG: 2/3c + spherical_circle intersection + closest-VP fallback
Change
Modify run_cbg_multilateration() in evaluate_million_scale.py to stop calling evaluate_cbg_probe() and instead inline spherical_circle intersection logic (same as run_vanilla_cbg()) but with estimate_location_fallback() as the fallback.

Specifically, replace the body of run_cbg_multilateration() with:

Same circle construction as run_vanilla_cbg(): LP distances → pre-filled 5-tuples (lat, lon, rtt, d, d/6371)
Same spherical_circle intersection: circle_intersections() → polygon_centroid() / get_middle_intersection()
Different fallback: when len(intersections) <= 1, use estimate_location_fallback() (inverse-radius weighted average) instead of closest-VP
This means evaluate_cbg_probe(), find_circles_intersection(), and create_circle_polygon() imports from filter_demonstration.py are no longer needed.

File to modify
scripts/analysis/million_scale/evaluate_million_scale.py

Changes:
Remove imports: evaluate_cbg_probe, find_circles_intersection (keep estimate_location_fallback)
Rewrite run_cbg_multilateration() body — model it after run_vanilla_cbg() but swap the fallback branch
Remove unused df parameter from run_cbg_multilateration() signature and its call in main()
Verification
python scripts/analysis/million_scale/evaluate_million_scale.py
All 3 methods should still evaluate 266 probes
Calibrated CBG results will change (no longer using planar_circle) — compare with previous run

STATUS: ABANDONED — decided to keep planar_circle intersection and instead change the fallback to closest-VP (same as Million-Scale).

---

Plan: Change Calibrated CBG Fallback to Closest-VP
Context
Calibrated CBG currently uses inverse-radius weighted average as fallback (estimate_location_fallback()).
We want it to use closest-VP fallback (same as Million-Scale CBG) so the only difference
between Calibrated and Million-Scale is the intersection method (planar_circle vs spherical_circle) and
RTT-distance model (LP vs 2/3c).

After this change:

Vanilla CBG: LP + spherical_circle intersection + closest-VP fallback
Calibrated CBG: LP + planar_circle intersection + closest-VP fallback
Million-Scale CBG: 2/3c + spherical_circle intersection + closest-VP fallback
