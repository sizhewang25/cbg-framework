# Todos

## In Progress

(none)

## Pending

(none)

## Completed

- [x] Create task structure: tasks/octant-rtt-model/ with README, todos, lessons, report
- [x] Write unit tests: test_octant_model.py (12 tests)
- [x] Implement exceptions: OctantFitError hierarchy
- [x] Implement hull functions: compute_convex_hull_bounds(), hull_rtt_to_distance()
- [x] Implement polynomial functions: fit_rtt_distance_polynomial(), find_delta_for_coverage()
- [x] Implement OctantRTTModel class
- [x] Run tests and iterate, document lessons (12/12 tests pass)
- [x] Write final report
- [x] Move octant code from cbg_feasibility/ to scripts/analysis/octant/
- [x] Add constrain_monotonic polynomial fitting (slope coeffs >= 0, intercept free)
- [x] Replace polynomial with piecewise linear spline (make_lsq_spline k=1)
- [x] Fix hull chain extraction: replace trend-line classification with monotone chain algorithm
- [x] Restrict spline fitting to reliable region (RTT <= cutoff_rtt)
- [x] Write new math README (convex hull, cutoff, spline, delta search)
- [x] Create octant_spline_visualization.py (RTT on x, distance on y)
- [x] Bilateral cutoff detection: low_cutoff_rtt (first dense bin) + cutoff_rtt (last dense bin)
- [x] Hull behavior below low_cutoff: upper→2/3c line, lower→0 (vertical)
- [x] Spline fit restricted to [low_cutoff_rtt, cutoff_rtt], extended with 2/3c outside
- [x] n_knots = max(upper_hull_count, lower_hull_count) within reliable region
- [x] All 15 tests pass
