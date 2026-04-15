# Todos

## In Progress

(none)

## Pending

- [ ] Reproduce the April 1 baseline results from commit `a9bb3d6`
- [ ] Record baseline Octant accuracy and runtime in this task folder
- [ ] Measure the effect of the zero-radius Shapely epsilon change alone
- [ ] Measure the effect of removing the high-RTT filter alone
- [ ] Measure the effect of changing shared `delta` target coverage from `0.90` to `0.80`
- [ ] Measure the effect of switching from the unweighted evaluator path to the weighted-region path
- [ ] Measure the effect of Sobol sampling alone
- [ ] Measure the effect of `geom-median` alone
- [ ] Check for interaction effects between weighted-region logic and new point-selection logic
- [ ] Save a compact comparison table of all experiment results
- [ ] Decide which change should be reverted, tuned, or kept

## Completed

- [x] Create dedicated debug task folder: `tasks/debug-octant-impl/`
- [x] Record the relevant Octant geolocation changes since the April 1 baseline
- [x] Record the observation that the newer implementation appears to regress final accuracy
