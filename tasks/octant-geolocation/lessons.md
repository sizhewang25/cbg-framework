# Lessons

- The paper-aligned part that mattered most in practice was restoring the
  weighted-region path in the evaluator instead of using only hard annulus
  intersection.
- The current weighted logic is still an approximation: it uses a grid over the
  intersection of outer-disk bounding boxes rather than exact sub-region boolean
  weighting.
- Removing the max-RTT filter changes geolocation semantics less than expected
  because high-RTT landmarks naturally receive low exponential weight.
- `n_pts` only affects the unweighted Shapely region path; it is irrelevant to
  the weighted grid path.
