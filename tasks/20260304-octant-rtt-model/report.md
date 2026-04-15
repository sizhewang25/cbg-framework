# Octant RTT-Distance Model - Final Report

## Summary

Successfully implemented a standalone `OctantRTTModel` class that provides Octant-style dual-bound RTT-to-distance modeling with polynomial-based iterative refinement. The implementation follows the Octant paper (Wong et al., NSDI 2007) while ablating features unsuitable for passive measurement data.

### Key Features Implemented

1. **Dual Convex Hull Bounds**
   - Upper hull facets define R_L (outer radius for positive constraints)
   - Lower hull facets define r_L (inner radius for negative constraints)
   - Enables annular constraints instead of simple circles

2. **Reliability Cutoff**
   - Count-based detection of sparse data regions
   - Transitions to conservative speed-of-light bounds beyond cutoff
   - Configurable threshold (default: 5 points per RTT bin)

3. **Polynomial Iterative Refinement**
   - Fits polynomial to (RTT, distance) scatter data
   - Delta search finds multiplier achieving target coverage
   - Binary search with dynamic delta_max discovery

4. **Exception Hierarchy**
   - `OctantFitError` (base class)
   - `PolynomialFitError` - fitting failures
   - `DeltaSearchError` - no valid delta found
   - `DeltaSearchTimeout` - search exceeded time limit

### Ablated Features

- Height computation (inelastic last-hop delays)
- Intermediate router localization
- Spline fitting (simplified to polynomial)

## Results

### Unit Test Results

All 12 tests pass:

| Test Class | Tests | Status |
|------------|-------|--------|
| TestConvexHullBounds | 4 | ✓ Pass |
| TestPolynomialFit | 2 | ✓ Pass |
| TestDeltaSearch | 3 | ✓ Pass |
| TestOctantRTTModel | 3 | ✓ Pass |

### Implementation Statistics

| Component | Lines of Code |
|-----------|---------------|
| octant_model.py | ~350 |
| test_octant_model.py | ~200 |
| Total | ~550 |

### Key Functions

| Function | Purpose |
|----------|---------|
| `compute_convex_hull_bounds()` | Extract upper/lower hull from scatter data |
| `hull_rtt_to_distance()` | Interpolate along hull facets |
| `fit_rtt_distance_polynomial()` | Fit polynomial to data |
| `find_delta_for_coverage()` | Binary search for target coverage |

## Conclusions

### Design Decisions Validated

1. **Standalone class works well** - Easier to test independently, adapter can be added later
2. **Polynomial over spline** - Simpler implementation, sufficient for iterative refinement
3. **Count-based cutoff** - Robust detection of sparse data regions
4. **No delta_max limit** - Binary search naturally finds appropriate range

### Lessons Learned

1. Hull separation requires linear trend classification to distinguish upper/lower
2. Timeout tests need guaranteed failure (negative timeout) rather than racing
3. 100% coverage is always achievable with large delta; use combined constraints

### Future Work

1. **Integration**: Create adapter class for RTTDistanceModel interface
2. **Visualization**: Add plotting utilities for hull and polynomial bounds
3. **Optimization**: Consider spline fitting for smoother bounds
4. **Validation**: Test on real RIPE Atlas measurement data

### Files

```
scripts/analysis/octant/octant_model.py              # Main implementation
scripts/analysis/octant/test_octant_model.py         # Unit tests
scripts/analysis/octant/references/                  # Octant paper & notes
scripts/analysis/octant/outputs/vultr-7922-octant/   # Model outputs (.pkl, .png)
tasks/octant-rtt-model/README.md                     # Task context
tasks/octant-rtt-model/todos.md                      # Progress tracking
tasks/octant-rtt-model/lessons.md                    # Intermediate findings
tasks/octant-rtt-model/report.md                     # This report
```
