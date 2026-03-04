# Lessons Learned

## Intermediate Findings

### Convex Hull Separation
- Separating the 2D convex hull into upper and lower chains requires care
- Simple approach: fit a linear trend and classify vertices as above/below
- Both chains need at least the min/max RTT endpoints for proper interpolation

### Cutoff Detection
- Count-based cutoff (points per RTT bin) is straightforward to implement
- Scanning from high RTT to low finds the boundary between dense and sparse regions
- Default bin size of 10ms works well for typical RTT distributions

### Delta Search
- Binary search is effective for finding delta achieving target coverage
- Need to dynamically discover delta_max by doubling (no fixed upper limit)
- Tolerance parameter is critical - too tight may never converge

## Mistakes & Corrections

### Test for Timeout
- **Mistake**: Initial test used very short timeout (0.0001s) but small dataset completed instantly
- **Fix**: Use negative timeout (-1.0) to guarantee immediate timeout, or use large dataset

### Test for No Solution
- **Mistake**: Expected DeltaSearchError when 100% coverage requested with zero tolerance
- **Issue**: With wide enough delta, any coverage is achievable (just make bounds huge)
- **Fix**: Combine tight tolerance + limited iterations + short timeout to trigger failure

### Polynomial Evaluation
- **Note**: np.polyval expects coefficients highest-degree-first: [c_n, c_{n-1}, ..., c_0]
- np.polyfit returns in same order, so they're compatible

## Design Insights

### Hull vs Polynomial Trade-off
- **Hull bounds**: Tight, data-driven, but can be jagged with sparse data
- **Polynomial bounds**: Smooth, but may over/under-estimate at extremes
- Both are useful: hull for conservative constraints, polynomial for iterative refinement

### Exception Hierarchy
- Three distinct exceptions cover the failure modes well:
  - `PolynomialFitError`: Insufficient data or numerical issues
  - `DeltaSearchError`: Coverage target unachievable
  - `DeltaSearchTimeout`: Search took too long
- All inherit from `OctantFitError` for unified handling

### Serialization
- pickle works well for dataclass with numpy arrays
- to_dict() needed for JSON (convert np.ndarray to list)
