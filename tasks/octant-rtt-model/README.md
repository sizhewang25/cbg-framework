# Task: Octant RTT-Distance Model Implementation

## Background

The **Octant** framework (Wong et al., NSDI 2007) provides a comprehensive approach to IP geolocation that improves upon simple Constraint-Based Geolocation (CBG) by:

1. **Dual Bounds**: Using convex hull of (RTT, distance) scatter to derive both upper (R_L) and lower (r_L) distance bounds
2. **Annular Constraints**: Producing rings (not just circles) for tighter multilateration
3. **Reliability Cutoff**: Transitioning to conservative speed-of-light bounds where data is sparse
4. **Iterative Refinement**: Using polynomial/spline fits with adjustable delta for coverage-based tightening

## Goals

Implement `OctantRTTModel` - a standalone RTT-to-distance model that:

1. Extracts **convex hull dual bounds** from calibration data
2. Applies **count-based reliability cutoff** for sparse RTT regions
3. Fits **polynomial** for iterative refinement
4. Provides **delta search** to achieve target coverage percentage

## Ablated Features

For scalability on passive measurement data, we **do not implement**:
- Height computation (requires active traceroute to all landmarks)
- Intermediate router localization (requires traceroute paths)

## Approach

### Phase 1: Convex Hull Bounds
- Compute 2D convex hull of (RTT, distance) points
- Separate into upper chain (R_L) and lower chain (r_L)
- Detect cutoff RTT where bins have < threshold points
- Extend with speed-of-light slope beyond cutoff

### Phase 2: Polynomial Refinement
- Fit polynomial (degree 2) to data
- Binary search for delta achieving target coverage
- Bounds: `[poly(rtt)/delta, poly(rtt)*delta]`

### Phase 3: Integration
- `OctantRTTModel` class combining hull bounds + polynomial
- Serialization support (pickle, JSON dict)

## Success Criteria

- [ ] All unit tests pass
- [ ] Hull correctly separates upper/lower boundaries
- [ ] Cutoff mechanism prevents overfitting to sparse data
- [ ] Delta search converges for reasonable coverage targets (50-95%)
- [ ] Model serialization round-trips correctly

## Related Files

- [Octant Paper](../scripts/analysis/cbg_feasibility/references/Wong%20et%20al.%20-%20Octant%20A%20Comprehensive%20Framework%20for%20the%20Geolocalization%20of%20Internet%20Hosts.pdf)
- [Existing RTT Model](../scripts/analysis/cbg_feasibility/rtt_model.py)
- [CBG Feasibility Task](../cbg-feasibility/README.md)
