"""
Octant RTT-Distance Model

Implements the Octant framework's dual-bound RTT-distance modeling approach
(Wong et al., NSDI 2007) for IP geolocation.

Key features:
- Convex hull dual bounds (R_L upper, r_L lower) for annular constraints
- Count-based reliability cutoff for sparse data regions
- Polynomial-based iterative refinement with delta search

Ablated features (for scalability on passive data):
- Height computation (requires active traceroute)
- Intermediate router localization (requires traceroute paths)
"""

import numpy as np
import pickle
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

from scipy.spatial import ConvexHull
from scipy.optimize import lsq_linear

# Constants (shared with rtt_model.py)
EARTH_RADIUS_KM = 6371.0
SPEED_OF_LIGHT_KM_S = 300_000.0
SPEED_OF_LIGHT_KM_MS = SPEED_OF_LIGHT_KM_S / 1000.0  # 300 km/ms
THEORETICAL_SLOPE = 2 / (SPEED_OF_LIGHT_KM_MS * (2/3))  # ~0.01 ms/km for 2/3 c


# =============================================================================
# Exception Classes
# =============================================================================

class OctantFitError(Exception):
    """Base exception for Octant model fitting errors."""
    pass


class PolynomialFitError(OctantFitError):
    """Raised when polynomial fitting fails."""
    pass


class DeltaSearchError(OctantFitError):
    """Raised when no delta satisfies the coverage requirement."""
    pass


class DeltaSearchTimeout(OctantFitError):
    """Raised when delta search exceeds timeout."""
    pass


# =============================================================================
# Convex Hull Functions
# =============================================================================

def compute_convex_hull_bounds(
    rtts: np.ndarray,
    distances: np.ndarray,
    cutoff_min_points: int = 5,
    bin_size_ms: float = 10.0,
    baseline_slope: float = THEORETICAL_SLOPE
) -> Dict[str, Any]:
    """
    Compute upper and lower convex hull facets from (RTT, distance) scatter.

    Following Octant Section 3.1:
    - Upper hull facets define R_L (outer radius for positive constraints)
    - Lower hull facets define r_L (inner radius for negative constraints)
    - Cutoff: where bin has < cutoff_min_points, transition to conservative bounds

    Args:
        rtts: RTT values in ms (x-axis in hull computation)
        distances: Geographic distances in km (y-axis in hull computation)
        cutoff_min_points: Min points in RTT bin to trust hull (default 5)
        bin_size_ms: Size of RTT bins for cutoff detection (default 10ms)
        baseline_slope: Slope for conservative bounds beyond cutoff (km/ms)

    Returns:
        Dictionary with:
        - hull_upper_rtts, hull_upper_distances: Upper hull vertices
        - hull_lower_rtts, hull_lower_distances: Lower hull vertices
        - cutoff_rtt: RTT threshold for sparse data
        - metadata: Additional info
    """
    rtts = np.asarray(rtts, dtype=float)
    distances = np.asarray(distances, dtype=float)

    # Filter invalid values
    valid_mask = (rtts > 0) & (distances > 0) & np.isfinite(rtts) & np.isfinite(distances)
    rtts = rtts[valid_mask]
    distances = distances[valid_mask]

    if len(rtts) < 3:
        return {
            'hull_upper_rtts': [],
            'hull_upper_distances': [],
            'hull_lower_rtts': [],
            'hull_lower_distances': [],
            'cutoff_rtt': 0.0,
            'success': False,
            'message': f'Need at least 3 valid points, got {len(rtts)}'
        }

    # Stack points as (RTT, distance)
    points = np.column_stack([rtts, distances])

    try:
        hull = ConvexHull(points)
    except Exception as e:
        return {
            'hull_upper_rtts': [],
            'hull_upper_distances': [],
            'hull_lower_rtts': [],
            'hull_lower_distances': [],
            'cutoff_rtt': 0.0,
            'success': False,
            'message': f'ConvexHull computation failed: {str(e)}'
        }

    # Extract hull vertices
    hull_vertices = points[hull.vertices]

    # Sort by RTT (x-axis)
    sorted_indices = np.argsort(hull_vertices[:, 0])
    hull_vertices = hull_vertices[sorted_indices]

    # Separate upper and lower chains
    # Find the leftmost and rightmost points
    min_rtt_idx = 0
    max_rtt_idx = len(hull_vertices) - 1

    # Split vertices into upper and lower chains
    # Upper chain: max distance at each RTT
    # Lower chain: min distance at each RTT
    hull_upper = []
    hull_lower = []

    # Use a sweep approach: for each unique RTT region, track max and min
    # Actually, the convex hull vertices naturally form upper and lower chains
    # We need to identify which vertices are on the upper vs lower boundary

    # Compute centroid of all points for reference
    centroid_dist = np.mean(distances)

    # Classify each hull vertex as upper or lower based on distance relative to trend
    # Simple approach: fit a line through all points, classify based on residual
    if len(rtts) >= 2:
        # Fit linear trend
        slope, intercept = np.polyfit(rtts, distances, 1)
        trend_at_hull = slope * hull_vertices[:, 0] + intercept

        for i, (rtt, dist) in enumerate(hull_vertices):
            if dist >= trend_at_hull[i]:
                hull_upper.append((rtt, dist))
            else:
                hull_lower.append((rtt, dist))

    # Ensure both chains have at least the endpoints
    if len(hull_upper) < 2:
        # Add min and max RTT points to upper
        hull_upper = [(hull_vertices[0, 0], hull_vertices[0, 1]),
                      (hull_vertices[-1, 0], hull_vertices[-1, 1])]
    if len(hull_lower) < 2:
        # Add min and max RTT points to lower
        hull_lower = [(hull_vertices[0, 0], hull_vertices[0, 1]),
                      (hull_vertices[-1, 0], hull_vertices[-1, 1])]

    # Sort by RTT
    hull_upper = sorted(hull_upper, key=lambda x: x[0])
    hull_lower = sorted(hull_lower, key=lambda x: x[0])

    # Detect cutoff RTT by scanning bins from high to low
    max_rtt = np.max(rtts)
    min_rtt = np.min(rtts)
    cutoff_rtt = max_rtt  # Default: no cutoff

    # Scan from high RTT to low, find where density drops
    current_rtt = max_rtt
    while current_rtt > min_rtt:
        bin_mask = (rtts >= current_rtt - bin_size_ms) & (rtts < current_rtt)
        bin_count = np.sum(bin_mask)
        if bin_count >= cutoff_min_points:
            cutoff_rtt = current_rtt
            break
        current_rtt -= bin_size_ms

    if cutoff_rtt == max_rtt:
        # Check if the last bin has enough points
        bin_mask = (rtts >= max_rtt - bin_size_ms) & (rtts <= max_rtt)
        if np.sum(bin_mask) >= cutoff_min_points:
            cutoff_rtt = max_rtt + bin_size_ms  # No cutoff needed

    return {
        'hull_upper_rtts': [x[0] for x in hull_upper],
        'hull_upper_distances': [x[1] for x in hull_upper],
        'hull_lower_rtts': [x[0] for x in hull_lower],
        'hull_lower_distances': [x[1] for x in hull_lower],
        'cutoff_rtt': cutoff_rtt,
        'baseline_slope': baseline_slope,
        'success': True,
        'message': f'Hull computed with {len(hull_upper)} upper and {len(hull_lower)} lower vertices'
    }


def hull_rtt_to_distance(
    rtt: float,
    hull_rtts: List[float],
    hull_distances: List[float],
    cutoff_rtt: float,
    baseline_slope: float = THEORETICAL_SLOPE,
    is_upper: bool = True
) -> float:
    """
    Convert RTT to distance using piecewise-linear hull boundary.

    Interpolates between hull vertices for RTT < cutoff_rtt,
    uses conservative slope for RTT >= cutoff_rtt.

    Args:
        rtt: Query RTT in ms
        hull_rtts: RTT values of hull vertices (sorted ascending)
        hull_distances: Distance values of hull vertices
        cutoff_rtt: RTT threshold for sparse data
        baseline_slope: Conservative slope beyond cutoff (km/ms)
        is_upper: True for upper hull (R_L), False for lower hull (r_L)

    Returns:
        Distance in km
    """
    if len(hull_rtts) == 0:
        # Fallback to speed-of-light conversion
        return rtt / baseline_slope if baseline_slope > 0 else 0.0

    hull_rtts = np.array(hull_rtts)
    hull_distances = np.array(hull_distances)

    # Handle RTT beyond cutoff: use conservative slope
    if rtt >= cutoff_rtt and cutoff_rtt > 0:
        # Find distance at cutoff and extend with baseline slope
        cutoff_dist = hull_rtt_to_distance(
            cutoff_rtt - 0.001, hull_rtts.tolist(), hull_distances.tolist(),
            cutoff_rtt + 1, baseline_slope, is_upper
        )
        extra_rtt = rtt - cutoff_rtt
        if is_upper:
            # Upper bound: extend upward with speed-of-light slope
            return cutoff_dist + extra_rtt / baseline_slope
        else:
            # Lower bound: extend more conservatively (or stay flat)
            return max(0, cutoff_dist)

    # Handle RTT below minimum hull RTT
    if rtt < hull_rtts[0]:
        if is_upper:
            # Extrapolate using slope from first two vertices
            if len(hull_rtts) >= 2:
                slope = (hull_distances[1] - hull_distances[0]) / (hull_rtts[1] - hull_rtts[0])
                return max(0, hull_distances[0] + slope * (rtt - hull_rtts[0]))
            return hull_distances[0]
        else:
            # Lower bound: return 0 for very small RTT
            return 0.0

    # Handle RTT above maximum hull RTT (but below cutoff)
    if rtt > hull_rtts[-1]:
        if is_upper:
            # Extrapolate using slope from last two vertices
            if len(hull_rtts) >= 2:
                slope = (hull_distances[-1] - hull_distances[-2]) / (hull_rtts[-1] - hull_rtts[-2])
                return hull_distances[-1] + slope * (rtt - hull_rtts[-1])
            return hull_distances[-1]
        else:
            return hull_distances[-1]

    # Linear interpolation between hull vertices
    idx = np.searchsorted(hull_rtts, rtt, side='right') - 1
    idx = max(0, min(idx, len(hull_rtts) - 2))

    rtt_low, rtt_high = hull_rtts[idx], hull_rtts[idx + 1]
    dist_low, dist_high = hull_distances[idx], hull_distances[idx + 1]

    if rtt_high == rtt_low:
        return dist_low

    # Linear interpolation
    t = (rtt - rtt_low) / (rtt_high - rtt_low)
    return dist_low + t * (dist_high - dist_low)


# =============================================================================
# Polynomial Fitting Functions
# =============================================================================

def fit_rtt_distance_polynomial(
    rtts: np.ndarray,
    distances: np.ndarray,
    degree: int = 2,
    constrain_monotonic: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fit polynomial minimizing squared error to data.

    Args:
        rtts: RTT values in ms (x-axis)
        distances: Geographic distances in km (y-axis)
        degree: Polynomial degree (default 2 = quadratic)
        constrain_monotonic: If True, constrain slope coefficients >= 0 to ensure
                            monotonically increasing curve, while allowing intercept
                            to be negative (accounts for processing delay). (default True)

    Returns:
        (coefficients, metadata_dict) where coefficients are for np.polyval
        (highest degree first: [c_n, c_{n-1}, ..., c_1, c_0])

    Raises:
        PolynomialFitError: If fitting fails (e.g., insufficient data)

    Note:
        The intercept (c_0) is allowed to be negative because at RTT=0, distance
        should be 0, but real measurements have processing/queuing delay. A negative
        intercept means "distance = 0 when RTT = processing_delay".
    """
    rtts = np.asarray(rtts, dtype=float)
    distances = np.asarray(distances, dtype=float)

    # Filter invalid values
    valid_mask = (rtts > 0) & (distances > 0) & np.isfinite(rtts) & np.isfinite(distances)
    rtts = rtts[valid_mask]
    distances = distances[valid_mask]

    min_points = degree + 2  # Need at least degree+2 points for meaningful fit
    if len(rtts) < min_points:
        raise PolynomialFitError(
            f'Need at least {min_points} valid points for degree {degree}, got {len(rtts)}'
        )

    try:
        if constrain_monotonic:
            # Constrain slope coefficients >= 0, but allow intercept to be any value
            # For degree=2: distance = c2*rtt^2 + c1*rtt + c0
            # Constraints: c2 >= 0, c1 >= 0, c0 can be negative (processing delay)

            # Build Vandermonde matrix: [rtt^2, rtt^1, rtt^0]
            A = np.vander(rtts, degree + 1)

            # Bounds: [0, 0, ..., 0, -inf] for lower, [inf, inf, ..., inf] for upper
            # All coefficients >= 0 except intercept (last one) which can be negative
            lower_bounds = np.zeros(degree + 1)
            lower_bounds[-1] = -np.inf  # Intercept can be negative
            upper_bounds = np.full(degree + 1, np.inf)

            result = lsq_linear(
                A, distances,
                bounds=(lower_bounds, upper_bounds),
                method='bvls'
            )

            if result.success:
                coefficients = result.x
            else:
                # Fall back to unconstrained
                coefficients = np.polyfit(rtts, distances, degree)
        else:
            # Unconstrained fit
            coefficients = np.polyfit(rtts, distances, degree)

        # Calculate R-squared
        predicted = np.polyval(coefficients, rtts)
        ss_res = np.sum((distances - predicted) ** 2)
        ss_tot = np.sum((distances - np.mean(distances)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Check if slope coefficients are non-negative (all except intercept)
        slope_coeffs_positive = all(c >= 0 for c in coefficients[:-1]) if len(coefficients) > 1 else True

        metadata = {
            'degree': degree,
            'n_points': len(rtts),
            'r_squared': r_squared,
            'residual_std': np.std(distances - predicted),
            'constrained': constrain_monotonic,
            'slope_coeffs_positive': slope_coeffs_positive,
            'intercept': coefficients[-1] if len(coefficients) > 0 else 0.0
        }

        return coefficients, metadata

    except Exception as e:
        raise PolynomialFitError(f'Polynomial fitting failed: {str(e)}')


def find_delta_for_coverage(
    rtts: np.ndarray,
    distances: np.ndarray,
    poly_coefficients: np.ndarray,
    target_coverage: float,
    tolerance: float = 0.01,
    max_iterations: int = 100,
    timeout_seconds: float = 10.0,
    delta_min: float = 1.0
) -> Tuple[float, Dict[str, Any]]:
    """
    Find delta such that target_coverage % of data points fall within
    [poly(rtt)/delta, poly(rtt)*delta] bounds.

    Uses binary search to find optimal delta. No upper limit on delta.

    Args:
        rtts: RTT values in ms
        distances: Geographic distances in km
        poly_coefficients: Polynomial coefficients from np.polyfit
        target_coverage: Desired fraction of points within bounds (e.g., 0.90)
        tolerance: Acceptable deviation from target (e.g., 0.01 = ±1%)
        max_iterations: Maximum binary search iterations
        timeout_seconds: Maximum wall-clock time
        delta_min: Minimum delta to search (must be >= 1.0)

    Returns:
        (delta, metadata_dict) where metadata includes actual_coverage, iterations

    Raises:
        PolynomialFitError: If poly_coefficients is None or invalid
        DeltaSearchError: If no delta achieves target coverage
        DeltaSearchTimeout: If search exceeds timeout_seconds
    """
    if poly_coefficients is None or len(poly_coefficients) == 0:
        raise PolynomialFitError('Invalid polynomial coefficients')

    rtts = np.asarray(rtts, dtype=float)
    distances = np.asarray(distances, dtype=float)

    # Filter invalid values
    valid_mask = (rtts > 0) & (distances > 0) & np.isfinite(rtts) & np.isfinite(distances)
    rtts = rtts[valid_mask]
    distances = distances[valid_mask]

    if len(rtts) == 0:
        raise DeltaSearchError('No valid data points')

    start_time = time.time()

    def compute_coverage(delta: float) -> float:
        """Compute fraction of points within [poly/delta, poly*delta]."""
        predicted = np.polyval(poly_coefficients, rtts)
        # Handle negative predictions
        predicted = np.maximum(predicted, 1.0)  # Minimum 1 km
        lower = predicted / delta
        upper = predicted * delta
        within_bounds = (distances >= lower) & (distances <= upper)
        return np.mean(within_bounds)

    # Find initial delta_max by doubling until coverage >= target
    delta_max = delta_min
    max_coverage_seen = compute_coverage(delta_max)

    while max_coverage_seen < target_coverage:
        if time.time() - start_time > timeout_seconds:
            raise DeltaSearchTimeout(
                f'Timeout finding initial delta_max after {time.time() - start_time:.2f}s'
            )
        delta_max *= 2
        max_coverage_seen = compute_coverage(delta_max)
        if delta_max > 1e10:  # Safety limit
            raise DeltaSearchError(
                f'Cannot achieve {target_coverage:.1%} coverage even with delta={delta_max:.0f}'
            )

    # Binary search for optimal delta
    delta_low = delta_min
    delta_high = delta_max
    best_delta = delta_max
    best_coverage = max_coverage_seen
    best_diff = abs(best_coverage - target_coverage)

    for iteration in range(max_iterations):
        if time.time() - start_time > timeout_seconds:
            raise DeltaSearchTimeout(
                f'Timeout after {iteration} iterations, best delta={best_delta:.4f} '
                f'with coverage={best_coverage:.3f}'
            )

        delta_mid = (delta_low + delta_high) / 2
        coverage = compute_coverage(delta_mid)

        # Track best result
        diff = abs(coverage - target_coverage)
        if diff < best_diff:
            best_delta = delta_mid
            best_coverage = coverage
            best_diff = diff

        # Check convergence
        if diff <= tolerance:
            return delta_mid, {
                'actual_coverage': coverage,
                'iterations': iteration + 1,
                'converged': True
            }

        # Binary search step
        if coverage < target_coverage:
            delta_low = delta_mid  # Need wider bounds
        else:
            delta_high = delta_mid  # Can tighten bounds

        # Check if we've converged numerically
        if delta_high - delta_low < 1e-10:
            break

    # Return best found even if not within tolerance
    if best_diff > tolerance:
        raise DeltaSearchError(
            f'Could not achieve {target_coverage:.1%} coverage within tolerance {tolerance:.1%}. '
            f'Best: delta={best_delta:.4f} with coverage={best_coverage:.3f}'
        )

    return best_delta, {
        'actual_coverage': best_coverage,
        'iterations': max_iterations,
        'converged': False
    }


# =============================================================================
# OctantRTTModel Class
# =============================================================================

@dataclass
class OctantRTTModel:
    """
    Octant-style RTT-distance model with dual bounds and polynomial refinement.

    Produces annular constraints (inner/outer radii) instead of single circles.
    Standalone class - no inheritance from RTTDistanceModel.
    """
    anchor_ip: str
    anchor_lat: float = 0.0
    anchor_lon: float = 0.0

    # Convex hull bounds (piecewise linear) - vertices sorted by RTT
    hull_upper_rtts: List[float] = field(default_factory=list)
    hull_upper_distances: List[float] = field(default_factory=list)
    hull_lower_rtts: List[float] = field(default_factory=list)
    hull_lower_distances: List[float] = field(default_factory=list)

    # Reliability cutoff
    cutoff_rtt: float = 0.0
    cutoff_min_points: int = 5
    baseline_slope: float = THEORETICAL_SLOPE

    # Polynomial for iterative refinement
    poly_coefficients: Optional[np.ndarray] = None
    poly_degree: int = 2

    # Metadata
    n_measurements: int = 0
    fitted: bool = False
    fit_message: str = ''

    def fit(
        self,
        rtts: np.ndarray,
        distances: np.ndarray,
        cutoff_min_points: int = 5,
        fit_polynomial: bool = True,
        poly_degree: int = 2
    ) -> bool:
        """
        Fit Octant model: convex hull bounds + optional polynomial.

        Args:
            rtts: RTT values in ms
            distances: Geographic distances in km
            cutoff_min_points: Threshold for sparse data detection
            fit_polynomial: Whether to fit polynomial for refinement
            poly_degree: Polynomial degree (default 2)

        Returns:
            True if fitting succeeded
        """
        self.n_measurements = len(rtts)
        self.cutoff_min_points = cutoff_min_points
        self.poly_degree = poly_degree

        # Compute convex hull bounds
        hull_result = compute_convex_hull_bounds(
            rtts, distances,
            cutoff_min_points=cutoff_min_points,
            baseline_slope=self.baseline_slope
        )

        if not hull_result['success']:
            self.fitted = False
            self.fit_message = hull_result['message']
            return False

        self.hull_upper_rtts = hull_result['hull_upper_rtts']
        self.hull_upper_distances = hull_result['hull_upper_distances']
        self.hull_lower_rtts = hull_result['hull_lower_rtts']
        self.hull_lower_distances = hull_result['hull_lower_distances']
        self.cutoff_rtt = hull_result['cutoff_rtt']

        # Fit polynomial for iterative refinement
        if fit_polynomial:
            try:
                self.poly_coefficients, poly_meta = fit_rtt_distance_polynomial(
                    rtts,
                    distances,
                    degree=poly_degree,
                    constrain_monotonic=True,
                )
                self.fit_message = (
                    f"Hull: {len(self.hull_upper_rtts)} upper, {len(self.hull_lower_rtts)} lower vertices. "
                    f"Poly R²={poly_meta['r_squared']:.3f}"
                )
            except PolynomialFitError as e:
                self.poly_coefficients = None
                self.fit_message = f"Hull OK, polynomial failed: {str(e)}"
        else:
            self.poly_coefficients = None
            self.fit_message = f"Hull: {len(self.hull_upper_rtts)} upper, {len(self.hull_lower_rtts)} lower vertices"

        self.fitted = True
        return True

    def predict_distance_bounds(
        self,
        rtt: float,
        use_polynomial: bool = False,
        delta: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Predict (min_distance, max_distance) bounds from RTT.

        Args:
            rtt: Round-trip time in ms
            use_polynomial: If True, use polynomial bounds instead of hull
            delta: Required if use_polynomial=True

        Returns:
            (r_L(rtt), R_L(rtt)) = (inner_radius, outer_radius)
        """
        if not self.fitted:
            raise OctantFitError('Model not fitted')

        if use_polynomial:
            if self.poly_coefficients is None:
                raise OctantFitError('Polynomial not fitted')
            if delta is None:
                raise ValueError('delta required when use_polynomial=True')

            predicted = np.polyval(self.poly_coefficients, rtt)
            predicted = max(predicted, 1.0)  # Minimum 1 km
            return (predicted / delta, predicted * delta)

        # Use convex hull bounds
        max_dist = hull_rtt_to_distance(
            rtt, self.hull_upper_rtts, self.hull_upper_distances,
            self.cutoff_rtt, self.baseline_slope, is_upper=True
        )
        min_dist = hull_rtt_to_distance(
            rtt, self.hull_lower_rtts, self.hull_lower_distances,
            self.cutoff_rtt, self.baseline_slope, is_upper=False
        )

        return (max(0, min_dist), max_dist)

    def get_refined_bounds(
        self,
        rtt: float,
        target_coverage: float,
        rtts: np.ndarray,
        distances: np.ndarray,
        timeout_seconds: float = 10.0
    ) -> Tuple[float, float, float]:
        """
        Get iteratively refined distance bounds for given coverage.

        Args:
            rtt: RTT measurement in ms
            target_coverage: Desired data coverage (e.g., 0.90)
            rtts, distances: Calibration data for delta search
            timeout_seconds: Maximum time for delta search

        Returns:
            (min_distance, max_distance, delta_used)

        Raises:
            PolynomialFitError, DeltaSearchError, DeltaSearchTimeout
        """
        if not self.fitted:
            raise OctantFitError('Model not fitted')
        if self.poly_coefficients is None:
            raise PolynomialFitError('Polynomial not fitted')

        delta, _ = find_delta_for_coverage(
            rtts, distances, self.poly_coefficients,
            target_coverage=target_coverage,
            timeout_seconds=timeout_seconds
        )

        min_dist, max_dist = self.predict_distance_bounds(
            rtt, use_polynomial=True, delta=delta
        )

        return (min_dist, max_dist, delta)

    def save(self, filepath: Path) -> None:
        """Save model to pickle file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> 'OctantRTTModel':
        """Load model from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for JSON serialization."""
        return {
            'anchor_ip': self.anchor_ip,
            'anchor_lat': self.anchor_lat,
            'anchor_lon': self.anchor_lon,
            'hull_upper_rtts': self.hull_upper_rtts,
            'hull_upper_distances': self.hull_upper_distances,
            'hull_lower_rtts': self.hull_lower_rtts,
            'hull_lower_distances': self.hull_lower_distances,
            'cutoff_rtt': self.cutoff_rtt,
            'cutoff_min_points': self.cutoff_min_points,
            'baseline_slope': self.baseline_slope,
            'poly_coefficients': self.poly_coefficients.tolist() if self.poly_coefficients is not None else None,
            'poly_degree': self.poly_degree,
            'n_measurements': self.n_measurements,
            'fitted': self.fitted,
            'fit_message': self.fit_message
        }

    def __repr__(self) -> str:
        if self.fitted:
            return (
                f"OctantRTTModel(anchor={self.anchor_ip}, "
                f"hull_upper={len(self.hull_upper_rtts)} vertices, "
                f"hull_lower={len(self.hull_lower_rtts)} vertices, "
                f"cutoff_rtt={self.cutoff_rtt:.1f}ms)"
            )
        else:
            return f"OctantRTTModel(anchor={self.anchor_ip}, fitted=False)"
