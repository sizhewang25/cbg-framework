"""
Octant RTT-Distance Model

Implements the Octant framework's dual-bound RTT-distance modeling approach
(Wong et al., NSDI 2007) for IP geolocation.

Key features:
- Convex hull dual bounds (R_L upper, r_L lower) for annular constraints
- Count-based reliability cutoff for sparse data regions
- Piecewise linear spline iterative refinement with delta search

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

from scipy.interpolate import make_lsq_spline

# Constants (shared with rtt_model.py)
EARTH_RADIUS_KM = 6371.0
SPEED_OF_LIGHT_KM_S = 300_000.0
SPEED_OF_LIGHT_KM_MS = SPEED_OF_LIGHT_KM_S / 1000.0  # 300 km/ms
THEORETICAL_SLOPE = 2 / (SPEED_OF_LIGHT_KM_MS * (2/3))  # ~0.01 ms/km for 2/3 c
VALID_CUTOFF_VARIANTS = ('none', 'high_only', 'low_only', 'both')


# =============================================================================
# Exception Classes
# =============================================================================

class OctantFitError(Exception):
    """Base exception for Octant model fitting errors."""
    pass


class SplineFitError(OctantFitError):
    """Raised when spline fitting fails."""
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
    bin_size_ms: float = 5.0,
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

    # Monotone chain algorithm for upper and lower convex hull chains.
    # Points sorted by RTT (x), then distance (y) for ties.
    # Upper chain = max-distance boundary (R_L, outer radius).
    # Lower chain = min-distance boundary (r_L, inner radius).
    def _cross(O, A, B):
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])

    pts = sorted(zip(rtts.tolist(), distances.tolist()))

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Both chains include the shared endpoints; reverse upper so it is
    # sorted by RTT ascending (leftmost → rightmost).
    hull_upper = upper[::-1]
    hull_lower = lower

    # Detect low and high cutoff RTT by scanning bins from low to high.
    # low_cutoff_rtt: left edge of first dense bin (below = sparse at low RTT)
    # cutoff_rtt: right edge of last dense bin (above = sparse at high RTT)
    # Scanning upward avoids isolated high-RTT clusters from inflating the cutoff.
    max_rtt = np.max(rtts)
    min_rtt = np.min(rtts)
    low_cutoff_rtt = min_rtt  # Default: no low cutoff
    cutoff_rtt = min_rtt      # Will be updated upward
    found_first_dense = False

    for bin_start in np.arange(min_rtt, max_rtt, bin_size_ms):
        bin_count = np.sum((rtts >= bin_start) & (rtts < bin_start + bin_size_ms))
        if bin_count >= cutoff_min_points:
            if not found_first_dense:
                low_cutoff_rtt = bin_start   # Left edge of first dense bin
                found_first_dense = True
            cutoff_rtt = bin_start + bin_size_ms  # Right edge of last dense bin seen

    return {
        'hull_upper_rtts': [x[0] for x in hull_upper],
        'hull_upper_distances': [x[1] for x in hull_upper],
        'hull_lower_rtts': [x[0] for x in hull_lower],
        'hull_lower_distances': [x[1] for x in hull_lower],
        'cutoff_rtt': cutoff_rtt,
        'low_cutoff_rtt': low_cutoff_rtt,
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
    is_upper: bool = True,
    low_cutoff_rtt: float = 0.0
) -> float:
    """
    Convert RTT to distance using piecewise-linear hull boundary.

    Interpolates between hull vertices for RTT in [low_cutoff_rtt, cutoff_rtt].
    Below low_cutoff_rtt (sparse at low RTT): upper→2/3c line, lower→0.
    Above cutoff_rtt (sparse at high RTT): upper extends with 2/3c slope, lower stays flat.

    Args:
        rtt: Query RTT in ms
        hull_rtts: RTT values of hull vertices (sorted ascending)
        hull_distances: Distance values of hull vertices
        cutoff_rtt: High RTT threshold for sparse data
        baseline_slope: Conservative slope beyond cutoff (km/ms)
        is_upper: True for upper hull (R_L), False for lower hull (r_L)
        low_cutoff_rtt: Low RTT threshold below which data is sparse (default 0 = disabled)

    Returns:
        Distance in km
    """
    # Below low cutoff: data too sparse to trust hull vertices.
    # Upper bound: linear ramp from (0,0) through hull value at low_cutoff.
    # Lower bound: 0 (no reliable lower constraint).
    if low_cutoff_rtt > 0 and rtt < low_cutoff_rtt:
        if is_upper and len(hull_rtts) > 0:
            hull_rtts_arr = np.array(hull_rtts)
            hull_dists_arr = np.array(hull_distances)
            upper_at_cut = float(np.interp(low_cutoff_rtt, hull_rtts_arr, hull_dists_arr))
            return (upper_at_cut / low_cutoff_rtt) * rtt
        return 0.0

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
# Spline Fitting Functions
# =============================================================================

def fit_rtt_distance_spline(
    rtts: np.ndarray,
    distances: np.ndarray,
    n_knots: int = 20
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fit piecewise linear spline minimizing squared error to RTT-distance data.

    Uses scipy.interpolate.make_lsq_spline with k=1 (linear), placing interior
    knots uniformly across the RTT range. Matches the Octant paper description:
    "interpolated spline that minimizes the square error to the data points."

    Args:
        rtts: RTT values in ms
        distances: Geographic distances in km
        n_knots: Number of interior knots (default 20)

    Returns:
        (knot_rtts, knot_dists, metadata) where the two arrays define the
        piecewise linear spline for evaluation via hull_rtt_to_distance() or
        np.interp(). Monotonicity (non-decreasing distance) is enforced.

    Raises:
        SplineFitError: If fitting fails (insufficient data or numerical issues)
    """
    rtts = np.asarray(rtts, dtype=float)
    distances = np.asarray(distances, dtype=float)

    # Filter invalid values
    valid_mask = (rtts > 0) & (distances > 0) & np.isfinite(rtts) & np.isfinite(distances)
    rtts = rtts[valid_mask]
    distances = distances[valid_mask]

    # make_lsq_spline with k=1 and n_knots interior knots needs > n_knots+2 points
    min_points = n_knots + 3
    if len(rtts) < min_points:
        raise SplineFitError(
            f'Need at least {min_points} valid points for {n_knots} interior knots, got {len(rtts)}'
        )

    # Sort by RTT (required for make_lsq_spline)
    sort_idx = np.argsort(rtts)
    rtts = rtts[sort_idx]
    distances = distances[sort_idx]

    try:
        # Build full knot vector for k=1: boundary knots repeated k+1=2 times,
        # with n_knots interior knots strictly inside (rtt_min, rtt_max)
        interior_knots = np.linspace(rtts[0], rtts[-1], n_knots + 2)[1:-1]
        t_full = np.r_[(rtts[0],) * 2, interior_knots, (rtts[-1],) * 2]

        # Global least-squares piecewise linear (k=1) fit
        spline = make_lsq_spline(rtts, distances, t=t_full, k=1)

        # Evaluate at uniform grid to extract plain arrays for storage
        knot_rtts = np.linspace(rtts[0], rtts[-1], n_knots + 2)
        knot_dists = spline(knot_rtts)

        # Enforce monotonicity: distance non-decreasing with RTT
        for i in range(1, len(knot_dists)):
            if knot_dists[i] < knot_dists[i - 1]:
                knot_dists[i] = knot_dists[i - 1]

        # Compute R² against original data
        predicted = spline(rtts)
        ss_res = np.sum((distances - predicted) ** 2)
        ss_tot = np.sum((distances - np.mean(distances)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        metadata = {
            'n_knots': len(knot_rtts),
            'n_points': len(rtts),
            'r_squared': r_squared,
            'residual_std': float(np.std(distances - predicted)),
        }

        return knot_rtts, knot_dists, metadata

    except Exception as e:
        raise SplineFitError(f'Spline fitting failed: {str(e)}')


def find_delta_for_coverage(
    rtts: np.ndarray,
    distances: np.ndarray,
    spline_rtt_knots: np.ndarray,
    spline_dist_knots: np.ndarray,
    target_coverage: float,
    tolerance: float = 0.01,
    max_iterations: int = 100,
    timeout_seconds: float = 10.0,
    delta_min: float = 1.0
) -> Tuple[float, Dict[str, Any]]:
    """
    Find delta such that target_coverage % of data points fall within
    [spline(rtt)/delta, spline(rtt)*delta] bounds.

    Uses binary search to find optimal delta. No upper limit on delta.

    Args:
        rtts: RTT values in ms
        distances: Geographic distances in km
        spline_rtt_knots: RTT knot positions from fit_rtt_distance_spline
        spline_dist_knots: Distance knot values from fit_rtt_distance_spline
        target_coverage: Desired fraction of points within bounds (e.g., 0.90)
        tolerance: Acceptable deviation from target (e.g., 0.01 = ±1%)
        max_iterations: Maximum binary search iterations
        timeout_seconds: Maximum wall-clock time
        delta_min: Minimum delta to search (must be >= 1.0)

    Returns:
        (delta, metadata_dict) where metadata includes actual_coverage, iterations

    Raises:
        SplineFitError: If spline knots are None or invalid
        DeltaSearchError: If no delta achieves target coverage
        DeltaSearchTimeout: If search exceeds timeout_seconds
    """
    if spline_rtt_knots is None or len(spline_rtt_knots) < 2:
        raise SplineFitError('Invalid spline knots')

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
        """Compute fraction of points within [spline/delta, spline*delta]."""
        predicted = np.interp(rtts, spline_rtt_knots, spline_dist_knots)
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
    Octant-style RTT-distance model with dual bounds and spline refinement.

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

    # Reliability cutoffs (bilateral: low and high)
    cutoff_rtt: float = 0.0        # High RTT cutoff — right edge of last dense bin
    low_cutoff_rtt: float = 0.0    # Low RTT cutoff — left edge of first dense bin
    cutoff_min_points: int = 5
    baseline_slope: float = THEORETICAL_SLOPE
    cutoff_variant: str = 'high_only'
    reliable_min_rtt: float = 0.0
    reliable_max_rtt: float = 0.0

    # Piecewise linear spline for iterative refinement
    spline_rtt_knots: Optional[List[float]] = None
    spline_dist_knots: Optional[List[float]] = None
    spline_n_knots: int = 20

    # Metadata
    n_measurements: int = 0
    fitted: bool = False
    fit_message: str = ''

    def _validate_cutoff_variant(self) -> None:
        if self.cutoff_variant not in VALID_CUTOFF_VARIANTS:
            raise ValueError(
                f"Invalid cutoff_variant={self.cutoff_variant!r}. "
                f"Expected one of {VALID_CUTOFF_VARIANTS}."
            )

    def _low_cutoff_enabled(self) -> bool:
        return self.cutoff_variant in ('low_only', 'both')

    def _high_cutoff_enabled(self) -> bool:
        return self.cutoff_variant in ('high_only', 'both')

    def _effective_low_cutoff_rtt(self) -> float:
        if not self._low_cutoff_enabled():
            return 0.0
        if self.reliable_min_rtt > 0:
            return self.reliable_min_rtt
        return float(self.low_cutoff_rtt)

    def _effective_high_cutoff_rtt(self) -> float:
        if not self._high_cutoff_enabled():
            return 0.0
        if self.reliable_max_rtt > 0:
            return self.reliable_max_rtt
        return float(self.cutoff_rtt)

    def _set_reliable_interval(self, min_valid_rtt: float, max_valid_rtt: float) -> None:
        self.reliable_min_rtt = float(min_valid_rtt)
        self.reliable_max_rtt = float(max_valid_rtt)

        if self._low_cutoff_enabled():
            self.reliable_min_rtt = max(self.reliable_min_rtt, float(self.low_cutoff_rtt))
        if self._high_cutoff_enabled():
            self.reliable_max_rtt = min(self.reliable_max_rtt, float(self.cutoff_rtt))

    def _fit_spline_on_reliable_region(
        self,
        rtts: np.ndarray,
        distances: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], int]:
        reliable_mask = (rtts >= self.reliable_min_rtt) & (rtts <= self.reliable_max_rtt)
        spline_rtts = rtts[reliable_mask]
        spline_distances = distances[reliable_mask]

        upper_count = sum(
            self.reliable_min_rtt <= r <= self.reliable_max_rtt
            for r in self.hull_upper_rtts
        )
        lower_count = sum(
            self.reliable_min_rtt <= r <= self.reliable_max_rtt
            for r in self.hull_lower_rtts
        )
        n_knots_used = max(3, max(upper_count, lower_count))

        knot_rtts, knot_dists, spline_meta = fit_rtt_distance_spline(
            spline_rtts, spline_distances, n_knots=n_knots_used
        )
        return knot_rtts, knot_dists, spline_meta, n_knots_used

    def _hull_distance(self, rtt: float, is_upper: bool) -> float:
        hull_rtts = self.hull_upper_rtts if is_upper else self.hull_lower_rtts
        hull_distances = self.hull_upper_distances if is_upper else self.hull_lower_distances
        return hull_rtt_to_distance(
            rtt,
            hull_rtts,
            hull_distances,
            self._effective_high_cutoff_rtt(),
            self.baseline_slope,
            is_upper=is_upper,
            low_cutoff_rtt=self._effective_low_cutoff_rtt(),
        )

    def _hull_distance_array(self, rtt_array: np.ndarray, is_upper: bool) -> np.ndarray:
        return np.array([self._hull_distance(float(rtt), is_upper=is_upper) for rtt in rtt_array])

    def _predict_distance_raw(self, rtt: float) -> float:
        knot_rtts = np.array(self.spline_rtt_knots)
        knot_dists = np.array(self.spline_dist_knots)
        high_cutoff_rtt = self._effective_high_cutoff_rtt()

        if self._high_cutoff_enabled() and high_cutoff_rtt > 0 and rtt > high_cutoff_rtt:
            cutoff_val = float(np.interp(high_cutoff_rtt, knot_rtts, knot_dists))
            return cutoff_val + (rtt - high_cutoff_rtt) / self.baseline_slope
        return float(np.interp(rtt, knot_rtts, knot_dists))

    def _predict_distance_array_raw(self, rtt_array: np.ndarray) -> np.ndarray:
        knot_rtts = np.array(self.spline_rtt_knots)
        knot_dists = np.array(self.spline_dist_knots)
        high_cutoff_rtt = self._effective_high_cutoff_rtt()

        base = np.interp(rtt_array, knot_rtts, knot_dists)
        if self._high_cutoff_enabled() and high_cutoff_rtt > 0:
            cutoff_val = float(np.interp(high_cutoff_rtt, knot_rtts, knot_dists))
            base = np.where(
                rtt_array > high_cutoff_rtt,
                cutoff_val + (rtt_array - high_cutoff_rtt) / self.baseline_slope,
                base,
            )
        return np.maximum(base, 0.0)

    def fit(
        self,
        rtts: np.ndarray,
        distances: np.ndarray,
        cutoff_min_points: int = 5,
        fit_spline: bool = True,
        spline_n_knots: int = 20,
        bin_size_ms=5,
    ) -> bool:
        """
        Fit Octant model: convex hull bounds + optional piecewise linear spline.

        Args:
            rtts: RTT values in ms
            distances: Geographic distances in km
            cutoff_min_points: Threshold for sparse data detection
            fit_spline: Whether to fit piecewise linear spline for refinement
            spline_n_knots: Number of interior knots for the spline (default 20)

        Returns:
            True if fitting succeeded
        """
        self._validate_cutoff_variant()
        self.cutoff_min_points = cutoff_min_points
        self.spline_n_knots = spline_n_knots
        rtts_arr = np.asarray(rtts, dtype=float)
        distances_arr = np.asarray(distances, dtype=float)

        valid_mask = (
            (rtts_arr > 0)
            & (distances_arr > 0)
            & np.isfinite(rtts_arr)
            & np.isfinite(distances_arr)
        )
        valid_rtts = rtts_arr[valid_mask]
        valid_distances = distances_arr[valid_mask]
        self.n_measurements = len(valid_rtts)

        # Compute convex hull bounds
        hull_result = compute_convex_hull_bounds(
            rtts, distances,
            cutoff_min_points=cutoff_min_points,
            baseline_slope=self.baseline_slope,
            bin_size_ms=bin_size_ms,
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
        self.low_cutoff_rtt = hull_result.get('low_cutoff_rtt', 0.0)
        self.spline_rtt_knots = None
        self.spline_dist_knots = None

        if len(valid_rtts) > 0:
            self._set_reliable_interval(float(np.min(valid_rtts)), float(np.max(valid_rtts)))
        else:
            self.reliable_min_rtt = 0.0
            self.reliable_max_rtt = 0.0

        # Fit piecewise linear spline for iterative refinement
        if fit_spline:
            try:
                knot_rtts, knot_dists, spline_meta, n_knots_used = self._fit_spline_on_reliable_region(
                    valid_rtts, valid_distances
                )

                if (
                    (self._low_cutoff_enabled() or self._high_cutoff_enabled())
                    and self.reliable_max_rtt > self.reliable_min_rtt
                ):
                    grid = np.linspace(self.reliable_min_rtt, self.reliable_max_rtt, 500)
                    spline_vals = np.interp(grid, knot_rtts, knot_dists)
                    upper_vals = self._hull_distance_array(grid, is_upper=True)
                    lower_vals = self._hull_distance_array(grid, is_upper=False)
                    in_both = (spline_vals <= upper_vals) & (spline_vals >= lower_vals)

                    if in_both.any():
                        if self._low_cutoff_enabled():
                            self.reliable_min_rtt = max(
                                self.reliable_min_rtt,
                                float(grid[np.where(in_both)[0][0]]),
                            )
                        if self._high_cutoff_enabled():
                            self.reliable_max_rtt = min(
                                self.reliable_max_rtt,
                                float(grid[np.where(in_both)[0][-1]]),
                            )

                        if self.reliable_max_rtt > self.reliable_min_rtt:
                            knot_rtts, knot_dists, spline_meta, n_knots_used = (
                                self._fit_spline_on_reliable_region(valid_rtts, valid_distances)
                            )

                self.spline_rtt_knots = knot_rtts.tolist()
                self.spline_dist_knots = knot_dists.tolist()
                self.spline_n_knots = n_knots_used
                self.fit_message = (
                    f"Hull: {len(self.hull_upper_rtts)} upper, {len(self.hull_lower_rtts)} lower vertices. "
                    f"Variant={self.cutoff_variant}. "
                    f"Reliable RTT=[{self.reliable_min_rtt:.3f}, {self.reliable_max_rtt:.3f}] ms. "
                    f"Spline: {spline_meta['n_knots']} knots, R²={spline_meta['r_squared']:.3f}"
                )
            except SplineFitError as e:
                self.spline_rtt_knots = None
                self.spline_dist_knots = None
                self.fit_message = (
                    f"Hull OK for variant={self.cutoff_variant}, spline failed: {str(e)}"
                )
        else:
            self.spline_rtt_knots = None
            self.spline_dist_knots = None
            self.fit_message = (
                f"Hull: {len(self.hull_upper_rtts)} upper, {len(self.hull_lower_rtts)} lower vertices. "
                f"Variant={self.cutoff_variant}. "
                f"Reliable RTT=[{self.reliable_min_rtt:.3f}, {self.reliable_max_rtt:.3f}] ms"
            )

        self.fitted = True
        return True

    def predict_distance(self, rtt: float) -> float:
        """
        Predict distance from RTT using the fitted spline.

        Applies the piecewise linear spline within [low_cutoff_rtt, cutoff_rtt]
        and extends with the 2/3c slope outside that range, clipping the
        estimate to the convex-hull envelope.

        Args:
            rtt: Round-trip time in ms

        Returns:
            Estimated distance in km

        Raises:
            OctantFitError: If model is not fitted or spline is not available
        """
        if not self.fitted:
            raise OctantFitError('Model not fitted')
        if self.spline_rtt_knots is None:
            raise SplineFitError('Spline not fitted')

        predicted = self._predict_distance_raw(rtt)

        if self.hull_upper_rtts and self.hull_lower_rtts:
            hull_lower = self._hull_distance(rtt, is_upper=False)
            hull_upper = self._hull_distance(rtt, is_upper=True)
            predicted = max(hull_lower, min(predicted, hull_upper))

        return max(predicted, 0.0)

    def predict_distance_array(
        self,
        rtt_array: np.ndarray,
    ) -> np.ndarray:
        """Vectorized distance prediction over an array of RTTs.

        Applies the piecewise linear spline, extending with baseline slope
        beyond cutoff_rtt, then clips the result between hull bounds.

        Args:
            rtt_array: Array of RTT values in ms

        Returns:
            Array of predicted distances in km
        """
        if not self.fitted:
            raise OctantFitError('Model not fitted')
        if self.spline_rtt_knots is None:
            raise SplineFitError('Spline not fitted')

        result = self._predict_distance_array_raw(rtt_array)
        upper = self._hull_distance_array(rtt_array, is_upper=True)
        lower = self._hull_distance_array(rtt_array, is_upper=False)
        result = np.clip(result, lower, upper)

        return result

    def predict_bounds_array(
        self,
        rtt_array: np.ndarray,
        delta: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized delta bounds over an array of RTTs.

        Computes (spline/delta, spline*delta) clamped by hull bounds,
        using hull-clamped spline as the base prediction.

        Args:
            rtt_array: Array of RTT values in ms
            delta: Multiplicative delta factor

        Returns:
            (lower_array, upper_array) both in km, clamped by hulls
        """
        spline = self.predict_distance_array(rtt_array)
        upper_hull = self._hull_distance_array(rtt_array, is_upper=True)
        lower_hull = self._hull_distance_array(rtt_array, is_upper=False)

        delta_upper = np.minimum(spline * delta, upper_hull)
        delta_lower = np.maximum(spline / delta, lower_hull)
        delta_lower = np.maximum(delta_lower, 0.0)
        high_cutoff_rtt = self._effective_high_cutoff_rtt()
        low_cutoff_rtt = self._effective_low_cutoff_rtt()

        # Outside the reliable interval on enabled cutoff sides, fall back to
        # hull bounds directly because the spline is not trusted there.
        if self._high_cutoff_enabled() and high_cutoff_rtt > 0:
            beyond = rtt_array > high_cutoff_rtt
            delta_upper[beyond] = upper_hull[beyond]
            delta_lower[beyond] = np.maximum(lower_hull[beyond], 0.0)
        if self._low_cutoff_enabled() and low_cutoff_rtt > 0:
            below = rtt_array < low_cutoff_rtt
            delta_upper[below] = upper_hull[below]
            delta_lower[below] = np.maximum(lower_hull[below], 0.0)

        return delta_lower, delta_upper

    def predict_distance_bounds(
        self,
        rtt: float,
        delta: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Predict (min_distance, max_distance) bounds from RTT.

        Without delta: returns convex hull bounds (r_L, R_L).
        With delta: returns multiplicative spline band (spline/delta, spline*delta),
        clamped by hull bounds so inner >= hull_lower and outer <= hull_upper.

        Args:
            rtt: Round-trip time in ms
            delta: If provided, use spline delta band instead of hull bounds

        Returns:
            (inner_radius, outer_radius) in km
        """
        if not self.fitted:
            raise OctantFitError('Model not fitted')
        high_cutoff_rtt = self._effective_high_cutoff_rtt()
        low_cutoff_rtt = self._effective_low_cutoff_rtt()

        if (
            (self._low_cutoff_enabled() and low_cutoff_rtt > 0 and rtt < low_cutoff_rtt)
            or (self._high_cutoff_enabled() and high_cutoff_rtt > 0 and rtt > high_cutoff_rtt)
        ):
            max_dist = self._hull_distance(rtt, is_upper=True)
            min_dist = self._hull_distance(rtt, is_upper=False)
            return (max(0, min_dist), max_dist)

        if delta is not None:
            predicted = self.predict_distance(rtt)

            # Clamp spline by hull bounds before delta expansion
            hull_lower = self._hull_distance(rtt, is_upper=False)
            hull_upper = self._hull_distance(rtt, is_upper=True)
            predicted = max(hull_lower, min(predicted, hull_upper))

            inner = predicted / delta
            outer = predicted * delta

            # Clamp bounds by hulls
            inner = max(inner, hull_lower)
            outer = min(outer, hull_upper)

            return (max(0.0, inner), outer)

        # Use convex hull bounds
        max_dist = self._hull_distance(rtt, is_upper=True)
        min_dist = self._hull_distance(rtt, is_upper=False)

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
            SplineFitError, DeltaSearchError, DeltaSearchTimeout
        """
        if not self.fitted:
            raise OctantFitError('Model not fitted')
        if self.spline_rtt_knots is None:
            raise SplineFitError('Spline not fitted')

        delta, _ = find_delta_for_coverage(
            rtts, distances,
            np.array(self.spline_rtt_knots), np.array(self.spline_dist_knots),
            target_coverage=target_coverage,
            timeout_seconds=timeout_seconds
        )

        min_dist, max_dist = self.predict_distance_bounds(rtt, delta=delta)

        return (min_dist, max_dist, delta)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        if 'cutoff_variant' not in self.__dict__:
            self.cutoff_variant = 'high_only'
        if 'reliable_min_rtt' not in self.__dict__:
            if self.spline_rtt_knots is not None and len(self.spline_rtt_knots) > 0:
                self.reliable_min_rtt = float(self.spline_rtt_knots[0])
            else:
                self.reliable_min_rtt = float(self.low_cutoff_rtt)
        if 'reliable_max_rtt' not in self.__dict__:
            if self.spline_rtt_knots is not None and len(self.spline_rtt_knots) > 0:
                self.reliable_max_rtt = float(self.spline_rtt_knots[-1])
            else:
                self.reliable_max_rtt = float(self.cutoff_rtt)

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
            'low_cutoff_rtt': self.low_cutoff_rtt,
            'cutoff_min_points': self.cutoff_min_points,
            'baseline_slope': self.baseline_slope,
            'cutoff_variant': self.cutoff_variant,
            'reliable_min_rtt': self.reliable_min_rtt,
            'reliable_max_rtt': self.reliable_max_rtt,
            'spline_rtt_knots': self.spline_rtt_knots,
            'spline_dist_knots': self.spline_dist_knots,
            'spline_n_knots': self.spline_n_knots,
            'n_measurements': self.n_measurements,
            'fitted': self.fitted,
            'fit_message': self.fit_message
        }

    def __repr__(self) -> str:
        if self.fitted:
            return (
                f"OctantRTTModel(anchor={self.anchor_ip}, "
                f"variant={self.cutoff_variant}, "
                f"hull_upper={len(self.hull_upper_rtts)} vertices, "
                f"hull_lower={len(self.hull_lower_rtts)} vertices, "
                f"cutoff_rtt={self.cutoff_rtt:.1f}ms, "
                f"reliable=[{self.reliable_min_rtt:.1f}, {self.reliable_max_rtt:.1f}]ms)"
            )
        else:
            return (
                f"OctantRTTModel(anchor={self.anchor_ip}, "
                f"variant={self.cutoff_variant}, fitted=False)"
            )
