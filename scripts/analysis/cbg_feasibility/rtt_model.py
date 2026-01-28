"""
RTT-Distance Modeling Module for CBG Feasibility Analysis

This module implements the core functions for Constraint-Based Geolocation (CBG)
using calibrated RTT-distance relationships instead of fixed speed thresholds.

Key Components:
- Haversine distance calculation
- LP-based bestline fitting (original CBG paper method)
- Binned percentile bestline fitting (simplified approximation)
- RTTDistanceModel class for per-anchor calibration
"""

import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# Constants
EARTH_RADIUS_KM = 6371.0
SPEED_OF_LIGHT_KM_S = 300_000.0
SPEED_OF_LIGHT_KM_MS = SPEED_OF_LIGHT_KM_S / 1000.0  # 300 km/ms
THEORETICAL_SLOPE = 2 / (SPEED_OF_LIGHT_KM_MS * (2/3))  # ~0.01 ms/km for 2/3 c


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.

    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)

    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_KM * c


def filter_rtt_data(
    distances: np.ndarray,
    rtts: np.ndarray,
    baseline_slope: Optional[float] = None,
    bin_size_km: float = 100.0,
    n_std: float = 1.0,
    global_n_std: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Filter RTT data to remove invalid and outlier measurements.

    Four-stage filtering (revised approach for robust LP bestline fitting):

    Stage 1: Remove invalid values (zero, negative, inf)

    Stage 2: For each distance bin, keep RTTs within [mean - n_std*σ, mean + n_std*σ]
             - SYMMETRIC filtering: removes both low AND high outliers
             - Low outliers are likely due to mislabeled coordinates or measurement errors
             - High outliers are due to congestion, queuing delays

    Stage 3: Remove RTTs below speed-of-light baseline (physical sanity check)
             min_valid_rtt = baseline_slope * distance
             Keep only: rtt >= min_valid_rtt

    Stage 4: Global filter on binned min RTTs
             - Compute min RTT per distance bin
             - Calculate global mean and std of these bin minimums
             - Remove ALL points in bins where min_rtt is outside [mean - global_n_std*σ, mean + global_n_std*σ]
             - This catches bins with mislabeled coordinates where even the best RTT is anomalous

    Key design decisions:
    - Larger bins (100km default) = more points per bin = more robust mean/std estimates
    - Symmetric mean ± 1σ filtering removes suspiciously low RTTs that would skew LP
    - Baseline filter applied AFTER statistical filtering as final sanity check
    - Global filter on bin minimums catches entire mislabeled distance ranges

    Args:
        distances: Array of distances in km
        rtts: Array of RTT values in ms
        baseline_slope: Minimum RTT/distance ratio (default: 2/3 speed of light)
        bin_size_km: Size of distance bins for outlier filtering (default 100km)
        n_std: Number of standard deviations for per-bin filtering (default 1.0)
        global_n_std: Number of standard deviations for global bin-min filter (default 2.0)

    Returns:
        Tuple of (filtered_distances, filtered_rtts, stats_dict)
    """
    distances = np.asarray(distances, dtype=float)
    rtts = np.asarray(rtts, dtype=float)

    if baseline_slope is None:
        baseline_slope = THEORETICAL_SLOPE

    initial_count = len(distances)
    stats = {
        'initial_count': initial_count,
        'removed_invalid': 0,
        'removed_low_outliers': 0,
        'removed_high_outliers': 0,
        'removed_below_baseline': 0,
        'removed_global_bin_outliers': 0,
        'final_count': 0
    }

    # Stage 1: Filter invalid values (zero, negative, inf)
    valid_mask = (distances > 0) & (rtts > 0) & np.isfinite(distances) & np.isfinite(rtts)
    stats['removed_invalid'] = int(np.sum(~valid_mask))
    distances = distances[valid_mask]
    rtts = rtts[valid_mask]

    if len(distances) == 0:
        stats['final_count'] = 0
        return np.array([]), np.array([]), stats

    # Stage 2: Distance-binned mean ± n_std filtering (SYMMETRIC)
    # This removes both low AND high outliers per bin
    bin_indices = (distances // bin_size_km).astype(int)
    unique_bins = np.unique(bin_indices)

    keep_mask = np.ones(len(distances), dtype=bool)
    low_outlier_count = 0
    high_outlier_count = 0

    for bin_idx in unique_bins:
        in_bin = bin_indices == bin_idx
        bin_rtts = rtts[in_bin]

        if len(bin_rtts) >= 3:  # Need enough points for meaningful mean/std
            mean_rtt = np.mean(bin_rtts)
            std_rtt = np.std(bin_rtts)

            lower_bound = mean_rtt - n_std * std_rtt
            upper_bound = mean_rtt + n_std * std_rtt

            # Mark outliers for removal
            low_outliers = bin_rtts < lower_bound
            high_outliers = bin_rtts > upper_bound

            bin_positions = np.where(in_bin)[0]
            keep_mask[bin_positions[low_outliers]] = False
            keep_mask[bin_positions[high_outliers]] = False

            low_outlier_count += np.sum(low_outliers)
            high_outlier_count += np.sum(high_outliers)

    stats['removed_low_outliers'] = int(low_outlier_count)
    stats['removed_high_outliers'] = int(high_outlier_count)
    distances = distances[keep_mask]
    rtts = rtts[keep_mask]

    if len(distances) == 0:
        stats['final_count'] = 0
        return np.array([]), np.array([]), stats

    # Stage 3: Filter RTTs below baseline (speed-of-light constraint)
    # Applied AFTER statistical filtering as final sanity check
    min_valid_rtt = baseline_slope * distances
    physics_mask = rtts >= min_valid_rtt
    stats['removed_below_baseline'] = int(np.sum(~physics_mask))
    distances = distances[physics_mask]
    rtts = rtts[physics_mask]

    if len(distances) == 0:
        stats['final_count'] = 0
        return np.array([]), np.array([]), stats

    # Stage 4: Global filter on binned min RTTs
    # This catches entire bins with mislabeled coordinates where even the best RTT is anomalous
    bin_indices = (distances // bin_size_km).astype(int)
    unique_bins = np.unique(bin_indices)

    if len(unique_bins) >= 3:  # Need enough bins for meaningful global statistics
        # Compute min RTT for each bin
        bin_min_rtts = {}
        for bin_idx in unique_bins:
            in_bin = bin_indices == bin_idx
            bin_min_rtts[bin_idx] = np.min(rtts[in_bin])

        # Calculate global mean and std of bin minimums
        min_rtt_values = np.array(list(bin_min_rtts.values()))
        global_mean = np.mean(min_rtt_values)
        global_std = np.std(min_rtt_values)

        # Identify outlier bins (where min RTT is outside global mean ± global_n_std*σ)
        lower_bound = global_mean - global_n_std * global_std
        upper_bound = global_mean + global_n_std * global_std

        outlier_bins = set()
        for bin_idx, min_rtt in bin_min_rtts.items():
            if min_rtt < lower_bound or min_rtt > upper_bound:
                outlier_bins.add(bin_idx)

        # Remove ALL points in outlier bins
        if outlier_bins:
            keep_mask = np.array([bin_idx not in outlier_bins for bin_idx in bin_indices])
            stats['removed_global_bin_outliers'] = int(np.sum(~keep_mask))
            distances = distances[keep_mask]
            rtts = rtts[keep_mask]

    stats['final_count'] = len(distances)

    return distances, rtts, stats


def fit_bestline_lp(
    distances: np.ndarray,
    rtts: np.ndarray,
    baseline_slope: Optional[float] = None,
    filter_outliers: bool = True,
    bin_size_km: float = 100.0,
    n_std: float = 1.0,
    global_n_std: float = 1.0
) -> Dict[str, Any]:
    """
    Fit lower envelope bestline using Linear Programming (original CBG paper method).

    From Gueye et al. "Constraint-Based Geolocation of Internet Hosts" (IMC 2004):

    The bestline y = m*x + b is defined as the line that is:
    1. Closest to, but BELOW, all data points (g_ij, d_ij)
    2. Has non-negative intercept (b >= 0)
    3. Has slope >= baseline slope (m >= m_baseline)

    Where:
    - g_ij = geographic distance (km) - plotted on x-axis
    - d_ij = network delay/RTT (ms) - plotted on y-axis
    - m = slope (ms/km)
    - b = intercept (ms) - represents fixed processing delays

    LP Formulation (from Section 3.2, Equations 1-2):

        Minimize: Σ [d_ij - (m * g_ij + b)]   (total slack above the line)

        Subject to:
            m * g_ij + b <= d_ij  for all j   (line below all points)
            m >= m_baseline                    (slope >= 2/3 speed of light)
            b >= 0                             (non-negative intercept)

    Since Σ d_ij is constant, minimizing total slack is equivalent to:
        Maximize: Σ (m * g_ij + b) = m * Σg_ij + n * b
        Or: Minimize: -m * Σg_ij - n * b

    Data Filtering (Four-Stage):
        Stage 1: Remove invalid values (zero, negative, inf)
        Stage 2: For each distance bin, keep RTTs within [mean - n_std*σ, mean + n_std*σ]
        Stage 3: Remove RTTs below speed-of-light baseline
        Stage 4: Remove entire bins where min RTT is outside global mean ± global_n_std*σ

    Args:
        distances: Array of geographic distances in km (x-axis)
        rtts: Array of RTT/delay values in ms (y-axis)
        baseline_slope: Minimum allowed slope (default: 2/3 speed of light ≈ 0.01 ms/km)
        filter_outliers: If True, apply four-stage filtering. Default True.
        bin_size_km: Size of distance bins for outlier filtering (default 100km)
        n_std: Number of standard deviations for per-bin filtering (default 1.0)
        global_n_std: Number of standard deviations for global bin-min filter (default 2.0)

    Returns:
        Dictionary containing:
        - slope: float (ms/km)
        - intercept: float (ms)
        - n_points: int
        - n_filtered: int (total number of points filtered)
        - filter_stats: dict (detailed filtering statistics)
        - success: bool
        - message: str
    """
    if not SCIPY_AVAILABLE:
        return {
            'slope': None,
            'intercept': None,
            'n_points': 0,
            'n_filtered': 0,
            'filter_stats': {},
            'success': False,
            'message': 'scipy not available, cannot use LP method'
        }

    distances = np.asarray(distances, dtype=float)
    rtts = np.asarray(rtts, dtype=float)

    # Validate inputs
    if len(distances) != len(rtts):
        return {
            'slope': None,
            'intercept': None,
            'n_points': 0,
            'n_filtered': 0,
            'filter_stats': {},
            'success': False,
            'message': 'Distance and RTT arrays must have same length'
        }

    # Filter out invalid values
    valid_mask = (distances > 0) & (rtts > 0) & np.isfinite(distances) & np.isfinite(rtts)
    distances = distances[valid_mask]
    rtts = rtts[valid_mask]
    n_points = len(distances)

    if n_points < 3:
        return {
            'slope': None,
            'intercept': None,
            'n_points': n_points,
            'n_filtered': 0,
            'filter_stats': {},
            'success': False,
            'message': f'Need at least 3 valid points, got {n_points}'
        }

    # Default baseline slope: theoretical minimum at 2/3 speed of light
    # RTT = 2 * distance / (2/3 * c) = distance * (2 / 200) = distance * 0.01 ms/km
    if baseline_slope is None:
        baseline_slope = THEORETICAL_SLOPE  # ~0.01 ms/km

    # Apply comprehensive data filtering if enabled
    filter_stats = {'initial_count': n_points, 'removed_invalid': 0,
                    'removed_low_outliers': 0, 'removed_high_outliers': 0,
                    'removed_below_baseline': 0, 'removed_global_bin_outliers': 0,
                    'final_count': n_points}
    n_filtered = 0

    if filter_outliers:
        distances, rtts, filter_stats = filter_rtt_data(
            distances, rtts,
            baseline_slope=baseline_slope,
            bin_size_km=bin_size_km,
            n_std=n_std,
            global_n_std=global_n_std
        )
        n_points = len(distances)
        n_filtered = filter_stats['initial_count'] - filter_stats['final_count']

        if n_points < 3:
            return {
                'slope': None,
                'intercept': None,
                'n_points': n_points,
                'n_filtered': n_filtered,
                'filter_stats': filter_stats,
                'success': False,
                'message': f'Only {n_points} valid points after filtering {n_filtered} (low: {filter_stats["removed_low_outliers"]}, high: {filter_stats["removed_high_outliers"]}, baseline: {filter_stats["removed_below_baseline"]}, global_bin: {filter_stats["removed_global_bin_outliers"]})'
            }

    # =========================================================================
    # LP Formulation (following CBG paper exactly)
    # =========================================================================
    # Variables: x = [m, b] where m = slope (ms/km), b = intercept (ms)
    #
    # Objective: Minimize -Σg_j * m - n * b  (push line up toward data)
    #
    # Constraints:
    #   (1) m * g_j + b <= d_j  for all j   (line must be below all points)
    #   (2) m >= baseline_slope              (physical speed limit)
    #   (3) b >= 0                           (non-negative intercept)
    # =========================================================================

    # Objective: minimize c @ x where x = [m, b]
    # We want to maximize m*Σg + n*b, so minimize -Σg*m - n*b
    c = np.array([-np.sum(distances), -float(n_points)])

    # Inequality constraints: A_ub @ x <= b_ub
    # Constraint (1): m * g_j + b <= d_j  =>  g_j * m + 1 * b <= d_j
    A_ub = np.zeros((n_points, 2))
    A_ub[:, 0] = distances   # coefficient for m (the g_j values)
    A_ub[:, 1] = 1.0         # coefficient for b
    b_ub = rtts              # RHS: d_j (the RTT values)

    # Bounds:
    # - m >= baseline_slope (no upper bound - let LP find optimal)
    # - b >= 0 (non-negative intercept per paper)
    bounds = [(baseline_slope, None), (0.0, None)]

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if result.success:
            slope = result.x[0]
            intercept = result.x[1]

            # Verify constraint satisfaction (all points on or above line)
            predicted = slope * distances + intercept
            violations = np.sum(rtts < predicted - 0.001)  # small tolerance for numerics

            msg = f'LP converged: {n_points} points'
            if n_filtered > 0:
                msg += f', {n_filtered} filtered'
            if violations > 0:
                msg += f', {violations} violations'

            return {
                'slope': float(slope),
                'intercept': float(intercept),
                'n_points': n_points,
                'n_filtered': n_filtered,
                'filter_stats': filter_stats,
                'success': True,
                'violations': int(violations),
                'message': msg
            }
        else:
            return {
                'slope': None,
                'intercept': None,
                'n_points': n_points,
                'n_filtered': n_filtered,
                'filter_stats': filter_stats,
                'success': False,
                'message': f'LP did not converge: {result.message}'
            }

    except Exception as e:
        return {
            'slope': None,
            'intercept': None,
            'n_points': n_points,
            'n_filtered': n_filtered,
            'filter_stats': filter_stats,
            'success': False,
            'message': f'LP error: {str(e)}'
        }


def fit_bestline(
    distances: np.ndarray,
    rtts: np.ndarray,
    bin_size_km: float = 50.0,
    percentile: float = 0.05,
    min_points_per_bin: int = 1
) -> Dict[str, Any]:
    """
    Fit lower envelope bestline using binned percentile approach.

    This implements the CBG calibration method:
    1. Bin data by distance (default 50km bins)
    2. Take specified percentile RTT in each bin (default 5th)
    3. Fit linear regression through bin centers and percentile RTTs
    4. Require ≥3 bins for valid fit

    Args:
        distances: Array of distances in km
        rtts: Array of RTT values in ms
        bin_size_km: Size of distance bins in km
        percentile: Percentile to use for lower envelope (0.05 = 5th percentile)
        min_points_per_bin: Minimum points required in a bin to include it

    Returns:
        Dictionary containing:
        - slope: float (ms/km)
        - intercept: float (ms, baseline latency)
        - n_bins: int (number of bins with data)
        - bin_centers: list (distance bin centers used)
        - bin_rtts: list (percentile RTT per bin)
        - success: bool (False if < 3 bins)
        - r_squared: float (fit quality, None if failed)
        - message: str (status message)
    """
    distances = np.asarray(distances)
    rtts = np.asarray(rtts)

    # Validate inputs
    if len(distances) != len(rtts):
        return {
            'slope': None,
            'intercept': None,
            'n_bins': 0,
            'bin_centers': [],
            'bin_rtts': [],
            'success': False,
            'r_squared': None,
            'message': 'Distance and RTT arrays must have same length'
        }

    if len(distances) == 0:
        return {
            'slope': None,
            'intercept': None,
            'n_bins': 0,
            'bin_centers': [],
            'bin_rtts': [],
            'success': False,
            'r_squared': None,
            'message': 'Empty input arrays'
        }

    # Filter out invalid values
    valid_mask = (distances > 0) & (rtts > 0) & np.isfinite(distances) & np.isfinite(rtts)
    distances = distances[valid_mask]
    rtts = rtts[valid_mask]

    if len(distances) == 0:
        return {
            'slope': None,
            'intercept': None,
            'n_bins': 0,
            'bin_centers': [],
            'bin_rtts': [],
            'success': False,
            'r_squared': None,
            'message': 'No valid data points after filtering'
        }

    # Create distance bins
    min_dist = distances.min()
    max_dist = distances.max()

    # Align bins to start from 0 for consistency
    bin_start = (min_dist // bin_size_km) * bin_size_km
    bin_end = ((max_dist // bin_size_km) + 1) * bin_size_km
    bin_edges = np.arange(bin_start, bin_end + bin_size_km, bin_size_km)

    # Compute percentile RTT for each bin
    bin_centers = []
    bin_rtts = []

    for i in range(len(bin_edges) - 1):
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
        bin_data = rtts[mask]

        if len(bin_data) >= min_points_per_bin:
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            bin_percentile_rtt = np.percentile(bin_data, percentile * 100)
            bin_centers.append(bin_center)
            bin_rtts.append(bin_percentile_rtt)

    n_bins = len(bin_centers)

    if n_bins < 3:
        return {
            'slope': None,
            'intercept': None,
            'n_bins': n_bins,
            'bin_centers': bin_centers,
            'bin_rtts': bin_rtts,
            'success': False,
            'r_squared': None,
            'message': f'Only {n_bins} bins with data, need at least 3'
        }

    # Fit linear regression: RTT = slope * distance + intercept
    bin_centers_arr = np.array(bin_centers)
    bin_rtts_arr = np.array(bin_rtts)

    # Use numpy polyfit (degree 1 = linear)
    coefficients = np.polyfit(bin_centers_arr, bin_rtts_arr, 1)
    slope = coefficients[0]  # ms/km
    intercept = coefficients[1]  # ms

    # Calculate R-squared
    predicted = slope * bin_centers_arr + intercept
    ss_res = np.sum((bin_rtts_arr - predicted) ** 2)
    ss_tot = np.sum((bin_rtts_arr - np.mean(bin_rtts_arr)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'n_bins': n_bins,
        'bin_centers': bin_centers,
        'bin_rtts': bin_rtts,
        'success': True,
        'r_squared': float(r_squared),
        'message': f'Successfully fitted bestline with {n_bins} bins'
    }


def rtt_to_distance(rtt: float, slope: float, intercept: float) -> float:
    """
    Convert RTT to maximum possible distance using calibrated bestline.

    The bestline represents the lower envelope: RTT = slope * distance + intercept
    Inverting: distance = (RTT - intercept) / slope

    This gives the MAXIMUM distance the target could be at, given the observed RTT.

    Args:
        rtt: Round-trip time in ms
        slope: Calibrated slope in ms/km
        intercept: Calibrated intercept in ms

    Returns:
        Maximum distance in km (0 if RTT < intercept or slope <= 0)
    """
    if slope <= 0:
        return 0.0

    distance = (rtt - intercept) / slope
    return max(0.0, distance)


def rtt_to_distance_fixed(rtt: float, speed_fraction: float = 2/3) -> float:
    """
    Convert RTT to distance using fixed speed threshold (for comparison).

    This is the method used in the million-scale paper:
    distance = (speed_fraction * c) * (RTT / 2)

    Args:
        rtt: Round-trip time in ms
        speed_fraction: Fraction of speed of light (default 2/3)

    Returns:
        Maximum distance in km
    """
    # RTT is round-trip, so divide by 2 for one-way time
    # Speed in km/ms = 300 * speed_fraction
    speed_km_ms = SPEED_OF_LIGHT_KM_MS * speed_fraction
    return speed_km_ms * (rtt / 2)


@dataclass
class RTTDistanceModel:
    """
    Per-anchor RTT-distance calibration model.

    Stores the calibrated parameters for converting RTT to distance
    for a specific anchor (vantage point).

    Supports two fitting methods:
    - 'lp': Linear Programming (original CBG paper method) - true lower bound
    - 'percentile': Binned percentile approach (approximation)
    """
    anchor_ip: str
    anchor_lat: float
    anchor_lon: float
    slope: Optional[float] = None
    intercept: Optional[float] = None
    r_squared: Optional[float] = None
    n_bins: int = 0
    n_measurements: int = 0
    bin_size_km: float = 50.0
    percentile: float = 0.05
    bin_centers: List[float] = field(default_factory=list)
    bin_rtts: List[float] = field(default_factory=list)
    fitted: bool = False
    fit_message: str = ''
    fit_method: str = 'lp'  # 'lp' or 'percentile'

    def fit(
        self,
        distances: np.ndarray,
        rtts: np.ndarray,
        method: str = 'lp',
        bin_size_km: Optional[float] = None,
        percentile: Optional[float] = None,
        baseline_slope: Optional[float] = None,
        filter_outliers: bool = True,
        n_std: float = 1.0,
        global_n_std: float = 1.0
    ) -> bool:
        """
        Fit the model to RTT-distance data.

        Args:
            distances: Array of distances in km
            rtts: Array of RTT values in ms
            method: 'lp' (Linear Programming) or 'percentile' (binned percentile)
            bin_size_km: Override default bin size (default 100km for LP, 50km for percentile)
            percentile: Override default percentile (percentile method only)
            baseline_slope: Minimum slope constraint (LP method only)
            filter_outliers: Filter outliers by four-stage filtering (LP method)
            n_std: Number of standard deviations for per-bin filtering (default 1.0)
            global_n_std: Number of standard deviations for global bin-min filter (default 2.0)

        Returns:
            True if fitting succeeded, False otherwise
        """
        if bin_size_km is not None:
            self.bin_size_km = bin_size_km
        elif method == 'lp':
            self.bin_size_km = 100.0  # Default for LP method
        if percentile is not None:
            self.percentile = percentile

        self.n_measurements = len(distances)
        self.fit_method = method

        if method == 'lp':
            result = fit_bestline_lp(
                distances=distances,
                rtts=rtts,
                baseline_slope=baseline_slope,
                filter_outliers=filter_outliers,
                bin_size_km=self.bin_size_km,
                n_std=n_std,
                global_n_std=global_n_std
            )
            # LP doesn't produce bins, but we can compute them for visualization
            self.slope = result['slope']
            self.intercept = result['intercept']
            self.r_squared = None  # LP doesn't have R² in traditional sense
            self.n_bins = 0
            self.bin_centers = []
            self.bin_rtts = []
            self.fitted = result['success']
            self.fit_message = result['message']

            # Optionally compute bins for visualization
            if self.fitted:
                bin_result = fit_bestline(
                    distances=distances,
                    rtts=rtts,
                    bin_size_km=self.bin_size_km,
                    percentile=self.percentile
                )
                self.n_bins = bin_result['n_bins']
                self.bin_centers = bin_result['bin_centers']
                self.bin_rtts = bin_result['bin_rtts']
                # Note: R² from binned fit, not LP
                self.r_squared = bin_result.get('r_squared')

        else:  # percentile method
            result = fit_bestline(
                distances=distances,
                rtts=rtts,
                bin_size_km=self.bin_size_km,
                percentile=self.percentile
            )
            self.slope = result['slope']
            self.intercept = result['intercept']
            self.r_squared = result['r_squared']
            self.n_bins = result['n_bins']
            self.bin_centers = result['bin_centers']
            self.bin_rtts = result['bin_rtts']
            self.fitted = result['success']
            self.fit_message = result['message']

        return self.fitted

    def predict_distance(self, rtt: float) -> Optional[float]:
        """
        Predict maximum distance from RTT.

        Args:
            rtt: Round-trip time in ms

        Returns:
            Maximum distance in km, or None if model not fitted
        """
        if not self.fitted:
            return None
        return rtt_to_distance(rtt, self.slope, self.intercept)

    def predict_rtt(self, distance: float) -> Optional[float]:
        """
        Predict expected minimum RTT for a given distance.

        Args:
            distance: Distance in km

        Returns:
            Expected minimum RTT in ms, or None if model not fitted
        """
        if not self.fitted:
            return None
        return self.slope * distance + self.intercept

    def save(self, filepath: Path) -> None:
        """Save model to pickle file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> 'RTTDistanceModel':
        """Load model from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for JSON serialization."""
        return {
            'anchor_ip': self.anchor_ip,
            'anchor_lat': self.anchor_lat,
            'anchor_lon': self.anchor_lon,
            'slope': self.slope,
            'intercept': self.intercept,
            'r_squared': self.r_squared,
            'n_bins': self.n_bins,
            'n_measurements': self.n_measurements,
            'bin_size_km': self.bin_size_km,
            'percentile': self.percentile,
            'bin_centers': self.bin_centers,
            'bin_rtts': self.bin_rtts,
            'fitted': self.fitted,
            'fit_message': self.fit_message,
            'fit_method': self.fit_method
        }

    def __repr__(self) -> str:
        if self.fitted:
            return (
                f"RTTDistanceModel(anchor={self.anchor_ip}, "
                f"slope={self.slope:.6f} ms/km, "
                f"intercept={self.intercept:.2f} ms, "
                f"R²={self.r_squared:.4f}, "
                f"n_bins={self.n_bins})"
            )
        else:
            return f"RTTDistanceModel(anchor={self.anchor_ip}, fitted=False, msg='{self.fit_message}')"
