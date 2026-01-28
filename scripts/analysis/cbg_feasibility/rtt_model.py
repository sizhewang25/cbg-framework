"""
RTT-Distance Modeling Module for CBG Feasibility Analysis

This module implements the core functions for Constraint-Based Geolocation (CBG)
using calibrated RTT-distance relationships instead of fixed speed thresholds.

Key Components:
- Haversine distance calculation
- Binned 5th percentile bestline fitting
- RTTDistanceModel class for per-anchor calibration
"""

import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any


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

    def fit(
        self,
        distances: np.ndarray,
        rtts: np.ndarray,
        bin_size_km: Optional[float] = None,
        percentile: Optional[float] = None
    ) -> bool:
        """
        Fit the model to RTT-distance data.

        Args:
            distances: Array of distances in km
            rtts: Array of RTT values in ms
            bin_size_km: Override default bin size
            percentile: Override default percentile

        Returns:
            True if fitting succeeded, False otherwise
        """
        if bin_size_km is not None:
            self.bin_size_km = bin_size_km
        if percentile is not None:
            self.percentile = percentile

        self.n_measurements = len(distances)

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
            'fit_message': self.fit_message
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
