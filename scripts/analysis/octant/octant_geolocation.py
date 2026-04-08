"""
Octant Geolocation Pipeline

Implements the core Octant geolocation algorithm (Wong et al., NSDI 2007):
1. Constraint formation: RTT measurements -> annular constraints (r_L, R_L)
2. Region computation: intersect positive constraints, subtract negative constraints
3. Point selection: Monte Carlo geometric median approximation

Builds on OctantRTTModel (octant_model.py) which provides the calibration layer.

Coordinate convention:
- Internal: (lat, lon) everywhere
- Shapely: (lon, lat) = (x, y) at the boundary
"""

import logging
import numpy as np
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Optional, Tuple, Any

from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union

from scripts.analysis.octant.octant_model import OctantRTTModel
from scripts.utils.helpers import haversine

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
SHAPELY_RADIUS_EPSILON_KM = 1e-3


# =============================================================================
# Layer 1: Constraint Formation
# =============================================================================

@dataclass
class AnnularConstraint:
    """A single landmark's constraint on the target location.

    Represents an annulus: the target is between inner_radius_km (r_L)
    and outer_radius_km (R_L) from the landmark.
    """
    landmark_lat: float
    landmark_lon: float
    landmark_ip: str
    rtt_ms: float
    inner_radius_km: float   # r_L(rtt) - lower bound on distance
    outer_radius_km: float   # R_L(rtt) - upper bound on distance
    weight: float             # exp(-rtt / tau)


def form_constraint(
    landmark_lat: float,
    landmark_lon: float,
    landmark_ip: str,
    rtt_ms: float,
    model: OctantRTTModel,
    weight_tau_ms: float = 50.0,
    delta: Optional[float] = None,
) -> AnnularConstraint:
    """Form a single annular constraint from one landmark's RTT measurement.

    Args:
        landmark_lat, landmark_lon: Landmark coordinates
        landmark_ip: Landmark IP identifier
        rtt_ms: Measured RTT to target in ms
        model: Fitted OctantRTTModel for this landmark
        weight_tau_ms: Decay constant for exponential weighting
        delta: If provided, use spline delta band instead of hull bounds

    Returns:
        AnnularConstraint with (r_L, R_L) bounds and weight
    """
    inner_km, outer_km = model.predict_distance_bounds(rtt_ms, delta=delta)
    weight = np.exp(-rtt_ms / weight_tau_ms)
    return AnnularConstraint(
        landmark_lat=landmark_lat,
        landmark_lon=landmark_lon,
        landmark_ip=landmark_ip,
        rtt_ms=rtt_ms,
        inner_radius_km=max(0.0, inner_km),
        outer_radius_km=max(0.0, outer_km),
        weight=weight,
    )


def form_constraints(
    target_ip: str,
    rtt_measurements: Dict[str, float],
    landmark_coords: Dict[str, Tuple[float, float]],
    models: Dict[str, OctantRTTModel],
    weight_tau_ms: float = 50.0,
    delta: Optional[float] = None,
    max_rtt_ms: float = 200.0,
) -> List[AnnularConstraint]:
    """Form all annular constraints for a target from its RTT measurements.

    Args:
        target_ip: Target IP (for logging only)
        rtt_measurements: {landmark_ip: min_rtt_ms}
        landmark_coords: {landmark_ip: (lat, lon)}
        models: {landmark_ip: fitted OctantRTTModel}
        weight_tau_ms: Decay constant for exponential weighting
        delta: If provided, use spline delta band instead of hull bounds
        max_rtt_ms: Ignore RTTs above this threshold

    Returns:
        List of AnnularConstraint sorted by weight descending
    """
    constraints = []
    for lm_ip, rtt in rtt_measurements.items():
        if rtt > max_rtt_ms:
            continue
        if lm_ip not in models or not models[lm_ip].fitted:
            continue
        if lm_ip not in landmark_coords:
            continue

        lat, lon = landmark_coords[lm_ip]
        try:
            c = form_constraint(
                lat, lon, lm_ip, rtt, models[lm_ip],
                weight_tau_ms=weight_tau_ms,
                delta=delta,
            )
            # Skip degenerate constraints
            if c.outer_radius_km > c.inner_radius_km:
                constraints.append(c)
            else:
                logger.debug(
                    "Skipping degenerate constraint for %s from %s: r_L=%.1f >= R_L=%.1f",
                    target_ip, lm_ip, c.inner_radius_km, c.outer_radius_km
                )
        except Exception as e:
            logger.debug("Failed to form constraint for %s from %s: %s", target_ip, lm_ip, e)

    constraints.sort(key=lambda c: c.weight, reverse=True)
    return constraints


# =============================================================================
# Layer 2: Region Computation (Shapely-based)
# =============================================================================

def _circle_to_shapely(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    n_pts: int = 100,
) -> Polygon:
    """Convert a geographic circle to a Shapely polygon.

    Uses degree-based approximation accounting for latitude.
    Shapely coordinates are (lon, lat) = (x, y).
    """
    # Shapely polygons need nonzero radius to avoid collapsing exact matches
    # into degenerate geometries during region intersection.
    radius_km = max(radius_km, SHAPELY_RADIUS_EPSILON_KM)

    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(center_lat))
    km_per_deg_lon = max(km_per_deg_lon, 1.0)  # avoid division by zero near poles

    r_lat = radius_km / km_per_deg_lat
    r_lon = radius_km / km_per_deg_lon

    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    lons = center_lon + r_lon * np.cos(angles)
    lats = center_lat + r_lat * np.sin(angles)

    coords = list(zip(lons, lats))  # Shapely uses (x, y) = (lon, lat)
    return Polygon(coords)


def _annulus_to_shapely(
    center_lat: float,
    center_lon: float,
    inner_radius_km: float,
    outer_radius_km: float,
    n_pts: int = 100,
) -> Optional[Polygon]:
    """Convert an annular constraint to a Shapely polygon (ring geometry).

    Returns outer disk minus inner disk, or None if degenerate.
    """
    if inner_radius_km >= outer_radius_km:
        return None

    outer = _circle_to_shapely(center_lat, center_lon, outer_radius_km, n_pts)
    if inner_radius_km <= 0:
        return outer

    inner = _circle_to_shapely(center_lat, center_lon, inner_radius_km, n_pts)
    result = outer.difference(inner)

    if result.is_empty:
        return None
    return result


def _haversine_vectorized(
    lats: np.ndarray,
    lons: np.ndarray,
    ref_lat: float,
    ref_lon: float,
) -> np.ndarray:
    """Compute haversine distances from arrays of (lat, lon) to a reference point.

    Returns distances in km.
    """
    lat1 = np.radians(lats)
    lon1 = np.radians(lons)
    lat2 = np.radians(ref_lat)
    lon2 = np.radians(ref_lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def _point_in_annulus(
    point_lat: float,
    point_lon: float,
    constraint: AnnularConstraint,
) -> bool:
    """Check if a point satisfies an annular constraint."""
    d = haversine(
        (point_lat, point_lon),
        (constraint.landmark_lat, constraint.landmark_lon),
    )
    return constraint.inner_radius_km <= d <= constraint.outer_radius_km


def _points_in_annulus_vectorized(
    lats: np.ndarray,
    lons: np.ndarray,
    constraint: AnnularConstraint,
) -> np.ndarray:
    """Vectorized point-in-annulus check. Returns boolean mask."""
    distances = _haversine_vectorized(
        lats, lons, constraint.landmark_lat, constraint.landmark_lon
    )
    return (distances >= constraint.inner_radius_km) & (distances <= constraint.outer_radius_km)


def compute_feasible_region_unweighted(
    constraints: List[AnnularConstraint],
    n_pts: int = 100,
) -> Optional[Any]:
    """Compute feasible region: intersect all outer disks, subtract all inner disks.

    Result = intersection(all outer disks) - union(all inner disks)

    Args:
        constraints: List of annular constraints
        n_pts: Points per circle polygon

    Returns:
        Shapely geometry (Polygon/MultiPolygon) or None if empty
    """
    if not constraints:
        return None

    # Intersect all outer disks (positive constraints)
    outer_disks = [
        _circle_to_shapely(c.landmark_lat, c.landmark_lon, c.outer_radius_km, n_pts)
        for c in constraints
    ]
    positive_region = reduce(lambda a, b: a.intersection(b), outer_disks)

    if positive_region.is_empty:
        return None

    # Subtract all inner disks (negative constraints)
    inner_disks = [
        _circle_to_shapely(c.landmark_lat, c.landmark_lon, c.inner_radius_km, n_pts)
        for c in constraints
        if c.inner_radius_km > 0
    ]

    if inner_disks:
        negative_region = unary_union(inner_disks)
        result = positive_region.difference(negative_region)
    else:
        result = positive_region

    if result.is_empty:
        return None
    return result


def compute_feasible_region_weighted(
    constraints: List[AnnularConstraint],
    weight_threshold: float = 0.5,
    n_pts: int = 100,
    grid_resolution_deg: float = 0.1,
) -> Optional[Any]:
    """Compute weighted feasible region using grid-based approach.

    For each grid point, accumulates weights from constraints whose annulus
    contains the point. Points with weight >= threshold * max_possible_weight
    form the feasible region.

    Args:
        constraints: List of annular constraints
        weight_threshold: Fraction of max possible weight required
        n_pts: Points per circle polygon
        grid_resolution_deg: Grid spacing in degrees

    Returns:
        Shapely geometry or None if empty
    """
    if not constraints:
        return None

    # Compute bounding box from outer circles
    min_lat, max_lat = 90.0, -90.0
    min_lon, max_lon = 180.0, -180.0
    for c in constraints:
        km_per_deg_lat = 111.0
        km_per_deg_lon = max(111.0 * np.cos(np.radians(c.landmark_lat)), 1.0)
        r_lat = c.outer_radius_km / km_per_deg_lat
        r_lon = c.outer_radius_km / km_per_deg_lon
        min_lat = min(min_lat, c.landmark_lat - r_lat)
        max_lat = max(max_lat, c.landmark_lat + r_lat)
        min_lon = min(min_lon, c.landmark_lon - r_lon)
        max_lon = max(max_lon, c.landmark_lon + r_lon)

    # Intersect bounding boxes for tighter bounds
    # Use the intersection of outer disks' bounding boxes
    for c in constraints:
        km_per_deg_lat = 111.0
        km_per_deg_lon = max(111.0 * np.cos(np.radians(c.landmark_lat)), 1.0)
        r_lat = c.outer_radius_km / km_per_deg_lat
        r_lon = c.outer_radius_km / km_per_deg_lon
        min_lat = max(min_lat, c.landmark_lat - r_lat)
        max_lat = min(max_lat, c.landmark_lat + r_lat)
        min_lon = max(min_lon, c.landmark_lon - r_lon)
        max_lon = min(max_lon, c.landmark_lon + r_lon)

    if min_lat >= max_lat or min_lon >= max_lon:
        return None

    # Create grid
    lat_grid = np.arange(min_lat, max_lat, grid_resolution_deg)
    lon_grid = np.arange(min_lon, max_lon, grid_resolution_deg)

    if len(lat_grid) == 0 or len(lon_grid) == 0:
        return None

    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    flat_lats = lat_mesh.ravel()
    flat_lons = lon_mesh.ravel()

    # Accumulate weights
    max_weight = sum(c.weight for c in constraints)
    if max_weight <= 0:
        return None

    weight_accum = np.zeros(len(flat_lats))
    for c in constraints:
        mask = _points_in_annulus_vectorized(flat_lats, flat_lons, c)
        weight_accum[mask] += c.weight

    # Select points above threshold
    threshold = weight_threshold * max_weight
    qualifying = weight_accum >= threshold

    if not np.any(qualifying):
        return None

    # Convert qualifying points to polygon via buffering and union
    half_cell = grid_resolution_deg / 2.0
    boxes = []
    q_lons = flat_lons[qualifying]
    q_lats = flat_lats[qualifying]

    for lon, lat in zip(q_lons, q_lats):
        box = Polygon([
            (lon - half_cell, lat - half_cell),
            (lon + half_cell, lat - half_cell),
            (lon + half_cell, lat + half_cell),
            (lon - half_cell, lat + half_cell),
        ])
        boxes.append(box)

    region = unary_union(boxes)
    if region.is_empty:
        return None
    return region


# =============================================================================
# Layer 3: Monte Carlo Point Selection
# =============================================================================

def sample_points_in_region(
    region,
    n_samples: int = 5000,
    rng: Optional[np.random.Generator] = None,
    max_attempts_factor: int = 20,
) -> np.ndarray:
    """Sample uniformly random points within a Shapely geometry.

    Uses rejection sampling within the bounding box.

    Args:
        region: Shapely geometry (Polygon or MultiPolygon)
        n_samples: Number of points to sample
        rng: Numpy random generator (for reproducibility)
        max_attempts_factor: Maximum attempts = n_samples * this factor

    Returns:
        Array of shape (n_collected, 2) with columns [lat, lon].
        May return fewer than n_samples if region is very small.
    """
    if rng is None:
        rng = np.random.default_rng()

    minx, miny, maxx, maxy = region.bounds  # (min_lon, min_lat, max_lon, max_lat)
    collected = []
    max_attempts = n_samples * max_attempts_factor
    attempts = 0

    while len(collected) < n_samples and attempts < max_attempts:
        batch_size = min(n_samples * 4, max_attempts - attempts)
        rand_lons = rng.uniform(minx, maxx, batch_size)
        rand_lats = rng.uniform(miny, maxy, batch_size)
        attempts += batch_size

        for lon, lat in zip(rand_lons, rand_lats):
            if region.contains(Point(lon, lat)):
                collected.append((lat, lon))  # Internal convention: (lat, lon)
                if len(collected) >= n_samples:
                    break

    return np.array(collected) if collected else np.empty((0, 2))


def geometric_median_approx(points: np.ndarray) -> Tuple[float, float]:
    """Approximate geometric median via minimum sum-of-distances.

    For each point, computes sum of haversine distances to all other points.
    Returns the point with the minimum sum.

    Args:
        points: Array of shape (n, 2) with columns [lat, lon]

    Returns:
        (lat, lon) of the approximate geometric median
    """
    n = len(points)
    if n == 0:
        raise ValueError("Cannot compute geometric median of empty point set")
    if n == 1:
        return (points[0, 0], points[0, 1])

    lats = points[:, 0]
    lons = points[:, 1]

    # Vectorized pairwise distance computation
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)

    # Compute all pairwise distances using broadcasting
    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]

    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) *
         np.sin(dlon / 2.0) ** 2)
    dist_matrix = EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    # Sum of distances for each point
    sum_distances = dist_matrix.sum(axis=1)
    best_idx = np.argmin(sum_distances)

    return (lats[best_idx], lons[best_idx])


# =============================================================================
# Layer 4: Orchestration
# =============================================================================

def _region_area_km2(region) -> float:
    """Approximate area of a Shapely region in km^2.

    Uses degree-based approximation at the centroid latitude.
    """
    if region is None or region.is_empty:
        return 0.0
    centroid = region.centroid
    center_lat = centroid.y  # Shapely: y = lat
    km_per_deg_lat = 111.0
    km_per_deg_lon = max(111.0 * np.cos(np.radians(center_lat)), 1.0)
    return region.area * km_per_deg_lat * km_per_deg_lon


def _weighted_centroid_fallback(
    constraints: List[AnnularConstraint],
) -> Tuple[float, float]:
    """Fallback: inverse-RTT weighted centroid of landmarks."""
    if not constraints:
        raise ValueError("No constraints for fallback")

    total_weight = sum(c.weight for c in constraints)
    if total_weight <= 0:
        # Equal weighting
        lat = np.mean([c.landmark_lat for c in constraints])
        lon = np.mean([c.landmark_lon for c in constraints])
        return (lat, lon)

    lat = sum(c.landmark_lat * c.weight for c in constraints) / total_weight
    lon = sum(c.landmark_lon * c.weight for c in constraints) / total_weight
    return (lat, lon)


def estimate_location(
    constraints: List[AnnularConstraint],
    method: str = 'weighted',
    n_samples: int = 5000,
    weight_threshold: float = 0.5,
    grid_resolution_deg: float = 0.1,
    n_pts: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Dict[str, Any]]:
    """Estimate target location from annular constraints.

    Computes feasible region, then selects a representative point
    via Monte Carlo geometric median.

    Fallback chain:
    1. weighted region at threshold
    2. weighted region at threshold/2
    3. unweighted region
    4. inverse-RTT weighted centroid of landmarks

    Args:
        constraints: List of AnnularConstraint
        method: 'weighted', 'unweighted', or 'centroid'
        n_samples: Monte Carlo samples
        weight_threshold: For weighted region
        grid_resolution_deg: For weighted region grid
        n_pts: Points per circle polygon
        rng: Random generator for reproducibility

    Returns:
        Dict with lat, lon, region_area_km2, n_constraints, method, fallback
        or None if no constraints
    """
    if not constraints:
        return None

    fallback = False
    actual_method = method
    region = None

    if method == 'centroid':
        lat, lon = _weighted_centroid_fallback(constraints)
        return {
            'lat': lat,
            'lon': lon,
            'region_area_km2': 0.0,
            'n_constraints': len(constraints),
            'method': 'centroid',
            'fallback': False,
        }

    # Try primary method
    if method == 'weighted':
        region = compute_feasible_region_weighted(
            constraints, weight_threshold=weight_threshold,
            n_pts=n_pts, grid_resolution_deg=grid_resolution_deg,
        )
        # Fallback: lower threshold
        if region is None:
            region = compute_feasible_region_weighted(
                constraints, weight_threshold=weight_threshold / 2.0,
                n_pts=n_pts, grid_resolution_deg=grid_resolution_deg,
            )
            if region is not None:
                fallback = True
                actual_method = 'weighted_low_threshold'

    if method == 'unweighted' or region is None:
        region = compute_feasible_region_unweighted(constraints, n_pts=n_pts)
        if region is not None and actual_method != 'unweighted':
            fallback = True
            actual_method = 'unweighted'

    # Final fallback: weighted centroid
    if region is None:
        lat, lon = _weighted_centroid_fallback(constraints)
        return {
            'lat': lat,
            'lon': lon,
            'region_area_km2': 0.0,
            'n_constraints': len(constraints),
            'method': 'centroid_fallback',
            'fallback': True,
        }

    # Sample points and find geometric median
    area_km2 = _region_area_km2(region)
    points = sample_points_in_region(region, n_samples=n_samples, rng=rng)

    if len(points) < 2:
        # Region too small for sampling; use Shapely centroid
        centroid = region.centroid
        lat, lon = centroid.y, centroid.x
    else:
        lat, lon = geometric_median_approx(points)

    return {
        'lat': lat,
        'lon': lon,
        'region_area_km2': area_km2,
        'n_constraints': len(constraints),
        'method': actual_method,
        'fallback': fallback,
        'n_samples': len(points),
    }


class OctantGeolocator:
    """Full Octant geolocation pipeline.

    Holds pre-fitted OctantRTTModel instances for all landmarks
    and provides a high-level API for geolocating targets.
    """

    def __init__(
        self,
        models: Dict[str, OctantRTTModel],
        landmark_coords: Dict[str, Tuple[float, float]],
        weight_tau_ms: float = 50.0,
        method: str = 'weighted',
        n_samples: int = 5000,
        weight_threshold: float = 0.5,
        max_rtt_ms: float = 200.0,
        delta: Optional[float] = None,
        grid_resolution_deg: float = 0.1,
    ):
        self.models = models
        self.landmark_coords = landmark_coords
        self.weight_tau_ms = weight_tau_ms
        self.method = method
        self.n_samples = n_samples
        self.weight_threshold = weight_threshold
        self.max_rtt_ms = max_rtt_ms
        self.delta = delta
        self.grid_resolution_deg = grid_resolution_deg

    def geolocate(
        self,
        target_ip: str,
        rtt_measurements: Dict[str, float],
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[Dict[str, Any]]:
        """Geolocate a single target.

        Args:
            target_ip: Target IP identifier
            rtt_measurements: {landmark_ip: min_rtt_ms}
            rng: Random generator for reproducibility

        Returns:
            Dict with lat, lon, region_area_km2, n_constraints, method, fallback
        """
        constraints = form_constraints(
            target_ip, rtt_measurements, self.landmark_coords, self.models,
            weight_tau_ms=self.weight_tau_ms,
            delta=self.delta,
            max_rtt_ms=self.max_rtt_ms,
        )
        return estimate_location(
            constraints,
            method=self.method,
            n_samples=self.n_samples,
            weight_threshold=self.weight_threshold,
            grid_resolution_deg=self.grid_resolution_deg,
            rng=rng,
        )

    def geolocate_batch(
        self,
        targets: Dict[str, Dict[str, float]],
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Geolocate multiple targets.

        Args:
            targets: {target_ip: {landmark_ip: min_rtt_ms}}
            rng: Random generator for reproducibility

        Returns:
            {target_ip: result_dict_or_None}
        """
        results = {}
        for target_ip, rtts in targets.items():
            results[target_ip] = self.geolocate(target_ip, rtts, rng=rng)
        return results

    def evaluate(
        self,
        targets: Dict[str, Dict[str, float]],
        ground_truth: Dict[str, Tuple[float, float]],
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """Evaluate geolocation accuracy against ground truth.

        Args:
            targets: {target_ip: {landmark_ip: min_rtt_ms}}
            ground_truth: {target_ip: (lat, lon)}
            rng: Random generator for reproducibility

        Returns:
            Dict with results, median_error_km, mean_error_km,
            accuracy_at_thresholds, error_cdf
        """
        results = []
        errors = []

        for target_ip, rtts in targets.items():
            if target_ip not in ground_truth:
                continue

            result = self.geolocate(target_ip, rtts, rng=rng)
            true_lat, true_lon = ground_truth[target_ip]

            if result is not None:
                error_km = haversine(
                    (result['lat'], result['lon']),
                    (true_lat, true_lon),
                )
                result['error_km'] = error_km
                result['true_lat'] = true_lat
                result['true_lon'] = true_lon
                result['target_ip'] = target_ip
                errors.append(error_km)
            else:
                result = {
                    'target_ip': target_ip,
                    'error_km': None,
                    'true_lat': true_lat,
                    'true_lon': true_lon,
                }

            results.append(result)

        if not errors:
            return {
                'results': results,
                'median_error_km': None,
                'mean_error_km': None,
                'accuracy_at_thresholds': {},
                'error_cdf': [],
            }

        sorted_errors = sorted(errors)
        thresholds = [40, 100, 500, 1000]
        accuracy_at = {}
        for t in thresholds:
            accuracy_at[t] = sum(1 for e in errors if e <= t) / len(errors)

        return {
            'results': results,
            'median_error_km': float(np.median(sorted_errors)),
            'mean_error_km': float(np.mean(sorted_errors)),
            'accuracy_at_thresholds': accuracy_at,
            'error_cdf': sorted_errors,
        }
