"""Per-anchor Octant RTT-distance annulus model (high-cutoff variant).

For each RTT query, returns `(inner_km, outer_km)` — an annular constraint:

    outer_km  =  upper convex hull facet at this RTT (or spline·δ, clipped)
    inner_km  =  lower convex hull facet at this RTT (or spline/δ, clipped)

A piecewise-linear LSQ spline through the in-cutoff scatter is the "best
guess" curve inside the hull band; multiplicative δ widening expands it to
a target coverage of the training points.

Cutoff semantics: bin the RTT axis by `bin_size_ms`; `cutoff_rtt` is the
right edge of the last bin with >= `cutoff_min_points` samples. Above
`cutoff_rtt`, the spline is not trusted — `predict_distance_bounds` falls
back to bare hull bounds there.

This is the simplified, production-only mirror of
`scripts/libs/octant/octant_model.py`. The legacy module keeps its
multi-variant cutoff configuration (none / high_only / low_only / both),
spline-hull refit pass, and full RTT-array prediction helpers for the
exploratory scripts that depend on them.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import make_lsq_spline


EARTH_RADIUS_KM = 6371.0
SPEED_OF_LIGHT_KM_S = 300_000.0
SPEED_OF_LIGHT_KM_MS = SPEED_OF_LIGHT_KM_S / 1000.0  # 300 km/ms
# RTT lower bound at 2/3 c: rtt = 2·d / (2/3·c) = 0.01 ms/km
THEORETICAL_SLOPE = 2 / (SPEED_OF_LIGHT_KM_MS * (2 / 3))


class OctantFitError(Exception):
    """Base exception for Octant model fitting errors."""


class SplineFitError(OctantFitError):
    """Raised when spline fitting fails."""


class DeltaSearchError(OctantFitError):
    """Raised when no δ satisfies the coverage requirement."""


class DeltaSearchTimeout(OctantFitError):
    """Raised when δ search exceeds the wall-clock timeout."""


# ----------------------------------------------------------------------------
# Convex hull
# ----------------------------------------------------------------------------


def compute_convex_hull_bounds(
    rtts: np.ndarray,
    distances: np.ndarray,
    cutoff_min_points: int = 5,
    bin_size_ms: float = 5.0,
) -> Dict[str, Any]:
    """Upper/lower convex hull chains over (rtt, distance) scatter.

    Monotone-chain on points sorted by (rtt, distance). Returns vertices in
    ascending-RTT order. `cutoff_rtt` is the right edge of the last RTT bin
    with at least `cutoff_min_points` samples — beyond it, data is too
    sparse to trust hull extrapolation.
    """
    rtts = np.asarray(rtts, dtype=float)
    distances = np.asarray(distances, dtype=float)

    valid = (rtts > 0) & (distances > 0) & np.isfinite(rtts) & np.isfinite(distances)
    rtts = rtts[valid]
    distances = distances[valid]

    if len(rtts) < 3:
        return {
            "hull_upper_rtts": [],
            "hull_upper_distances": [],
            "hull_lower_rtts": [],
            "hull_lower_distances": [],
            "cutoff_rtt": 0.0,
            "success": False,
            "message": f"need at least 3 valid points, got {len(rtts)}",
        }

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

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

    hull_upper = upper[::-1]  # sort ascending by RTT
    hull_lower = lower

    # Right edge of the last dense bin — scanning low → high so an isolated
    # high-RTT cluster doesn't inflate the cutoff. Clamped at `max_rtt`: the
    # bin scan can run past the last data point (bin_start + bin_size_ms),
    # but the trusted region can never extend beyond the data itself.
    min_rtt = float(np.min(rtts))
    max_rtt = float(np.max(rtts))
    cutoff_rtt = min_rtt
    for bin_start in np.arange(min_rtt, max_rtt, bin_size_ms):
        bin_count = np.sum((rtts >= bin_start) & (rtts < bin_start + bin_size_ms))
        if bin_count >= cutoff_min_points:
            cutoff_rtt = bin_start + bin_size_ms
    cutoff_rtt = min(float(cutoff_rtt), max_rtt)

    return {
        "hull_upper_rtts": [x[0] for x in hull_upper],
        "hull_upper_distances": [x[1] for x in hull_upper],
        "hull_lower_rtts": [x[0] for x in hull_lower],
        "hull_lower_distances": [x[1] for x in hull_lower],
        "cutoff_rtt": cutoff_rtt,
        "success": True,
        "message": f"hull: {len(hull_upper)} upper, {len(hull_lower)} lower vertices",
    }


def _interpolate_in_hull(
    rtt: float,
    hull_rtts_arr: np.ndarray,
    hull_dists_arr: np.ndarray,
) -> float:
    """Linear interpolation between adjacent hull vertices at `rtt`.

    Shared between the outer and inner boundaries. Caller is responsible for
    ensuring `rtt` is in the trusted range — no extrapolation handling here.
    """
    idx = np.searchsorted(hull_rtts_arr, rtt, side="right") - 1
    idx = max(0, min(idx, len(hull_rtts_arr) - 2))
    rtt_low, rtt_high = hull_rtts_arr[idx], hull_rtts_arr[idx + 1]
    dist_low, dist_high = hull_dists_arr[idx], hull_dists_arr[idx + 1]
    if rtt_high == rtt_low:
        return float(dist_low)
    t = (rtt - rtt_low) / (rtt_high - rtt_low)
    return float(dist_low + t * (dist_high - dist_low))


def hull_outer_distance(
    rtt: float,
    hull_rtts: List[float],
    hull_distances: List[float],
    cutoff_rtt: float,
    baseline_slope: float = THEORETICAL_SLOPE,
) -> float:
    """Upper hull boundary distance at `rtt`.

    - Below the leftmost vertex: line through origin (physical bound — all
      training RTTs already respect the speed-of-internet constraint).
    - Strictly above `cutoff_rtt`: pin to the hull at `cutoff_rtt`, extend
      with the baseline (2/3·c) slope. The cutoff itself uses plain
      interpolation — only RTTs strictly larger than the cutoff are in the
      extension regime, matching `predict_distance_bounds`'s gate.
    - Otherwise: linear interpolation between adjacent hull vertices.

    `compute_convex_hull_bounds` clamps `cutoff_rtt ≤ max(data_rtt)`, so the
    cutoff lookup is always plain interpolation.
    """
    if len(hull_rtts) == 0:
        return rtt / baseline_slope if baseline_slope > 0 else 0.0

    hull_rtts_arr = np.array(hull_rtts)
    hull_dists_arr = np.array(hull_distances)

    if rtt > cutoff_rtt and cutoff_rtt > 0:
        cutoff_dist = _interpolate_in_hull(cutoff_rtt, hull_rtts_arr, hull_dists_arr)
        return cutoff_dist + (rtt - cutoff_rtt) / baseline_slope

    x0 = hull_rtts_arr[0]
    if rtt < x0:
        if x0 <= 0:
            return float(hull_dists_arr[0])
        return max(0.0, (hull_dists_arr[0] / x0) * rtt)

    return _interpolate_in_hull(rtt, hull_rtts_arr, hull_dists_arr)


def hull_inner_distance(
    rtt: float,
    hull_rtts: List[float],
    hull_distances: List[float],
    cutoff_rtt: float,
) -> float:
    """Lower hull boundary distance at `rtt`.

    - Below the leftmost vertex: 0 (no useful lower constraint at very low RTT).
    - Strictly above `cutoff_rtt`: hold flat at the hull's value at
      `cutoff_rtt`. The cutoff itself uses plain interpolation — only RTTs
      strictly larger than the cutoff are in the flat regime, matching
      `predict_distance_bounds`'s gate.
    - Otherwise: linear interpolation between adjacent hull vertices.

    Takes no `baseline_slope` — the inner bound never extends, only holds flat.
    """
    if len(hull_rtts) == 0:
        return 0.0

    hull_rtts_arr = np.array(hull_rtts)
    hull_dists_arr = np.array(hull_distances)

    if rtt > cutoff_rtt and cutoff_rtt > 0:
        return max(0.0, _interpolate_in_hull(cutoff_rtt, hull_rtts_arr, hull_dists_arr))

    if rtt < hull_rtts_arr[0]:
        return 0.0

    return _interpolate_in_hull(rtt, hull_rtts_arr, hull_dists_arr)


# ----------------------------------------------------------------------------
# Spline fit
# ----------------------------------------------------------------------------


def fit_rtt_distance_spline(
    rtts: np.ndarray,
    distances: np.ndarray,
    n_knots: int = 20,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Piecewise-linear LSQ spline (k=1) through the (rtt, distance) scatter.

    Interior knots uniformly placed across the RTT range. Monotonicity in
    distance is enforced post-fit. Needs >= `n_knots + 3` valid points.

    Returns `(knot_rtts, knot_dists, metadata)` — two arrays suitable for
    `np.interp(rtt, knot_rtts, knot_dists)` plus a meta dict with n_knots,
    n_points, r_squared, residual_std.
    """
    rtts = np.asarray(rtts, dtype=float)
    distances = np.asarray(distances, dtype=float)

    valid = (rtts > 0) & (distances > 0) & np.isfinite(rtts) & np.isfinite(distances)
    rtts = rtts[valid]
    distances = distances[valid]

    min_points = n_knots + 3
    if len(rtts) < min_points:
        raise SplineFitError(
            f"need at least {min_points} valid points for {n_knots} interior knots, "
            f"got {len(rtts)}"
        )

    sort_idx = np.argsort(rtts)
    rtts = rtts[sort_idx]
    distances = distances[sort_idx]

    try:
        # k=1: boundary knots repeated k+1=2 times, n_knots interior knots strictly inside.
        interior = np.linspace(rtts[0], rtts[-1], n_knots + 2)[1:-1]
        # Knots. Knots and data points must satisfy Schoenberg-Whitney conditions.
        t_full = np.r_[(rtts[0],) * 2, interior, (rtts[-1],) * 2]
        spline = make_lsq_spline(rtts, distances, t=t_full, k=1)
    except Exception as e:
        raise SplineFitError(f"spline fitting failed: {e}")

    knot_rtts = np.linspace(rtts[0], rtts[-1], n_knots + 2)
    knot_dists = spline(knot_rtts)

    if not np.all(np.isfinite(knot_dists)):
        raise SplineFitError(
            f"spline produced non-finite values (ill-conditioned LSQ with "
            f"{n_knots} interior knots over {len(rtts)} points)"
        )

    # Monotonicity enforcement 
    # This is not isotonic regression (which would minimize squared error subject to a monotonicity constraint). 
    # It's a one-pass post-hoc fix that doesn't preserve the LSQ objective. 
    # The trade-off: cheap and predictable, but the post-fix curve is no longer the LSQ-optimal monotonic fit. 
    # For Octant's use case — where the spline is the "best guess" inside a hull band 
    # that will clip wild values anyway, this approximation is fine.
    for i in range(1, len(knot_dists)):
        if knot_dists[i] < knot_dists[i - 1]:
            knot_dists[i] = knot_dists[i - 1]

    predicted = spline(rtts)
    ss_res = float(np.sum((distances - predicted) ** 2))
    ss_tot = float(np.sum((distances - np.mean(distances)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return knot_rtts, knot_dists, {
        "n_knots": len(knot_rtts),
        "n_points": len(rtts),
        "r_squared": r_squared,
        "residual_std": float(np.std(distances - predicted)),
    }


# ----------------------------------------------------------------------------
# δ search
# ----------------------------------------------------------------------------


def find_delta_for_coverage(
    rtts: np.ndarray,
    distances: np.ndarray,
    spline_rtt_knots: np.ndarray,
    spline_dist_knots: np.ndarray,
    target_coverage: float,
    tolerance: float = 0.01,
    max_iterations: int = 100,
    timeout_seconds: float = 10.0,
    delta_min: float = 1.0,
) -> Tuple[float, Dict[str, Any]]:
    """Binary-search δ ≥ 1 so that `target_coverage` of points lie within
    `(spline(rtt)/δ, spline(rtt)·δ)`.

    Returns `(delta, metadata)`. Raises `DeltaSearchError` if no δ in
    [delta_min, 1e10] reaches `target_coverage`, or `DeltaSearchTimeout`
    if wall time exceeds `timeout_seconds`.
    """
    if spline_rtt_knots is None or len(spline_rtt_knots) < 2:
        raise SplineFitError("invalid spline knots")

    rtts = np.asarray(rtts, dtype=float)
    distances = np.asarray(distances, dtype=float)
    valid = (rtts > 0) & (distances > 0) & np.isfinite(rtts) & np.isfinite(distances)
    rtts = rtts[valid]
    distances = distances[valid]
    if len(rtts) == 0:
        raise DeltaSearchError("no valid data points")

    start = time.time()

    def coverage(d: float) -> float:
        pred = np.interp(rtts, spline_rtt_knots, spline_dist_knots)
        pred = np.maximum(pred, 1.0)
        return float(np.mean((distances >= pred / d) & (distances <= pred * d)))

    # Double δ until the band covers enough.
    delta_max = delta_min
    cov_max = coverage(delta_max)
    while cov_max < target_coverage:
        if time.time() - start > timeout_seconds:
            raise DeltaSearchTimeout(
                f"timeout finding delta_max after {time.time() - start:.2f}s"
            )
        delta_max *= 2
        cov_max = coverage(delta_max)
        if delta_max > 1e10:
            raise DeltaSearchError(
                f"cannot achieve {target_coverage:.1%} coverage at delta={delta_max:.0f}"
            )

    lo, hi = delta_min, delta_max
    best_delta = delta_max
    best_cov = cov_max
    best_diff = abs(best_cov - target_coverage)

    for i in range(max_iterations):
        if time.time() - start > timeout_seconds:
            raise DeltaSearchTimeout(
                f"timeout after {i} iterations; best delta={best_delta:.4f} cov={best_cov:.3f}"
            )
        mid = (lo + hi) / 2
        cov = coverage(mid)
        diff = abs(cov - target_coverage)
        if diff < best_diff:
            best_delta = mid
            best_cov = cov
            best_diff = diff
        if diff <= tolerance:
            return mid, {"actual_coverage": cov, "iterations": i + 1, "converged": True}
        if cov < target_coverage:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break

    if best_diff > tolerance:
        raise DeltaSearchError(
            f"could not achieve {target_coverage:.1%} within tolerance {tolerance:.1%}; "
            f"best delta={best_delta:.4f} cov={best_cov:.3f}"
        )

    return best_delta, {
        "actual_coverage": best_cov,
        "iterations": max_iterations,
        "converged": False,
    }


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


@dataclass
class OctantRTTModel:
    """Per-anchor Octant annulus model.

    `predict_distance_bounds(rtt, delta)` returns `(inner_km, outer_km)`.
    Without δ: convex hull facets at `rtt`. With δ: multiplicative spline
    band, clipped by the hull.
    """

    anchor_ip: str
    anchor_lat: float = 0.0
    anchor_lon: float = 0.0

    hull_upper_rtts: List[float] = field(default_factory=list)
    hull_upper_distances: List[float] = field(default_factory=list)
    hull_lower_rtts: List[float] = field(default_factory=list)
    hull_lower_distances: List[float] = field(default_factory=list)

    cutoff_rtt: float = 0.0
    cutoff_min_points: int = 5
    baseline_slope: float = THEORETICAL_SLOPE

    spline_rtt_knots: Optional[List[float]] = None
    spline_dist_knots: Optional[List[float]] = None
    spline_n_knots: int = 20

    n_measurements: int = 0
    fitted: bool = False
    fit_message: str = ""

    def _outer(self, rtt: float) -> float:
        return hull_outer_distance(
            rtt,
            self.hull_upper_rtts,
            self.hull_upper_distances,
            self.cutoff_rtt,
            self.baseline_slope,
        )

    def _inner(self, rtt: float) -> float:
        return hull_inner_distance(
            rtt,
            self.hull_lower_rtts,
            self.hull_lower_distances,
            self.cutoff_rtt,
        )

    def fit(
        self,
        rtts: np.ndarray,
        distances: np.ndarray,
        cutoff_min_points: int = 5,
        fit_spline: bool = True,
        spline_n_knots: int = 20,
        bin_size_ms: float = 5.0,
    ) -> bool:
        """Fit hull bounds + (optional) LSQ spline through the in-cutoff region.

        Returns `True` if hull fit succeeded (the spline is best-effort —
        a spline failure leaves `spline_rtt_knots = None` but still
        marks the model fitted).
        """
        self.cutoff_min_points = cutoff_min_points
        self.spline_n_knots = spline_n_knots

        rtts_arr = np.asarray(rtts, dtype=float)
        dists_arr = np.asarray(distances, dtype=float)
        # Filter NaN / non-positive rows AND rows below the speed-of-internet line
        # (rtt < baseline_slope · distance is physically impossible at 2/3·c — those
        # rows are mislabeled coordinates or measurement artifacts and would distort
        # both the hull and the spline).
        valid = (
            (rtts_arr > 0)
            & (dists_arr > 0)
            & np.isfinite(rtts_arr)
            & np.isfinite(dists_arr)
            & (rtts_arr >= self.baseline_slope * dists_arr)
        )
        valid_rtts = rtts_arr[valid]
        valid_dists = dists_arr[valid]
        self.n_measurements = int(len(valid_rtts))

        hull = compute_convex_hull_bounds(
            valid_rtts,
            valid_dists,
            cutoff_min_points=cutoff_min_points,
            bin_size_ms=bin_size_ms,
        )
        if not hull["success"]:
            self.fitted = False
            self.fit_message = hull["message"]
            return False

        self.hull_upper_rtts = hull["hull_upper_rtts"]
        self.hull_upper_distances = hull["hull_upper_distances"]
        self.hull_lower_rtts = hull["hull_lower_rtts"]
        self.hull_lower_distances = hull["hull_lower_distances"]
        self.cutoff_rtt = hull["cutoff_rtt"]
        self.spline_rtt_knots = None
        self.spline_dist_knots = None

        if fit_spline:
            if self.cutoff_rtt > 0:
                mask = valid_rtts <= self.cutoff_rtt
            else:
                mask = np.ones_like(valid_rtts, dtype=bool)
            spline_rtts = valid_rtts[mask]
            spline_dists = valid_dists[mask]
            # Cap n_knots by (distinct RTTs − 2). With k=1 splines, each basis
            # function spans two inter-knot intervals; when the data clusters
            # at a small set of distinct RTT values, asking for more knots
            # than the data shape can support gives an ill-conditioned LSQ
            # system that returns NaN/inf coefficients.
            n_distinct = int(np.unique(spline_rtts).size)
            n_knots_used = max(3, min(spline_n_knots, max(3, n_distinct - 2)))
            try:
                knot_rtts, knot_dists, spline_meta = fit_rtt_distance_spline(
                    spline_rtts, spline_dists, n_knots=n_knots_used
                )
                self.spline_rtt_knots = knot_rtts.tolist()
                self.spline_dist_knots = knot_dists.tolist()
                self.spline_n_knots = n_knots_used
                self.fit_message = (
                    f"hull: {len(self.hull_upper_rtts)} upper, "
                    f"{len(self.hull_lower_rtts)} lower; "
                    f"cutoff_rtt={self.cutoff_rtt:.3f} ms; "
                    f"spline: {spline_meta['n_knots']} knots, "
                    f"R²={spline_meta['r_squared']:.3f}"
                )
            except SplineFitError as e:
                self.fit_message = f"hull OK, spline failed: {e}"
        else:
            self.fit_message = (
                f"hull: {len(self.hull_upper_rtts)} upper, "
                f"{len(self.hull_lower_rtts)} lower; "
                f"cutoff_rtt={self.cutoff_rtt:.3f} ms"
            )

        self.fitted = True
        return True

    def predict_distance(self, rtt: float) -> float:
        """Spline value at `rtt`, clipped to the hull band.

        Above `cutoff_rtt`: pin to the spline value at cutoff and extend
        with the baseline (2/3·c) slope.
        """
        if not self.fitted:
            raise OctantFitError("model not fitted")
        if self.spline_rtt_knots is None:
            raise SplineFitError("spline not fitted")

        knot_rtts = np.array(self.spline_rtt_knots)
        knot_dists = np.array(self.spline_dist_knots)
        if rtt > self.cutoff_rtt and self.cutoff_rtt > 0:
            cutoff_val = float(np.interp(self.cutoff_rtt, knot_rtts, knot_dists))
            predicted = cutoff_val + (rtt - self.cutoff_rtt) / self.baseline_slope
        else:
            predicted = float(np.interp(rtt, knot_rtts, knot_dists))

        if self.hull_upper_rtts and self.hull_lower_rtts:
            predicted = max(self._inner(rtt), min(predicted, self._outer(rtt)))

        return max(predicted, 0.0)

    def predict_distance_bounds(
        self,
        rtt: float,
        delta: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Annular `(inner_km, outer_km)` for `rtt`.

        - Above `cutoff_rtt` or no δ: bare convex-hull bounds.
        - With δ: `(spline/δ, spline·δ)` clipped by the hull.
        """
        if not self.fitted:
            raise OctantFitError("model not fitted")

        if rtt > self.cutoff_rtt and self.cutoff_rtt > 0:
            return (max(0.0, self._inner(rtt)), self._outer(rtt))

        if delta is not None:
            predicted = self.predict_distance(rtt)
            inner = max(predicted / delta, self._inner(rtt))
            outer = min(predicted * delta, self._outer(rtt))
            return (max(0.0, inner), outer)

        return (max(0.0, self._inner(rtt)), self._outer(rtt))

    def save(self, filepath: Path) -> None:
        """Persist the fitted model as JSON for checkpoint recovery."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "OctantRTTModel":
        with open(filepath) as f:
            return cls(**json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor_ip": self.anchor_ip,
            "anchor_lat": self.anchor_lat,
            "anchor_lon": self.anchor_lon,
            "hull_upper_rtts": self.hull_upper_rtts,
            "hull_upper_distances": self.hull_upper_distances,
            "hull_lower_rtts": self.hull_lower_rtts,
            "hull_lower_distances": self.hull_lower_distances,
            "cutoff_rtt": self.cutoff_rtt,
            "cutoff_min_points": self.cutoff_min_points,
            "baseline_slope": self.baseline_slope,
            "spline_rtt_knots": self.spline_rtt_knots,
            "spline_dist_knots": self.spline_dist_knots,
            "spline_n_knots": self.spline_n_knots,
            "n_measurements": self.n_measurements,
            "fitted": self.fitted,
            "fit_message": self.fit_message,
        }

    def __repr__(self) -> str:
        if self.fitted:
            return (
                f"OctantRTTModel(anchor={self.anchor_ip}, "
                f"hull_upper={len(self.hull_upper_rtts)}, "
                f"hull_lower={len(self.hull_lower_rtts)}, "
                f"cutoff_rtt={self.cutoff_rtt:.1f} ms)"
            )
        return f"OctantRTTModel(anchor={self.anchor_ip}, fitted=False)"
