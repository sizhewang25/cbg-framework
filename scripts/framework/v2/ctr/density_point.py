"""Density-aware point estimates — Spotter's §III-B options, one class each.

§III-B ends by naming three ways to collapse the probability surface to a single
location: "the center of the estimated region", "the maximum", or "the mean
value of the probability distribution". Those are genuinely different estimators
on a skewed posterior, so they are three CTRs rather than one with a mode flag —
the benchmark treats CTR as a factor, and a flag would hide the choice inside a
kwarg instead of naming it in the combo id.

`DensityMLECTR` is a fourth, and is not in the paper: it takes the grid argmax
as a seed and finishes the job continuously. Worth having because the exact MAP
is available in closed form as a least-squares problem —

    argmax log P(x) = argmin Σ_i ((s_i(x) − µ_i) / σ_i)²

— which is 2 free parameters and `O(n)` per iteration, i.e. microseconds. It is
the *reference* against which any discretised or geometric approximation of
Spotter should be scored, and it removes the grid's own quantisation error
(an H3-4 cell is tens of km across, which is the same order as the accuracy
thresholds the paper reports at).

All four refuse to run on an `MTLResult` with no `density`, rather than falling
back to the `intersection` coords: a density-blind reading of a density field is
what the geometric CTRs already do, and silently substituting it here would make
the combo id a lie about which estimator produced the number.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from scripts.framework.v2.ctr.base import CTRMethod, CTRResult
from scripts.framework.v2.mtl.base import DensityField, MTLResult
from scripts.framework.v2.registry import register_ctr
from scripts.framework.v2.types import Coord, Error

EARTH_RADIUS_KM = 6371.0


def _require_density(mtl: MTLResult) -> Optional[DensityField]:
    if mtl.density is None or not mtl.density.cells:
        return None
    return mtl.density


def _no_density() -> CTRResult:
    return CTRResult(success=False, error=Error.INSUFFICIENT_DATA)


def _spherical_mean(
    lats: np.ndarray, lons: np.ndarray, weights: np.ndarray
) -> Optional[Coord]:
    """Weighted mean direction on the sphere, back-projected to lat/lon.

    Averaging lat/lon arithmetically is wrong across the antimeridian and near
    the poles; averaging unit vectors is not. A field spread symmetrically
    enough for the resultant to vanish has no mean direction, and that returns
    None rather than an arbitrary point.
    """
    p, l = np.radians(lats), np.radians(lons)
    x = float(np.sum(weights * np.cos(p) * np.cos(l)))
    y = float(np.sum(weights * np.cos(p) * np.sin(l)))
    z = float(np.sum(weights * np.sin(p)))
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 1e-12:
        return None
    x, y, z = x / norm, y / norm, z / norm
    return Coord(
        lat=math.degrees(math.asin(max(-1.0, min(1.0, z)))),
        lon=math.degrees(math.atan2(y, x)),
    )


@register_ctr("density_argmax")
class DensityArgmaxCTR(CTRMethod):
    """The single most probable cell's centre — §III-B's "the maximum".

    The MAP estimate up to the grid's own resolution, and the only one of the
    three that is invariant to how far the field was truncated: adding or
    dropping low-probability cells cannot move it. That makes it the right
    default when the MTL pruned coarse-to-fine.
    """

    def _select_centroid(self, mtl: MTLResult) -> CTRResult:
        field = _require_density(mtl)
        if field is None:
            return _no_density()
        best = max(range(len(field.log_density)), key=lambda i: field.log_density[i])
        return CTRResult(success=True, tg_coord=field.cells[best])


@register_ctr("density_mean")
class DensityMeanCTR(CTRMethod):
    """Probability-weighted mean position — §III-B's "the mean value".

    Pulled toward secondary modes and toward whatever the field's truncation
    left in, so it is the estimator most sensitive to the MTL's pruning
    settings. That is a property of the estimator, not a defect: on a broad
    unimodal posterior it is more stable than the argmax, and on a bimodal one
    it lands between the modes, which is the honest answer to "where is the
    mean" and the wrong answer to "where is the target".
    """

    def _select_centroid(self, mtl: MTLResult) -> CTRResult:
        field = _require_density(mtl)
        if field is None:
            return _no_density()
        weights = np.asarray(field.probabilities(), dtype=float)
        if not weights.size or not np.isfinite(weights).all() or weights.sum() <= 0:
            return CTRResult(success=False, error=Error.NUMERICAL_FAILURE)
        lats = np.fromiter((c.lat for c in field.cells), float, len(field.cells))
        lons = np.fromiter((c.lon for c in field.cells), float, len(field.cells))
        coord = _spherical_mean(lats, lons, weights)
        if coord is None:
            return CTRResult(success=False, error=Error.DEGENERATE_REGION)
        return CTRResult(success=True, tg_coord=coord)


@register_ctr("density_region_center")
class DensityRegionCenterCTR(CTRMethod):
    """Unweighted centre of the credible region — §III-B's "center of the region".

    Deliberately ignores `log_density` inside the region and averages the cells
    the MTL selected, which is what "centre of the estimated region" means: the
    region is already the confidence statement, and re-weighting inside it
    would make this a duplicate of `density_mean` under a different name.
    """

    def _select_centroid(self, mtl: MTLResult) -> CTRResult:
        field = _require_density(mtl)
        if field is None:
            return _no_density()
        region = mtl.intersection
        if not isinstance(region, list) or not region:
            return CTRResult(success=False, error=Error.EMPTY_REGION)
        lats = np.fromiter((c.lat for c in region), float, len(region))
        lons = np.fromiter((c.lon for c in region), float, len(region))
        coord = _spherical_mean(lats, lons, np.ones(len(region), dtype=float))
        if coord is None:
            return CTRResult(success=False, error=Error.DEGENERATE_REGION)
        return CTRResult(success=True, tg_coord=coord)


@register_ctr("density_mle")
class DensityMLECTR(CTRMethod):
    """Exact continuous MAP: `min Σ ((s_i(x) − µ_i)/σ_i)²`, seeded at the argmax.

    Not the paper's — the paper reads its estimate off the grid — but the exact
    optimum of the paper's own objective, so it is what the grid and every
    geometric approximation should be measured against.

    Optimised in a **local tangent plane** at the seed rather than in raw
    (lat, lon): degrees of longitude are not degrees of latitude away from the
    equator, and an unscaled parameterisation makes the problem badly
    conditioned in a way that shows up as premature convergence rather than as
    an error. Falls back to the seed if SciPy is unavailable or the solve fails,
    so this can never do worse than `density_argmax` by more than one grid cell.
    """

    def __init__(self, max_nfev: int = 200) -> None:
        if max_nfev < 1:
            raise ValueError(f"max_nfev must be >= 1, got {max_nfev}")
        self.max_nfev = max_nfev

    def _select_centroid(self, mtl: MTLResult) -> CTRResult:
        field = _require_density(mtl)
        if field is None:
            return _no_density()
        if not field.constraints:
            return CTRResult(success=False, error=Error.INSUFFICIENT_DATA)
        seed = field.cells[
            max(range(len(field.log_density)), key=lambda i: field.log_density[i])
        ]
        try:
            from scipy.optimize import least_squares
        except ImportError:
            return CTRResult(success=True, tg_coord=seed)

        lat0, lon0 = seed.lat, seed.lon
        # km per degree at the seed. The longitude scale collapses at the poles,
        # so it is floored: a zero scale would make the east-west parameter
        # unidentifiable and the Jacobian singular.
        km_per_deg_lat = math.pi * EARTH_RADIUS_KM / 180.0
        km_per_deg_lon = max(km_per_deg_lat * math.cos(math.radians(lat0)), 1e-6)
        vp = np.array([[c.lat, c.lon] for c, _, _ in field.constraints], dtype=float)
        mu = np.array([m for _, m, _ in field.constraints], dtype=float)
        sigma = np.array([s for _, _, s in field.constraints], dtype=float)

        def residuals(delta: np.ndarray) -> np.ndarray:
            lat = lat0 + delta[0] / km_per_deg_lat
            lon = lon0 + delta[1] / km_per_deg_lon
            p0, l0 = math.radians(lat), math.radians(lon)
            p1, l1 = np.radians(vp[:, 0]), np.radians(vp[:, 1])
            a = (
                np.sin((p1 - p0) / 2.0) ** 2
                + math.cos(p0) * np.cos(p1) * np.sin((l1 - l0) / 2.0) ** 2
            )
            s = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.minimum(1.0, a)))
            return (s - mu) / sigma

        try:
            fit = least_squares(residuals, x0=np.zeros(2), max_nfev=self.max_nfev)
        except Exception:
            return CTRResult(success=True, tg_coord=seed)
        if not np.isfinite(fit.x).all():
            return CTRResult(success=True, tg_coord=seed)

        lat = lat0 + float(fit.x[0]) / km_per_deg_lat
        lon = lon0 + float(fit.x[1]) / km_per_deg_lon
        if not (-90.0 <= lat <= 90.0):
            return CTRResult(success=True, tg_coord=seed)
        # Wrap rather than reject: the tangent plane is free to step across the
        # antimeridian, and that is a valid position, not a failure.
        lon = ((lon + 180.0) % 360.0) - 180.0
        return CTRResult(success=True, tg_coord=Coord(lat=lat, lon=lon))
