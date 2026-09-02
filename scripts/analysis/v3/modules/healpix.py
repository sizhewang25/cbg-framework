"""HEALPix quantizer for the target answer space (paper §7.3).

The grid is a *quantizer*, not the answer space: its only job is to merge target
points that sit close enough to count as one place. Each occupied cell then
yields one class, seeded at the centroid of the targets it merged (§7.4).

`nside=128` is the paper's setting — 196,608 equal-area cells of 2,594 km²,
nominally 51 km across, chosen because prior work puts metro granularity near
40 km. The NESTED ordering is used throughout so that coarsening is a bit shift
(`pix >> 2k`), which makes the §7.3 multi-scale concentration curve over
nside 128 / 64 / 32 / 16 a single pass with no re-projection.

Because HEALPix cell ids are global, class sets from different datasets land in
one frame automatically and their footprints can be intersected rather than only
compared in aggregate.
"""

from __future__ import annotations

import numpy as np

from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM

#: Paper §7.3 setting.
DEFAULT_NSIDE = 128

#: Coarsening hierarchy for the multi-scale concentration diagnostic (§7.3).
#: Nested ordering makes each step a right shift by 2 bits.
NSIDE_HIERARCHY: tuple[int, ...] = (128, 64, 32, 16)

_ORDER = "nested"


def _healpix(nside: int):
    # Imported lazily: astropy is a heavy import and only the answer-space
    # builder needs it.
    from astropy_healpix import HEALPix

    return HEALPix(nside=nside, order=_ORDER)


def validate_nside(nside: int) -> int:
    """`nside` must be a power of two for the NESTED scheme to nest."""
    if nside < 1 or (nside & (nside - 1)) != 0:
        raise ValueError(f"nside must be a positive power of two, got {nside}")
    return int(nside)


def npix(nside: int) -> int:
    """Total cells: `12 * nside**2` (196,608 at nside=128)."""
    return 12 * validate_nside(nside) ** 2


def pixel_area_km2(nside: int) -> float:
    """Equal-area cell size in km² (2,594 km² at nside=128)."""
    return 4.0 * np.pi * EARTH_RADIUS_KM**2 / npix(nside)


def nominal_cell_km(nside: int) -> float:
    """Nominal cell pitch, `sqrt(area)` (51 km at nside=128).

    This is the merge scale — the distance below which two targets are treated
    as the same place — and the pitch the §7.3 straddle probability is stated
    against.
    """
    return float(np.sqrt(pixel_area_km2(nside)))


def ang2pix(lat_deg, lon_deg, nside: int = DEFAULT_NSIDE) -> np.ndarray:
    """Cell id (NESTED) for each (lat, lon) in degrees."""
    import astropy.units as u

    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    pix = _healpix(validate_nside(nside)).lonlat_to_healpix(lon * u.deg, lat * u.deg)
    return np.asarray(pix, dtype=np.int64)


def degrade(pix, nside_from: int, nside_to: int) -> np.ndarray:
    """Map NESTED cell ids from a finer grid to a coarser one by bit shift.

    Valid only for NESTED ordering and `nside_to <= nside_from`, both powers of
    two — which is what makes the hierarchy diagnostic free.
    """
    validate_nside(nside_from)
    validate_nside(nside_to)
    if nside_to > nside_from:
        raise ValueError(f"nside_to ({nside_to}) must not exceed nside_from ({nside_from})")
    shift = 2 * int(np.log2(nside_from // nside_to))
    return np.asarray(pix, dtype=np.int64) >> shift


def occupied_cell_hierarchy(
    lat_deg, lon_deg, nsides: tuple[int, ...] = NSIDE_HIERARCHY
) -> dict[int, int]:
    """Distinct occupied cell count at each nside, computed in one pass.

    A steep climb toward fine cells means the point set only separates at
    intra-metro scales; a flat curve means genuinely distinct metros (§7.3).
    """
    ordered = sorted(set(int(n) for n in nsides), reverse=True)
    finest = ordered[0]
    pix = ang2pix(lat_deg, lon_deg, finest)
    return {n: int(np.unique(degrade(pix, finest, n)).size) for n in ordered}


def spherical_centroid(lat_deg, lon_deg) -> tuple[float, float]:
    """Centroid of points on the sphere: normalized mean of unit vectors.

    Averaging lat/lon directly is wrong near the dateline and at high latitude;
    this is the projection-free form, matching §7.4's "centroid of the targets
    inside the cell".
    """
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    vx, vy, vz = float(x.mean()), float(y.mean()), float(z.mean())
    norm = np.sqrt(vx * vx + vy * vy + vz * vz)
    if norm == 0.0:
        # Antipodal cancellation — impossible within a 51 km cell, but a mean
        # of zero has no direction so there is no centroid to return.
        raise ValueError("degenerate point set: unit vectors cancel to zero")
    return (
        float(np.degrees(np.arcsin(vz / norm))),
        float(np.degrees(np.arctan2(vy, vx))),
    )
