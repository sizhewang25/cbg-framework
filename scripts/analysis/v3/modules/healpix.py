"""HEALPix quantizer for the target answer space (paper §7.3).

One of two grids behind the `grid.Grid` interface (`h3grid.H3Grid` is the other,
and the default). This was the paper's original choice and stays fully
supported; `HealpixGrid` at the bottom of this file is the adapter, and the
functions above it remain usable on their own.

The grid is a *quantizer*, not the answer space: its only job is to merge target
points that sit close enough to count as one place. Each occupied cell then
yields one class, seeded at the centre of the cell itself (§7.4).

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

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from scripts.analysis.v3.modules.grid import Grid, ring_lonlat
from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM

if TYPE_CHECKING:
    import pandas as pd

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
    as the same place — and the pitch the §7.3 merge scale is stated
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


class HealpixGrid(Grid):
    """HEALPix as an answer-space quantizer.

    Thin adapter over the functions above, which stay usable on their own — the
    module-level `degrade` in particular is HEALPix-only and deliberately *not*
    on the `Grid` interface, because no other tessellation can offer it.
    """

    name: ClassVar[str] = "healpix"
    resolution_arg: ClassVar[str] = "nside"
    DEFAULT_RESOLUTION: ClassVar[int] = DEFAULT_NSIDE
    HIERARCHY: ClassVar[tuple[int, ...]] = NSIDE_HIERARCHY

    def validate_resolution(self, resolution: int) -> int:
        return validate_nside(resolution)

    def cell_ids(self, lat_deg, lon_deg, resolution: int) -> np.ndarray:
        return ang2pix(lat_deg, lon_deg, self.validate_resolution(resolution))

    def coerce_cell_ids(self, values) -> "pd.Series":
        import pandas as pd

        return pd.Series(values).astype("int64")

    def cell_area_km2(self, resolution: int) -> float:
        return pixel_area_km2(resolution)

    def nominal_cell_km(self, resolution: int) -> float:
        return nominal_cell_km(resolution)

    def n_cells(self, resolution: int) -> int:
        return npix(resolution)

    def cell_centers(self, cell_ids, resolution: int) -> np.ndarray:
        """One `(lat, lon)` degree pair per cell, as an `(N, 2)` array.

        **`(lat, lon)`, the opposite order from `cell_boundaries`.** Rings are
        `(lon, lat)` for plotting; a seed is written to `seeds.csv` as `seed_lat` then
        `seed_lon`. `test_grid.py` pins the round trip `cell_ids(cell_centers(c, r), r)
        == c`, which a swapped pair cannot survive.

        `resolution` is required rather than inferred, for the same reason as
        `cell_boundaries`: HEALPix ids are not self-describing.

        `healpix_to_lonlat` hands back an astropy `Longitude`, which wraps to `[0, 360)`
        — so a Chicago pixel arrives as `271.77`, not `-88.23`. Normalizing here is not
        cosmetic: nothing would crash, because `pairwise_km` goes through `cos`/`sin` and
        stays correct, but `mapping.seed_voronoi` picks its projection centre from
        `lon.mean()` and every map would render in the wrong hemisphere. Same expression
        as `ring_lonlat`; a centre is a point, not a ring, so the relative unwrapping
        `ring_lonlat` also does would be wrong here. A cell centred at exactly 180
        normalizes to -180, which is the half-open convention and harmless.
        """
        pix = np.asarray(cell_ids, dtype=np.int64).ravel()
        if pix.size == 0:
            return np.zeros((0, 2), dtype=float)
        nside = self.validate_resolution(resolution)
        lon, lat = _healpix(nside).healpix_to_lonlat(pix)
        lon = np.asarray(lon.to_value("deg"), dtype=float)
        lat = np.asarray(lat.to_value("deg"), dtype=float)
        return np.column_stack([lat, ((lon + 180.0) % 360.0) - 180.0])

    def cell_boundaries(
        self, cell_ids, resolution: int, *, step: int = 8
    ) -> list[np.ndarray]:
        """Cell rings, `step` points per edge so curvature on the sphere shows.

        `resolution` is required rather than inferred: HEALPix ids are not
        self-describing (cell 5 exists at every nside), so the same id array means
        a different set of cells at each one.

        `boundaries_lonlat` returns a rectangular `(K, 4*step)` pair of astropy
        `Quantity` arrays; unwrapping the units and splitting into per-cell rings
        happens here so no plotting code has to import astropy.
        """
        pix = np.asarray(cell_ids, dtype=np.int64).ravel()
        if pix.size == 0:
            return []
        nside = self.validate_resolution(resolution)
        lon, lat = _healpix(nside).boundaries_lonlat(pix, step=step)
        lon = np.asarray(lon.to_value("deg"))
        lat = np.asarray(lat.to_value("deg"))
        return [ring_lonlat(lon[i], lat[i]) for i in range(pix.size)]

    def describe(self, resolution: int) -> dict:
        meta = super().describe(resolution)
        # The ordering is load-bearing, not decorative: RING ids would not
        # coarsen by bit shift, so an answer space built under it could not be
        # read back with `degrade`.
        meta["order"] = _ORDER
        meta["nominal_cell_km_note"] = (
            "sqrt(area); HEALPix cells are exactly equal-area"
        )
        return meta
