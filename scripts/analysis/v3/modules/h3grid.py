"""H3 (Uber) hexagonal quantizer for the target answer space — the default grid.

Hexagons buy uniform neighbour distance and unambiguous adjacency: every neighbour is
edge-adjacent at the same centre-to-centre distance, with no corner-vs-edge distinction.
That is why H3 is the working grid in telecom RF analytics, where "signal reaches the
neighbours" has to mean one thing, and it is available natively in ClickHouse, Postgres,
BigQuery, Snowflake and Spark, so quantization can move into SQL later without a
reimplementation.

`res=4` is the default: 1,770 km² average, ~45 km centre-to-centre, the closest rung to
the paper's original HEALPix `nside=128` (~51 km), so switching the default grid does not
quietly move the merge scale. Resolutions 2-5 are supported — 5 is Starlink's
service-cell resolution (~17 km), 3 is metro (~120 km), 2 is macro-region (~316 km).

**Two costs, both measured rather than argued.** H3 is not exactly equal-area (cells vary
within a resolution, and there are 12 pentagons per resolution, placed over ocean), and
its hierarchy is aperture-7: hexagons cannot tile hexagons, so a parent's six outer
children each straddle its boundary. `cell_to_parent` is exact on the *index* but is not
a geometric container, which means

    cell_to_parent(latlng_to_cell(p, fine), coarse) != latlng_to_cell(p, coarse)

for points near a boundary. HEALPix's `pix >> 2k` has no such gap. The consequences are
handled rather than hidden: `Grid.occupied_cell_hierarchy` re-bins from coordinates
instead of walking lineage, and `occupancy_diagnostics` reports how many targets the two
routes actually disagree on.

What H3 does *not* fix is straddling itself. Hexagons have boundaries too, and a facility
group spanning one is still split into two cells and two classes. The gain is marginal
and structural: three cells meet at a hexagon vertex versus four at a quad corner, so
worst-case fragmentation of a tight cluster goes 4 -> 3.

`h3` is imported lazily inside functions, as `healpix.py` does with astropy — only the
answer-space builder needs it.

Verified against h3-py 4.5.0 (the v4 API renamed everything from v3).
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pandas as pd

from scripts.analysis.v3.modules.grid import Grid, ring_lonlat

#: Closest rung to HEALPix `nside=128`; see module docstring.
DEFAULT_RES = 4

#: Sweep rungs, finest first: ~17 / 45 / 120 / 316 km centre-to-centre.
RES_HIERARCHY: tuple[int, ...] = (5, 4, 3, 2)

#: H3 itself allows 0-15. We accept only the band that is meaningful for metro-to-region
#: geolocation: res 6 is ~6 km (finer than any ground truth we have) and res 1 is ~840 km
#: (coarser than a country). Widening this is a one-line change.
SUPPORTED_RESOLUTIONS: tuple[int, ...] = (2, 3, 4, 5)


def _h3():
    """Lazy import — `h3` is only needed when a grid is actually built."""
    import h3

    return h3


class H3Grid(Grid):
    """Uber H3 hexagonal tessellation."""

    name: ClassVar[str] = "h3"
    resolution_arg: ClassVar[str] = "res"
    DEFAULT_RESOLUTION: ClassVar[int] = DEFAULT_RES
    HIERARCHY: ClassVar[tuple[int, ...]] = RES_HIERARCHY

    # ---- resolution -------------------------------------------------------

    def validate_resolution(self, resolution: int) -> int:
        r = int(resolution)
        if r not in SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"h3 resolution must be one of {list(SUPPORTED_RESOLUTIONS)}, got {r}"
            )
        return r

    # ---- quantization -----------------------------------------------------

    def cell_ids(self, lat_deg, lon_deg, resolution: int) -> np.ndarray:
        """Canonical H3 hex-string ids, one per coordinate.

        Strings rather than the 64-bit integer form because that is what ClickHouse,
        BigQuery and the H3 docs all display, and because every id carries `f` padding
        for its unused resolution digits, so pandas can never mistake one for a number
        on a CSV round-trip.

        `latlng_to_cell` is scalar-only in h3-py, hence the loop. Target sets here are
        hundreds of rows, so this is not worth vectorizing around.
        """
        h3 = _h3()
        r = self.validate_resolution(resolution)
        lat = np.asarray(lat_deg, dtype=float).ravel()
        lon = np.asarray(lon_deg, dtype=float).ravel()
        return np.asarray(
            [h3.latlng_to_cell(float(a), float(o), r) for a, o in zip(lat, lon)],
            dtype=object,
        )

    def coerce_cell_ids(self, values) -> pd.Series:
        return pd.Series(values).astype(str)

    # ---- scale ------------------------------------------------------------

    def cell_area_km2(self, resolution: int) -> float:
        """*Average* cell area — H3 cells are not equal-area.

        `occupancy_diagnostics` reports the actual spread over the occupied cells, so
        this mean is never the only area number on the record.
        """
        return float(_h3().average_hexagon_area(self.validate_resolution(resolution), unit="km^2"))

    def nominal_cell_km(self, resolution: int) -> float:
        """Centre-to-centre distance: `edge * sqrt(3)` for a regular hexagon.

        Deliberately not `sqrt(area)`, which is what the HEALPix side uses. For a
        hexagon `sqrt(area)` understates the spacing by about 7%, and this number's job
        is to be compared against seed-to-seed distances, which now *are* distances
        between adjacent cell centres — so it has to be one too, not an area proxy.
        """
        edge = _h3().average_hexagon_edge_length(
            self.validate_resolution(resolution), unit="km"
        )
        return float(edge) * math.sqrt(3.0)

    def n_cells(self, resolution: int) -> int:
        return int(_h3().get_num_cells(self.validate_resolution(resolution)))

    # ---- geometry ---------------------------------------------------------

    def cell_centers(self, cell_ids, resolution: int) -> np.ndarray:
        """One `(lat, lon)` degree pair per cell, as an `(N, 2)` array.

        **`(lat, lon)`, the opposite order from `cell_boundaries`.** Rings are
        `(lon, lat)` because that is what every plotting call wants; a seed is written to
        `seeds.csv` as `seed_lat` then `seed_lon`, so it is stored the other way round.
        Two orders in one class is a swap-bug magnet, which is why `test_grid.py` pins
        the round trip `cell_ids(cell_centers(c, r), r) == c` — a swapped pair lands in a
        different cell, or is rejected outright for a longitude past ±90.

        `resolution` is accepted and unused, as in `cell_boundaries`: H3 ids encode their
        own resolution, and the parameter exists so HEALPix, which needs it, can share
        one signature.

        `cell_to_latlng` is scalar-only in h3-py, hence the loop — same reasoning as
        `cell_ids`. Pentagons are handled with no special case.
        """
        h3 = _h3()
        cells = np.asarray(cell_ids, dtype=object).ravel()
        if cells.size == 0:
            return np.zeros((0, 2), dtype=float)
        return np.array([h3.cell_to_latlng(c) for c in cells], dtype=float)

    def cell_boundaries(
        self, cell_ids, resolution: int, *, step: int = 8
    ) -> list[np.ndarray]:
        """Exact cell vertices as `(lon, lat)` rings.

        Two traps handled here. `cell_to_boundary` returns **(lat, lng)** pairs, the
        opposite order from every plotting call, and pentagons return 5 vertices where
        hexagons return 6 — so the result is a ragged list and must not be treated as an
        array.

        `resolution` and `step` are both accepted and unused: H3 ids already encode their
        resolution, and H3 hands back true vertices with nothing to interpolate. The
        parameters exist so the HEALPix side, which needs both, can share one signature.
        """
        h3 = _h3()
        rings: list[np.ndarray] = []
        for cell in np.asarray(cell_ids, dtype=object).ravel():
            verts = h3.cell_to_boundary(cell)
            lat = [v[0] for v in verts]
            lon = [v[1] for v in verts]
            rings.append(ring_lonlat(lon, lat))
        return rings

    # ---- self description -------------------------------------------------

    def describe(self, resolution: int) -> dict:
        h3 = _h3()
        r = self.validate_resolution(resolution)
        meta = super().describe(r)
        meta["avg_edge_km"] = round(
            float(h3.average_hexagon_edge_length(r, unit="km")), 3
        )
        meta["n_pentagons"] = len(h3.get_pentagons(r))
        meta["nominal_cell_km_note"] = (
            "centre-to-centre (edge * sqrt(3)), not sqrt(area): H3 cells are hexagons "
            "and are not equal-area"
        )
        return meta

    def occupancy_diagnostics(
        self, cell_ids, lat_deg, lon_deg, resolution: int
    ) -> dict:
        """What H3 costs on *this* target set.

        Three numbers, each answering an objection that would otherwise be a matter of
        opinion:

        * `cell_area_km2_{min,max,ratio}` — the equal-area property HEALPix has and H3
          does not, measured over the cells actually occupied rather than over the whole
          sphere.
        * `n_occupied_pentagons` — H3's 12 pentagons per resolution break both equal area
          and the uniform-neighbour claim. They sit over ocean by construction, so this
          should be 0 for land targets; reporting it is how we know.
        * `parent_lineage_disagreements` — targets for which `cell_to_parent` and
          re-binning land in different coarse cells, per coarser rung. This is the
          non-nesting caveat as a count. Zero is a legitimate result on a small target
          set; it is measured so the claim does not have to be taken on trust.
        """
        h3 = _h3()
        r = self.validate_resolution(resolution)
        cells = np.asarray(cell_ids, dtype=object).ravel()
        occupied = list(dict.fromkeys(cells))

        areas = np.array([float(h3.cell_area(c, unit="km^2")) for c in occupied])
        lineage: dict[str, int] = {}
        for coarse in self.coarsening_ladder(r)[1:]:
            by_lineage = np.asarray(
                [h3.cell_to_parent(c, coarse) for c in cells], dtype=object
            )
            by_rebin = self.cell_ids(lat_deg, lon_deg, coarse)
            lineage[str(coarse)] = int(np.sum(by_lineage != by_rebin))

        return {
            "n_occupied_cells": len(occupied),
            "cell_area_km2_min": round(float(areas.min()), 3) if areas.size else None,
            "cell_area_km2_max": round(float(areas.max()), 3) if areas.size else None,
            "cell_area_km2_max_over_min": (
                round(float(areas.max() / areas.min()), 4) if areas.size else None
            ),
            "n_occupied_pentagons": int(sum(1 for c in occupied if h3.is_pentagon(c))),
            "parent_lineage_disagreements": lineage,
            "parent_lineage_note": (
                "H3 is aperture-7 and hexagons cannot tile hexagons, so a parent's outer "
                "children straddle its boundary: cell_to_parent is exact on the index but "
                "is not a geometric container. Counts are targets where index lineage and "
                "re-binning from coordinates disagree. The hierarchy diagnostic re-bins."
            ),
        }
