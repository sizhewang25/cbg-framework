"""Answer-space grid interface: one contract, two tessellations.

The answer space (§7.3/§7.4) needs a grid only as a **quantizer** — something that
decides which target coordinates are close enough to count as one place. Everything
downstream (seeds at target centroids, Voronoi labelling, distance-to-all-seeds
scoring) is pure spherical geometry and does not care which tessellation produced the
equivalence classes. This module is where that indifference is made explicit.

Two implementations:

* **HEALPix** (`healpix.HealpixGrid`) — equal-area quads, exactly nested, so coarsening
  is a bit shift. The paper's original setting (`nside=128`, ~51 km).
* **H3** (`h3grid.H3Grid`) — hexagons, uniform neighbour distance, no ambiguous
  edge/corner adjacency, and the de-facto standard in telecom analytics (native in
  ClickHouse, Postgres, BigQuery, Snowflake, Spark). The default.

Neither grid is better in the abstract and the choice is not free — see
`Grid.coarsening_ladder` and `Grid.occupancy_diagnostics` for where the two genuinely
differ, and `h3grid` for the non-nesting caveat H3 carries.

**Contract on resolution numbers.** Both grids use a single integer where a *smaller
number means a coarser grid* (HEALPix `nside`, H3 `res`). That shared convention is what
lets `coarsening_ladder` be written once here instead of twice in the implementations.
It is a coincidence of the two APIs, not a law, so a third grid must be checked against
it rather than assumed to comply.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import pandas as pd

#: Grid used when the caller does not say. H3 for operator-facing reasons: hexagons are
#: what RF/telecom analytics and SQL engines speak.
DEFAULT_GRID = "h3"

#: Selectable names, in `--help` listing order.
GRID_NAMES: tuple[str, ...] = ("h3", "healpix")


def ring_lonlat(lon_deg, lat_deg) -> np.ndarray:
    """One cell boundary as an `(V, 2)` array of `(lon, lat)` degrees.

    Does the one piece of bookkeeping both grids need: making a ring **contiguous in
    longitude**. A cell straddling the antimeridian comes back from either library with
    mixed-sign longitudes (H3 gives e.g. `[179.85, 179.75, -179.96, ...]`), and drawing
    that as a single polygon smears it clear across the map instead of rendering a
    hexagon. Every vertex is therefore placed within half a turn of the first one, which
    puts the ring slightly outside [-180, 180] when it straddles — correct for plotting,
    and what cartopy expects.
    """
    lon = np.asarray(lon_deg, dtype=float).ravel()
    lat = np.asarray(lat_deg, dtype=float).ravel()
    lon = ((lon + 180.0) % 360.0) - 180.0
    lon = lon[0] + ((lon - lon[0] + 180.0) % 360.0) - 180.0
    return np.column_stack([lon, lat])


class Grid(ABC):
    """A tessellation of the sphere usable as an answer-space quantizer.

    Subclasses supply the grid-specific parts; the hierarchy walk and the static
    self-description are implemented once here.
    """

    #: Value written to `seeds.csv`'s `grid_scheme` and used in directory slugs.
    name: ClassVar[str]

    #: What this grid calls its resolution knob (`"nside"` / `"res"`). Used only in help
    #: text and echoes, so a HEALPix run still reads `nside=128` rather than a
    #: generic-sounding `resolution=128`.
    resolution_arg: ClassVar[str]

    #: Resolution used when the caller does not say.
    DEFAULT_RESOLUTION: ClassVar[int]

    #: Sweep rungs, finest first.
    HIERARCHY: ClassVar[tuple[int, ...]]

    # ---- resolution -------------------------------------------------------

    @abstractmethod
    def validate_resolution(self, resolution: int) -> int:
        """Return `resolution` as an int, or raise `ValueError` naming what is allowed."""

    # ---- quantization -----------------------------------------------------

    @abstractmethod
    def cell_ids(self, lat_deg, lon_deg, resolution: int) -> np.ndarray:
        """Cell id per input coordinate.

        Ids need only be hashable, sortable and stable — HEALPix returns `int64`, H3
        returns its canonical hex strings. Nothing downstream does arithmetic on them.
        """

    @abstractmethod
    def coerce_cell_ids(self, values) -> pd.Series:
        """Restore the id dtype after a CSV round-trip.

        `seeds.csv` is text, so `cell_id` comes back as whatever pandas inferred. This
        is the one place that knows which dtype was meant.
        """

    # ---- scale ------------------------------------------------------------

    @abstractmethod
    def cell_area_km2(self, resolution: int) -> float:
        """Cell area. Exact for HEALPix; the mean for H3, whose cells vary."""

    @abstractmethod
    def nominal_cell_km(self, resolution: int) -> float:
        """The merge scale: how far apart two points can be and still share a cell.

        This is the number the §7.3 straddle diagnostic is stated against, so each grid
        picks the form that is honest for its own cell shape rather than a common
        formula.
        """

    @abstractmethod
    def n_cells(self, resolution: int) -> int:
        """Total cells covering the sphere at this resolution."""

    # ---- geometry ---------------------------------------------------------

    @abstractmethod
    def cell_boundaries(
        self, cell_ids, resolution: int, *, step: int = 8
    ) -> list[np.ndarray]:
        """One `(V, 2)` `(lon, lat)`-degree ring per cell, for drawing.

        **Ragged by design.** HEALPix rings are 4-sided (times `step`), H3 rings are 6
        for hexagons and 5 for the 12 pentagons per resolution, so callers must not
        index this as a rectangular array.

        `resolution` is passed even though H3 ids already encode it, because HEALPix ids
        do not: cell 5 exists at every nside, so ids alone cannot say which cells are
        meant. Inferring it would be a silent-wrong-answer bug rather than a crash.

        `step` is the number of interpolated points per edge, which HEALPix uses to
        follow the cell's curvature on the sphere. H3 returns exact vertices and ignores
        it.
        """

    # ---- self description -------------------------------------------------

    def describe(self, resolution: int) -> dict:
        """Static facts about the grid at this resolution, for `meta["grid"]`.

        Static means "derivable from the resolution alone" — subclasses extend this with
        their own invariants (HEALPix's ordering, H3's average edge). Anything that
        depends on *which* cells the targets landed in belongs in
        `occupancy_diagnostics` instead.
        """
        r = self.validate_resolution(resolution)
        return {
            "scheme": self.name,
            "resolution": r,
            "n_cells": self.n_cells(r),
            "cell_area_km2": round(self.cell_area_km2(r), 3),
            "nominal_cell_km": round(self.nominal_cell_km(r), 3),
        }

    def occupancy_diagnostics(
        self, cell_ids, lat_deg, lon_deg, resolution: int
    ) -> dict:
        """Grid-specific caveats measured against *this* target set.

        Empty for a grid with nothing to disclose. HEALPix is exactly equal-area and
        exactly nested, so it has nothing; H3 reports its area spread, its pentagons,
        and how often index lineage disagrees with re-binning. The point is that a
        grid's weaknesses get *measured on the data at hand* rather than argued about in
        prose.
        """
        return {}

    # ---- hierarchy --------------------------------------------------------

    def coarsening_ladder(self, resolution: int) -> tuple[int, ...]:
        """`resolution`, then every coarser rung of `HIERARCHY`, coarsening.

        Starting at the resolution in use is the whole point: the multi-scale
        concentration curve (§7.3) describes what happens when you *coarsen* the grid
        you built, so reporting rungs finer than it is not a diagnostic, it is a
        different grid. An earlier version walked the fixed `HIERARCHY` regardless of
        the resolution built, which at the coarsest rung reported counts for grids that
        were never used.
        """
        r = self.validate_resolution(resolution)
        return (r,) + tuple(x for x in self.HIERARCHY if x < r)

    def occupied_cell_hierarchy(
        self, lat_deg, lon_deg, resolutions: tuple[int, ...]
    ) -> dict[int, int]:
        """`{resolution: distinct occupied cells}`, by **re-binning** at each rung.

        Deliberately not "coarsen the ids from the finest rung", even though HEALPix
        makes that a bit shift and this a little slower. H3 is aperture-7 and hexagons
        cannot tile hexagons, so a parent's outer children straddle its boundary and
        `cell_to_parent(cell_ids(p, fine), coarse) != cell_ids(p, coarse)` for points
        near a boundary. Re-binning from the coordinates is the definition of occupancy
        at a resolution and is correct on both grids; the shift is merely an
        optimization that happens to be exact on one of them
        (`test_degrade_matches_direct_ang2pix` pins that).
        """
        return {
            int(r): int(np.unique(self.cell_ids(lat_deg, lon_deg, r)).size)
            for r in resolutions
        }


def get_grid(name: str) -> Grid:
    """Resolve a grid by name.

    Implementations are imported inside the body for two reasons: `healpix` and `h3grid`
    both need `Grid` from this module, so a top-level import here would be circular; and
    it keeps `astropy` and `h3` off the import path of anyone who only wanted the ABC,
    matching the lazy-import convention used throughout this package for heavy deps.
    """
    key = str(name).strip().lower()
    if key == "healpix":
        from scripts.analysis.v3.modules.healpix import HealpixGrid

        return HealpixGrid()
    if key == "h3":
        from scripts.analysis.v3.modules.h3grid import H3Grid

        return H3Grid()
    raise ValueError(f"unknown grid {name!r}; choose one of {list(GRID_NAMES)}")


def resolutions_for(
    grid: Grid, resolution: list[int] | tuple[int, ...] | None, *, sweep: bool
) -> list[int]:
    """Shared `--resolution` / `--sweep` resolution for the four CLI commands.

    Lives here rather than being copy-pasted into each command module so that "no
    resolution given" means the same thing everywhere: the grid's own default, not a
    hard-coded number that would silently disagree with the grid in use.
    """
    if sweep:
        return list(grid.HIERARCHY)
    if not resolution:
        return [grid.DEFAULT_RESOLUTION]
    return list(dict.fromkeys(int(r) for r in resolution))


def resolve_cli_grid(
    grid_name: str, resolution: list[int] | None, *, sweep: bool
) -> tuple[Grid, list[int]]:
    """Front door for the four commands: validate `--grid` / `--resolution`.

    One place rather than four so a bad value gets the same clean CLI error
    everywhere instead of a traceback, and so the `--grid` whitelist cannot
    drift between commands. `typer` is imported inside the body to keep this
    module usable (and testable) without it.
    """
    import typer

    if grid_name not in GRID_NAMES:
        raise typer.BadParameter(f"--grid must be one of {list(GRID_NAMES)}")
    g = get_grid(grid_name)
    try:
        return g, [
            g.validate_resolution(r)
            for r in resolutions_for(g, resolution, sweep=sweep)
        ]
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


# ---- shared CLI help text ---------------------------------------------------
# Kept here rather than repeated in the four command modules so the four
# `--help` screens cannot drift apart as the grids change.

GRID_HELP = (
    "Answer-space grid: h3 (hexagons, uniform neighbour distance, the default) "
    "or healpix (equal-area quads, exactly nested)."
)

RESOLUTION_HELP = (
    "Grid resolution (repeatable). Defaults to the chosen grid's own default: "
    "h3 res 4 (~45 km) or healpix nside 128 (~51 km). h3 supports 2-5 "
    "(5=~17 km, 3=~120 km, 2=~316 km); healpix takes any power of two."
)

SWEEP_HELP = (
    "Shorthand for the chosen grid's full hierarchy: h3 5/4/3/2, "
    "healpix 128/64/32/16."
)
