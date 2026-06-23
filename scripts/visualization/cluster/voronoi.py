"""Landmass-clipped Voronoi partition of the answer-space centroids.

The answer space (`scripts.benchmark.v2.sources.cluster_ground_truth`) reframes
CBG validation as classification: each region's centroid is one answer-space
point, and a coordinate is assigned to its *nearest* centroid. The Voronoi
diagram of those centroids is therefore the literal decision boundary of that
nearest-centroid classifier — the **target answer-space partition**. This module
builds it, restricted to (and clipped against) a named landmass.

It is a pure-geometry primitive (no matplotlib): resolve a landmass name to a
boundary polygon, then compute the centroid Voronoi cells clipped to it.
Rendering lives in `plot_ground_truth_clusters.py`.

Landmass names accepted by `resolve_landmass` (Natural Earth 110m, offline via
cartopy):
  * a continent  — "Europe", "North America", "Asia", ... (the ``CONTINENT``
    column; case-insensitive);
  * a country    — ISO_A2 ("US"), ISO_A3 ("USA"), or admin/name ("France").

No new dependency: cells come from ``shapely.ops.voronoi_diagram`` (shapely 2)
and are intersected with the boundary. Cells are computed in lon/lat
(PlateCarree) to match the cluster map — a visualization-grade approximation,
not an equal-area construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPoint, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union, voronoi_diagram
from shapely.strtree import STRtree

logger = logging.getLogger(__name__)

# Natural Earth 110m sets ISO_A2 to this sentinel for disputed/unmapped rows
# (notably France, Norway), so ISO lookups fall back to ADMIN/NAME.
_NE_ISO_SENTINEL = "-99"

# Default membership buffer (degrees): a coastal/island centroid can sit just
# outside the generalized 110m boundary, so we seed from a slightly grown
# region while still clipping cells to the exact boundary.
_DEFAULT_BUFFER_DEG = 0.3


@dataclass(frozen=True)
class LandmassVoronoi:
    """The answer-space partition over one landmass, ready to draw.

    Attributes:
        cells: GeoDataFrame (EPSG:4326), one clipped Voronoi cell per seeding
            centroid, carrying ``cluster_id``, ``n_members``, ``is_singleton``,
            and the cell ``geometry``.
        boundary: The landmass polygon the cells were clipped to (EPSG:4326).
        label: Human-readable match label, e.g. ``"US — United States of
            America"`` or ``"continent — Europe"``.
    """

    cells: gpd.GeoDataFrame
    boundary: BaseGeometry
    label: str

    def focus_extent(self, *, pct: float = 2.0, pad_frac: float = 0.08) -> list[float]:
        """A readable ``[lon_min, lon_max, lat_min, lat_max]`` for the partition.

        Derived from the seeding centroids (cell representative points) using a
        ``[pct, 100-pct]`` percentile clip, then padded by ``pad_frac``. Using
        the centroid mass rather than the raw boundary keeps the view tight when
        the landmass has far-flung members — e.g. Natural Earth's "Europe"
        includes Russia out to ~180°E, and "US" includes Alaska/Aleutians — which
        would otherwise squash the dense region. Outlier cells may fall outside
        this extent; pass an explicit extent to override.
        """
        reps = self.cells.geometry.representative_point()
        lon = np.array([p.x for p in reps])
        lat = np.array([p.y for p in reps])
        lon_min, lon_max = np.percentile(lon, [pct, 100 - pct])
        lat_min, lat_max = np.percentile(lat, [pct, 100 - pct])
        pad_x = (lon_max - lon_min) * pad_frac or 1.0
        pad_y = (lat_max - lat_min) * pad_frac or 1.0
        return [lon_min - pad_x, lon_max + pad_x, lat_min - pad_y, lat_max + pad_y]


def _load_admin0() -> gpd.GeoDataFrame:
    """Load the Natural Earth 110m country polygons (cartopy-cached, offline)."""
    import cartopy.io.shapereader as shpreader

    path = shpreader.natural_earth(
        resolution="110m", category="cultural", name="admin_0_countries"
    )
    return gpd.read_file(path)


def resolve_landmass(name: str) -> tuple[BaseGeometry, str]:
    """Resolve a landmass name to ``(boundary_polygon, label)`` in EPSG:4326.

    Matching order: continent (``CONTINENT``), then country by ISO_A2, ISO_A3,
    ADMIN, NAME — all case-insensitive. A continent returns the union of its
    member countries. Raises ``ValueError`` with nearby suggestions on no match.
    """
    gdf = _load_admin0()
    query = name.strip()
    up = query.upper()

    cont = gdf["CONTINENT"].astype(str).str.upper() == up
    if cont.any():
        return unary_union(gdf.loc[cont].geometry.values), f"continent — {query}"

    # ISO_A2/A3 carry the "-99" sentinel for some states (France, Norway); the
    # "_EH" variants fill those in, so check them too.
    for col in ("ISO_A2", "ISO_A2_EH", "ISO_A3", "ISO_A3_EH", "ADMIN", "NAME"):
        if col not in gdf.columns:
            continue
        col_up = gdf[col].astype(str).str.upper()
        match = (col_up == up) & (gdf[col].astype(str) != _NE_ISO_SENTINEL)
        if match.any():
            admin = str(gdf.loc[match, "ADMIN"].iloc[0])
            return unary_union(gdf.loc[match].geometry.values), f"{query} — {admin}"

    raise ValueError(
        f"unknown landmass {name!r}. Use a continent "
        f"({', '.join(sorted(gdf['CONTINENT'].dropna().unique()))}) "
        f"or a country code/name (e.g. 'US', 'USA', 'France')."
    )


def clipped_voronoi_cells(
    lons: np.ndarray, lats: np.ndarray, boundary: BaseGeometry
) -> gpd.GeoDataFrame:
    """Voronoi cells of the given seed points, clipped to ``boundary``.

    Returns a GeoDataFrame (EPSG:4326) with a ``seed_index`` column (position in
    the input arrays) and the clipped cell ``geometry``. Cells whose clipped
    geometry is empty (seed outside the boundary's reach) are dropped. Each cell
    is matched to its seed by point-in-cell containment, so the result is robust
    to GEOS cell ordering.
    """
    seeds = [Point(float(x), float(y)) for x, y in zip(lons, lats)]
    if len(seeds) < 2:
        raise ValueError(f"need >=2 seed centroids for a Voronoi diagram, got {len(seeds)}")

    diagram = voronoi_diagram(MultiPoint(seeds), envelope=boundary)
    tree = STRtree(seeds)

    records = []
    for cell in diagram.geoms:
        clipped = cell.intersection(boundary)
        if clipped.is_empty:
            continue
        contained = tree.query(cell, predicate="contains")
        if len(contained) == 0:
            # Fall back to nearest seed (degenerate sliver); shouldn't normally hit.
            seed_index = int(tree.nearest(cell.representative_point()))
        else:
            seed_index = int(contained[0])
        records.append({"seed_index": seed_index, "geometry": clipped})

    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def build_landmass_voronoi(
    clusters: pd.DataFrame, landmass: str, *, buffer_deg: float = _DEFAULT_BUFFER_DEG
) -> LandmassVoronoi:
    """Build the answer-space Voronoi partition for one landmass.

    Seeds from every cluster centroid inside the landmass (singletons included —
    they are valid answer-space points), clips each cell to the boundary, and
    carries each cell's ``cluster_id`` / ``n_members`` / ``is_singleton`` back
    for styling.

    Args:
        clusters: ``clusters.csv`` frame with ``cluster_id``, ``centroid_lat``,
            ``centroid_lon``, ``n_members``.
        landmass: Continent or country name/code (see `resolve_landmass`).
        buffer_deg: Degrees to grow the boundary for the seed-membership test
            (cells stay clipped to the exact boundary); 0 disables.
    """
    boundary, label = resolve_landmass(landmass)

    lon = clusters["centroid_lon"].to_numpy(dtype=float)
    lat = clusters["centroid_lat"].to_numpy(dtype=float)
    region = boundary.buffer(buffer_deg) if buffer_deg else boundary
    inside = shapely.contains_xy(region, lon, lat)
    sub = clusters.loc[inside].reset_index(drop=True)
    logger.info(
        "%s: %d/%d centroids inside landmass seed the partition",
        label, len(sub), len(clusters),
    )

    cells = clipped_voronoi_cells(
        sub["centroid_lon"].to_numpy(dtype=float),
        sub["centroid_lat"].to_numpy(dtype=float),
        boundary,
    )
    n_members = sub["n_members"].to_numpy(dtype=int)
    cells["cluster_id"] = sub["cluster_id"].to_numpy()[cells["seed_index"].to_numpy()]
    cells["n_members"] = n_members[cells["seed_index"].to_numpy()]
    cells["is_singleton"] = cells["n_members"] <= 1
    cells = cells.drop(columns="seed_index")

    return LandmassVoronoi(cells=cells, boundary=boundary, label=label)
