"""Cell polygons: the Voronoi cells of the seeds, bounded by the landmass.

`classify` never needs these. It labels a prediction by its nearest seed
(great-circle) and by one landmass containment test, which is exact and costs
one distance matrix. The polygons exist to **draw** the cell partition, so the
map shows the boundary `cell_label` is scored against.

## Planar, in the landmass's own projection

The cells are built with `shapely.voronoi_polygons` in EPSG:5070, the plane the
landmass is buffered in, then intersected with the buffered landmass and
brought back to `(lon, lat)`. A planar Voronoi in an equal-area conic is not
the great-circle nearest-seed partition: the projection's ~1-2% distance
distortion shifts a boundary hundreds of km from its seeds by several km.
`agreement_with_nearest_seed` measures it on a point sample -- 99.4-99.7% of
inland points on the three meshes at every rung -- and the map's manifest
records the number, so the figure states how faithfully it draws the rule
rather than asserting it. Without edge densification (`_SEGMENT_M`) it was
~98%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

from scripts.analysis.v5.modules.geodesy import pairwise_km
from scripts.analysis.v5.modules.landmass import Landmass

#: Edge densification before leaving the projection. A straight edge in
#: EPSG:5070 is a curve in lon/lat; unsegmented, a 1,000 km cell edge would be
#: redrawn as a lon/lat chord and bow away from the boundary it represents.
_SEGMENT_M = 10_000.0


def cell_polygons(seeds: pd.DataFrame, landmass: Landmass) -> dict[int, shapely.Geometry]:
    """`seed_id -> cell polygon` in `(lon, lat)` degrees, clipped to the landmass.

    A seed whose clipped cell is empty is omitted -- impossible while every
    site is inland (the guard in `build_answer_space`), but a polygon with no
    area is not something to draw.
    """
    x, y = landmass.project(seeds["seed_lat"], seeds["seed_lon"])
    ids = seeds["seed_id"].to_numpy(dtype=int)
    bound = landmass.geometry
    if len(ids) == 1:
        return {int(ids[0]): landmass.to_lonlat(shapely.segmentize(bound, _SEGMENT_M))}

    points = shapely.points(x, y)
    # Extend past the landmass so no cell is cut by the diagram's own frame
    # before it is cut by the landmass.
    frame = shapely.box(*bound.bounds).buffer(1e6)
    diagram = shapely.voronoi_polygons(shapely.multipoints(points), extend_to=frame)
    out: dict[int, shapely.Geometry] = {}
    for poly in shapely.get_parts(diagram):
        # Each Voronoi polygon holds exactly one generator; find which.
        hit = np.flatnonzero(shapely.contains_xy(poly, x, y))
        if hit.size != 1:
            raise AssertionError(f"a Voronoi polygon holds {hit.size} seeds, not 1")
        clipped = poly.intersection(bound)
        if not clipped.is_empty:
            out[int(ids[hit[0]])] = landmass.to_lonlat(shapely.segmentize(clipped, _SEGMENT_M))
    return out


def agreement_with_nearest_seed(
    seeds: pd.DataFrame,
    landmass: Landmass,
    polygons: dict[int, shapely.Geometry],
    *,
    n: int = 4000,
    seed: int = 0,
) -> float:
    """Share of random inland points whose drawn cell is their nearest seed.

    The drawn polygons against the rule `classify` scores with. Sampled
    uniformly in the landmass's lon/lat bounding box, keeping inland points.
    """
    minx, miny, maxx, maxy = landmass.to_lonlat(landmass.geometry).bounds
    rng = np.random.default_rng(seed)
    lon = rng.uniform(minx, maxx, n * 3)
    lat = rng.uniform(miny, maxy, n * 3)
    keep = landmass.contains(lat, lon)
    lat, lon = lat[keep][:n], lon[keep][:n]
    if lat.size == 0:
        return float("nan")

    d = pairwise_km(lat, lon, seeds["seed_lat"], seeds["seed_lon"])
    nearest = seeds["seed_id"].to_numpy(dtype=int)[d.argmin(axis=1)]
    drawn = np.full(lat.size, -1)
    for seed_id, poly in polygons.items():
        drawn[shapely.contains_xy(poly, lon, lat)] = seed_id
    return float((drawn == nearest).mean())
