"""Cell polygons: the Voronoi cells of the seeds, unbounded.

`classify` never needs these. It labels a prediction by its nearest seed, a
great-circle argmin over one distance matrix, with no geometry and no bound of
any kind. The polygons exist to **draw** the cell partition.

## Unbounded, and what that means for the drawing

Every point on Earth is nearest to some seed, so the partition the map draws
covers the frame edge to edge -- over ocean, over Canada, arbitrarily far from
any site. That overreach is the property the evaluation section is about: it
is why a nearest-seed verdict alone credits a prediction 2,385 km from the
truth, and why `ring` is read beside `cell_label` rather than instead of it.

`shapely.voronoi_polygons` cannot return a genuinely unbounded cell, so the
diagram is built against a frame derived from the **drawn extent** and then
cut to it. That cut is a rendering bound and nothing else: every cell reaches
the edge of the frame and would keep going if the frame were larger. Do not
read the frame edge as a cell edge -- unlike the landmass it replaces, it
carries no meaning and `classify` has no equivalent of it.

The frame is sized from the extent rather than by a fixed pad around the
seeds because EPSG:5070 is a CONUS projection: a box a few thousand km past
the seeds has corners outside the projection's usable domain, and they come
back from the inverse transform as `inf` or wrapped past the antimeridian,
drawn as straight lines across the whole map. Note `extend_to` alone is not
enough -- GEOS treats it as a lower bound and returns polygons well past it,
so the explicit intersection is what keeps every vertex in-domain.

## Planar, in EPSG:5070

Built with `shapely.voronoi_polygons` in the projected plane, then brought
back to `(lon, lat)`. A planar Voronoi in an equal-area conic is not the
great-circle nearest-seed partition: the projection's ~1-2% distance
distortion shifts a boundary far from its seeds by several km.
`agreement_with_nearest_seed` measures it on a point sample and the map's
manifest records the number, so the figure states how faithfully it draws the
rule rather than asserting it. Without edge densification (`_SEGMENT_M`) it
was ~98% even against the old clipped cells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

from scripts.analysis.v5.modules.geodesy import pairwise_km
from scripts.analysis.v5.modules.projection import project, to_lonlat

#: Edge densification before leaving the projection. A straight edge in
#: EPSG:5070 is a curve in lon/lat; unsegmented, a 1,000 km cell edge would be
#: redrawn as a lon/lat chord and bow away from the boundary it represents.
_SEGMENT_M = 10_000.0

#: Pad on the frame, in metres: enough that no cell edge coincides with the
#: drawn boundary, small enough to stay inside EPSG:5070's usable domain.
_FRAME_PAD_M = 5e5

#: Samples per extent edge when projecting it. The extent is a lon/lat box and
#: EPSG:5070 is conic, so its edges are curves in the plane; the corners alone
#: understate the projected bounding box.
_EDGE_SAMPLES = 25


def _frame(
    x: np.ndarray, y: np.ndarray, extent: tuple[float, float, float, float]
) -> shapely.Geometry:
    """The finite box the diagram is extended to, in projected metres.

    Covers the drawn extent and the seeds both, padded. See the module
    docstring on why this is a rendering bound and not a semantic one.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    lons = np.linspace(lon_min, lon_max, _EDGE_SAMPLES)
    lats = np.linspace(lat_min, lat_max, _EDGE_SAMPLES)
    edge_lon = np.concatenate([lons, lons, np.full(_EDGE_SAMPLES, lon_min),
                               np.full(_EDGE_SAMPLES, lon_max)])
    edge_lat = np.concatenate([np.full(_EDGE_SAMPLES, lat_min),
                               np.full(_EDGE_SAMPLES, lat_max), lats, lats])
    ex, ey = project(edge_lat, edge_lon)
    xs = np.concatenate([x, ex])
    ys = np.concatenate([y, ey])
    return shapely.box(
        float(xs.min()) - _FRAME_PAD_M, float(ys.min()) - _FRAME_PAD_M,
        float(xs.max()) + _FRAME_PAD_M, float(ys.max()) + _FRAME_PAD_M,
    )


def cell_polygons(
    seeds: pd.DataFrame, extent: tuple[float, float, float, float]
) -> dict[int, shapely.Geometry]:
    """`seed_id -> cell polygon` in `(lon, lat)` degrees, cut to the frame.

    `extent` is `(lon_min, lon_max, lat_min, lat_max)`, the frame the map
    draws; the cells cover all of it and are cut there for rendering only.
    Every seed gets a cell: an unbounded partition leaves none empty, which is
    the one simplification the landmass removal buys.
    """
    x, y = project(seeds["seed_lat"], seeds["seed_lon"])
    ids = seeds["seed_id"].to_numpy(dtype=int)
    frame = _frame(x, y, extent)
    if len(ids) == 1:
        # One seed owns everything. The frame is the whole drawing, which is
        # the honest picture: its edge is the figure's, not the partition's.
        return {int(ids[0]): to_lonlat(shapely.segmentize(frame, _SEGMENT_M))}

    points = shapely.points(x, y)
    diagram = shapely.voronoi_polygons(shapely.multipoints(points), extend_to=frame)
    out: dict[int, shapely.Geometry] = {}
    for poly in shapely.get_parts(diagram):
        # Each Voronoi polygon holds exactly one generator; find which.
        hit = np.flatnonzero(shapely.contains_xy(poly, x, y))
        if hit.size != 1:
            raise AssertionError(f"a Voronoi polygon holds {hit.size} seeds, not 1")
        clipped = poly.intersection(frame)
        out[int(ids[hit[0]])] = to_lonlat(shapely.segmentize(clipped, _SEGMENT_M))
    if len(out) != len(ids):
        raise AssertionError(f"{len(out)} cells for {len(ids)} seeds; every seed must get one")
    bad = sorted(k for k, v in out.items() if not v.is_valid or not np.isfinite(v.bounds).all())
    if bad:
        # An out-of-domain frame corner comes back from the inverse transform
        # as inf or wrapped past the antimeridian, and draws as a line across
        # the map rather than as a cell. Fail instead.
        raise AssertionError(f"cells {bad} reprojected to an invalid polygon; frame too large")
    return out


def agreement_with_nearest_seed(
    seeds: pd.DataFrame,
    polygons: dict[int, shapely.Geometry],
    extent: tuple[float, float, float, float],
    *,
    n: int = 4000,
    seed: int = 0,
) -> float:
    """Share of random points in `extent` whose drawn cell is their nearest seed.

    The drawn polygons against the rule `classify` scores with. `extent` is
    `(lon_min, lon_max, lat_min, lat_max)` -- the frame the map actually
    draws, so the number describes the picture the reader sees.

    Expect this to sit below what the old landmass-clipped cells reported.
    That version rejection-sampled to inland points only, and planar/spherical
    disagreement is worst far from the seeds -- exactly the ocean and
    cross-border area an unbounded partition now covers.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    rng = np.random.default_rng(seed)
    lon = rng.uniform(lon_min, lon_max, n)
    lat = rng.uniform(lat_min, lat_max, n)

    d = pairwise_km(lat, lon, seeds["seed_lat"], seeds["seed_lon"])
    nearest = seeds["seed_id"].to_numpy(dtype=int)[d.argmin(axis=1)]
    drawn = np.full(lat.size, -1)
    for seed_id, poly in polygons.items():
        drawn[shapely.contains_xy(poly, lon, lat)] = seed_id
    return float((drawn == nearest).mean())
