"""Cell polygons: one per seed, tiling the frame, faithful to nearest seed."""

from __future__ import annotations

import pandas as pd
import pytest
import shapely

from scripts.analysis.v5.modules import cells as CL
from scripts.analysis.v5.modules import mapping as M

SEEDS = pd.DataFrame(
    {
        "seed_id": [0, 1, 2, 3],
        "seed_lat": [47.449, 41.2565, 41.8781, 25.7932],
        "seed_lon": [-122.309, -95.9345, -87.6298, -80.29],
    }
)

EXTENT = M.US_MAINLAND_EXTENT


@pytest.fixture(scope="module")
def polys():
    return CL.cell_polygons(SEEDS, EXTENT)


def test_one_polygon_per_seed_holding_its_seed(polys):
    assert sorted(polys) == [0, 1, 2, 3]
    for sid, poly in polys.items():
        row = SEEDS.loc[SEEDS["seed_id"] == sid].iloc[0]
        assert shapely.contains_xy(poly, row["seed_lon"], row["seed_lat"])


def test_cells_cover_the_drawn_extent_without_overlap(polys):
    """The partition is unbounded, so it covers everything that is drawn.

    The old landmass version asserted the union *was* the land. There is no
    land now: the union must contain the whole frame instead.
    """
    lon_min, lon_max, lat_min, lat_max = EXTENT
    frame = shapely.box(lon_min, lat_min, lon_max, lat_max)
    union = shapely.union_all(list(polys.values()))
    # Area of the shortfall, not `contains`: the cells meet along shared
    # edges, and a union built from them fails a strict containment test on
    # boundary slivers even when it covers every point.
    assert frame.difference(union).area / frame.area < 1e-9
    total = sum(p.area for p in polys.values())
    assert total == pytest.approx(union.area, rel=1e-6)


def test_no_cell_stops_short_of_the_frame(polys):
    """Every cell here touches the frame edge: with four seeds and no clip,
    none of them is interior. This is the property the landmass used to
    destroy, and the one the figure is drawn to show."""
    lon_min, lon_max, lat_min, lat_max = EXTENT
    frame = shapely.box(lon_min, lat_min, lon_max, lat_max)
    for sid, poly in polys.items():
        assert poly.intersects(frame.exterior), f"cell {sid} does not reach the frame"


def test_every_cell_reprojects_to_a_finite_valid_polygon(polys):
    """An out-of-domain frame corner comes back from the inverse transform as
    `inf` or wrapped past the antimeridian, and draws as a line across the
    whole map rather than as a cell. It happened; this pins the fix."""
    for sid, poly in polys.items():
        assert poly.is_valid, f"cell {sid} is not valid"
        assert all(abs(v) < 1e4 for v in poly.bounds), f"cell {sid} has runaway bounds"
        assert poly.bounds[0] >= -180.0 and poly.bounds[2] <= 180.0


def test_drawn_cells_agree_with_nearest_seed(polys):
    assert CL.agreement_with_nearest_seed(SEEDS, polys, EXTENT) > 0.99


def test_a_single_seed_owns_the_whole_frame():
    polys = CL.cell_polygons(SEEDS.iloc[:1], EXTENT)
    assert list(polys) == [0]
    lon_min, lon_max, lat_min, lat_max = EXTENT
    frame = shapely.box(lon_min, lat_min, lon_max, lat_max)
    assert frame.difference(polys[0]).area / frame.area < 1e-9
