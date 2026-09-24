"""Cell polygons: one per seed, inside the landmass, faithful to nearest seed."""

from __future__ import annotations

import pandas as pd
import pytest
import shapely

from scripts.analysis.v5.modules import cells as CL
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.landmass import load_landmass

SEEDS = pd.DataFrame(
    {
        "seed_id": [0, 1, 2, 3],
        "seed_lat": [47.449, 41.2565, 41.8781, 25.7932],
        "seed_lon": [-122.309, -95.9345, -87.6298, -80.29],
    }
)


@pytest.fixture(scope="module")
def drawn():
    lm = load_landmass(G.grid_km(128))
    return lm, CL.cell_polygons(SEEDS, lm)


def test_one_polygon_per_seed_holding_its_seed(drawn):
    _, polys = drawn
    assert sorted(polys) == [0, 1, 2, 3]
    for sid, poly in polys.items():
        row = SEEDS.loc[SEEDS["seed_id"] == sid].iloc[0]
        assert shapely.contains_xy(poly, row["seed_lon"], row["seed_lat"])


def test_cells_tile_the_landmass_without_overlap(drawn):
    lm, polys = drawn
    land = lm.to_lonlat(lm.geometry)
    union = shapely.union_all(list(polys.values()))
    assert union.symmetric_difference(land).area / land.area < 1e-3
    total = sum(p.area for p in polys.values())
    assert total == pytest.approx(union.area, rel=1e-6)


def test_drawn_cells_agree_with_nearest_seed(drawn):
    lm, polys = drawn
    assert CL.agreement_with_nearest_seed(SEEDS, lm, polys) > 0.99


def test_a_single_seed_owns_the_whole_landmass():
    lm = load_landmass(G.grid_km(16))
    polys = CL.cell_polygons(SEEDS.iloc[:1], lm)
    assert list(polys) == [0]
    assert polys[0].area == pytest.approx(lm.to_lonlat(lm.geometry).area, rel=1e-3)
