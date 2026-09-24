"""The landmass decides `outland`. Points chosen so each rung's buffer matters."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.landmass import load_landmass

LAKE_MICHIGAN = (44.0, -87.0)
#: Spotter's prediction for tg-e1a1545 (truth Seattle), the case v3 credited.
ARCTIC = (63.7369, -97.4685)
HONOLULU = (21.31, -157.86)
ANCHORAGE = (61.2, -149.9)
MID_PACIFIC = (30.0, -150.0)
#: 150 km north of the 49th parallel: outside the 51 and 102 km buffers,
#: inside the 204 and 407 km ones. (Winnipeg is only ~100 km from the border.)
NORTH_OF_BORDER_150KM = (50.35, -97.14)
COASTAL_SITES = {
    "seattle": (47.449, -122.309),
    "miami": (25.7932, -80.29),
    "boston": (42.3656, -71.0052),
}


def _inside(point, nside) -> bool:
    return bool(load_landmass(G.grid_km(nside)).contains(*point)[0])


@pytest.mark.parametrize("nside", G.NSIDE_LADDER)
def test_the_great_lakes_are_inland(nside):
    """Territory, not coastline: the country polygon runs through the lakes."""
    assert _inside(LAKE_MICHIGAN, nside)


@pytest.mark.parametrize("nside", G.NSIDE_LADDER)
@pytest.mark.parametrize("point", [ARCTIC, HONOLULU, ANCHORAGE, MID_PACIFIC])
def test_off_the_mainland_is_outland_at_every_rung(point, nside):
    assert not _inside(point, nside)


@pytest.mark.parametrize("nside", G.NSIDE_LADDER)
@pytest.mark.parametrize("name", sorted(COASTAL_SITES))
def test_coastal_sites_are_inland(name, nside):
    assert _inside(COASTAL_SITES[name], nside)


def test_the_buffer_is_the_rung_grid_km():
    assert [_inside(NORTH_OF_BORDER_150KM, n) for n in (128, 64, 32, 16)] == [
        False, False, True, True,
    ]


def test_inland_stays_inland_as_grids_grow():
    rng = np.random.default_rng(7)
    lat = rng.uniform(20, 58, 3000)
    lon = rng.uniform(-135, -60, 3000)
    fine_to_coarse = [load_landmass(G.grid_km(n)).contains(lat, lon) for n in G.NSIDE_LADDER]
    for finer, coarser in zip(fine_to_coarse, fine_to_coarse[1:]):
        assert not np.any(finer & ~coarser)


def test_missing_coordinates_are_not_inland():
    assert not load_landmass(G.grid_km(128)).contains([np.nan], [np.nan])[0]


def test_negative_buffer_is_refused():
    with pytest.raises(ValueError):
        load_landmass(-1.0)
