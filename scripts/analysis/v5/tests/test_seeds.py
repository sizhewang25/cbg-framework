"""Seeds: complete linkage caps the group diameter at `grid_km`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.geodesy import pairwise_km, spherical_centroid
from scripts.analysis.v5.modules.seeds import build_seeds, group_sites

EWR = (40.6895, -74.1745)
JFK = (40.6413, -73.7781)
SEATTLE = (47.449, -122.309)
OMAHA = (41.2565, -95.9345)


def _sites(coords) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": range(len(coords)),
            "site_lat": [c[0] for c in coords],
            "site_lon": [c[1] for c in coords],
            "n_tgs": [3] * len(coords),
        }
    )


def _chain(n=3, step_km=40.0, lat=40.0, lon0=-100.0):
    """Sites due east of each other, `step_km` apart."""
    dlon = step_km / (111.195 * np.cos(np.radians(lat)))
    return [(lat, lon0 + i * dlon) for i in range(n)]


def test_ewr_and_jfk_share_a_seed_at_the_finest_rung():
    assert pairwise_km([EWR[0]], [EWR[1]], [JFK[0]], [JFK[1]])[0, 0] < G.grid_km(128)
    ids = group_sites([EWR[0], JFK[0]], [EWR[1], JFK[1]], G.grid_km(128))
    assert ids[0] == ids[1]


@pytest.mark.parametrize("nside", G.NSIDE_LADDER)
def test_seattle_and_omaha_never_share(nside):
    ids = group_sites([SEATTLE[0], OMAHA[0]], [SEATTLE[1], OMAHA[1]], G.grid_km(nside))
    assert ids[0] != ids[1]


def test_complete_linkage_does_not_chain():
    """Three sites 40 km apart: every neighbour pair is within 51 km, the ends
    are 80 km apart. Single linkage would make one seed; complete must not."""
    chain = _chain()
    ids = group_sites([c[0] for c in chain], [c[1] for c in chain], G.grid_km(128))
    assert len(set(ids)) == 2
    assert len(set(group_sites([c[0] for c in chain], [c[1] for c in chain], 100.0))) == 1


@pytest.mark.parametrize("nside", G.NSIDE_LADDER)
def test_every_seed_is_at_most_one_grid_across(nside):
    rng = np.random.default_rng(nside)
    coords = list(zip(rng.uniform(30, 45, 60), rng.uniform(-120, -75, 60)))
    ids, seeds = build_seeds(_sites(coords), G.grid_km(nside))
    assert (seeds["seed_diameter_km"] <= G.grid_km(nside) + 1e-6).all()
    assert seeds["n_sites"].sum() == len(coords)
    assert sorted(set(ids)) == list(range(len(seeds)))


def test_ids_follow_site_order_not_scipy_labels():
    coords = [SEATTLE, EWR, OMAHA, JFK]
    ids = group_sites([c[0] for c in coords], [c[1] for c in coords], G.grid_km(128))
    assert ids.tolist() == [0, 1, 2, 1]


def test_seed_is_the_spherical_centroid_of_its_sites():
    ids, seeds = build_seeds(_sites([EWR, JFK, SEATTLE]), G.grid_km(128))
    lat, lon = spherical_centroid([EWR[0], JFK[0]], [EWR[1], JFK[1]])
    row = seeds.loc[seeds["seed_id"] == ids[0]].iloc[0]
    assert (row["seed_lat"], row["seed_lon"]) == pytest.approx((lat, lon))
    assert row["n_sites"] == 2 and row["n_tgs"] == 6


def test_a_single_site_is_one_seed():
    ids, seeds = build_seeds(_sites([SEATTLE]), G.grid_km(128))
    assert ids.tolist() == [0]
    assert len(seeds) == 1 and np.isnan(seeds["nearest_seed_km"].iloc[0])
