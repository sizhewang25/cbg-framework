"""Answer-space geometry: quantization, seed derivation, and class adjacency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules.answer_space import (
    build_answer_space,
    geodesic_points,
    pairwise_km,
    seed_crossing_matrix,
)


# ---- class adjacency, walked rather than triangulated ------------------------


def _latlon_seeds(points):
    """Seeds at explicit coordinates, with the margins the crossing walk steps by."""
    lat = [p[0] for p in points]
    lon = [p[1] for p in points]
    d = pairwise_km(np.array(lat, float), np.array(lon, float))
    np.fill_diagonal(d, np.inf)
    return pd.DataFrame(
        {
            "seed_id": range(len(points)),
            "seed_lat": lat,
            "seed_lon": lon,
            "nearest_seed_km": d.min(axis=1),
            "margin_km": d.min(axis=1) / 2.0,
        }
    )


def test_geodesic_points_hit_both_endpoints():
    lat, lon = geodesic_points(40.0, -100.0, 45.0, -80.0, 9)
    assert (lat[0], lon[0]) == pytest.approx((40.0, -100.0))
    assert (lat[-1], lon[-1]) == pytest.approx((45.0, -80.0))
    assert len(lat) == 9


def test_geodesic_points_are_evenly_spaced_along_the_great_circle():
    """Spherical interpolation, not lat/lon interpolation — at these longitudes
    the straight line in degrees bows hundreds of km off the actual path."""
    lat, lon = geodesic_points(40.0, -120.0, 40.0, -74.0, 21)
    steps = np.array(
        [
            float(pairwise_km(lat[i : i + 1], lon[i : i + 1], lat[i + 1 : i + 2], lon[i + 1 : i + 2])[0, 0])
            for i in range(len(lat) - 1)
        ]
    )
    assert steps.std() / steps.mean() < 1e-6
    # A great circle between two equal-latitude points bows poleward.
    assert lat.max() > 40.0


def test_two_seeds_are_one_boundary_apart():
    c = seed_crossing_matrix(_latlon_seeds([(40.0, -100.0), (41.0, -100.0)]))
    assert c[0, 1] == 1
    assert c[0, 0] == 0


def test_a_seed_in_the_way_adds_a_crossing():
    """Three collinear cells: the ends are two boundaries apart, not one."""
    c = seed_crossing_matrix(
        _latlon_seeds([(38.0, -100.0), (40.0, -100.0), (42.0, -100.0)])
    )
    assert c[0, 1] == 1 and c[1, 2] == 1
    assert c[0, 2] == 2


def test_the_matrix_is_symmetric_with_a_zero_diagonal():
    c = seed_crossing_matrix(
        _latlon_seeds([(38.0, -100.0), (40.0, -100.0), (42.0, -100.0), (40.0, -96.0)])
    )
    assert (c == c.T).all()
    assert (np.diag(c) == 0).all()


def test_a_neighbouring_cell_need_not_be_the_nearest_seed():
    """The whole reason `seeds_crossed` exists.

    C sits north of A with nothing between them — one boundary away — while B
    sits closer to the east. Ranking by distance calls C "second nearest" and
    misses that the estimate only slipped across a single line.
    """
    seeds = _latlon_seeds([(40.0, -100.0), (40.0, -99.0), (41.2, -100.0)])
    c = seed_crossing_matrix(seeds)
    d = pairwise_km(
        seeds.seed_lat.to_numpy(float), seeds.seed_lon.to_numpy(float)
    )
    assert d[0, 1] < d[0, 2]          # B is nearer to A than C is
    assert c[0, 2] == 1               # ...but C is one boundary away
    assert c[0, 1] == 1


def test_outer_seeds_around_a_centre_are_not_adjacent_to_each_other():
    """The failure mode that ruled out the convex-hull triangulation.

    Five seeds in a small cap: a centre with four around it. Walking the path
    from north to south passes through the centre cell, so they are two
    boundaries apart, and the correctly projected 2-D Delaunay agrees — it gives
    the north seed degree 3 (centre, east, west).

    A removed `_delaunay_degree` gave it 4, counting *south* as a neighbour: the
    hull of the unit vectors triangulates the whole sphere, so it had to close
    around the far side and joined the outer seeds across the empty hemisphere.
    This is the controlled case that retired that column.
    """
    seeds = _latlon_seeds(
        [
            (40.0, -100.0),   # centre
            (42.0, -100.0),   # N
            (38.0, -100.0),   # S
            (40.0, -97.4),    # E
            (40.0, -102.6),   # W
        ]
    )
    c = seed_crossing_matrix(seeds)
    assert (c[0, 1:] == 1).all()      # the centre borders all four
    assert c[1, 2] == 2               # N to S goes through the centre
    assert c[3, 4] == 2               # E to W likewise
    # The walk gives the north seed degree 3 — centre, east and west — which is
    # exactly what the correctly projected 2-D Delaunay gives. The hull's 4 was
    # the outlier, not the walk's answer.
    assert (c == 1).sum(axis=1)[1] == 3
    assert sorted(np.flatnonzero(c[1] == 1)) == [0, 3, 4]


def test_class_adjacency_degree_is_emitted_and_matches_the_walk():
    """`seeds.csv`'s per-seed count and `confusion`'s per-pair column are one walk.

    Two numbers for one relation is exactly how the retired `delaunay_degree`
    came to disagree with everything else, so the degree is derived from the same
    matrix rather than computed a second way.
    """
    space = build_answer_space(
        pd.DataFrame(
            {
                "target_id": [f"t{i}" for i in range(4)],
                "target_lat": [41.97, 37.46, 40.70, 29.76],
                "target_lon": [-87.90, -121.92, -74.00, -95.36],
            }
        )
    )
    assert "delaunay_degree" not in space.seeds.columns
    c = seed_crossing_matrix(space.seeds)
    assert space.seeds["class_adjacency_degree"].tolist() == (c == 1).sum(axis=1).tolist()
    assert space.meta["class_adjacency_degree"]["n"] == space.n_seeds
    # §7.4's edge-length distribution: one entry per adjacent pair.
    assert space.meta["adjacency_edge_km"]["n"] == int((np.triu(c == 1, k=1)).sum())


def test_every_seed_borders_at_least_one_other():
    """A class no answer can be confused with would be unreachable by error alone;
    on a connected seed set every cell has a neighbour."""
    space = build_answer_space(
        pd.DataFrame(
            {
                "target_id": [f"t{i}" for i in range(5)],
                "target_lat": [41.97, 37.46, 40.70, 29.76, 47.60],
                "target_lon": [-87.90, -121.92, -74.00, -95.36, -122.33],
            }
        )
    )
    assert (space.seeds["class_adjacency_degree"] >= 1).all()
