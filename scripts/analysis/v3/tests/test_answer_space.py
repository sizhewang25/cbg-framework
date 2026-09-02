"""Answer-space construction invariants (paper §7.3/§7.4).

Almost everything here is parametrized over **both grids**. That is the point of
the `grid.Grid` abstraction: steps 2 and 3 of the construction (seeds at target
centroids, nearest-seed labelling) are pure spherical geometry, so every
invariant they guarantee has to hold whichever tessellation did the merging. A
test that passes on one grid and fails on the other marks a place where grid
detail leaked out of the quantizer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules.answer_space import (
    build_answer_space,
    load_answer_space,
    pairwise_km,
    spherical_centroid,
)


@pytest.fixture(params=("h3", "healpix"))
def grid(request):
    """Every construction invariant, on both tessellations."""
    return request.param


def _targets(coords: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_id": [f"tg-{i}" for i in range(len(coords))],
            "target_lat": [c[0] for c in coords],
            "target_lon": [c[1] for c in coords],
        }
    )


def test_colocated_targets_collapse_to_one_seed(grid):
    """The grid's whole job: merge points close enough to count as one place."""
    space = build_answer_space(_targets([(41.9742, -87.9073)] * 5), grid=grid)
    assert space.n_seeds == 1
    assert space.seeds.loc[0, "n_targets"] == 5
    assert space.seeds.loc[0, "intra_seed_spread_km"] == 0.0


def test_distant_targets_get_distinct_seeds(grid):
    space = build_answer_space(
        _targets([(41.97, -87.90), (37.46, -121.92), (40.71, -74.01)]), grid=grid
    )
    assert space.n_seeds == 3
    assert set(space.assignments["seed_id"]) == {0, 1, 2}


def test_seed_is_centroid_of_its_members(grid):
    space = build_answer_space(_targets([(41.90, -87.90), (41.91, -87.91)]), grid=grid)
    assert space.n_seeds == 1
    lat, lon = spherical_centroid([41.90, 41.91], [-87.90, -87.91])
    assert space.seeds.loc[0, "centroid_lat"] == pytest.approx(lat)
    assert space.seeds.loc[0, "centroid_lon"] == pytest.approx(lon)


def test_every_target_is_assigned_exactly_once(grid):
    coords = [(25 + i * 0.7, -120 + i * 1.3) for i in range(40)]
    space = build_answer_space(_targets(coords), grid=grid)
    assert len(space.assignments) == 40
    assert space.assignments["target_id"].is_unique
    assert space.assignments["seed_id"].isin(space.seeds["seed_id"]).all()


def test_seed_ids_are_contiguous_and_order_independent(grid):
    """seed_id must not depend on input row order, or scoring is unstable."""
    coords = [(41.97, -87.90), (37.46, -121.92), (40.71, -74.01)]
    a = build_answer_space(_targets(coords), grid=grid)
    shuffled = _targets(coords).iloc[::-1].reset_index(drop=True)
    b = build_answer_space(shuffled, grid=grid)
    assert list(a.seeds["seed_id"]) == list(range(a.n_seeds))
    pd.testing.assert_frame_equal(
        a.seeds.drop(columns=["seed_id"]).reset_index(drop=True),
        b.seeds.drop(columns=["seed_id"]).reset_index(drop=True),
    )


def test_margin_is_half_the_nearest_seed_distance(grid):
    """§7.4: the class boundary bisects the geodesic joining two seeds."""
    space = build_answer_space(_targets([(41.97, -87.90), (40.71, -74.01)]), grid=grid)
    assert space.seeds["margin_km"].to_numpy() == pytest.approx(
        space.seeds["nearest_seed_km"].to_numpy() / 2.0, rel=1e-6
    )


def test_intra_seed_spread_is_max_pairwise_within_the_seed(grid):
    """§7.4 calls this the floor under any error-distance figure."""
    coords = [(41.900, -87.900), (41.930, -87.930), (41.910, -87.905)]
    space = build_answer_space(_targets(coords), grid=grid)
    assert space.n_seeds == 1
    d = pairwise_km([c[0] for c in coords], [c[1] for c in coords])
    assert space.seeds.loc[0, "intra_seed_spread_km"] == pytest.approx(
        d.max(), abs=1e-3
    )


def test_duplicate_target_id_is_rejected(grid):
    t = _targets([(41.97, -87.90), (40.71, -74.01)])
    t.loc[1, "target_id"] = t.loc[0, "target_id"]
    with pytest.raises(ValueError, match="duplicate target_id"):
        build_answer_space(t, grid=grid)


def test_empty_targets_is_rejected(grid):
    with pytest.raises(ValueError, match="empty"):
        build_answer_space(_targets([]), grid=grid)


@pytest.mark.parametrize(
    "grid_name,fine,coarse", [("h3", 5, 2), ("healpix", 128, 16)]
)
def test_finer_resolution_cannot_reduce_seed_count(grid_name, fine, coarse):
    coords = [(30 + i * 0.4, -100 + i * 0.4) for i in range(30)]
    t = _targets(coords)
    n_coarse = build_answer_space(t, grid=grid_name, resolution=coarse).n_seeds
    n_fine = build_answer_space(t, grid=grid_name, resolution=fine).n_seeds
    assert n_fine >= n_coarse


def test_resolution_defaults_to_the_grids_own(grid):
    """A grid and a resolution from a *different* grid must never be pairable.

    `nside=128` and `res=4` are both "the default", but only one is valid per
    grid, so the default has to come from the grid rather than from a constant.
    """
    from scripts.analysis.v3.modules.grid import get_grid

    space = build_answer_space(_targets([(41.97, -87.90)]), grid=grid)
    assert space.meta["grid"]["resolution"] == get_grid(grid).DEFAULT_RESOLUTION
    assert space.meta["grid"]["scheme"] == grid


def test_roundtrip_through_disk(tmp_path, grid):
    """`cell_id` must survive CSV untyped storage on both grids.

    HEALPix ids are int64 and H3's are hex strings; pandas infers per column, so
    without `Grid.coerce_cell_ids` one of the two comes back as the wrong dtype
    and stops matching `assignments`.
    """
    space = build_answer_space(
        _targets([(41.97, -87.90), (37.46, -121.92), (40.71, -74.01)]), grid=grid
    )
    space.write(tmp_path)
    back = load_answer_space(tmp_path)
    assert back.n_seeds == space.n_seeds
    pd.testing.assert_frame_equal(back.seeds, space.seeds)
    pd.testing.assert_frame_equal(back.assignments, space.assignments)
    assert back.meta["grid"]["resolution"] == space.meta["grid"]["resolution"]
    assert back.meta["grid"]["scheme"] == space.meta["grid"]["scheme"]


def test_mesh_is_symmetric_with_zero_diagonal(grid):
    space = build_answer_space(
        _targets([(41.97, -87.90), (37.46, -121.92), (40.71, -74.01)]), grid=grid
    )
    m = space.seed_mesh_km.to_numpy()
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 0.0)


def test_hierarchy_reports_only_the_grid_built_and_coarser(grid):
    """Regression: the diagnostic used to describe grids that were never built.

    `occupied_cells_by_resolution` walked a fixed hierarchy regardless of the
    resolution in use, so at the coarsest rung every count came from *finer*
    grids than the answer space itself.
    """
    from scripts.analysis.v3.modules.grid import get_grid

    g = get_grid(grid)
    coarsest = g.HIERARCHY[-1]
    space = build_answer_space(
        _targets([(41.97, -87.90), (37.46, -121.92)]), grid=grid, resolution=coarsest
    )
    reported = [int(k) for k in space.meta["occupied_cells_by_resolution"]]
    assert reported == [coarsest]


def test_spherical_centroid_handles_dateline():
    """Averaging lon directly would put this at 0°, halfway around the world."""
    lat, lon = spherical_centroid([0.0, 0.0], [179.0, -179.0])
    assert lat == pytest.approx(0.0, abs=1e-9)
    assert abs(lon) == pytest.approx(180.0, abs=1e-6)


def test_spherical_centroid_of_identical_points_is_that_point():
    lat, lon = spherical_centroid([41.9742] * 3, [-87.9073] * 3)
    assert lat == pytest.approx(41.9742, abs=1e-9)
    assert lon == pytest.approx(-87.9073, abs=1e-9)
