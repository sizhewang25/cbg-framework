"""Answer-space construction invariants (paper §7.3/§7.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules import healpix as hx
from scripts.analysis.v3.modules.answer_space import (
    build_answer_space,
    load_answer_space,
    pairwise_km,
)


def _targets(coords: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_id": [f"tg-{i}" for i in range(len(coords))],
            "target_lat": [c[0] for c in coords],
            "target_lon": [c[1] for c in coords],
        }
    )


def test_colocated_targets_collapse_to_one_seed():
    """The grid's whole job: merge points close enough to count as one place."""
    space = build_answer_space(_targets([(41.9742, -87.9073)] * 5))
    assert space.n_seeds == 1
    assert space.seeds.loc[0, "n_targets"] == 5
    assert space.seeds.loc[0, "intra_seed_spread_km"] == 0.0


def test_distant_targets_get_distinct_seeds():
    space = build_answer_space(
        _targets([(41.97, -87.90), (37.46, -121.92), (40.71, -74.01)])
    )
    assert space.n_seeds == 3
    assert set(space.assignments["seed_id"]) == {0, 1, 2}


def test_seed_is_centroid_of_its_members():
    space = build_answer_space(_targets([(41.90, -87.90), (41.91, -87.91)]))
    assert space.n_seeds == 1
    lat, lon = hx.spherical_centroid([41.90, 41.91], [-87.90, -87.91])
    assert space.seeds.loc[0, "centroid_lat"] == pytest.approx(lat)
    assert space.seeds.loc[0, "centroid_lon"] == pytest.approx(lon)


def test_every_target_is_assigned_exactly_once():
    coords = [(25 + i * 0.7, -120 + i * 1.3) for i in range(40)]
    space = build_answer_space(_targets(coords))
    assert len(space.assignments) == 40
    assert space.assignments["target_id"].is_unique
    assert space.assignments["seed_id"].isin(space.seeds["seed_id"]).all()


def test_seed_ids_are_contiguous_and_order_independent():
    """seed_id must not depend on input row order, or scoring is unstable."""
    coords = [(41.97, -87.90), (37.46, -121.92), (40.71, -74.01)]
    a = build_answer_space(_targets(coords))
    shuffled = _targets(coords).iloc[::-1].reset_index(drop=True)
    b = build_answer_space(shuffled)
    assert list(a.seeds["seed_id"]) == list(range(a.n_seeds))
    pd.testing.assert_frame_equal(
        a.seeds.drop(columns=["seed_id"]).reset_index(drop=True),
        b.seeds.drop(columns=["seed_id"]).reset_index(drop=True),
    )


def test_margin_is_half_the_nearest_seed_distance():
    """§7.4: the class boundary bisects the geodesic joining two seeds."""
    space = build_answer_space(_targets([(41.97, -87.90), (40.71, -74.01)]))
    assert space.seeds["margin_km"].to_numpy() == pytest.approx(
        space.seeds["nearest_seed_km"].to_numpy() / 2.0, rel=1e-6
    )


def test_intra_seed_spread_is_max_pairwise_within_the_seed():
    """§7.4 calls this the floor under any error-distance figure."""
    coords = [(41.900, -87.900), (41.930, -87.930), (41.910, -87.905)]
    space = build_answer_space(_targets(coords))
    assert space.n_seeds == 1
    d = pairwise_km([c[0] for c in coords], [c[1] for c in coords])
    assert space.seeds.loc[0, "intra_seed_spread_km"] == pytest.approx(
        d.max(), abs=1e-3
    )


def test_duplicate_target_id_is_rejected():
    t = _targets([(41.97, -87.90), (40.71, -74.01)])
    t.loc[1, "target_id"] = t.loc[0, "target_id"]
    with pytest.raises(ValueError, match="duplicate target_id"):
        build_answer_space(t)


def test_empty_targets_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        build_answer_space(_targets([]))


def test_finer_nside_cannot_reduce_seed_count():
    coords = [(30 + i * 0.4, -100 + i * 0.4) for i in range(30)]
    t = _targets(coords)
    coarse = build_answer_space(t, nside=16).n_seeds
    fine = build_answer_space(t, nside=128).n_seeds
    assert fine >= coarse


def test_roundtrip_through_disk(tmp_path):
    space = build_answer_space(
        _targets([(41.97, -87.90), (37.46, -121.92), (40.71, -74.01)])
    )
    space.write(tmp_path)
    back = load_answer_space(tmp_path)
    assert back.n_seeds == space.n_seeds
    pd.testing.assert_frame_equal(back.seeds, space.seeds)
    pd.testing.assert_frame_equal(back.assignments, space.assignments)
    assert back.meta["grid"]["nside"] == space.meta["grid"]["nside"]


def test_mesh_is_symmetric_with_zero_diagonal():
    space = build_answer_space(
        _targets([(41.97, -87.90), (37.46, -121.92), (40.71, -74.01)])
    )
    m = space.seed_mesh_km.to_numpy()
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 0.0)
