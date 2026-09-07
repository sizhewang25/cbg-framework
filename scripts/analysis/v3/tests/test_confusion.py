"""Confusion geometry: density bins, neighbour ranks, and the boundary margin."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules.answer_space import pairwise_km
from scripts.analysis.v3.modules.confusion import (
    confusion_by_density,
    confusion_pairs,
    density_bins,
    geodesic_points,
    seed_crossing_matrix,
)


def _seeds(nearest):
    return pd.DataFrame(
        {
            "seed_id": range(len(nearest)),
            "grid_scheme": "h3",
            "grid_resolution": 4,
            "seed_lat": 40.0,
            "seed_lon": -80.0,
            "nearest_seed_km": nearest,
            "margin_km": [n / 2 for n in nearest],
        }
    )


def _mesh(matrix):
    ids = list(range(len(matrix)))
    return pd.DataFrame(matrix, index=pd.Index(ids, name="seed_id"), columns=ids)


def _scored(rows):
    """`(target_id, status, tg_seed, pred_seed, rank, e_tgt, e_tg_seed, e_pred_seed)`."""
    return pd.DataFrame(
        [
            dict(
                zip(
                    [
                        "target_id", "status", "tg_seed_id", "pred_seed_id",
                        "tg_seed_rank", "error_to_target_km",
                        "error_to_tg_seed_km", "error_to_pred_seed_km",
                    ],
                    r,
                )
            )
            for r in rows
        ]
    )


# ---- binning ----------------------------------------------------------------


def test_quantile_bins_keep_every_bucket_populated():
    v = pd.Series([10.0, 20, 30, 40, 50, 60, 70, 80, 90])
    bins, edges = density_bins(v, n_bins=3)
    assert len(edges) == 4
    assert sorted(bins.value_counts().tolist()) == [3, 3, 3]


def test_ties_collapse_bins_rather_than_raising():
    """Evenly spaced seeds genuinely yield fewer distinct quantiles than asked.

    That is a fact about the grid worth reporting, not a failure worth raising —
    `h3` at one resolution puts many seeds at exactly the same pitch.
    """
    bins, edges = density_bins(pd.Series([50.0] * 8), n_bins=3)
    assert (bins == 0).all()
    # Two identical edges, not one: callers label a bin with edges[b]/edges[b+1]
    # and a one-element list would make the degenerate case the only one needing
    # a special path — which is the case that actually occurs on as01 and as03.
    assert edges == [50.0, 50.0]


def test_bins_come_from_the_true_seed_not_the_predicted_one():
    """The question is whether crowding *caused* the mistake, so the bin must be
    assigned upstream of it. Two rows share a true seed and differ in prediction;
    both must land in the same bin."""
    seeds = _seeds([50.0, 400.0])
    scored = {
        "m": _scored(
            [
                ("t1", "SUCCESS", 0, 0, 0, 5.0, 6.0, 6.0),
                ("t2", "SUCCESS", 0, 1, 1, 500.0, 501.0, 20.0),
            ]
        )
    }
    out, _ = confusion_by_density(scored, seeds, ns=(1,), n_bins=2)
    assert len(out) == 1
    assert out.iloc[0]["n_targets"] == 2


def test_density_bin_accuracy_uses_the_fallback_policy():
    """A FALLBACK row is a failure even at rank 0 (§7.2) — the same rule
    `topn_accuracy.csv` applies, re-applied here rather than re-decided."""
    seeds = _seeds([100.0])
    scored = {
        "m": _scored(
            [
                ("t1", "SUCCESS", 0, 0, 0, 5.0, 6.0, 6.0),
                ("t2", "FALLBACK", 0, 0, 0, 5.0, 6.0, 6.0),
            ]
        )
    }
    out, _ = confusion_by_density(scored, seeds, ns=(1,), n_bins=1)
    assert out.iloc[0]["accuracy_top1"] == 0.5


# ---- where the wrong answers land -------------------------------------------


def test_neighbour_rank_counts_from_the_true_seed():
    """Rank 1 is the true seed's *nearest* neighbour, so the number reads as
    "nth nearest" — the diagonal self-distance is rank 0 and never appears."""
    seeds = _seeds([100.0, 100.0, 300.0])
    mesh = _mesh([[0, 100, 300], [100, 0, 250], [300, 250, 0]])
    scored = {
        "m": _scored(
            [
                ("t1", "SUCCESS", 0, 1, 1, 90.0, 95.0, 10.0),   # nearest neighbour
                ("t2", "SUCCESS", 0, 2, 2, 290.0, 295.0, 12.0),  # second nearest
            ]
        )
    }
    out = confusion_pairs(scored, seeds, mesh, top_n=1)
    assert out.set_index("target_id")["pred_seed_neighbour_rank"].to_dict() == {
        "t1": 1,
        "t2": 2,
    }
    assert out.set_index("target_id")["tg_to_pred_seed_km"].to_dict() == {
        "t1": 100.0,
        "t2": 300.0,
    }


def test_correct_rows_are_absent_and_fallbacks_are_present():
    """§7.2 makes a FALLBACK a failure, so it belongs in the confusion table with
    its predicted seed intact — where it landed is exactly what this file is for."""
    seeds = _seeds([100.0, 100.0])
    mesh = _mesh([[0, 100], [100, 0]])
    scored = {
        "m": _scored(
            [
                ("ok", "SUCCESS", 0, 0, 0, 5.0, 6.0, 6.0),
                ("fb", "FALLBACK", 0, 0, 0, 5.0, 6.0, 6.0),
                ("bad", "SUCCESS", 0, 1, 1, 90.0, 95.0, 10.0),
            ]
        )
    }
    out = confusion_pairs(scored, seeds, mesh, top_n=1)
    assert sorted(out["target_id"]) == ["bad", "fb"]


def test_boundary_margin_separates_a_tie_break_from_a_mislocation():
    """`error_to_tg_seed_km - error_to_pred_seed_km` near zero means the estimate
    sat almost equidistant from both classes; large means it was confidently in
    the wrong cell. The two are the same wrong answer in an accuracy column."""
    seeds = _seeds([100.0, 100.0])
    mesh = _mesh([[0, 100], [100, 0]])
    scored = {
        "m": _scored(
            [
                ("tie", "SUCCESS", 0, 1, 1, 52.0, 51.0, 49.0),
                ("gross", "SUCCESS", 0, 1, 1, 95.0, 98.0, 3.0),
            ]
        )
    }
    out = confusion_pairs(scored, seeds, mesh, top_n=1).set_index("target_id")
    assert out.loc["tie", "boundary_margin_km"] == pytest.approx(2.0)
    assert out.loc["gross", "boundary_margin_km"] == pytest.approx(95.0)


def test_top_n_widens_what_counts_as_correct():
    seeds = _seeds([100.0, 100.0, 300.0])
    mesh = _mesh([[0, 100, 300], [100, 0, 250], [300, 250, 0]])
    scored = {
        "m": _scored(
            [
                ("near", "SUCCESS", 0, 1, 1, 90.0, 95.0, 10.0),
                ("far", "SUCCESS", 0, 2, 2, 290.0, 295.0, 12.0),
            ]
        )
    }
    assert len(confusion_pairs(scored, seeds, mesh, top_n=1)) == 2
    assert confusion_pairs(scored, seeds, mesh, top_n=2)["target_id"].tolist() == ["far"]


def test_a_method_with_no_mistakes_contributes_no_rows():
    seeds = _seeds([100.0])
    mesh = _mesh([[0]])
    scored = {"m": _scored([("t1", "SUCCESS", 0, 0, 0, 5.0, 6.0, 6.0)])}
    out = confusion_pairs(scored, seeds, mesh, top_n=1)
    assert out.empty
    # The schema still stands, so a downstream concat cannot lose columns.
    assert "pred_seed_neighbour_rank" in out.columns


def test_a_missing_prediction_keeps_the_row_with_null_geometry():
    """An ERROR row has no coordinate, so its neighbour rank is undefined — but
    dropping it would shrink the denominator the failure rate is taken over."""
    seeds = _seeds([100.0])
    mesh = _mesh([[0]])
    scored = {"m": _scored([("t1", "ERROR", 0, -1, -1, np.nan, np.nan, np.nan)])}
    out = confusion_pairs(scored, seeds, mesh, top_n=1)
    assert len(out) == 1
    assert out.iloc[0]["pred_seed_neighbour_rank"] == -1
    assert np.isnan(out.iloc[0]["boundary_margin_km"])


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
    boundaries apart. `answer_space._delaunay_degree` takes the hull of the unit
    vectors, which triangulates the whole sphere and therefore joins the four
    outer seeds to each other straight across the far side — reporting a degree
    for them that no amount of real geometry supports.
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

    from scripts.analysis.v3.modules.answer_space import _delaunay_degree

    hull_degree = _delaunay_degree(
        seeds.seed_lat.to_numpy(float), seeds.seed_lon.to_numpy(float)
    )
    walked_degree = (c == 1).sum(axis=1)
    assert hull_degree[1] > walked_degree[1]


def test_seeds_crossed_lands_in_the_pairs_table():
    seeds = _latlon_seeds([(38.0, -100.0), (40.0, -100.0), (42.0, -100.0)])
    mesh = pd.DataFrame(
        pairwise_km(seeds.seed_lat.to_numpy(float), seeds.seed_lon.to_numpy(float)),
        index=pd.Index([0, 1, 2], name="seed_id"),
        columns=[0, 1, 2],
    )
    scored = {
        "m": _scored(
            [
                ("near", "SUCCESS", 0, 1, 1, 90.0, 95.0, 10.0),
                ("far", "SUCCESS", 0, 2, 2, 290.0, 295.0, 12.0),
            ]
        )
    }
    out = confusion_pairs(scored, seeds, mesh, top_n=1).set_index("target_id")
    assert out.loc["near", "seeds_crossed"] == 1
    assert out.loc["far", "seeds_crossed"] == 2


def test_a_missing_prediction_has_no_crossing_count():
    seeds = _latlon_seeds([(40.0, -100.0), (41.0, -100.0)])
    mesh = pd.DataFrame(
        pairwise_km(seeds.seed_lat.to_numpy(float), seeds.seed_lon.to_numpy(float)),
        index=pd.Index([0, 1], name="seed_id"),
        columns=[0, 1],
    )
    scored = {"m": _scored([("t1", "ERROR", 0, -1, -1, np.nan, np.nan, np.nan)])}
    out = confusion_pairs(scored, seeds, mesh, top_n=1)
    assert out.iloc[0]["seeds_crossed"] == -1
