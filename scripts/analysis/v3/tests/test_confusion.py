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
)


def _latlon_seeds(points):
    """Seeds at explicit coordinates, with the margins the crossing walk steps by.

    `seed_crossing_matrix` lives in `answer_space` and is exercised there
    (`test_seed_adjacency.py`); these two only check that its output reaches
    `confusion_pairs.csv` intact.
    """
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


# ---- seeds_crossed in the pairs table ---------------------------------------


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
