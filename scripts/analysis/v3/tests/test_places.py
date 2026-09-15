"""Region ids and cluster-aware statistics.

The invariant worth pinning is that a *rate* and its *effective sample size*
come apart. Twenty replicas of one coordinate make a target-level rate look
twenty times better resourced than it is, and nothing in the number itself says
so. These tests fix the three places that gap can be reintroduced: id
assignment must not depend on row order, the clustered interval must widen with
the cluster count rather than the row count, and the fold statistic must count
distinct folds per region rather than per target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.v3.modules.places import (
    REGION_COL,
    assign_region_ids,
    fold_region_overlap,
    region_diagnostics,
    region_level_accuracy,
)

CHI = (41.9742, -87.9073)
SJC = (37.4675, -121.9215)
NYC = (40.7085, -74.0095)


def _replicated(coords, per_region):
    """`per_region` IP replicas at each coordinate, in a deliberately odd order."""
    rows = []
    for i, (lat, lon) in enumerate(coords):
        for j in range(per_region):
            rows.append({"target_id": f"tg-{i}-{j}", "target_lat": lat, "target_lon": lon})
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=7).reset_index(drop=True)


def test_ids_are_stable_under_row_order():
    """Sorted-coordinate assignment, not order of first appearance.

    Two frames holding the same targets must agree on ids, or two artifacts
    built from differently-sorted inputs would disagree about which region is
    which while both looking self-consistent.
    """
    df = _replicated([CHI, SJC, NYC], 4)
    a = assign_region_ids(df)
    b = assign_region_ids(df.sort_values("target_id").reset_index(drop=True))

    key_a = dict(zip(df["target_id"], a))
    shuffled = df.sort_values("target_id").reset_index(drop=True)
    key_b = dict(zip(shuffled["target_id"], b))
    assert key_a == key_b


def test_replicas_collapse_and_missing_coords_stay_in_the_denominator():
    df = _replicated([CHI, SJC], 5)
    df.loc[len(df)] = {"target_id": "tg-nan", "target_lat": np.nan, "target_lon": 1.0}

    ids = assign_region_ids(df)
    assert set(ids[ids >= 0]) == {0, 1}
    assert (ids == -1).sum() == 1
    # The row is kept, not dropped: a rate's denominator must not change
    # silently because a coordinate was missing.
    assert len(ids) == len(df)


def test_diagnostics_report_rounding_stability():
    """Exactly-equal replicas are stable; jittered ones announce themselves."""
    exact = region_diagnostics(_replicated([CHI, SJC], 6))
    assert exact["n_regions"] == 2
    assert exact["region_count_is_rounding_stable"] is True
    assert exact["accuracy_quantum"] == 0.5

    jittered = pd.DataFrame(
        {
            "target_id": ["a", "b"],
            "target_lat": [CHI[0], CHI[0] + 0.0002],
            "target_lon": [CHI[1], CHI[1]],
        }
    )
    d = region_diagnostics(jittered)
    # Two sites at 6 dp, one at 2 dp -- the count depends on the cutoff, which
    # is the case the flag exists to surface.
    assert d["region_count_is_rounding_stable"] is False
    assert d["n_distinct_by_decimals"][2] < d["n_distinct_by_decimals"][6]


def test_clustered_interval_tracks_clusters_not_rows():
    """Replicating every target 20x must not shrink the interval.

    This is the whole point of the module. A binomial interval over rows would
    narrow by ~sqrt(20); the clustered one must not move materially, because no
    new independent information was added.
    """
    coords = [(40.0 + i, -100.0) for i in range(12)]
    correct = [1] * 8 + [0] * 4

    def _frame(per_region):
        df = _replicated(coords, per_region)
        df[REGION_COL] = assign_region_ids(df)
        df["correct"] = df[REGION_COL].map(dict(enumerate(correct)))
        return df

    thin = region_level_accuracy(_frame(1), correct_col="correct")
    fat = region_level_accuracy(_frame(20), correct_col="correct")

    assert thin["n_regions"] == fat["n_regions"] == 12
    assert thin["n_rows"] == 12 and fat["n_rows"] == 240
    assert thin["region_accuracy"] == fat["region_accuracy"]

    width_thin = thin["ci_high"] - thin["ci_low"]
    width_fat = fat["ci_high"] - fat["ci_low"]
    assert abs(width_thin - width_fat) < 1e-9
    # 12 clusters cannot support an asymptotic interval, and the dict says so.
    assert fat["ci_is_underpowered"] is True


def test_region_and_target_rates_diverge_on_unequal_replica_counts():
    """The two estimators answer different questions; both are reported."""
    df = pd.DataFrame(
        {
            "target_id": [f"t{i}" for i in range(11)],
            "target_lat": [40.0] * 10 + [41.0],
            "target_lon": [-100.0] * 11,
            "correct": [1] * 10 + [0],
        }
    )
    df[REGION_COL] = assign_region_ids(df)
    out = region_level_accuracy(df, correct_col="correct")

    assert out["n_regions"] == 2
    assert out["target_accuracy"] == 10 / 11       # replica-weighted
    assert out["region_accuracy"] == 0.5           # one site right, one wrong
    assert out["region_homogeneity"] == 1.0


def test_region_homogeneity_catches_replicas_that_disagree():
    df = pd.DataFrame(
        {
            "target_id": ["a", "b", "c", "d"],
            "target_lat": [40.0, 40.0, 41.0, 41.0],
            "target_lon": [-100.0, -100.0, -100.0, -100.0],
            "correct": [1, 0, 1, 1],
        }
    )
    df[REGION_COL] = assign_region_ids(df)
    out = region_level_accuracy(df, correct_col="correct")
    # One of two regions splits, so the region rate is a mean of fractions.
    assert out["region_homogeneity"] == 0.5
    assert out["region_accuracy"] == 0.75


def test_fold_overlap_counts_distinct_folds_per_region():
    """Complete leakage and clean separation are both representable."""
    coords = [(40.0 + i, -100.0) for i in range(3)]
    df = _replicated(coords, 5)
    df[REGION_COL] = assign_region_ids(df)

    # One replica of every region into every fold -- keyed on a within-region
    # counter, since `_replicated` deliberately returns the rows shuffled.
    leaky = df.copy()
    leaky["fold"] = leaky.groupby(REGION_COL).cumcount() % 5
    out = fold_region_overlap(leaky)
    assert out["n_folds"] == 5
    assert out["share_regions_in_all_folds"] == 1.0
    assert out["folds_per_region_min"] == 5

    clean = df.copy()
    clean["fold"] = clean[REGION_COL]
    out = fold_region_overlap(clean)
    assert out["share_regions_in_all_folds"] == 0.0
    assert out["share_regions_in_one_fold"] == 1.0
    assert out["folds_per_region_max"] == 1


def test_empty_inputs_return_a_shaped_dict_rather_than_raising():
    """This layer reports, it does not raise -- see the module docstring."""
    empty = pd.DataFrame({REGION_COL: [], "correct": [], "fold": []})
    acc = region_level_accuracy(empty, correct_col="correct")
    assert acc["n_rows"] == 0 and np.isnan(acc["target_accuracy"])
    assert fold_region_overlap(empty)["n_regions"] == 0
