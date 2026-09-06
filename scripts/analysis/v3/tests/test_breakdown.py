"""Crossing accuracy with the proximity strata: partitions, phi, the tautology."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules.breakdown import (
    TAUTOLOGICAL_CELL,
    _phi,
    breakdown_by_flag,
    breakdown_by_taxonomy,
    taxonomy_of,
)
from scripts.analysis.v3.modules.classify import SHORTEST_PING
from scripts.analysis.v3.modules.proximity import FLAGS, TAXONOMY


def _labels(rows):
    """`(target_id, prox_vp, disc_vp, prox_sping, disc_sping)` -> a labels frame."""
    return pd.DataFrame(
        [dict(zip(["target_id", *FLAGS], r)) for r in rows]
    )


def _membership(targets, cols, *, n=1):
    return {n: pd.DataFrame(cols, index=pd.Index(targets, name="target_id"))}


def test_taxonomy_is_a_partition():
    labels = _labels(
        [
            ("t1", False, False, False, False),   # geometry_only
            ("t2", True, True, False, False),     # selection_miss
            ("t3", True, True, True, True),       # selection_hit
            ("t4", True, False, False, False),    # selection_miss
        ]
    )
    term = taxonomy_of(labels)
    assert term.tolist() == [
        TAXONOMY[0], TAXONOMY[1], TAXONOMY[2], TAXONOMY[1]
    ]
    assert set(term) <= set(TAXONOMY)


def test_taxonomy_counts_sum_to_the_target_count():
    """Three terms, every target in exactly one — the check that makes the table
    readable as a decomposition of the headline accuracy rather than three
    unrelated rates."""
    labels = _labels(
        [
            ("t1", False, False, False, False),
            ("t2", True, True, False, False),
            ("t3", True, True, True, True),
        ]
    )
    m = _membership(["t1", "t2", "t3"], {"m": [True, False, True]})
    out = breakdown_by_taxonomy(m, labels)
    assert out["n_targets"].sum() == 3
    # Shares are rounded to 4 dp at the artifact boundary, so three thirds sum
    # to 0.9999. The counts are the exact statement; the shares are presentation.
    assert out["share_of_targets"].sum() == pytest.approx(1.0, abs=1e-3)


def test_geometry_only_accuracy_is_not_forced_to_zero():
    """A method can be right where no VP was proximate — that is the point.

    Pins the reading the `geometry_only` rename exists to protect: the stratum
    is a ceiling on Shortest-Ping, not on a variant whose answer comes from
    multilateration. Measured on as02, Octant-Hull scores 0.975 here.
    """
    labels = _labels(
        [("t1", False, False, False, False), ("t2", False, False, False, False)]
    )
    m = _membership(["t1", "t2"], {"cbg": [True, True], SHORTEST_PING: [False, False]})
    out = breakdown_by_taxonomy(m, labels)
    geo = out[out.term == TAXONOMY[0]].set_index("method")
    assert geo.loc["cbg", "accuracy"] == 1.0
    assert geo.loc[SHORTEST_PING, "accuracy"] == 0.0


def test_the_tautological_cell_is_marked():
    labels = _labels(
        [("t1", True, True, True, True), ("t2", True, True, False, False)]
    )
    m = _membership(["t1", "t2"], {SHORTEST_PING: [True, False], "cbg": [True, True]})
    out = breakdown_by_flag(m, labels)
    marked = out[out.is_tautological]
    assert len(marked) == 1
    row = marked.iloc[0]
    assert (row.flag, row.method, row.top_n) == TAUTOLOGICAL_CELL
    # ...and it is exactly the perfect-separation cell the annotation warns about.
    assert row.accuracy_when_true == 1.0 and row.accuracy_when_false == 0.0
    assert row.phi == 1.0


def test_the_tautology_is_not_marked_at_top3():
    """The flag stays top-1, so the same cell at top-3 is a real measurement."""
    labels = _labels([("t1", True, True, True, True)])
    m = {3: pd.DataFrame({SHORTEST_PING: [True]}, index=pd.Index(["t1"], name="target_id"))}
    out = breakdown_by_flag(m, labels)
    assert not out["is_tautological"].any()


def test_a_constant_flag_reports_no_variance_not_a_zero_difference():
    """`has_proximate_vp` is genuinely constant on as01 and as03.

    A `rate_difference` of 0.0 would say "this flag does not matter"; NaN plus
    `zero_variance` says "there was nothing to compare", which is the true
    statement and a different one.
    """
    labels = _labels(
        [("t1", True, True, True, True), ("t2", True, True, True, True)]
    )
    m = _membership(["t1", "t2"], {"cbg": [True, False]})
    out = breakdown_by_flag(m, labels)
    row = out[out.flag == "has_proximate_vp"].iloc[0]
    assert row.zero_variance
    assert row.n_false == 0
    assert np.isnan(row.rate_difference)
    assert np.isnan(row.phi)
    # The populated side is still reported: the rate exists, the contrast does not.
    assert row.accuracy_when_true == 0.5


def test_phi_matches_pearson_on_the_booleans():
    """Phi is the Pearson correlation of two indicator vectors; pinned against
    numpy so the hand-rolled 2x2 form cannot drift from the definition."""
    rng = np.random.default_rng(7)
    x = rng.random(200) < 0.4
    y = (x & (rng.random(200) < 0.8)) | (~x & (rng.random(200) < 0.2))
    a = int((x & y).sum()); b = int((x & ~y).sum())
    c = int((~x & y).sum()); d = int((~x & ~y).sum())
    assert _phi(a, b, c, d) == pytest.approx(
        np.corrcoef(x.astype(float), y.astype(float))[0, 1]
    )


def test_all_four_flags_are_reported_per_method():
    labels = _labels([("t1", True, True, True, True), ("t2", True, False, False, False)])
    m = _membership(["t1", "t2"], {"a": [True, False], "b": [False, True]})
    out = breakdown_by_flag(m, labels)
    assert set(out["flag"]) == set(FLAGS)
    assert len(out) == len(FLAGS) * 2
