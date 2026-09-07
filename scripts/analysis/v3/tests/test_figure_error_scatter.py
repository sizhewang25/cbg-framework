"""`plot-error-vs-cells` — the y axis, the two disagreement regions, the axes shared with the CDF.

The figure exists to count where accuracy and error distance disagree, so the
tests that matter are the ones guarding the counting: that the `y == 0` column
is present at all (it is absent from `confusion_pairs.csv`, which is why this
recomputes), that the margin comparison is not off by an endpoint, and that the
error axis is the same one `plot-error-cdf` uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules import figure_error_cdf as CDF
from scripts.analysis.v3.modules import figure_error_scatter as S
from scripts.analysis.v3.modules import io
from scripts.analysis.v3.modules.classify import SHORTEST_PING


def _points(rows):
    """`[(method, error_km, seeds_crossed), ...]` -> a `crossings_per_target` frame."""
    df = pd.DataFrame(rows, columns=["method", "error_km", "seeds_crossed"])
    df["method_label"] = df["method"]
    df["target_id"] = [f"tg-{i}" for i in range(len(df))]
    df["level"] = np.clip(df["seeds_crossed"], 0, S.MAX_LEVEL)
    return df


# ---------------------------------------------------------------------------
# the y axis
# ---------------------------------------------------------------------------


def test_the_correct_cell_is_level_zero():
    """The column `confusion_pairs.csv` cannot supply — it keeps wrong rows only."""
    p = _points([("m", 10.0, 0)])
    assert int(p.loc[0, "level"]) == 0


def test_levels_past_the_top_fold_into_the_bucket():
    p = _points([("m", 1.0, 3), ("m", 1.0, 4), ("m", 1.0, 9)])
    assert p["level"].tolist() == [S.MAX_LEVEL] * 3


def test_the_top_level_is_labelled_as_a_bucket_only_when_it_is_one():
    assert S._level_labels(S.MAX_LEVEL)[-1] == str(S.MAX_LEVEL)
    assert S._level_labels(S.MAX_LEVEL + 1)[-1] == f"{S.MAX_LEVEL}+"


def test_the_zero_level_is_named_not_just_numbered():
    assert "correct cell" in S._level_labels(4)[0]


# ---------------------------------------------------------------------------
# the two disagreement regions
# ---------------------------------------------------------------------------


def test_right_cell_beyond_the_margin_is_counted():
    p = _points([("m", 500.0, 0), ("m", 10.0, 0)])
    row = S.summarise(p, margin_km=100.0).iloc[0]
    assert int(row["n_right_cell_beyond_margin"]) == 1


def test_wrong_cell_within_the_margin_is_counted():
    p = _points([("m", 10.0, 1), ("m", 900.0, 2)])
    s = S.summarise(p, margin_km=100.0)
    assert int(s["n_wrong_cell_within_margin"].iloc[0]) == 1


def test_a_point_exactly_on_the_margin_is_inside_it():
    """`>` for the far side and `<=` for the near side, so the two are disjoint
    and every point falls in exactly one of them or neither."""
    p = _points([("m", 100.0, 0), ("m", 100.0, 1)])
    row = S.summarise(p, margin_km=100.0).iloc[0]
    assert int(row["n_right_cell_beyond_margin"]) == 0
    assert int(row["n_wrong_cell_within_margin"]) == 1


def test_the_two_regions_never_double_count_a_point():
    rng = np.random.default_rng(0)
    p = _points(
        [("m", float(e), int(c)) for e, c in zip(rng.uniform(1, 3000, 200),
                                                 rng.integers(0, 5, 200))]
    )
    row = S.summarise(p, margin_km=150.0).iloc[0]
    far = (p["level"] == 0) & (p["error_km"] > 150.0)
    near = (p["level"] >= 1) & (p["error_km"] <= 150.0)
    assert not (far & near).any()
    assert int(row["n_right_cell_beyond_margin"]) == int(far.sum())
    assert int(row["n_wrong_cell_within_margin"]) == int(near.sum())


def test_the_counts_are_per_method_not_pooled():
    p = _points([("a", 500.0, 0), ("b", 10.0, 0)])
    s = S.summarise(p, margin_km=100.0).set_index("method")
    assert int(s.loc["a", "n_right_cell_beyond_margin"]) == 1
    assert int(s.loc["b", "n_right_cell_beyond_margin"]) == 0


# ---------------------------------------------------------------------------
# the summary
# ---------------------------------------------------------------------------


def test_shares_sum_to_one_per_method():
    p = _points([("m", 1.0, 0), ("m", 2.0, 1), ("m", 3.0, 1), ("m", 4.0, 2)])
    s = S.summarise(p, margin_km=100.0)
    assert s["share"].sum() == pytest.approx(1.0)
    assert s["n"].sum() == 4


def test_each_level_reports_its_own_error_percentiles():
    p = _points([("m", 10.0, 0), ("m", 20.0, 0), ("m", 900.0, 2)])
    s = S.summarise(p, margin_km=100.0).set_index("level")
    assert s.loc[0, "error_km_p50"] == pytest.approx(15.0)
    assert s.loc[2, "error_km_p50"] == pytest.approx(900.0)


# ---------------------------------------------------------------------------
# row policy, shared with the CDF
# ---------------------------------------------------------------------------


def test_the_error_column_and_x_range_are_the_cdfs():
    """Both figures put error distance on x; a second set of bounds here would
    be free to drift and the two could no longer be read against each other."""
    assert S.ERROR_COLUMN is CDF.ERROR_COLUMN
    assert S.X_MIN_KM == CDF.X_MIN_KM
    assert S.DEFAULT_X_MAX_KM == CDF.DEFAULT_X_MAX_KM


def test_fallbacks_are_excluded_by_the_shared_predicate():
    df = pd.DataFrame({"status": ["SUCCESS", "FALLBACK", "BASELINE"]})
    assert io.solved_mask(df).tolist() == [True, False, False]


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def test_the_jitter_is_reproducible():
    """Two renders of one dataset must not look like two datasets."""
    a = np.random.default_rng(S.JITTER_SEED).uniform(-S.JITTER, S.JITTER, 50)
    b = np.random.default_rng(S.JITTER_SEED).uniform(-S.JITTER, S.JITTER, 50)
    assert np.array_equal(a, b)


def test_the_figure_renders_to_a_non_empty_png(tmp_path):
    p = _points(
        [(SHORTEST_PING, 10.0, 0), (SHORTEST_PING, 800.0, 2),
         ("octant_cbg_hull", 3.0, 0), ("octant_cbg_hull", 400.0, 1)]
    )
    s = S.summarise(p, margin_km=160.0)
    out = S.plot_error_vs_cells(
        p, s, tmp_path / "scatter.png",
        title="t", subtitle="s", margin_km=160.0, max_crossings=2,
    )
    assert out.exists() and out.stat().st_size > 5_000


def test_a_single_method_renders_without_empty_panels_erroring(tmp_path):
    p = _points([("m", 10.0, 0)])
    s = S.summarise(p, margin_km=100.0)
    out = S.plot_error_vs_cells(
        p, s, tmp_path / "one.png",
        title="t", subtitle="s", margin_km=100.0, max_crossings=0,
    )
    assert out.exists()
