"""`plot-error-vs-cells` / `plot-error-vs-rank` — the denominator and the bands.

The figures exist so band 0 can be read as the method's accuracy, which makes
the share denominator the load-bearing detail: dividing by solved rows instead
of by every target would make band 0 disagree with `topn_accuracy.csv` on any
method that falls back. Most of these tests guard that, the rest guard the band
bookkeeping and that the two y modes stay distinguishable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules import figure_error_cdf as CDF
from scripts.analysis.v3.modules import figure_error_scatter as S
from scripts.analysis.v3.modules import io
from scripts.analysis.v3.modules.classify import SHORTEST_PING


def _points(rows, mode=S.CELLS):
    """`[(method, error_km, class_error), ...]` -> a `load_points` frame."""
    df = pd.DataFrame(rows, columns=["method", "error_km", "class_error"])
    df["method_label"] = df["method"]
    df["target_id"] = [f"tg-{i}" for i in range(len(df))]
    df["level"] = np.clip(df["class_error"], 0, mode.max_level)
    return df


# ---------------------------------------------------------------------------
# the share denominator — why this figure exists in this shape
# ---------------------------------------------------------------------------


def test_shares_divide_by_every_target_not_by_the_drawn_rows():
    """as02 Vanilla: 123 band-0 rows / 412 targets = 0.298 = its top-1 accuracy.

    Over its 337 solved rows the same count reads 0.365 and matches nothing.
    """
    p = _points([("m", 10.0, 0)] * 123 + [("m", 500.0, 1)] * 214)
    t = S.band_table(p, {"m": {"n_targets": 412}}, S.CELLS)
    assert t.loc[t["level"] == 0, "share"].iloc[0] == pytest.approx(0.2985, abs=5e-4)


def test_the_bands_sum_to_one_minus_the_fallback_share():
    p = _points([("m", 10.0, 0)] * 80 + [("m", 500.0, 1)] * 20)
    t = S.band_table(p, {"m": {"n_targets": 125}}, S.CELLS)
    assert t["share"].sum() == pytest.approx(0.8, abs=1e-3)  # 25/125 fell back


def test_a_method_with_no_fallbacks_sums_to_one():
    p = _points([("m", 10.0, 0)] * 50 + [("m", 500.0, 2)] * 50)
    t = S.band_table(p, {"m": {"n_targets": 100}}, S.CELLS)
    assert t["share"].sum() == pytest.approx(1.0, abs=1e-3)


def test_a_missing_count_falls_back_to_the_drawn_rows_rather_than_dividing_by_zero():
    p = _points([("m", 10.0, 0), ("m", 20.0, 1)])
    t = S.band_table(p, {}, S.CELLS)
    assert t["share"].sum() == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# cumulative share — the top-N reading on the rank mode
# ---------------------------------------------------------------------------


def test_cumulative_share_is_running_and_ends_at_the_total():
    p = _points(
        [("m", 1.0, 0)] * 40 + [("m", 2.0, 1)] * 30 + [("m", 3.0, 2)] * 30,
        mode=S.RANK,
    )
    t = S.band_table(p, {"m": {"n_targets": 100}}, S.RANK).set_index("level")
    assert t.loc[0, "cumulative_share"] == pytest.approx(0.40)
    assert t.loc[2, "cumulative_share"] == pytest.approx(1.00)
    assert t["cumulative_share"].is_monotonic_increasing


def test_cumulative_through_rank_two_is_the_top_three_reading():
    """`tg_seed_rank < N` is top-N, so rank <= 2 is top-3 by construction."""
    p = _points([("m", 1.0, r) for r in [0, 0, 1, 2, 3, 4]], mode=S.RANK)
    t = S.band_table(p, {"m": {"n_targets": 6}}, S.RANK).set_index("level")
    # stored at 4 decimals, matching every other rate in this layer
    assert t.loc[2, "cumulative_share"] == pytest.approx(4 / 6, abs=5e-5)


# ---------------------------------------------------------------------------
# bands
# ---------------------------------------------------------------------------


def test_every_band_gets_a_row_even_when_empty():
    """The share axis puts a tick on every band, so every band needs a row."""
    p = _points([("m", 1.0, 0)])
    t = S.band_table(p, {"m": {"n_targets": 1}}, S.CELLS)
    assert t["level"].tolist() == list(range(S.CELLS.max_level + 1))
    assert int(t.loc[t["level"] == 3, "n"].iloc[0]) == 0
    assert np.isnan(t.loc[t["level"] == 3, "error_km_p50"].iloc[0])


def test_values_past_the_top_band_fold_into_it():
    p = _points([("m", 1.0, 3), ("m", 1.0, 4), ("m", 1.0, 12)])
    assert p["level"].tolist() == [S.CELLS.max_level] * 3


def test_each_band_reports_its_own_median():
    p = _points([("m", 10.0, 0), ("m", 20.0, 0), ("m", 900.0, 2)])
    t = S.band_table(p, {"m": {"n_targets": 3}}, S.CELLS).set_index("level")
    assert t.loc[0, "error_km_p50"] == pytest.approx(15.0)
    assert t.loc[2, "error_km_p50"] == pytest.approx(900.0)


def test_counts_are_per_method():
    p = _points([("a", 1.0, 0), ("b", 1.0, 1)])
    t = S.band_table(p, {"a": {"n_targets": 1}, "b": {"n_targets": 1}}, S.CELLS)
    a = t[(t["method"] == "a") & (t["level"] == 0)]
    b = t[(t["method"] == "b") & (t["level"] == 0)]
    assert int(a["n"].iloc[0]) == 1
    assert int(b["n"].iloc[0]) == 0


# ---------------------------------------------------------------------------
# the two y modes stay distinguishable
# ---------------------------------------------------------------------------


def test_cells_walks_the_answer_space_and_rank_reads_a_column():
    assert S.CELLS.column is None
    assert S.RANK.column == "tg_seed_rank"


def test_the_modes_have_different_depths_matching_their_observed_ranges():
    """Crossings reach 4 at h3-4, rank reaches 7 — hence 4 bands versus 6."""
    assert S.CELLS.max_level == 3
    assert S.RANK.max_level == 5


def test_both_modes_are_registered_under_their_key():
    assert S.MODES["cells"] is S.CELLS
    assert S.MODES["rank"] is S.RANK


def test_each_mode_names_level_zero_for_what_it_means():
    assert "true class" in S.CELLS.level0_label
    assert "top-1" in S.RANK.level0_label


def test_the_modes_write_to_different_files():
    assert S.CELLS.stem != S.RANK.stem


def test_the_top_label_is_a_bucket_only_when_it_buckets():
    assert S.level_labels(S.CELLS, S.CELLS.max_level)[-1] == "3"
    assert S.level_labels(S.CELLS, S.CELLS.max_level + 1)[-1] == "3+"
    assert S.level_labels(S.RANK, 7)[-1] == "5+"


def test_level_labels_cover_every_band():
    assert len(S.level_labels(S.RANK, 7)) == S.RANK.max_level + 1


# ---------------------------------------------------------------------------
# shared with the error CDF
# ---------------------------------------------------------------------------


def test_the_error_column_and_x_bounds_are_the_cdfs():
    """A second set of bounds here and the figures could not be read as a set."""
    assert S.ERROR_COLUMN is CDF.ERROR_COLUMN
    assert S.X_MIN_KM == CDF.X_MIN_KM
    assert S.DEFAULT_X_MAX_KM == CDF.DEFAULT_X_MAX_KM


def test_fallbacks_are_excluded_from_the_drawn_lines():
    df = pd.DataFrame({"status": ["SUCCESS", "FALLBACK", "BASELINE"]})
    assert io.solved_mask(df).tolist() == [True, False, False]


# ---------------------------------------------------------------------------
# formatting and rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(1448.0, "1,448"), (516.0, "516"), (51.0, "51"), (2.4, "2.4"), (np.nan, "—")],
)
def test_median_readouts_are_formatted_for_their_magnitude(value, expected):
    assert S._fmt_km(value) == expected


def test_both_modes_render_to_a_non_empty_png(tmp_path):
    for mode in (S.CELLS, S.RANK):
        p = _points(
            [(SHORTEST_PING, 10.0, 0), (SHORTEST_PING, 800.0, 2),
             ("octant_cbg_hull", 3.0, 0), ("octant_cbg_hull", 400.0, 1)],
            mode=mode,
        )
        counts = {m: {"n_targets": 2} for m in p["method"].unique()}
        t = S.band_table(p, counts, mode)
        out = S.plot_bands(
            p, t, tmp_path / f"{mode.stem}.png",
            mode=mode, title="t", subtitle="s", max_observed=2,
        )
        assert out.exists() and out.stat().st_size > 5_000


def test_a_method_whose_bands_do_not_sum_to_one_still_renders(tmp_path):
    """Vanilla's case: the panel header names the shortfall as fallbacks."""
    p = _points([("vanilla_cbg", 10.0, 0)])
    t = S.band_table(p, {"vanilla_cbg": {"n_targets": 4}}, S.CELLS)
    out = S.plot_bands(
        p, t, tmp_path / "v.png",
        mode=S.CELLS, title="t", subtitle="s", max_observed=0,
    )
    assert out.exists()
    assert t["share"].sum() == pytest.approx(0.25)
