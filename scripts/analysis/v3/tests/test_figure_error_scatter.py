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
    df["level"] = np.clip(
        df["class_error"] + mode.index_base, mode.index_base, mode.top_level
    )
    return df


# ---------------------------------------------------------------------------
# the share denominator — why this figure exists in this shape
# ---------------------------------------------------------------------------


def _correct(table, mode):
    return table.loc[table["level"] == mode.index_base]


def test_shares_divide_by_every_target_not_by_the_drawn_rows():
    """as02 Vanilla: 123 band-0 rows / 412 targets = 0.298 = its top-1 accuracy.

    Over its 337 solved rows the same count reads 0.365 and matches nothing.
    """
    p = _points([("m", 10.0, 0)] * 123 + [("m", 500.0, 1)] * 214)
    t = S.band_table(p, {"m": {"n_targets": 412}}, S.CELLS)
    assert _correct(t, S.CELLS)["share"].iloc[0] == pytest.approx(0.2985, abs=5e-4)


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
    assert t.loc[1, "cumulative_share"] == pytest.approx(0.40)
    assert t.loc[3, "cumulative_share"] == pytest.approx(1.00)
    assert t["cumulative_share"].is_monotonic_increasing


def test_the_rank_index_is_one_based_so_band_n_sums_to_top_n():
    """`index <= N` is top-N once the axis is 1-indexed.

    `tg_seed_rank < N` needed translating; the shifted index does not, which is
    the whole reason the rank mode is numbered from 1.
    """
    p = _points([("m", 1.0, r) for r in [0, 0, 1, 2, 3, 4]], mode=S.RANK)
    t = S.band_table(p, {"m": {"n_targets": 6}}, S.RANK).set_index("level")
    assert t.loc[1, "share"] == pytest.approx(2 / 6, abs=5e-5)  # top-1
    assert t.loc[3, "cumulative_share"] == pytest.approx(4 / 6, abs=5e-5)  # top-3


# ---------------------------------------------------------------------------
# bands
# ---------------------------------------------------------------------------


def test_every_band_gets_a_row_even_when_empty():
    """The share axis puts a tick on every band, so every band needs a row."""
    p = _points([("m", 1.0, 0)])
    t = S.band_table(p, {"m": {"n_targets": 1}}, S.CELLS)
    assert t["level"].tolist() == list(S.CELLS.levels)
    top = S.CELLS.top_level
    assert int(t.loc[t["level"] == top, "n"].iloc[0]) == 0
    assert np.isnan(t.loc[t["level"] == top, "error_km_p50"].iloc[0])


def test_values_past_the_top_band_fold_into_it():
    p = _points([("m", 1.0, 3), ("m", 1.0, 4), ("m", 1.0, 12)])
    assert p["level"].tolist() == [S.CELLS.top_level] * 3


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


def test_cells_is_zero_indexed_and_rank_is_one_indexed():
    """Crossings count boundaries, so 0 means none crossed. Rank is an index
    into the cells ordered by distance, so 1 means the true cell is nearest."""
    assert S.CELLS.index_base == 0
    assert list(S.CELLS.levels) == [0, 1, 2, 3]
    assert S.RANK.index_base == 1
    assert list(S.RANK.levels) == [1, 2, 3, 4]


def test_both_modes_are_registered_under_their_key():
    assert S.MODES["cells"] is S.CELLS
    assert S.MODES["rank"] is S.RANK


def test_the_band_ticks_are_bare_integers():
    """What band 1 means lives in the axis label and footnote, not the tick."""
    assert S.level_labels(S.CELLS, 4) == ["0", "1", "2", "3+"]
    assert S.level_labels(S.RANK, 7) == ["1", "2", "3", "4+"]


def test_the_modes_write_to_different_files():
    assert S.CELLS.stem != S.RANK.stem


def test_the_top_label_is_a_bucket_only_when_it_buckets():
    assert S.level_labels(S.CELLS, 3)[-1] == "3"
    assert S.level_labels(S.CELLS, 4)[-1] == "3+"
    # raw rank 3 shifts to index 4, the top band, so it is not yet a bucket
    assert S.level_labels(S.RANK, 3)[-1] == "4"
    assert S.level_labels(S.RANK, 7)[-1] == "4+"


def test_level_labels_cover_every_band():
    assert len(S.level_labels(S.RANK, 7)) == S.RANK.n_bands


# ---------------------------------------------------------------------------
# the row grid
# ---------------------------------------------------------------------------


def test_rows_are_contiguous_unit_cells_starting_at_the_x_axis():
    """The bottom band's lower edge is y=0 exactly — resting on the axis like
    the tracks in a spectrum-allocation chart, not hovering above it."""
    for mode in (S.CELLS, S.RANK):
        first_lo, _ = S.band_span(mode, mode.index_base)
        assert first_lo == 0.0
        for k, level in enumerate(mode.levels):
            lo, hi = S.band_span(mode, level)
            assert lo == pytest.approx(k + S.BAND_BOTTOM)
            assert hi == pytest.approx(k + S.BAND_TOP)


def test_the_gutter_is_only_above_the_band():
    """A band rests on its row's floor; the free strip is the ceiling side."""
    assert S.BAND_BOTTOM == 0.0
    assert S.BAND_TOP < 1.0
    for mode in (S.CELLS, S.RANK):
        for k, level in enumerate(mode.levels):
            lo, _ = S.band_span(mode, level)
            assert lo == pytest.approx(k)


def test_the_median_readout_stays_inside_its_own_row():
    """Above the band but below the next separator — otherwise the number would
    sit in the row above and name the wrong band."""
    for mode in (S.CELLS, S.RANK):
        for k, level in enumerate(mode.levels):
            _, hi = S.band_span(mode, level)
            text_y = k + S.MEDIAN_TEXT_Y
            assert hi < text_y < k + 1


def test_the_tick_sits_at_the_bands_middle_not_the_rows():
    """The tick names the data, so it aligns with the band rather than with the
    grid cell that also holds the readout gutter."""
    lo, hi = S.band_span(S.RANK, 1)
    assert S.band_centre(S.RANK, 1) == pytest.approx((lo + hi) / 2)
    assert S.band_centre(S.RANK, 1) < 0.5  # below the row's own centre


def test_bands_do_not_touch_so_the_grid_lines_stay_visible():
    for mode in (S.CELLS, S.RANK):
        levels = list(mode.levels)
        for a, b in zip(levels, levels[1:]):
            assert S.band_span(mode, a)[1] < S.band_span(mode, b)[0]


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
