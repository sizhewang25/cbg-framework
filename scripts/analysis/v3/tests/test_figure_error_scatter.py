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
import typer

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


def test_the_gutter_leaves_physical_room_for_the_readout():
    """The compaction knob is `ROW_INCHES`; below ~0.13 in of gutter the 7 pt
    median number starts touching the band it labels."""
    gutter_inches = (1.0 - S.BAND_TOP) * S.ROW_INCHES
    assert gutter_inches >= 0.13


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


# ---------------------------------------------------------------------------
# the cross-dataset grid
# ---------------------------------------------------------------------------


class _FakeRun:
    """Stand-in for `RunPaths`: `load_points` is stubbed, so only the id is read."""

    def __init__(self, run_id: str, setup: str = "anchors_to_probes") -> None:
        self.run_id = run_id
        self.setup = setup


def _stub_load_points(monkeypatch, per_run):
    """`{run_id: (rows, n_targets)}` -> a `load_points` that serves each run."""

    def fake(run, mode, **_):
        rows, n_targets = per_run[run.run_id]
        counts = {m: {"n_targets": n_targets} for m in {r[0] for r in rows}}
        return _points(rows, mode=mode), counts

    monkeypatch.setattr(S, "load_points", fake)
    return {rid: _FakeRun(rid) for rid in per_run}


def test_each_dataset_keeps_its_own_denominator_rather_than_a_pooled_one(monkeypatch):
    """The same 100 correct rows are a different accuracy on a 399- and a 458-
    target run, and each panel has to report its own."""
    runs = _stub_load_points(
        monkeypatch,
        {
            "as01-260728-260802": ([("m", 10.0, 0)] * 100, 399),
            "as03-260728-260802": ([("m", 10.0, 0)] * 100, 458),
        },
    )
    _, table, _ = S.load_cross_points(
        runs, S.CELLS, analysis_root=None, grid="h3", resolution=4
    )
    share = table.set_index(["dataset", "level"])["share"]
    assert share[("as01", 0)] == pytest.approx(100 / 399, abs=5e-5)
    assert share[("as03", 0)] == pytest.approx(100 / 458, abs=5e-5)
    # A pooled denominator would give both the same 200/857.
    assert share[("as01", 0)] != share[("as03", 0)]


def test_the_dataset_key_is_the_short_name_not_the_run_id(monkeypatch):
    runs = _stub_load_points(
        monkeypatch, {"as02-260728-260802": ([("m", 10.0, 0)], 1)}
    )
    points, table, counts = S.load_cross_points(
        runs, S.CELLS, analysis_root=None, grid="h3", resolution=4
    )
    assert set(points["dataset"]) == {"as02"}
    assert set(table["dataset"]) == {"as02"}
    # The full run_id survives on the rows and in the counts, so a panel can
    # still be traced back to the run it came from.
    assert set(points["run_id"]) == {"as02-260728-260802"}
    assert counts["as02"]["run_id"] == "as02-260728-260802"


def test_the_columns_are_dataset_sorted_not_argument_ordered(monkeypatch):
    runs = _stub_load_points(
        monkeypatch,
        {
            "as03-260728-260802": ([("m", 10.0, 0)], 1),
            "as01-260728-260802": ([("m", 10.0, 0)], 1),
        },
    )
    points, _, _ = S.load_cross_points(
        runs, S.CELLS, analysis_root=None, grid="h3", resolution=4
    )
    assert list(points["dataset"]) == ["as01", "as03"]


def test_nothing_is_dropped_or_duplicated_when_the_runs_are_concatenated(monkeypatch):
    runs = _stub_load_points(
        monkeypatch,
        {
            "as01-260728-260802": ([("m", 10.0, 0)] * 7, 10),
            "as02-260728-260802": ([("m", 20.0, 1)] * 3, 10),
        },
    )
    points, _, _ = S.load_cross_points(
        runs, S.CELLS, analysis_root=None, grid="h3", resolution=4
    )
    assert len(points) == 10
    assert points["dataset"].value_counts().to_dict() == {"as01": 7, "as02": 3}


def test_methods_are_rows_and_datasets_are_columns_not_the_transpose():
    points = _points([("m1", 10.0, 0), ("m2", 20.0, 0), ("m3", 30.0, 0)])
    points["dataset"] = ["as01", "as01", "as02"]
    assert S.cross_grid_shape(points) == (3, 2)


def test_the_cross_grid_keeps_the_per_run_bands_geometry():
    """A band in the grid is the same height as a band in the per-run figure.

    That is what lets the two figures be read against each other; only the
    per-row chrome shrinks, because the grid shares its x axis down a column
    and carries the method as a row label instead of a panel title.
    """
    w1, h1 = S.cross_figsize(1, 1, S.RANK)
    w2, h2 = S.cross_figsize(2, 3, S.RANK)
    assert h2 - h1 == pytest.approx(S.ROW_INCHES * S.RANK.n_bands
                                    + S.CROSS_ROW_CHROME_INCHES)
    assert w2 == pytest.approx(3 * w1)
    assert S.CROSS_ROW_CHROME_INCHES < S.PANEL_CHROME_INCHES


def test_the_cross_figure_renders_to_a_non_empty_png(tmp_path):
    rows = [
        (SHORTEST_PING, 10.0, 0),
        (SHORTEST_PING, 800.0, 2),
        ("octant_cbg_hull", 3.0, 0),
        ("octant_cbg_hull", 400.0, 1),
    ]
    frames = []
    tables = []
    for dataset, n in (("as01", 4), ("as02", 5)):
        p = _points(rows, mode=S.RANK).assign(dataset=dataset)
        counts = {m: {"n_targets": n} for m in p["method"].unique()}
        frames.append(p)
        tables.append(S.band_table(p, counts, S.RANK).assign(dataset=dataset))
    out = S.plot_cross_bands(
        pd.concat(frames, ignore_index=True),
        pd.concat(tables, ignore_index=True),
        tmp_path / "cross.png",
        mode=S.RANK,
        title="t",
        subtitle="s",
        max_observed=2,
    )
    assert out.exists() and out.stat().st_size > 5_000


def test_the_cross_artifacts_land_under_their_own_kind(monkeypatch, tmp_path):
    runs = _stub_load_points(
        monkeypatch,
        {
            "as01-260728-260802": ([("m", 10.0, 0)], 2),
            "as02-260728-260802": ([("m", 20.0, 1)], 2),
        },
    )
    (built,) = S.build_cross(
        runs,
        mode=S.RANK,
        layouts=[S.COMPARE],
        analysis_root=tmp_path,
        grid="h3",
        resolution=4,
    )
    png, manifest = built["png"], built["manifest"]
    assert png.parent == tmp_path / "_cross" / S.CROSS_KIND / "as01+as02"
    # The grid slug is in the filename because the cross directory is keyed by
    # dataset set alone, so h3-4 and healpix-128 would otherwise collide.
    assert png.name == "error_vs_rank_by_dataset.h3-4.png"
    assert manifest["datasets"] == ["as01", "as02"]
    assert manifest["run_ids"] == sorted(runs)
    assert manifest["cross_layout"] == S.COMPARE


def test_the_panel_header_names_the_fallback_shortfall():
    p = _points([("vanilla_cbg", 10.0, 0)])
    t = S.band_table(p, {"vanilla_cbg": {"n_targets": 4}}, S.CELLS)
    header = S.panel_header(t)
    assert header.startswith("n = 4")
    assert "75.0% fell back" in header


def test_a_panel_with_no_fallbacks_prints_only_its_n():
    p = _points([("m", 10.0, 0)] * 4)
    t = S.band_table(p, {"m": {"n_targets": 4}}, S.CELLS)
    assert S.panel_header(t) == "n = 4"


def test_the_cumulative_clause_is_printed_only_where_it_holds():
    """Bands sum to top-N by rank; crossings have no such property (SCHEMA.md)."""
    assert "top-3" in S.footnote_text(S.RANK)
    assert "top-3" not in S.footnote_text(S.CELLS)


# ---------------------------------------------------------------------------
# the pooled layout
# ---------------------------------------------------------------------------


def _disjoint(monkeypatch, per_run):
    """Like `_stub_load_points`, but every run's target ids are its own.

    Pooling is only defined over disjoint target sets, so the fixture that
    exercises it has to produce them — `_points` numbers from zero every time.
    """
    offsets = {rid: i * 1000 for i, rid in enumerate(sorted(per_run))}

    def fake(run, mode, **_):
        rows, n_targets = per_run[run.run_id]
        pts = _points(rows, mode=mode)
        pts["target_id"] = [
            f"tg-{offsets[run.run_id] + i}" for i in range(len(pts))
        ]
        counts = {m: {"n_targets": n_targets} for m in {r[0] for r in rows}}
        return pts, counts

    monkeypatch.setattr(S, "load_points", fake)
    return {rid: _FakeRun(rid) for rid in per_run}


def test_pooling_adds_the_runs_denominators():
    counts = {
        "as01": {"run_id": "r1", "methods": {"m": {"n_targets": 399, "n_solved": 300}}},
        "as02": {"run_id": "r2", "methods": {"m": {"n_targets": 412, "n_solved": 340}}},
        "as03": {"run_id": "r3", "methods": {"m": {"n_targets": 458, "n_solved": 360}}},
    }
    pooled = S.pool_counts(counts)
    assert pooled["m"]["n_targets"] == 1269
    assert pooled["m"]["n_solved"] == 1000
    assert pooled["m"]["datasets"] == ["as01", "as02", "as03"]


def test_a_method_missing_from_one_run_is_scored_only_on_the_runs_it_ran_on():
    """Not penalized for targets it never saw — `pareto`'s rule for a gap."""
    counts = {
        "as01": {"run_id": "r1", "methods": {"m": {"n_targets": 400}, "x": {"n_targets": 400}}},
        "as02": {"run_id": "r2", "methods": {"m": {"n_targets": 400}}},
    }
    pooled = S.pool_counts(counts)
    assert pooled["m"]["n_targets"] == 800
    assert pooled["x"]["n_targets"] == 400
    assert pooled["x"]["datasets"] == ["as01"]


def test_the_pooled_share_is_target_weighted_not_the_mean_of_the_datasets(monkeypatch):
    """100/400 and 100/100 pool to 200/500, not to the mean of 0.25 and 1.0."""
    runs = _disjoint(
        monkeypatch,
        {
            "as01-260728-260802": ([("m", 10.0, 0)] * 100, 400),
            "as02-260728-260802": ([("m", 10.0, 0)] * 100, 100),
        },
    )
    points, per_dataset, counts = S.load_cross_points(
        runs, S.CELLS, analysis_root=None, grid="h3", resolution=4
    )
    pooled = S.band_table(points, S.pool_counts(counts), S.CELLS)
    micro = _correct(pooled, S.CELLS)["share"].iloc[0]
    assert micro == pytest.approx(200 / 500, abs=5e-5)
    macro = _correct(per_dataset, S.CELLS)["share"].mean()
    assert macro == pytest.approx((0.25 + 1.0) / 2, abs=5e-5)
    assert micro != pytest.approx(macro, abs=1e-3)


def test_the_weighting_check_records_both_averages(monkeypatch):
    runs = _disjoint(
        monkeypatch,
        {
            "as01-260728-260802": ([("m", 10.0, 0)] * 100, 400),
            "as02-260728-260802": ([("m", 10.0, 0)] * 100, 100),
        },
    )
    points, per_dataset, counts = S.load_cross_points(
        runs, S.CELLS, analysis_root=None, grid="h3", resolution=4
    )
    pooled = S.band_table(points, S.pool_counts(counts), S.CELLS)
    check = S.weighting_check(per_dataset, pooled, S.CELLS)["m"]
    assert check["pooled_micro"] == pytest.approx(0.4, abs=5e-5)
    assert check["dataset_macro_mean"] == pytest.approx(0.625, abs=5e-5)
    assert check["delta"] == pytest.approx(0.4 - 0.625, abs=5e-5)
    assert check["per_dataset"] == {"as01": 0.25, "as02": 1.0}


def test_runs_sharing_a_target_are_refused_rather_than_double_counted(
    monkeypatch, tmp_path
):
    """`_stub_load_points` numbers both runs from zero, which is the bug case."""
    runs = _stub_load_points(
        monkeypatch,
        {
            "as01-260728-260802": ([("m", 10.0, 0)] * 3, 4),
            "as02-260728-260802": ([("m", 20.0, 1)] * 3, 4),
        },
    )
    with pytest.raises(typer.BadParameter, match="share targets"):
        S.build_cross(
            runs,
            mode=S.CELLS,
            layouts=[S.POOLED],
            analysis_root=tmp_path,
            grid="h3",
            resolution=4,
        )
    # The comparison layout is unaffected: it never shares a denominator.
    (built,) = S.build_cross(
        runs,
        mode=S.CELLS,
        layouts=[S.COMPARE],
        analysis_root=tmp_path,
        grid="h3",
        resolution=4,
    )
    assert built["png"].exists()


def test_the_pooled_layout_draws_the_per_run_grid_over_merged_targets(
    monkeypatch, tmp_path
):
    runs = _disjoint(
        monkeypatch,
        {
            "as01-260728-260802": ([(SHORTEST_PING, 10.0, 0)] * 2, 3),
            "as02-260728-260802": ([(SHORTEST_PING, 900.0, 2)] * 2, 3),
        },
    )
    (built,) = S.build_cross(
        runs,
        mode=S.RANK,
        layouts=[S.POOLED],
        analysis_root=tmp_path,
        grid="h3",
        resolution=4,
    )
    assert built["png"].name == "error_vs_rank.h3-4.png"
    assert built["png"].exists() and built["png"].stat().st_size > 5_000
    # One denominator over the merged population, and no dataset column: the
    # pooled table is a per-method table, like a single run's.
    assert int(built["table"]["n_targets"].max()) == 6
    assert "dataset" not in built["table"].columns
    assert built["manifest"]["cross_layout"] == S.POOLED


def test_the_two_layouts_write_different_files_so_both_can_coexist():
    assert S.layout_stem(S.RANK, S.POOLED) == "error_vs_rank"
    assert S.layout_stem(S.RANK, S.COMPARE) == "error_vs_rank_by_dataset"
    assert S.layout_stem(S.CELLS, S.POOLED) != S.layout_stem(S.RANK, S.POOLED)


def test_an_unknown_layout_is_refused_by_name(monkeypatch, tmp_path):
    runs = _disjoint(
        monkeypatch, {"as01-260728-260802": ([("m", 10.0, 0)], 1)}
    )
    with pytest.raises(typer.BadParameter, match="unknown layout"):
        S.build_cross(
            runs,
            mode=S.CELLS,
            layouts=["mean"],
            analysis_root=tmp_path,
            grid="h3",
            resolution=4,
        )
    with pytest.raises(ValueError, match="unknown layout"):
        S.layout_stem(S.CELLS, "mean")


def test_both_layouts_come_from_one_read_of_the_data(monkeypatch, tmp_path):
    """Loading walks the answer space, so asking for both must not read twice."""
    runs = _disjoint(
        monkeypatch,
        {
            "as01-260728-260802": ([("m", 10.0, 0)], 2),
            "as02-260728-260802": ([("m", 20.0, 1)], 2),
        },
    )
    calls = []
    inner = S.load_points

    def counting(run, mode, **kw):
        calls.append(run.run_id)
        return inner(run, mode, **kw)

    monkeypatch.setattr(S, "load_points", counting)
    built = S.build_cross(
        runs,
        mode=S.CELLS,
        layouts=[S.POOLED, S.COMPARE],
        analysis_root=tmp_path,
        grid="h3",
        resolution=4,
    )
    assert len(built) == 2
    assert sorted(calls) == sorted(runs)  # once per run, not once per layout
