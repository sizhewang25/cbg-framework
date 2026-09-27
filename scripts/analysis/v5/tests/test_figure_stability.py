"""Stability: the unanimity rule, the solved mask, and the violin that lied."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import contest as CT
from scripts.analysis.v5.modules import figure_stability as F
from scripts.analysis.v5.tests.conftest import (
    ALPHA_CORRECT,
    BETA_CORRECT,
    BETA_FALLBACK_PLACE,
    NSIDE,
    REPLICAS,
)


@pytest.fixture(scope="module")
def run(contest_run):
    return contest_run


@pytest.fixture(scope="module")
def data(run):
    return CT.load(
        [run], method_a="alpha", method_b="beta", nside=NSIDE, analysis_root=run.root
    )


@pytest.fixture(scope="module")
def ratio(data):
    """The success-ratio table: one row per **cell** per method."""
    return F.ratio_rows(CT.paired_targets(data), "alpha", "beta")


@pytest.fixture(scope="module")
def spread(data):
    """The spread table: one row per **site** per method."""
    return F.spread_rows(CT.contest_table(data), "alpha", "beta")


@pytest.fixture(scope="module")
def rows(ratio, spread):
    return {F.RATIO: ratio, F.SPREAD: spread}


# -- the table ------------------------------------------------------------


def test_the_two_figures_count_different_things(ratio, spread):
    """The ratio is per cell because `correct` is a cell verdict; the spread
    is per site because it asks whether identical coordinates get identical
    answers, and two sites in one cell are not identical coordinates."""
    assert F.UNIT == {F.RATIO: "cell", F.SPREAD: "site"}
    assert list(ratio.columns) == F.csv_columns(F.RATIO)
    assert list(spread.columns) == F.csv_columns(F.SPREAD)
    assert ratio.groupby(["run_id", "tg_seed_id", "method"]).size().eq(1).all()
    assert spread.groupby(["run_id", "site_id", "method"]).size().eq(1).all()


def test_a_cell_pools_the_sites_that_share_its_seed(data):
    """Counting them separately would weight a merged facility twice and
    measure something the metric never grades."""
    targets = CT.paired_targets(data)
    got = F.ratio_rows(targets, "alpha", "beta")
    per_cell = got[got["method"] == "alpha"]
    assert per_cell["n_tgs"].sum() == len(targets)
    assert (per_cell["n_sites"] >= 1).all()


def test_the_ratio_is_the_correct_count_over_the_total(ratio):
    alpha = ratio[ratio["method"] == "alpha"]
    assert alpha["k"].sum() == sum(ALPHA_CORRECT)
    assert (alpha[F.RATIO_COL] == alpha["k"] / alpha["n_tgs"]).all()


def test_a_declined_cell_counts_as_unanimous_at_zero(ratio, data, place_to_site):
    """`beta` answers every replica of the fallback site correctly and is
    credited with none of them, so its cell reads 0 of n -- unanimous, and
    unanimously wrong. Against the cell's total, not its solved count."""
    targets = CT.paired_targets(data)
    site = place_to_site[BETA_FALLBACK_PLACE]
    seed = int(targets.loc[targets["site_id"] == site, "tg_seed_id"].iloc[0])
    row = ratio[(ratio["method"] == "beta") & (ratio["tg_seed_id"] == seed)].iloc[0]
    assert row["k"] == 0
    assert row["unanimous"]
    assert row[F.RATIO_COL] == 0.0


def test_a_site_with_no_solved_row_has_no_spread(spread, place_to_site):
    """NaN, not zero: zero would read as perfect agreement."""
    site = place_to_site[BETA_FALLBACK_PLACE]
    row = spread[(spread["method"] == "beta") & (spread["site_id"] == site)].iloc[0]
    assert np.isnan(row[F.SPREAD_COL])
    assert F.method_stats(spread, F.SPREAD, "beta")["n_units_without_spread"] == 1
    assert F.method_stats(spread, F.SPREAD, "alpha")["n_units_without_spread"] == 0


def test_method_stats_reproduce_the_unanimity_rate(ratio):
    for method in ("alpha", "beta"):
        mine = ratio[ratio["method"] == method]
        got = F.method_stats(ratio, F.RATIO, method)
        assert got["unit"] == "cell"
        assert got["n_units"] == len(mine)
        assert got["n_unanimous"] == int(mine["unanimous"].sum())
        assert got["n_split"] == got["n_units"] - got["n_unanimous"]
        assert got["unanimity_rate"] == pytest.approx(
            got["n_unanimous"] / got["n_units"], abs=1e-4
        )


# -- the drawing ----------------------------------------------------------


def _panel(draw, series, *extra):
    """One panel on a throwaway figure, so a draw function can be inspected."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw(ax, series, ["#e34948", "#17890b"], *extra)
    return fig, ax


def test_each_twin_holds_only_its_own_figures_columns(rows):
    """A twin carrying columns its figure never drew invites a number to be
    quoted from the wrong file -- and here the keys differ too, so a twin
    holding both would not say which unit it counted."""
    r, s = set(F.csv_columns(F.RATIO)), set(F.csv_columns(F.SPREAD))
    assert F.RATIO_COL in r and F.RATIO_COL not in s
    assert F.SPREAD_COL in s and F.SPREAD_COL not in r
    assert "site_id" in s and "site_id" not in r
    for figure in F.FIGURES:
        assert list(rows[figure].columns) == F.csv_columns(figure)


def test_an_unknown_figure_is_refused():
    with pytest.raises(ValueError, match="unknown figure"):
        F.csv_columns("something-else")


def test_the_two_figures_are_named_apart():
    a = F.artifact_names(F.RATIO, "spotter_cbg", "octant_cbg_hull")
    b = F.artifact_names(F.SPREAD, "spotter_cbg", "octant_cbg_hull")
    assert a["png"] == "paired_ratio_of_success.spo_vs_octh.png"
    assert b["png"] == "paired_std_grid_offset.spo_vs_octh.png"
    assert set(a.values()).isdisjoint(b.values())


# -- the ratio CDF -------------------------------------------------------


def test_the_ecdf_is_exact_at_every_observation():
    """No bins, no bandwidth, no choices -- which is why this figure is a CDF
    and not the violin it was twice."""
    values = np.array([0.0, 0.0, 0.5, 1.0])
    x, y = F.ecdf(values)
    for v in np.unique(values):
        share = float((values <= v).mean())
        assert y[np.searchsorted(x, v, side="right") - 1] == pytest.approx(share)


def test_the_curve_spans_the_whole_unit_interval():
    x, y = F.ecdf(np.array([0.4, 0.6]))
    assert (x[0], x[-1]) == F.RATIO_BOUNDS
    assert (y[0], y[-1]) == (0.0, 1.0)


def test_the_two_jumps_are_the_unanimity_rate():
    """What makes the figure readable: the jump at 0 is the share of sites a
    method missed entirely, the jump at 1 the share it swept, and together
    they are the unanimity rate the manifest reports."""
    values = np.array([0.0] * 3 + [0.5] * 2 + [1.0] * 5)
    x, y = F.ecdf(values)
    at_zero = y[np.searchsorted(x, 0.0, side="right") - 1]
    below_one = y[np.searchsorted(x, 1.0, side="left") - 1]
    assert at_zero + (1.0 - below_one) == pytest.approx(
        float(((values == 0) | (values == 1)).mean())
    )


def test_a_method_that_swept_every_site_is_still_drawn():
    """One value repeated has no density a KDE can take, and it is the success
    case for this claim. A staircase does not care."""
    import matplotlib.pyplot as plt

    fig, ax = _panel(F.draw_ratio, [np.ones(6), np.linspace(0, 1, 6)], ["A", "B"])
    ys = [ln.get_ydata() for ln in ax.lines]
    plt.close(fig)
    assert len(ys) == 2 and all(len(v) for v in ys)


def test_each_curve_takes_its_methods_hue_and_is_named():
    import matplotlib.pyplot as plt

    fig, ax = _panel(
        F.draw_ratio, [np.array([0.0, 1.0]), np.array([0.3, 0.7])], ["SPO", "OCT-H"]
    )
    colours = [ln.get_color() for ln in ax.lines]
    names = [txt.get_text() for txt in ax.get_legend().get_texts()]
    plt.close(fig)
    assert colours == ["#e34948", "#17890b"]
    assert names == ["SPO", "OCT-H"]


# -- the spread box -------------------------------------------------------


def test_the_spread_column_is_the_substrates():
    assert F.SPREAD_COL == CT.OFFSET_SD


def test_the_drawn_whisker_ends_are_reported_as_numbers(spread):
    """A figure that states a bound nothing else does is a figure nobody can
    quote. The whisker percentiles are in `QUANTILES` for that reason."""
    for whisker in F.SPREAD_WHIS:
        assert whisker / 100.0 in F.QUANTILES
    got = F.method_stats(spread, F.SPREAD, "alpha")
    for whisker in F.SPREAD_WHIS:
        assert f"spread_p{int(whisker)}" in got
    assert got["spread_min"] <= got["spread_p5"]
    assert got["spread_p95"] <= got["spread_max"]


def test_the_whiskers_are_percentiles_and_nothing_is_drawn_past_them():
    """p5 and p95, not 1.5x IQR and not the extremes. The consequence is that
    the worst site is off the page -- which is the half of this claim that
    runs the other way, and why `spread_max` is in the twin."""
    import matplotlib.pyplot as plt

    tail = np.concatenate([np.zeros(20), [15.8]])
    fig, ax = _panel(F.draw_spread, [tail, np.linspace(0, 6, 21)])
    fliers = [ln for ln in ax.lines if ln.get_linestyle() == "None" and len(ln.get_ydata())]
    top = max(np.max(ln.get_ydata()) for ln in ax.lines if len(ln.get_ydata()))
    plt.close(fig)

    assert not fliers, "no outliers are drawn"
    assert top < 15.8, "the extreme site is deliberately off the page"
    assert top == pytest.approx(np.percentile(tail, F.SPREAD_WHIS[1]), abs=6.0)


@pytest.mark.parametrize("figure", F.FIGURES)
def test_a_figure_is_one_untitled_panel(rows, run, figure, monkeypatch):
    import matplotlib.pyplot as plt

    captured = {}
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: captured.setdefault("fig", fig))
    F.render(rows[figure], ["alpha", "beta"], figure, run.root / f"{figure}.png")
    fig = captured["fig"]
    ax = fig.axes[0]
    titles = [a.get_title() for a in fig.axes]
    sup = fig._suptitle
    # The two name their methods differently: the spread box puts them on the
    # x axis, the ratio CDF has a continuous x and names its curves instead.
    named = (
        [txt.get_text() for txt in ax.get_legend().get_texts()]
        if figure == F.RATIO
        else [txt.get_text() for txt in ax.get_xticklabels()]
    )
    real_close(fig)

    assert len(fig.axes) == 1, "one figure, one panel -- they are filed apart"
    assert titles == [""] and sup is None
    assert named == ["alpha", "beta"]


def test_build_writes_a_triple_per_figure(run, monkeypatch):
    import matplotlib.pyplot as plt

    captured = []
    real_close = plt.close
    monkeypatch.setattr(plt, "close", captured.append)
    pngs = F.build_for_runs(
        [run], method_a="alpha", method_b="beta", analysis_root=run.root
    )
    for fig in captured:
        real_close(fig)

    assert len(pngs) == len(F.FIGURES)
    out = pngs[0].parent
    assert out.parent.name == F.KIND
    for figure in F.FIGURES:
        names = F.artifact_names(figure, "alpha", "beta")
        for key in ("png", "csv", "man"):
            assert (out / names[key]).exists()
        body = json.loads((out / names["man"]).read_text())
        assert body["kind"] == figure
        assert body["companion"] == [n for n in F.FIGURES if n != figure]
        assert body["source_nside"] == NSIDE
        assert [s["method"] for s in body["stats"]] == ["alpha", "beta"]
        assert "solved_mask" in body["policy"]
        assert "caption names it" in body["subject_note"]
        header = pd.read_csv(out / names["csv"], nrows=0)
        assert list(header.columns) == F.csv_columns(figure)


def test_only_the_named_figure_is_built(run, monkeypatch):
    import matplotlib.pyplot as plt

    captured = []
    real_close = plt.close
    monkeypatch.setattr(plt, "close", captured.append)
    pngs = F.build_for_runs(
        [run], method_a="alpha", method_b="beta", figures=[F.SPREAD],
        analysis_root=run.root,
    )
    for fig in captured:
        real_close(fig)
    assert [p.name for p in pngs] == [
        F.artifact_names(F.SPREAD, "alpha", "beta")["png"]
    ]


def test_an_unknown_figure_is_refused_before_anything_is_read(run):
    with pytest.raises(ValueError, match="unknown figure"):
        F.build_for_runs(
            [run], method_a="alpha", method_b="beta", figures=["nope"],
            analysis_root=run.root,
        )


@pytest.mark.parametrize("figure", F.FIGURES)
def test_the_axis_labels_fit_inside_the_figure(rows, run, figure, monkeypatch):
    """The figure is saved at a fixed canvas, so a label that overruns is
    clipped without complaint rather than shrinking the axes. It happened on
    `figure_peripherality`, which lost the last character of its x-label."""
    import matplotlib.pyplot as plt

    captured = {}
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: captured.setdefault("fig", fig))
    F.render(rows[figure], ["alpha", "beta"], figure, run.root / f"fit-{figure}.png")
    fig = captured["fig"]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    canvas = fig.get_window_extent()
    boxes = {
        "x": ax.xaxis.get_label().get_window_extent(renderer=renderer),
        "y": ax.yaxis.get_label().get_window_extent(renderer=renderer),
    }
    real_close(fig)

    for axis, box in boxes.items():
        assert box.x0 >= canvas.x0 - 0.5 and box.x1 <= canvas.x1 + 0.5, (
            f"{figure} {axis}-label runs off the canvas horizontally"
        )
        assert box.y0 >= canvas.y0 - 0.5 and box.y1 <= canvas.y1 + 0.5, (
            f"{figure} {axis}-label runs off the canvas vertically"
        )
