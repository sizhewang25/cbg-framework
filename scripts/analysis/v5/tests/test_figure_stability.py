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
def long(run):
    data = CT.load(
        [run], method_a="alpha", method_b="beta", nside=NSIDE, analysis_root=run.root
    )
    return F.long_table(CT.contest_table(data), "alpha", "beta")


# -- the table ------------------------------------------------------------


def test_one_row_per_site_per_method(long):
    assert list(long.columns) == list(F.LONG_COLUMNS)
    assert len(long) == len(ALPHA_CORRECT) * 2
    assert long.groupby(["run_id", "site_id", "method"]).size().eq(1).all()


def test_the_ratio_is_the_correct_count_over_the_total(long):
    alpha = long[long["method"] == "alpha"].set_index("site_id")
    for site, k in enumerate(ALPHA_CORRECT):
        assert alpha.loc[site, "k"] == k
        assert alpha.loc[site, F.RATIO_COL] == pytest.approx(k / REPLICAS)


def test_a_declined_site_counts_as_unanimous_at_zero(long, place_to_site):
    """`beta` answers every replica of the fallback site correctly and is
    credited with none of them, so the site reads 0 of 4 -- unanimous, and
    unanimously wrong. Against the total, not the solved count: a method that
    declined a site did not agree with itself about it."""
    site = place_to_site[BETA_FALLBACK_PLACE]
    row = long[(long["method"] == "beta") & (long["site_id"] == site)].iloc[0]
    assert row["k"] == 0 and row["n_tgs"] == REPLICAS and row["n_solved"] == 0
    assert row["unanimous"]
    assert row[F.RATIO_COL] == 0.0


def test_a_site_with_no_solved_row_has_no_spread(long, place_to_site):
    """NaN, not zero: zero would read as perfect agreement."""
    site = place_to_site[BETA_FALLBACK_PLACE]
    row = long[(long["method"] == "beta") & (long["site_id"] == site)].iloc[0]
    assert np.isnan(row[F.SPREAD_COL])
    assert F.method_stats(long, "beta")["n_sites_without_spread"] == 1
    assert F.method_stats(long, "alpha")["n_sites_without_spread"] == 0


def test_method_stats_reproduce_the_unanimity_rate(long):
    for method, correct in (("alpha", ALPHA_CORRECT), ("beta", BETA_CORRECT)):
        got = F.method_stats(long, method)
        expected = sum(1 for k in correct if k in (0, REPLICAS))
        if method == "beta":
            # the fallback place is credited 0, so it is unanimous whatever
            # `BETA_CORRECT` says it answered
            expected = sum(
                1
                for i, k in enumerate(correct)
                if (0 if i == BETA_FALLBACK_PLACE else k) in (0, REPLICAS)
            )
        assert got["n_unanimous"] == expected
        assert got["n_split"] == got["n_sites"] - expected
        assert got["unanimity_rate"] == pytest.approx(expected / got["n_sites"], abs=1e-4)


# -- the drawing ----------------------------------------------------------


def _panel(draw, series):
    """One panel on a throwaway figure, so a draw function can be inspected."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw(ax, series, ["#e34948", "#17890b"])
    return fig, ax


def test_each_twin_holds_only_its_own_figures_columns(long):
    """A twin carrying columns its figure never drew invites a number to be
    quoted from the wrong file."""
    ratio = set(F.csv_columns(F.RATIO))
    spread = set(F.csv_columns(F.SPREAD))
    assert F.RATIO_COL in ratio and F.RATIO_COL not in spread
    assert F.SPREAD_COL in spread and F.SPREAD_COL not in ratio
    assert "unanimous" in ratio and "unanimous" not in spread
    for figure in F.FIGURES:
        assert set(F.csv_columns(figure)) <= set(long.columns)


def test_an_unknown_figure_is_refused():
    with pytest.raises(ValueError, match="unknown figure"):
        F.csv_columns("something-else")


def test_the_two_figures_are_named_apart():
    a = F.artifact_names(F.RATIO, "spotter_cbg", "octant_cbg_hull")
    b = F.artifact_names(F.SPREAD, "spotter_cbg", "octant_cbg_hull")
    assert a["png"] == "paired_ratio_of_success.spo_vs_octh.png"
    assert b["png"] == "paired_std_grid_offset.spo_vs_octh.png"
    assert set(a.values()).isdisjoint(b.values())


# -- the violin -----------------------------------------------------------


def test_the_density_is_bounded_and_keeps_its_mass():
    """Reflection, not clipping: the estimate is confined to [0, 1] by
    construction rather than chopped after the fact, and still integrates to
    one."""
    values = np.array([0.0] * 9 + [1.0] * 9)
    got = F.violin_stats(values)
    lo, hi = F._VIOLIN_BOUNDS
    assert got["coords"].min() == pytest.approx(lo)
    assert got["coords"].max() == pytest.approx(hi)
    assert np.trapz(got["vals"], got["coords"]) == pytest.approx(1.0, abs=0.02)


def test_the_summary_lines_are_the_samples_not_the_densitys():
    values = np.array([0.0, 0.0, 0.25, 1.0, 1.0])
    got = F.violin_stats(values)
    assert got["median"] == pytest.approx(0.25)
    assert (got["min"], got["max"]) == (0.0, 1.0)
    assert got["quantiles"] == pytest.approx([0.0, 1.0])


def test_the_violin_cannot_show_a_share_outside_the_unit_interval():
    """A KDE over data stacked at 0 and 1 spills past both bounds, and density
    at 1.15 on a share is a drawing that cannot be true."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    fig, ax = _panel(
        F.draw_ratio, [np.array([0.0] * 9 + [1.0] * 9), np.linspace(0, 1, 18)]
    )
    bodies = [c for c in ax.collections if isinstance(c, PolyCollection)]
    ys = np.concatenate([p.vertices[:, 1] for c in bodies for p in c.get_paths()])
    plt.close(fig)
    assert ys.min() >= -1e-9 and ys.max() <= 1 + 1e-9


def test_a_perfectly_uniform_method_is_drawn_rather_than_raised():
    """`gaussian_kde` cannot take a series with no variance -- and a method
    unanimous and uniform on every site is the success case for this claim."""
    import matplotlib.pyplot as plt

    fig, ax = _panel(F.draw_ratio, [np.ones(6), np.linspace(0, 1, 6)])
    spans = [c for c in ax.collections if c.__class__.__name__ == "LineCollection"]
    plt.close(fig)
    assert spans, "the constant series must still be drawn"


def test_a_narrow_bandwidth_keeps_the_waist_the_data_has():
    """Scott's rule smears four split sites into a waist as wide as
    twenty-six's, which denies the claim the figure is drawn for."""
    values = np.array([0.0] * 30 + [0.5] * 2 + [1.0] * 30)
    mid = np.searchsorted(F.violin_stats(values)["coords"], 0.5)
    narrow = F.violin_stats(values)["vals"][mid]
    wide = F.violin_stats(values, bw=None)["vals"][mid]
    assert narrow < wide, "the default bandwidth fills the waist"


# -- the spread box -------------------------------------------------------


def test_the_spread_column_is_the_substrates(long):
    assert F.SPREAD_COL == CT.OFFSET_SD


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
def test_a_figure_is_one_untitled_panel(long, run, figure, monkeypatch):
    import matplotlib.pyplot as plt

    captured = {}
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: captured.setdefault("fig", fig))
    F.render(long, ["alpha", "beta"], figure, run.root / f"{figure}.png")
    fig = captured["fig"]
    titles = [ax.get_title() for ax in fig.axes]
    sup = fig._suptitle
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    real_close(fig)

    assert len(fig.axes) == 1, "one figure, one panel -- they are filed apart"
    assert titles == [""] and sup is None
    assert labels == ["alpha", "beta"]


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
