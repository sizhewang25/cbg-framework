"""Peripherality: the origin, the normalisation, and what the twin must hold."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import contest as CT
from scripts.analysis.v5.modules import figure_peripherality as F
from scripts.analysis.v5.tests.conftest import NSIDE


@pytest.fixture(scope="module")
def run(contest_run):
    return contest_run


@pytest.fixture(scope="module")
def data(run):
    return CT.load(
        [run], method_a="alpha", method_b="beta", nside=NSIDE, analysis_root=run.root
    )


@pytest.fixture(scope="module")
def normalised(data):
    return F.normalise(CT.contest_table(data))


# -- the axis -------------------------------------------------------------


def test_the_axis_spans_every_site_not_just_the_drawn_ones(normalised):
    """0 and 1 are real sites, and the axis does not move when a site changes
    hands. Normalising over the drawn categories alone would rescale the
    figure every time the contest did."""
    table, norm = normalised
    assert norm["over_n_sites"] == len(table)
    assert table[F.NORM_COL].min() == pytest.approx(0.0)
    assert table[F.NORM_COL].max() == pytest.approx(1.0)
    drawn = table[table["category"].isin(F.DRAWN)]
    assert len(drawn) < len(table), "the fixture must have an undrawn category"


def test_the_raw_kilometres_survive_the_normalisation(normalised):
    """A normalised axis cannot be quoted, so the twin has to invert it."""
    table, norm = normalised
    span = norm["max_km"] - norm["min_km"]
    back = table[F.NORM_COL] * span + norm["min_km"]
    assert np.allclose(back, table[F.DISTANCE_COL], atol=0.05)


def test_a_flat_answer_space_is_refused():
    table = pd.DataFrame({F.DISTANCE_COL: [500.0, 500.0], "category": [CT.TIED] * 2})
    with pytest.raises(ValueError, match="nothing to normalise"):
        F.normalise(table)


def test_the_distance_is_measured_against_this_runs_seeds(data):
    """The wiring, pinned where it is drawn. `contest.centroid_km` owns the
    rule that the origin is the seed cloud's and not the site set's; what
    matters here is that the column comes from *this* run's answer space.
    """
    table = CT.contest_table(data)
    seeds = data.seeds[data.run_ids[0]]
    expected = CT.seed_cloud_centroid_km(
        table["tg_lat"], table["tg_lon"], seeds["seed_lat"], seeds["seed_lon"]
    )
    assert np.allclose(table[F.DISTANCE_COL], expected)
    # A different answer space is a different origin, so a different column.
    shifted = CT.seed_cloud_centroid_km(
        table["tg_lat"], table["tg_lon"], seeds["seed_lat"] + 5.0, seeds["seed_lon"]
    )
    assert not np.allclose(table[F.DISTANCE_COL], shifted)


# -- the twin -------------------------------------------------------------


def test_csv_columns_are_all_emitted(normalised):
    table, norm = normalised
    got = F.build_csv(table, norm)
    assert list(got.columns) == F.csv_columns()
    assert len(got) == len(table)


def test_the_twin_keeps_the_sites_the_figure_does_not_draw(normalised):
    """A twin holding only the drawn rows could not reproduce its own axis."""
    table, norm = normalised
    got = F.build_csv(table, norm)
    assert set(got["drawn"]) == {True, False}
    assert got.loc[got["drawn"], "category"].isin(F.DRAWN).all()
    assert got[F.DISTANCE_COL].min() == pytest.approx(got["norm_min_km"].iloc[0], abs=0.05)


def test_the_stats_report_the_minimum_not_only_the_quartiles(normalised):
    """'No Spotter win is closer in than 1,018 km' is the claim, and a whisker
    is a drawing, not a number."""
    table, _ = normalised
    got = F.category_stats(table, CT.A_WINS)
    assert got["min_km"] == pytest.approx(
        table.loc[table["category"] == CT.A_WINS, F.DISTANCE_COL].min()
    )
    assert got["min_km"] <= got["p25_km"] <= got["p50_km"] <= got["max_km"]


def test_an_empty_category_still_reports_its_count(normalised):
    table, _ = normalised
    empty = table[table["category"] == "nothing-is-this"]
    got = F.category_stats(empty.assign(category="x"), "x")
    assert got == {"category": "x", "n_sites": 0}


# -- the encoding ---------------------------------------------------------


def test_each_box_takes_its_own_methods_hue():
    from scripts.analysis.v5.modules.methods import LABEL_HUES

    ink = F.category_ink("spotter_cbg", "octant_cbg_hull")
    assert ink[CT.A_WINS] == LABEL_HUES["SPO"]
    assert ink[CT.B_WINS] == LABEL_HUES["OCT-H"]


def test_only_the_two_decisive_categories_are_drawn():
    assert F.DRAWN == (CT.A_WINS, CT.B_WINS)
    assert set(F.DRAWN) < set(CT.CATEGORIES)


# -- the drawing ----------------------------------------------------------


def test_the_figure_carries_no_title(run, monkeypatch):
    """`figure_outcome_bars` settled this: the paper's caption names the
    figure, and redrawing it costs a third of the height."""
    import matplotlib.pyplot as plt

    captured = {}
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: captured.setdefault("fig", fig))
    data = CT.load(
        [run], method_a="alpha", method_b="beta", nside=NSIDE, analysis_root=run.root
    )
    table, _ = F.normalise(CT.contest_table(data))
    F.render(data, table, run.root / "peripherality.png")
    fig = captured["fig"]
    ax = fig.axes[0]
    title, sup = ax.get_title(), fig._suptitle
    real_close(fig)

    assert title == "" and sup is None


def test_the_boxes_read_top_down_in_drawn_order(run, monkeypatch):
    import matplotlib.pyplot as plt

    captured = {}
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: captured.setdefault("fig", fig))
    data = CT.load(
        [run], method_a="alpha", method_b="beta", nside=NSIDE, analysis_root=run.root
    )
    table, _ = F.normalise(CT.contest_table(data))
    F.render(data, table, run.root / "order.png")
    ax = captured["fig"].axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    inverted = ax.get_ylim()[0] > ax.get_ylim()[1]
    real_close(captured["fig"])

    names = F.category_names("alpha", "beta")
    assert labels == [names[c] for c in F.DRAWN]
    assert inverted, "position 1 is the bottom in matplotlib; the axis is flipped"


def test_build_writes_the_triple(run, monkeypatch):
    import matplotlib.pyplot as plt

    captured = []
    real_close = plt.close
    monkeypatch.setattr(plt, "close", captured.append)
    png = F.build_for_runs(
        [run], method_a="alpha", method_b="beta", analysis_root=run.root
    )
    for fig in captured:
        real_close(fig)

    pair = F.pair_slug("alpha", "beta")
    out = png.parent
    assert out.parent.name == F.KIND
    for template in (F.PNG_NAME, F.CSV_NAME, F.MANIFEST_NAME):
        assert (out / template.format(pair=pair)).exists()
    body = json.loads((out / F.MANIFEST_NAME.format(pair=pair)).read_text())
    assert body["source_nside"] == NSIDE
    assert [s["category"] for s in body["stats"]] == list(CT.CATEGORIES)
    assert body["normalisation"]["over_n_sites"] == 4
    assert "collinear" in body["caveat"]
    assert "caption names it" in body["subject_note"]
