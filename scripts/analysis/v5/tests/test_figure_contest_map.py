"""The contest map: the twin, the pooled line, the relaxation, the empty figure."""

from __future__ import annotations

import json
import numpy as np
import pytest

from scripts.analysis.v5.modules import contest as CT
from scripts.analysis.v5.modules import figure_contest_map as F
from scripts.analysis.v5.modules.paths import MissingArtifactError
from scripts.analysis.v5.tests.conftest import NSIDE, REPLICAS


@pytest.fixture(scope="module")
def run(contest_run):
    return contest_run


@pytest.fixture(scope="module")
def two_runs(contest_two_runs):
    return contest_two_runs


@pytest.fixture(scope="module")
def data(run):
    return F.load(
        [run], method_a="alpha", method_b="beta", nside=NSIDE, analysis_root=run.root
    )


@pytest.fixture(scope="module")
def table(data):
    return F.contest_table(data)


# -- the table ------------------------------------------------------------


def test_each_site_lands_in_the_category_its_counts_say(table):
    by_site = table.set_index("site_id")["category"].to_dict()
    counts = table.set_index("site_id")[["k_a", "k_b"]].to_dict("index")
    assert counts[0] == {"k_a": REPLICAS, "k_b": 0} and by_site[0] == CT.A_WINS
    assert counts[1] == {"k_a": 2, "k_b": REPLICAS} and by_site[1] == CT.B_WINS
    assert counts[2] == {"k_a": 3, "k_b": 3} and by_site[2] == CT.TIED


def test_the_fallback_site_is_a_win_not_a_tie(table):
    """`beta` answers correctly on all four Miami replicas and is credited
    with none of them: they are FALLBACK, so they are the baseline's."""
    row = table[table["site_id"] == 3].iloc[0]
    assert (row["k_a"], row["k_b"], row["n_b"]) == (REPLICAS, 0, REPLICAS)
    assert row["category"] == CT.A_WINS


# -- the twin -------------------------------------------------------------


def test_csv_columns_are_all_emitted(data, table):
    got = F.build_csv(table, F.panel_counts(table, data))
    assert list(got.columns) == F.csv_columns()
    assert len(got) == len(table)


def test_the_twin_reproduces_its_panels_counts(data, table):
    """The panel numbers are derivable from the per-site rows beside them, so
    a reader never has to trust the broadcast columns."""
    got = F.build_csv(table, F.panel_counts(table, data))
    for _, panel in got.groupby("dataset"):
        for category in CT.CATEGORIES:
            assert (panel["category"] == category).sum() == panel[f"n_{category}"].iloc[0]
        assert len(panel) == panel["n_sites"].iloc[0]


def test_the_pooled_line_is_in_the_manifest_and_not_the_twin(data, table):
    counts = F.panel_counts(table, data)
    assert counts[-1]["run_id"] == "__pooled__"
    assert counts[-1]["n_sites"] == len(table)
    got = F.build_csv(table, counts)
    assert "__pooled__" not in set(got["run_id"]), "a non-site row somebody will sum"


def test_the_panel_mcnemar_matches_its_category_counts(data, table):
    panel = F.panel_counts(table, data)[0]
    assert panel["mcnemar_b"] == panel[f"n_{CT.A_WINS}"]
    assert panel["mcnemar_c"] == panel[f"n_{CT.B_WINS}"]
    assert panel["mcnemar_p_floor"] <= panel["mcnemar_p"]


# -- the encoding ---------------------------------------------------------


def test_each_win_takes_its_own_methods_hue():
    from scripts.analysis.v5.modules.methods import LABEL_HUES

    ink = F.category_ink("spotter_cbg", "octant_cbg_hull")
    assert ink[CT.A_WINS] == LABEL_HUES["SPO"]
    assert ink[CT.B_WINS] == LABEL_HUES["OCT-H"]
    assert ink[CT.TIED] not in LABEL_HUES.values()
    assert ink[CT.NEITHER] not in LABEL_HUES.values()


def test_every_category_has_a_shape_of_its_own():
    """The figure must survive greyscale and colour-vision deficiency, where
    SPO's red and OCT-H's green land within 0.03 of each other in luminance."""
    shapes = [F.CATEGORY_MARKER[c][0] for c in CT.CATEGORIES]
    assert len(set(shapes)) == len(CT.CATEGORIES)


def test_the_pair_names_the_file():
    assert F.pair_slug("spotter_cbg", "octant_cbg_hull") == "SPO-vs-OCT-H"


def test_the_source_rung_comes_off_the_ladder():
    from scripts.analysis.v5.modules import grid as G

    assert F.SOURCE_NSIDE in G.NSIDE_LADDER


# -- the relaxation -------------------------------------------------------


def _dense(n=6):
    """Six sites inside three degrees -- as03's north-east, near enough."""
    rng = np.random.default_rng(0)
    return np.c_[-75.0 + rng.uniform(0, 3, n), 39.5 + rng.uniform(0, 3, n)]


def test_no_two_labels_overlap_even_in_a_dense_cluster():
    """Boxes, in the Chebyshev metric -- which is what overlapping means and
    what the eye sees. The relaxation separates in Euclidean, which is why
    `MIN_SEP_UNITS` is the diagonal and not 1.0."""
    a = _dense()
    got = F.label_crowding(a, F.place_site_labels(a, F.DEFAULT_EXTENT), F.DEFAULT_EXTENT)
    assert got["n_overlapping_label_pairs"] == 0


def test_a_lone_label_stays_under_its_own_marker():
    a = np.array([[-95.0, 41.0]])
    placed = F.place_site_labels(a, F.DEFAULT_EXTENT)
    assert placed[0][0] == pytest.approx(a[0][0], abs=0.05)
    assert placed[0][1] < a[0][1], "the label sits below the marker it labels"


def test_a_label_is_pushed_clear_of_a_neighbours_marker():
    """The fix `figure_outcome_map` documents as its known limitation. Without
    `obstacles=` a label comes to rest on top of another site's marker."""
    a = _dense()
    ext = F.DEFAULT_EXTENT
    scale = F.label_scale(ext)
    free = F.place_site_labels(a, ext)
    naive = (
        F.FOM.place_labels(
            a * scale,
            tuple(np.array(ext) * np.repeat(scale, 2)),
            min_sep=F.MIN_SEP_UNITS,
            pad=F.FRAME_PAD_UNITS,
            offset=F.LABEL_OFFSET_UNITS,
        )
        / scale
    )
    assert (
        F.label_crowding(a, free, ext)["min_label_marker_units"]
        > F.label_crowding(a, naive, ext)["min_label_marker_units"]
    )


def test_labels_stay_inside_the_frame():
    edge = np.array([[-127.5, 54.5], [-127.4, 54.4], [-63.5, 21.5]])
    placed = F.place_site_labels(edge, F.DEFAULT_EXTENT)
    lon_min, lon_max, lat_min, lat_max = F.DEFAULT_EXTENT
    assert (placed[:, 0] > lon_min).all() and (placed[:, 0] < lon_max).all()
    assert (placed[:, 1] > lat_min).all() and (placed[:, 1] < lat_max).all()


# -- the guards -----------------------------------------------------------


def test_one_method_against_itself_is_refused(run):
    with pytest.raises(ValueError, match="two methods"):
        F.load([run], method_a="alpha", method_b="alpha", analysis_root=run.root)


def test_a_method_missing_from_a_run_names_the_command(run):
    with pytest.raises(MissingArtifactError, match="classify --run-id"):
        F.load([run], method_a="alpha", method_b="nobody", analysis_root=run.root)


# -- the drawing ----------------------------------------------------------


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
    assert png.name == F.PNG_NAME.format(pair=pair)
    for template in (F.PNG_NAME, F.CSV_NAME, F.MANIFEST_NAME):
        assert (out / template.format(pair=pair)).exists()
    body = json.loads((out / F.MANIFEST_NAME.format(pair=pair)).read_text())
    assert body["source_nside"] == NSIDE
    assert body["panels"][-1]["run_id"] == "__pooled__"
    assert "solved_mask" in body["policy"]["correct"]
    assert "Los Angeles" in body["policy"]["category"]
    assert body["label_crowding"][0]["n_labels"] == 4


def test_the_figure_carries_no_statistics(run, monkeypatch):
    """McNemar and the category counts moved to the manifest. A p-value in a
    corner invites the reading that the map is the test; the test is over
    sites and the map is over places."""
    import matplotlib.pyplot as plt

    captured = {}
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: captured.setdefault("fig", fig))
    data = F.load(
        [run], method_a="alpha", method_b="beta", nside=NSIDE, analysis_root=run.root
    )
    png, crowding = F.render(data, F.contest_table(data), run.root / "contest.png")
    fig = captured["fig"]
    words = " ".join(t.get_text() for t in fig.findobj(plt.Text))
    real_close(fig)

    assert png.exists() and png.stat().st_size > 10_000
    assert len(crowding) == 1
    for banned in ("McNemar", "p=", "sites=", "tied="):
        assert banned not in words, f"{banned!r} belongs in the manifest, not the figure"


def test_ncols_stacks_the_panels(two_runs, monkeypatch):
    """The 1xN default is a 16-inch figure at three meshes; scaled to a paper's
    textwidth its labels fall under 3 pt. `--ncols 1` is the way out, so the
    figure keeps its panel width and grows downwards instead."""
    import matplotlib.pyplot as plt

    figs = []
    real_close = plt.close
    monkeypatch.setattr(plt, "close", figs.append)
    data = F.load(
        two_runs, method_a="alpha", method_b="beta", nside=NSIDE,
        analysis_root=two_runs[0].root,
    )
    table = F.contest_table(data)
    root = two_runs[0].root
    wide, _ = F.render(data, table, root / "wide.png")
    tall, _ = F.render(data, table, root / "tall.png", ncols=1)
    sizes = [f.get_size_inches() for f in figs]
    grids = [len(f.axes) for f in figs]
    for fig in figs:
        real_close(fig)

    assert grids == [2, 2]
    assert sizes[0][0] > sizes[1][0], "one row is wider"
    assert sizes[0][1] < sizes[1][1], "one column is taller"
    assert sizes[1][0] == pytest.approx(F._PANEL_W_IN), "a stacked panel keeps its width"
    assert wide.exists() and tall.exists()


def test_a_spare_panel_is_switched_off(run, monkeypatch):
    """One mesh in a two-wide grid leaves a hole; an empty cartopy axes draws
    a framed blank map, which reads as a mesh with no sites in it."""
    import matplotlib.pyplot as plt

    captured = {}
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: captured.setdefault("fig", fig))
    data = F.load(
        [run], method_a="alpha", method_b="beta", nside=NSIDE, analysis_root=run.root
    )
    F.render(data, F.contest_table(data), run.root / "spare.png", ncols=2)
    fig = captured["fig"]
    drawn = [ax for ax in fig.axes if ax.axison]
    real_close(fig)
    assert len(fig.axes) == 2 and len(drawn) == 1


def test_a_narrower_panel_makes_the_label_bigger_in_map_units():
    """The label box is a point size, so it does not shrink with the panel --
    which is why a three-column figure at textwidth cannot carry labels."""
    wide = F.label_scale(F.DEFAULT_EXTENT, panel_width=5.4)
    narrow = F.label_scale(F.DEFAULT_EXTENT, panel_width=2.3)
    assert (narrow < wide).all()
    span = F.DEFAULT_EXTENT[1] - F.DEFAULT_EXTENT[0]
    assert 1.0 / narrow[0] == pytest.approx(F.LABEL_W_IN * span / 2.3)


def test_no_labels_draws_the_markers_alone(run, monkeypatch):
    import matplotlib.pyplot as plt

    captured = {}
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: captured.setdefault("fig", fig))
    data = F.load(
        [run], method_a="alpha", method_b="beta", nside=NSIDE, analysis_root=run.root
    )
    _, crowding = F.render(
        data, F.contest_table(data), run.root / "bare.png", labels=False
    )
    fig = captured["fig"]
    words = " ".join(t.get_text() for t in fig.findobj(plt.Text))
    real_close(fig)

    assert crowding[0]["n_labels"] == 0
    assert "|" not in words
