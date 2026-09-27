"""The paired error difference: the cohort, the sign, and the curve's geometry."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import contest as CT
from scripts.analysis.v5.modules import figure_error_diff as F
from scripts.analysis.v5.tests.conftest import NSIDE


def paired(diffs, datasets=None, sites=None) -> pd.DataFrame:
    """A minimal cohort frame: only what the statistics read."""
    n = len(diffs)
    d = np.asarray(diffs, dtype=int)
    return pd.DataFrame(
        {
            "run_id": [f"r{x}" for x in (datasets or ["a"] * n)],
            "dataset": datasets or ["a"] * n,
            "tg_id": [f"tg-{i}" for i in range(n)],
            "site_id": sites if sites is not None else list(range(n)),
            "offset_a": np.maximum(d, 0),
            "offset_b": np.maximum(-d, 0),
            F.DIFF_COL: d,
            "closer": np.where(
                d < 0, CT.CLOSER_A, np.where(d > 0, CT.CLOSER_B, CT.CLOSER_EQUAL)
            ),
        }
    )


@pytest.fixture(scope="module")
def run(contest_run):
    return contest_run


# -- the numbers ----------------------------------------------------------


def test_the_three_shares_partition_the_cohort():
    got = F.shares(paired([-2, -1, 0, 0, 0, 1, 2, 3]))
    assert got["n_targets"] == 8
    assert got["share_a"] + got["share_b"] + got["share_equal"] == pytest.approx(1.0)
    assert (got["share_a"], got["share_equal"], got["share_b"]) == (0.25, 0.375, 0.375)


def test_the_curves_geometry_is_the_shares():
    """What makes the figure readable without annotation: the height left of
    zero is `share_a`, and the jump at zero is `share_equal`."""
    table = paired([-2, -1, 0, 0, 0, 1, 2, 3])
    got = F.shares(table)
    x, y = F.ecdf(table[F.DIFF_COL].to_numpy())
    below_zero = y[np.searchsorted(x, 0.0, side="left") - 1]
    at_zero = y[np.searchsorted(x, 0.0, side="right") - 1]
    assert below_zero == pytest.approx(got["share_a"])
    assert at_zero - below_zero == pytest.approx(got["share_equal"])
    assert 1.0 - at_zero == pytest.approx(got["share_b"])


def test_the_percentiles_are_reported_not_a_mean():
    """The difference is integral and skewed; the median is the number the
    prose quotes."""
    got = F.diff_stats(paired([0, 1, 2, 3, 19]))
    assert got["diff_p50"] == 2.0
    assert (got["diff_min"], got["diff_max"]) == (0, 19)
    assert "diff_mean" not in got


def test_per_mesh_covers_every_dataset():
    table = paired([1, 2, -1, 3], datasets=["as01", "as01", "as02", "as02"])
    got = F.per_mesh(table)
    assert [g["dataset"] for g in got] == ["as01", "as02"]
    assert sum(g["n_targets"] for g in got) == len(table)


def test_per_site_counts_sites_not_targets():
    """565 targets are ~20 replicas of 34 sites, so the target-level share is
    not that many independent observations."""
    table = paired([5, 5, 5, -5, -5, -5], sites=[0, 0, 0, 1, 1, 1])
    got = F.per_site(table)
    assert got["n_site_pairs"] == 2
    assert got[f"n_sites_{CT.CLOSER_B}"] == 1
    assert got[f"n_sites_{CT.CLOSER_A}"] == 1


def test_the_axis_carries_the_sign_convention():
    """The figure turns on which way the difference runs, so the axis says
    it rather than leaving it to the caption."""
    got = F.x_label("spotter_cbg", "octant_cbg_hull")
    assert "SPO" in got and "OCT-H" in got
    assert got.index("SPO") < got.index("OCT-H"), "the minuend reads first"


def test_each_direction_names_the_method_it_favours():
    got = F.direction_labels("spotter_cbg", "octant_cbg_hull")
    assert got[CT.CLOSER_A].startswith("SPO")
    assert got[CT.CLOSER_B].startswith("OCT-H")


def test_the_arrows_sit_above_anything_the_curve_can_reach():
    """A CDF stops at 1, so a band above it is empty whatever the data does --
    which is why the arrows are placed in data units and not guessed at."""
    assert F._ARROW_Y > 1.0
    assert 1.0 + F._HEADROOM > F._ARROW_Y


def test_the_ecdf_is_exact_at_every_observation():
    values = np.array([-2, 0, 0, 3])
    x, y = F.ecdf(values)
    for v in np.unique(values):
        share = float((values <= v).mean())
        assert y[np.searchsorted(x, v, side="right") - 1] == pytest.approx(share)


def test_ticks_are_spelled_not_left_to_the_log_locator():
    """A symlog locator labels decades and leaves the 1-to-3 band unmarked --
    which is where the median sits."""
    got = F.visible_ticks(np.array([-6, 19]))
    assert 0 in got and 1 in got and 2 in got
    assert max(got) <= 20 and min(got) >= -10


# -- the drawing ----------------------------------------------------------


def _drawn(table, monkeypatch, **kw):
    import matplotlib.pyplot as plt

    captured = {}
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: captured.setdefault("fig", fig))
    F.render(table, "spotter_cbg", "octant_cbg_hull", kw.pop("out"), **kw)
    return captured["fig"], real_close


def test_the_two_arrows_point_opposite_ways(tmp_path, monkeypatch):
    """Arrows, not tinted half-planes: a direction is what the axis encodes,
    and an arrow reads in greyscale where two pale tints do not."""
    fig, close = _drawn(paired([-2, 0, 3]), monkeypatch, out=tmp_path / "arrows.png")
    ax = fig.axes[0]
    arrows = [a for a in ax.texts if a.arrow_patch is not None]
    spans = [(a.xy[0], a.xyann[0]) for a in arrows]
    names = sorted(a.get_text() for a in arrows)
    close(fig)

    assert len(arrows) == 2
    assert names == sorted(F.direction_labels("spotter_cbg", "octant_cbg_hull").values())
    # one reaches left of where its text sits, the other right
    assert any(tip < anchor for tip, anchor in spans)
    assert any(tip > anchor for tip, anchor in spans)


def test_nothing_is_drawn_in_colour(tmp_path, monkeypatch):
    """The direction was tinted half-planes first; the arrows replaced them,
    so no method hue is left on this figure to be read as data."""
    fig, close = _drawn(paired([-2, 0, 3]), monkeypatch, out=tmp_path / "grey.png")
    ax = fig.axes[0]
    patches = [p for p in ax.patches if p.get_alpha() not in (None, 0)]
    legend = ax.get_legend()
    close(fig)
    assert not patches and legend is None


def test_the_axis_is_symlog_and_centred_on_zero(tmp_path, monkeypatch):
    fig, close = _drawn(paired([-6] + [0] * 3 + [19]), monkeypatch, out=tmp_path / "a.png")
    ax = fig.axes[0]
    scale, (lo, hi) = ax.get_xscale(), ax.get_xlim()
    close(fig)
    assert scale == "symlog"
    assert lo == pytest.approx(-hi), "zero sits at the centre"


def test_a_cohort_that_never_differs_still_draws(tmp_path, monkeypatch):
    """Range zero would collapse the limits onto a point."""
    fig, close = _drawn(paired([0] * 5), monkeypatch, out=tmp_path / "b.png")
    lo, hi = fig.axes[0].get_xlim()
    close(fig)
    assert lo < 0 < hi


def test_the_figure_carries_no_title(tmp_path, monkeypatch):
    fig, close = _drawn(paired([-1, 0, 2]), monkeypatch, out=tmp_path / "c.png")
    title, sup = fig.axes[0].get_title(), fig._suptitle
    close(fig)
    assert title == "" and sup is None


def test_the_axis_labels_fit_inside_the_figure(tmp_path, monkeypatch):
    """Saved at a fixed canvas, so an overrunning label is clipped silently."""
    fig, close = _drawn(paired([-6, 0, 19]), monkeypatch, out=tmp_path / "d.png")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax = fig.axes[0]
    canvas = fig.get_window_extent()
    boxes = [
        ax.xaxis.get_label().get_window_extent(renderer=r),
        ax.yaxis.get_label().get_window_extent(renderer=r),
    ]
    close(fig)
    for box in boxes:
        assert box.x0 >= canvas.x0 - 0.5 and box.x1 <= canvas.x1 + 0.5
        assert box.y0 >= canvas.y0 - 0.5 and box.y1 <= canvas.y1 + 0.5


# -- end to end -----------------------------------------------------------


def test_a_win_site_one_method_never_solved_is_not_judged(run, monkeypatch):
    """The companion compares per-site medians, and a site where one of them
    solved nothing has no median to compare. The fixture has exactly that."""
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "close", lambda fig=None: None)
    data = CT.load(
        [run], method_a="alpha", method_b="beta", nside=NSIDE, analysis_root=run.root
    )
    got = F.win_site_companion(data, CT.contest_table(data))
    plt.close("all")
    assert got["n_sites_compared"] < got["n_win_sites"]
    assert (
        got["n_sites_a_more_accurate"] + got["n_sites_b_more_accurate"]
        <= got["n_sites_compared"]
    )


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
    assert "caption names it" in body["subject_note"]
    assert "different populations" in body["policy"]["cohort"]
    assert "does not move" in body["policy"]["sign"]
    assert body["companion_win_sites"]["cohort"].startswith("every solved target")
    twin = pd.read_csv(out / F.CSV_NAME.format(pair=pair))
    assert list(twin.columns) == F.csv_columns()
    assert len(twin) == body["pooled"]["n_targets"]


def test_a_cohort_of_nothing_is_refused(run, monkeypatch):
    """Two methods that never agree on a correct answer have no paired
    comparison, and an empty CDF is not the way to say so."""
    monkeypatch.setattr(
        F.CT, "both_correct_targets", lambda data: pd.DataFrame(columns=CT.PAIRED_COLUMNS)
    )
    with pytest.raises(ValueError, match="no paired cohort"):
        F.build_for_runs(
            [run], method_a="alpha", method_b="beta", analysis_root=run.root
        )
