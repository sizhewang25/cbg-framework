"""The exclusive cohorts: which offset is read, and against which denominator."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import contest as CT
from scripts.analysis.v5.modules import figure_exclusive_error as F
from scripts.analysis.v5.tests.conftest import NSIDE


def targets(rows) -> pd.DataFrame:
    """A minimal per-target frame: `(cohort, offset_a, offset_b)` triples."""
    return pd.DataFrame(
        [
            {"tg_id": f"tg-{i}", "cohort": c, "offset_a": a, "offset_b": b}
            for i, (c, a, b) in enumerate(rows)
        ]
    )


@pytest.fixture(scope="module")
def run(contest_run):
    return contest_run


# -- which offset each cohort reads ---------------------------------------


def test_a_cohort_reads_the_offset_of_the_method_that_was_right():
    """The other method's offset there measures nothing: it never reached the
    cell, so how far out it landed is not a property of this cohort."""
    table = targets(
        [(CT.COHORT_ONLY_A, 7, 99), (CT.COHORT_ONLY_B, 99, 3)]
    )
    assert list(F.cohort_offsets(table, CT.COHORT_ONLY_A)) == [7]
    assert list(F.cohort_offsets(table, CT.COHORT_ONLY_B)) == [3]


def test_the_shared_and_missed_cohorts_are_not_drawn():
    """On `both` the offsets are comparable and `figure_error_diff` compares
    them; on `neither` there is no correct prediction to measure."""
    assert set(F.DRAWN) == {CT.COHORT_ONLY_A, CT.COHORT_ONLY_B}
    assert CT.COHORT_BOTH not in F.DRAWN and CT.COHORT_NEITHER not in F.DRAWN


def test_a_non_exclusive_cohort_is_refused():
    with pytest.raises(ValueError, match="not an exclusive cohort"):
        F.cohort_offsets(targets([(CT.COHORT_BOTH, 1, 1)]), CT.COHORT_BOTH)


def test_the_partition_accounts_for_every_target():
    table = targets(
        [(CT.COHORT_BOTH, 1, 1), (CT.COHORT_ONLY_A, 2, 9),
         (CT.COHORT_ONLY_B, 9, 3), (CT.COHORT_NEITHER, 9, 9)]
    )
    got = F.partition(table)
    assert sum(got[f"n_{c}"] for c in CT.COHORTS) == got["n_targets"] == 4
    assert sum(got[f"share_{c}"] for c in CT.COHORTS) == pytest.approx(1.0)


# -- the tail, and its denominators ---------------------------------------


def test_the_tail_is_reported_against_both_denominators():
    """"A small portion" means nothing without a stated denominator, and the
    two differ by a factor of five on the real data."""
    table = targets(
        [(CT.COHORT_ONLY_A, 30, 0)] + [(CT.COHORT_ONLY_A, 1, 0)] * 3
        + [(CT.COHORT_NEITHER, 0, 0)] * 4
    )
    got = F.cohort_stats(table, CT.COHORT_ONLY_A)
    assert got["n_beyond_20"] == 1
    assert got["share_of_cohort_beyond_20"] == pytest.approx(0.25)
    assert got["share_of_all_beyond_20"] == pytest.approx(0.125)
    assert got["share_of_cohort_beyond_20"] > got["share_of_all_beyond_20"]


def test_an_empty_cohort_reports_its_absence_not_a_shape():
    got = F.cohort_stats(targets([(CT.COHORT_BOTH, 1, 1)]), CT.COHORT_ONLY_A)
    assert got["n_targets"] == 0
    assert "offset_p50" not in got


def test_zeros_are_counted_because_a_log_axis_cannot_draw_them():
    table = targets([(CT.COHORT_ONLY_B, 9, 0), (CT.COHORT_ONLY_B, 9, 4)])
    got = F.cohort_stats(table, CT.COHORT_ONLY_B)
    assert got["n_at_zero"] == 1 and got["offset_min"] == 0


# -- the drawing ----------------------------------------------------------


def _drawn(table, monkeypatch, out):
    import matplotlib.pyplot as plt

    captured = {}
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: captured.setdefault("fig", fig))
    F.render(table, "spotter_cbg", "octant_cbg_hull", out)
    return captured["fig"], real_close


def _both_cohorts():
    return targets(
        [(CT.COHORT_ONLY_A, o, 99) for o in (1, 2, 4, 35)]
        + [(CT.COHORT_ONLY_B, 99, o) for o in (0, 1, 2, 8)]
    )


def test_each_curve_takes_its_methods_hue_and_names_its_cohort(tmp_path, monkeypatch):
    from scripts.analysis.v5.modules.methods import LABEL_HUES

    fig, close = _drawn(_both_cohorts(), monkeypatch, tmp_path / "a.png")
    ax = fig.axes[0]
    colours = [ln.get_color() for ln in ax.lines]
    names = [txt.get_text() for txt in ax.get_legend().get_texts()]
    close(fig)
    assert colours == [LABEL_HUES["SPO"], LABEL_HUES["OCT-H"]]
    assert names == ["SPO Correct Only", "OCT-H Correct Only"]


def test_each_curve_is_normalised_to_its_own_cohort(tmp_path, monkeypatch):
    """A common denominator would compress the tail the figure exists to
    show, and the cohorts are nearly the same size anyway."""
    fig, close = _drawn(_both_cohorts(), monkeypatch, tmp_path / "b.png")
    tops = [float(np.max(ln.get_ydata())) for ln in fig.axes[0].lines]
    close(fig)
    assert tops == pytest.approx([1.0, 1.0])


def test_each_curve_is_marked_where_it_ends(tmp_path, monkeypatch):
    """The two ceilings are the claim -- 35 grids against 8 on the real data
    -- and a step curve's last riser carrying one target is easy to miss."""
    fig, close = _drawn(_both_cohorts(), monkeypatch, tmp_path / "ends.png")
    ax = fig.axes[0]
    marked = sorted(txt.get_text() for txt in ax.texts)
    rules = [c for c in ax.collections if c.get_linestyle()[0][1] is not None]
    heights = [txt.xy[1] for txt in ax.texts]
    close(fig)

    assert marked == ["35", "8"], "each cohort's longest offset is named"
    assert len(rules) == 2, "one dashed rule per cohort"
    assert all(h > 1.0 for h in heights), "the numbers clear anything a CDF can reach"


def test_the_end_markers_take_their_cohorts_hue(tmp_path, monkeypatch):
    from scripts.analysis.v5.modules.methods import LABEL_HUES

    fig, close = _drawn(_both_cohorts(), monkeypatch, tmp_path / "hue.png")
    ax = fig.axes[0]
    by_text = {txt.get_text(): txt.get_color() for txt in ax.texts}
    close(fig)
    assert by_text["35"] == LABEL_HUES["SPO"]
    assert by_text["8"] == LABEL_HUES["OCT-H"]


def test_the_axis_says_it_is_symlog(tmp_path, monkeypatch):
    """A log-looking axis that shows zero needs to say why it can."""
    fig, close = _drawn(_both_cohorts(), monkeypatch, tmp_path / "say.png")
    label = fig.axes[0].get_xlabel()
    close(fig)
    assert "symlog" in label.lower()


def test_the_axis_is_symlog_so_a_zero_offset_is_drawn(tmp_path, monkeypatch):
    fig, close = _drawn(_both_cohorts(), monkeypatch, tmp_path / "c.png")
    ax = fig.axes[0]
    scale, lo = ax.get_xscale(), ax.get_xlim()[0]
    ticks = list(ax.get_xticks())
    close(fig)
    assert scale == "symlog"
    assert lo < 0, "zero must sit inside the frame, not on its edge"
    assert 0 in ticks


def test_a_cohort_with_no_exclusive_target_is_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="no exclusive cohort"):
        _drawn(targets([(CT.COHORT_BOTH, 1, 1)]), monkeypatch, tmp_path / "d.png")


def test_one_empty_cohort_still_draws_the_other(tmp_path, monkeypatch):
    table = targets([(CT.COHORT_ONLY_A, o, 99) for o in (1, 5)])
    fig, close = _drawn(table, monkeypatch, tmp_path / "e.png")
    n = len(fig.axes[0].lines)
    close(fig)
    assert n == 1


def test_the_figure_carries_no_title_and_its_labels_fit(tmp_path, monkeypatch):
    fig, close = _drawn(_both_cohorts(), monkeypatch, tmp_path / "f.png")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax = fig.axes[0]
    canvas = fig.get_window_extent()
    title, sup = ax.get_title(), fig._suptitle
    boxes = [
        ax.xaxis.get_label().get_window_extent(renderer=r),
        ax.yaxis.get_label().get_window_extent(renderer=r),
    ]
    close(fig)
    assert title == "" and sup is None
    for box in boxes:
        assert box.x0 >= canvas.x0 - 0.5 and box.x1 <= canvas.x1 + 0.5
        assert box.y0 >= canvas.y0 - 0.5 and box.y1 <= canvas.y1 + 0.5


# -- end to end -----------------------------------------------------------


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
    assert body["drawn"] == list(F.DRAWN)
    assert "stated denominator" in body["policy"]["tail"]
    twin = pd.read_csv(out / F.CSV_NAME.format(pair=pair))
    assert list(twin.columns) == F.csv_columns()
    # the twin is every target, so the partition is checkable from it alone
    assert len(twin) == body["partition"]["n_targets"]
    assert twin["cohort"].value_counts().to_dict() == {
        c: body["partition"][f"n_{c}"]
        for c in CT.COHORTS
        if body["partition"][f"n_{c}"]
    }
