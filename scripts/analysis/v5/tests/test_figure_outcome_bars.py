"""Outcome bars: cell label outer, ring tier inner, ranked by correct-region share."""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import answer_space as A
from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import figure_outcome_bars as F

SEATTLE = (47.449, -122.309)
OMAHA = (41.2565, -95.9345)
CHICAGO = (41.8781, -87.6298)
MIAMI = (25.7932, -80.29)
ARCTIC = (63.7369, -97.4685)


def _offset(point, north_km=0.0, east_km=0.0):
    lat, lon = point
    return (lat + north_km / 111.195, lon + east_km / (111.195 * np.cos(np.radians(lat))))


@pytest.fixture(scope="module")
def space():
    t = pd.DataFrame(
        {
            "tg_id": ["tg-0", "tg-1", "tg-2", "tg-3"],
            "tg_lat": [c[0] for c in (SEATTLE, OMAHA, CHICAGO, MIAMI)],
            "tg_lon": [c[1] for c in (SEATTLE, OMAHA, CHICAGO, MIAMI)],
        }
    )
    return A.build_answer_space(t, nside=128, run_id="syn")


def _scored(space, preds, statuses=None):
    tg = space.tgs.iloc[0]
    n = len(preds)
    frame = pd.DataFrame(
        {
            "tg_id": [f"{tg['tg_id']}"] * n,
            "tg_lat": [tg["tg_lat"]] * n,
            "tg_lon": [tg["tg_lon"]] * n,
            "pred_lat": [p[0] if p else np.nan for p in preds],
            "pred_lon": [p[1] if p else np.nan for p in preds],
            "status": statuses or ["SUCCESS"] * n,
        }
    )
    return C.score_method(frame, space)


@pytest.fixture(scope="module")
def table(space):
    # "tight": in its grid, but few; "regional": far off yet in the right
    # serving region more often, plus one unanswered. Ranking by ring would put
    # tight first; ranking by serving region puts regional first.
    tight = [_offset(SEATTLE, north_km=2)] * 3 + [OMAHA] * 7
    regional = [_offset(SEATTLE, east_km=400)] * 5 + [ARCTIC] * 4 + [None]
    summary = C.summarize(
        {
            "tight": _scored(space, tight),
            "regional": _scored(space, regional, ["SUCCESS"] * 9 + ["ERROR"]),
        },
        128,
    )
    summary.insert(0, "run_id", "syn-x")
    summary.insert(1, "dataset", "syn")
    return F._add_shares(summary.reset_index(drop=True))


@pytest.mark.parametrize("mode", F.MODES)
def test_the_drawn_segments_close_at_one(table, mode):
    cols = [f"share_{F.segment_col(t, lab)}" for t, lab in F.segments(mode)]
    assert np.allclose(table[cols].sum(axis=1), 1.0)


def test_group_shares_close_at_one(table):
    cols = [f"share_{F.group_col(g)}" for g in F.GROUPS]
    assert np.allclose(table[cols].sum(axis=1), 1.0)


def test_bounded_segments_are_cell_label_outer_ring_inner():
    segs = F.segments(F.BOUNDED)
    assert segs[:4] == [
        ("ring0", "correct"), ("ring1", "correct"), ("ring2", "correct"), ("beyond", "correct"),
    ]
    assert segs[-1] == (F.UNANSWERED, F.UNANSWERED)
    # Only the two graded labels get a tier breakdown; `unanswered` has no
    # ring, so it takes one ungraded slot.
    assert len(segs) == len(F.TIERS) * len(F.GRADED) + 1


def test_unbounded_segments_are_the_cell_axis_alone():
    segs = F.segments(F.UNBOUNDED)
    assert [lab for _, lab in segs] == list(F.GROUPS)
    assert len(segs) == len(C.CELL_LABELS)
    # No ring tier anywhere: that is the whole difference between the modes.
    assert all(tier in (None, F.UNANSWERED) for tier, _ in segs)


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="unknown mode"):
        F.segments("landmass")


def test_hatches_are_empty_right_empty():
    assert F.CELL_HATCH == {"correct": "", "wrong": "//", F.UNANSWERED: ""}


def test_only_the_unbounded_figure_takes_a_filename_token():
    assert F.mode_token(F.BOUNDED) == ""
    assert F.mode_token(F.UNBOUNDED) == ".unbounded"


def test_each_panel_ranks_by_serving_region_first(table):
    assert F.panel_order(table, "syn") == ["regional", "tight"]


def _tied_pair(sp_ring0, soi_ring0):
    """One dataset, S-P and SOI on the same correct-cell share, differing only
    in how tight their correct predictions are."""
    rows = []
    for method, ring0 in (("shortest_ping", sp_ring0), ("million_scale_cbg", soi_ring0)):
        rows.append({
            "dataset": "syn", "method": method, "n_tgs": 100,
            "n_cell_correct": 42, "n_cell_wrong": 58, "n_cell_unanswered": 0,
            "n_ring0_cell_correct": ring0, "n_ring1_cell_correct": 0,
            "n_ring2_cell_correct": 0, "n_beyond_cell_correct": 42 - ring0,
            "n_ring0_cell_wrong": 0, "n_ring1_cell_wrong": 0,
            "n_ring2_cell_wrong": 0, "n_beyond_cell_wrong": 58,
        })
    return pd.DataFrame(rows)


def test_soi_is_seated_above_shortest_ping_on_a_tie():
    # S-P is the tighter of the two, so the ring keys would seat it first --
    # but the drawn share is 42% for both, so the tie reads SOI first.
    order = F.panel_order(_tied_pair(sp_ring0=18, soi_ring0=15), "syn")
    assert [F.method_label(m) for m in order] == ["SOI", "S-P"]


def test_a_real_gap_still_seats_shortest_ping_above_soi():
    frame = _tied_pair(sp_ring0=18, soi_ring0=15)
    frame.loc[frame["method"] == "shortest_ping", ["n_cell_correct", "n_cell_wrong"]] = [50, 50]
    assert [F.method_label(m) for m in F.panel_order(frame, "syn")] == ["S-P", "SOI"]


def test_panel_tags_are_drawn_only_when_there_is_more_than_one_panel(table, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {}
    real = plt.Figure.savefig

    def capture(self, *args, **kwargs):
        titles[len(self.axes)] = [ax.get_title() for ax in self.axes]
        return real(self, *args, **kwargs)

    plt.Figure.savefig = capture
    try:
        F.render(table, 64, tmp_path, png_name="one.png")
        two = pd.concat([table, table.assign(dataset="syn2")], ignore_index=True)
        F.render(two, 64, tmp_path, png_name="two.png")
    finally:
        plt.Figure.savefig = real
    assert not titles[1][0].startswith("(")
    assert [t[:4] for t in titles[2]] == ["(a) ", "(b) "]


def test_render_borders_wrap_cell_groups_and_rails_carry_stripes(table, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured = {}
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: captured.setdefault("fig", fig))
    png = F.render(table, 128, tmp_path)
    real_close(captured["fig"])
    assert png.name == "outcome_bars.healpix-128.png" and png.exists()

    ax = captured["fig"].axes[0]
    borders = [p for p in ax.patches if p.get_facecolor()[3] == 0 and p.get_linewidth() > 1.0]
    # tight: correct + wrong. regional: correct + no answer -- it has no wrong
    # rows at all, because the far-flung predictions that used to be `outland`
    # are nearest the right seed and the unbounded rule now credits them.
    assert len(borders) == 4
    rails = [p for p in ax.patches if p.get_width() == pytest.approx(F._RAIL_W)]
    assert {p.get_hatch() or "" for p in rails} == {"", "//"}
    # No rail for "no answer": 2 for tight, 1 for regional.
    assert len(rails) == 3


def test_unbounded_draws_no_rail(table, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured = {}
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: captured.setdefault("fig", fig))
    png = F.render(
        table, 128, tmp_path, mode=F.UNBOUNDED,
        png_name="outcome_bars.unbounded.healpix-128.png",
    )
    real_close(captured["fig"])
    assert png.name == "outcome_bars.unbounded.healpix-128.png" and png.exists()

    ax = captured["fig"].axes[0]
    rails = [p for p in ax.patches if p.get_width() == pytest.approx(F._RAIL_W)]
    assert rails == [], "the bar is already the cell breakdown; a rail repeats it"
    # One border per drawn group, and the fills are the cell labels rather
    # than the ring tiers.
    fills = {tuple(round(v, 3) for v in p.get_facecolor()[:3]) for p in ax.patches
             if p.get_facecolor()[3] > 0}
    assert not (fills & {tuple(round(v, 3) for v in mcolors.to_rgb(F.TIER_INK[t]))
                         for t in ("ring1", "ring2", "beyond")})


def test_rail_labels_stay_inside_the_axis():
    ys = F._spread([0.01, 0.02, 0.97, 0.99], F._RAIL_LABEL_GAP)
    assert ys[0] >= F._RAIL_LABEL_GAP / 2 - 1e-9 and ys[-1] <= 1 - F._RAIL_LABEL_GAP / 2 + 1e-9
    assert all(b - a >= F._RAIL_LABEL_GAP - 1e-9 for a, b in zip(ys, ys[1:]))


def test_guard_rejects_a_table_whose_cross_tab_does_not_close(table, tmp_path):
    bad = table.copy()
    bad.loc[0, "n_beyond_cell_wrong"] += 1
    bad.loc[0, "n_beyond"] += 1
    bad.loc[0, "n_failed"] -= 1
    C.guard_partition(bad)
    bad.loc[0, "n_beyond_cell_correct"] -= 1
    with pytest.raises(ValueError, match="does not close"):
        C.guard_cross_tab(bad)


@pytest.mark.parametrize("mode", F.MODES)
def test_csv_columns_are_all_emitted(table, mode):
    missing = [
        c for c in F.csv_columns(mode)
        if c not in table.columns and c not in ("run_id", "dataset")
    ]
    assert missing == []
    for g in F.GROUPS:
        assert F.group_col(g) in F.csv_columns(mode)
