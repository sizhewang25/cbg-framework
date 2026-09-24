"""Outcome bars: cell label outer, ring tier inner, ranked by correct-region share."""

from __future__ import annotations

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


def test_the_drawn_segments_close_at_one(table):
    cols = [f"share_{F.count_col(t, lab)}" for t, lab in F.segments()]
    assert np.allclose(table[cols].sum(axis=1), 1.0)


def test_group_shares_close_at_one(table):
    cols = [f"share_{F.group_col(g)}" for g in F.GROUPS]
    assert np.allclose(table[cols].sum(axis=1), 1.0)


def test_segments_are_cell_label_outer_ring_inner():
    segs = F.segments()
    assert segs[:4] == [("ring0", "true"), ("ring1", "true"), ("ring2", "true"), ("beyond", "true")]
    assert segs[-1] == (F.FAILED, None)
    assert len(segs) == len(F.TIERS) * len(C.CELL_LABELS) + 1


def test_hatches_are_empty_right_left():
    assert F.CELL_HATCH == {"true": "", "wrong": "//", "outland": "\\\\"}


def test_each_panel_ranks_by_serving_region_first(table):
    assert F.panel_order(table, "syn") == ["regional", "tight"]


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
    # tight: true + wrong; regional: true + outland + no answer.
    assert len(borders) == 5
    rails = [p for p in ax.patches if p.get_width() == pytest.approx(F._RAIL_W)]
    assert {p.get_hatch() or "" for p in rails} == {"", "//", "\\\\"}
    # No rail for "no answer": 2 groups for tight, 2 for regional.
    assert len(rails) == 4


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
    bad.loc[0, "n_beyond_cell_true"] -= 1
    with pytest.raises(ValueError, match="do not partition"):
        C.guard_cross_tab(bad)


def test_csv_columns_are_all_emitted(table):
    missing = [c for c in F.csv_columns() if c not in table.columns and c not in ("run_id", "dataset")]
    assert missing == []
    for g in F.GROUPS:
        assert F.group_col(g) in F.csv_columns()
