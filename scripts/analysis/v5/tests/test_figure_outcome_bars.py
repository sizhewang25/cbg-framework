"""Outcome bars: the stack partitions, the encoding holds, the order is the ladder."""

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
    good = [_offset(SEATTLE, north_km=2)] * 6 + [OMAHA] * 2 + [ARCTIC] * 2
    bad = [_offset(SEATTLE, east_km=400)] * 3 + [OMAHA] * 4 + [ARCTIC] * 2 + [None]
    summary = C.summarize(
        {"good": _scored(space, good), "bad": _scored(space, bad, ["SUCCESS"] * 9 + ["ERROR"])},
        128,
    )
    summary.insert(0, "run_id", "syn-x")
    summary.insert(1, "dataset", "syn")
    return F._add_shares(summary.reset_index(drop=True))


def test_the_drawn_segments_close_at_one(table):
    cols = [f"share_{F.count_col(t, lab)}" for t, lab in F.segments()]
    assert np.allclose(table[cols].sum(axis=1), 1.0)


def test_segments_are_tiers_by_cell_label_then_failed():
    segs = F.segments()
    assert segs[-1] == (F.FAILED, None)
    assert segs[:3] == [("ring0", "true"), ("ring0", "wrong"), ("ring0", "outland")]
    assert len(segs) == len(F.TIERS) * len(C.CELL_LABELS) + 1


def test_hatches_are_empty_right_left():
    assert F.CELL_HATCH == {"true": "", "wrong": "//", "outland": "\\\\"}


def test_each_panel_ranks_down_the_ring_ladder(table):
    assert F.panel_order(table, "syn") == ["good", "bad"]


def test_render_draws_hatches_and_one_border_per_tier(table, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured = {}
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: captured.setdefault("fig", fig))
    png = F.render(table, 128, tmp_path)
    real_close(captured["fig"])
    assert png.exists()

    ax = captured["fig"].axes[0]
    patches = [p for p in ax.patches]
    borders = [p for p in patches if p.get_facecolor()[3] == 0 and p.get_linewidth() > 1.0]
    hatched = {p.get_hatch() for p in patches if p.get_facecolor()[3] > 0}
    # good: ring0 + beyond; bad: beyond + failed.
    assert len(borders) == 4
    assert {"//", "\\\\"} <= hatched and ("" in hatched or None in hatched)


def test_guard_rejects_a_table_whose_cross_tab_does_not_close(table, tmp_path):
    bad = table.copy()
    bad.loc[0, "n_beyond_cell_wrong"] += 1
    bad.loc[0, "n_beyond"] += 1
    bad.loc[0, "n_failed"] -= 1
    C.guard_partition(bad)
    bad.loc[0, "n_beyond_cell_true"] -= 1
    with pytest.raises(ValueError, match="do not partition"):
        C.guard_cross_tab(bad)


def test_csv_columns_are_all_emitted_by_summarize(table):
    missing = [c for c in F.csv_columns() if c not in table.columns and c not in ("run_id", "dataset")]
    assert missing == []
