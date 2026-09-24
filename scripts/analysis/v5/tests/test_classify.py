"""Two labels per prediction, and the cross-tab that partitions them.

`TestTheArcticCase` pins the reason the cell partition must be bounded: the
prediction v3 credited to Seattle 2,360 km away is beyond on the grid axis and
outland on the cell axis, at every rung.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import answer_space as A
from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import grid as G

SEATTLE = (47.449, -122.309)
OMAHA = (41.2565, -95.9345)
CHICAGO = (41.8781, -87.6298)
MIAMI = (25.7932, -80.29)
ARCTIC = (63.7369, -97.4685)


def _space(nside=128, coords=(SEATTLE, OMAHA, CHICAGO, MIAMI)):
    t = pd.DataFrame(
        {
            "tg_id": [f"tg-{i}" for i in range(len(coords))],
            "tg_lat": [c[0] for c in coords],
            "tg_lon": [c[1] for c in coords],
        }
    )
    return A.build_answer_space(t, nside=nside, run_id="test-run")


def _frame(space, preds, statuses=None):
    """One row per prediction, all against the Seattle TG (`tg-0`)."""
    tg = space.tgs.iloc[0]
    n = len(preds)
    return pd.DataFrame(
        {
            "tg_id": [tg["tg_id"]] * n,
            "tg_lat": [tg["tg_lat"]] * n,
            "tg_lon": [tg["tg_lon"]] * n,
            "pred_lat": [p[0] if p else np.nan for p in preds],
            "pred_lon": [p[1] if p else np.nan for p in preds],
            "status": statuses or ["SUCCESS"] * n,
        }
    )


def _offset(point, north_km=0.0, east_km=0.0):
    lat, lon = point
    return (lat + north_km / 111.195, lon + east_km / (111.195 * np.cos(np.radians(lat))))


class TestCellLabel:
    def test_near_the_site_is_true(self):
        s = _space()
        out = C.score_method(_frame(s, [_offset(SEATTLE, north_km=2)]), s)
        assert out["cell_label"].iloc[0] == "true"
        assert out["ring"].iloc[0] == 0

    def test_in_another_serving_region_is_wrong(self):
        s = _space()
        out = C.score_method(_frame(s, [OMAHA]), s)
        assert out["cell_label"].iloc[0] == "wrong"
        assert out["pred_seed_id"].iloc[0] != out["tg_seed_id"].iloc[0]

    def test_no_prediction_is_none(self):
        s = _space()
        out = C.score_method(_frame(s, [None]), s)
        assert out["cell_label"].iloc[0] == C.NO_PREDICTION
        assert out["ring"].iloc[0] == -1 and out["pred_seed_id"].iloc[0] == -1

    def test_far_but_in_the_right_serving_region(self):
        """Beyond on the grid axis, true on the cell axis: 400 km east of
        Seattle is still nearer Seattle's seed than Omaha's. The case the
        cross-tab exists to show."""
        s = _space()
        out = C.score_method(_frame(s, [_offset(SEATTLE, east_km=400)]), s)
        assert out["ring"].iloc[0] == -1
        assert out["cell_label"].iloc[0] == "true"

    def test_distances_to_tg_and_to_seed(self):
        s = _space()
        out = C.score_method(_frame(s, [_offset(SEATTLE, north_km=10)]), s)
        # A single-site seed sits on its site, so both distances agree.
        assert out["pred_dist_to_tg_km"].iloc[0] == pytest.approx(10, abs=0.05)
        assert out["pred_dist_to_seed_km"].iloc[0] == pytest.approx(10, abs=0.05)


class TestTheArcticCase:
    @pytest.mark.parametrize("nside", G.NSIDE_LADDER)
    def test_it_is_beyond_and_outland_at_every_rung(self, nside):
        s = _space(nside=nside)
        out = C.score_method(_frame(s, [ARCTIC]), s)
        assert out["ring"].iloc[0] == -1
        assert out["cell_label"].iloc[0] == "outland"
        assert not out["pred_in_landmass"].iloc[0]


class TestSummary:
    def _summary(self, preds, statuses=None):
        s = _space()
        scored = C.score_method(_frame(s, preds, statuses), s)
        return C.summarize({"m": scored}, s.nside).iloc[0]

    def test_cross_tab_partitions_every_tier(self):
        row = self._summary(
            [_offset(SEATTLE, north_km=2), OMAHA, ARCTIC, _offset(SEATTLE, east_km=400), None],
            ["SUCCESS"] * 4 + ["ERROR"],
        )
        assert row["n_tgs"] == 5 and row["n_solved"] == 4 and row["n_failed"] == 1
        assert row["n_ring0_cell_true"] == 1
        assert row["n_beyond_cell_wrong"] == 1  # Omaha
        assert row["n_beyond_cell_outland"] == 1  # Arctic
        assert row["n_beyond_cell_true"] == 1  # 400 km east
        assert row["n_cell_true"] == 2 and row["accuracy_cell_true"] == pytest.approx(0.4)

    def test_fallback_is_labelled_but_not_counted(self):
        s = _space()
        scored = C.score_method(_frame(s, [_offset(SEATTLE, north_km=2)], ["FALLBACK"]), s)
        assert scored["cell_label"].iloc[0] == "true"
        row = C.summarize({"m": scored}, s.nside).iloc[0]
        assert row["n_cell_true"] == 0 and row["n_failed"] == 1
        assert row["accuracy_ring0"] == 0.0

    def test_answered_without_a_coordinate_is_failed_not_beyond(self):
        """A BASELINE row with no eval-source match: answered, nothing to label."""
        row = self._summary([SEATTLE, None], ["BASELINE", "BASELINE"])
        assert row["n_failed"] == 1 and row["n_beyond"] == 0

    def test_guard_catches_a_broken_cross_tab(self):
        row = self._summary([OMAHA])
        bad = pd.DataFrame([row])
        bad["n_beyond_cell_wrong"] = 0
        with pytest.raises(ValueError, match="do not partition the ring tiers"):
            C.guard_cross_tab(bad)


class TestPopulationContract:
    def test_unknown_tg_is_refused(self):
        s = _space()
        f = _frame(s, [SEATTLE])
        f["tg_id"] = "nobody"
        with pytest.raises(ValueError, match="not in the nside=128 answer space"):
            C.score_method(f, s)


class TestGridAxisIsMonotone:
    def test_ring0_never_falls_as_grids_grow(self):
        rng = np.random.default_rng(3)
        preds = [
            _offset(SEATTLE, north_km=float(n), east_km=float(e))
            for n, e in rng.normal(0, 120, (200, 2))
        ]
        rows = []
        for nside in G.NSIDE_LADDER:
            s = _space(nside=nside)
            rows.append(C.summarize({"m": C.score_method(_frame(s, preds), s)}, nside))
        assert C.monotonicity_violations(pd.concat(rows, ignore_index=True)).empty
