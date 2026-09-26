"""Two labels per prediction, and the cross-tab that partitions them.

`TestTheArcticCase` pins the reason the two labels must be read together. The
prediction v3 credited to Seattle 2,360 km away is `correct` on the cell axis
-- the cell partition is unbounded, so it says so -- and `beyond` on the grid
axis. v4 suppressed that with a landmass polygon and a fourth label; v5 keeps
it, because a nearest-seed verdict crediting a 2,360 km miss is the finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import answer_space as A
from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.paths import MissingArtifactError

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
    def test_near_the_site_is_correct(self):
        s = _space()
        out = C.score_method(_frame(s, [_offset(SEATTLE, north_km=2)]), s)
        assert out["cell_label"].iloc[0] == "correct"
        assert out[C.GRID_OFFSET].iloc[0] == 0

    def test_in_another_serving_region_is_wrong(self):
        s = _space()
        out = C.score_method(_frame(s, [OMAHA]), s)
        assert out["cell_label"].iloc[0] == "wrong"
        assert out["pred_seed_id"].iloc[0] != out["tg_seed_id"].iloc[0]

    def test_no_prediction_is_unanswered(self):
        """The `has_pred` guard in `score_method` is what makes this pass. A
        row with no prediction keeps `pred_seed_id == -1`, which never equals
        a real `tg_seed_id`, so masking on inequality alone would label it
        `wrong` -- a refusal silently recorded as a wrong answer."""
        s = _space()
        out = C.score_method(_frame(s, [None]), s)
        assert out["cell_label"].iloc[0] == C.UNANSWERED
        assert out["pred_seed_id"].iloc[0] == -1
        # The only thing -1 means on the grid axis now.
        assert out[C.GRID_OFFSET].iloc[0] == -1

    def test_far_but_in_the_right_serving_region(self):
        """Beyond on the grid axis, correct on the cell axis: 400 km east of
        Seattle is still nearer Seattle's seed than Omaha's. The case the
        cross-tab exists to show."""
        s = _space()
        out = C.score_method(_frame(s, [_offset(SEATTLE, east_km=400)]), s)
        assert out[C.GRID_OFFSET].iloc[0] > C.MAX_RING
        assert out["cell_label"].iloc[0] == "correct"

    def test_distances_to_tg_and_to_seed(self):
        s = _space()
        out = C.score_method(_frame(s, [_offset(SEATTLE, north_km=10)]), s)
        # A single-site seed sits on its site, so both distances agree.
        assert out["pred_dist_to_tg_km"].iloc[0] == pytest.approx(10, abs=0.05)
        assert out["pred_dist_to_seed_km"].iloc[0] == pytest.approx(10, abs=0.05)


class TestTheArcticCase:
    def test_the_cell_axis_credits_it_and_the_ring_axis_does_not(self):
        """2,360 km from the truth, in the Canadian Arctic, and the nearest of
        the US seeds is still Seattle's -- so the unbounded cell rule calls it
        `correct`. That is not a bug to patch here: it is why `ring` is read
        beside `cell_label`, and why `cell_label` alone is not a verdict."""
        s = _space()
        out = C.score_method(_frame(s, [ARCTIC]), s)
        assert out["cell_label"].iloc[0] == "correct"
        assert out["pred_dist_to_tg_km"].iloc[0] > 2000
        # What v4's `-1` was hiding: tens of grids out, not merely "further".
        assert out[C.GRID_OFFSET].iloc[0] > 20


class TestUncappedGridAxis:
    """The offset is exact; `_tier` is the only thing that bands it."""

    def test_the_tier_bands_follow_the_offset(self):
        assert list(C._tier(np.array([0, 1, 2]))) == ["ring0", "ring1", "ring2"]

    def test_anything_past_the_cap_is_beyond(self):
        past = np.array([C.MAX_RING + 1, 17, 68])
        assert (C._tier(past) == "beyond").all()

    def test_a_row_with_no_prediction_also_lands_in_beyond(self):
        """It is excluded by `answered` before any count, so the tier it would
        have taken never matters -- but it must not crash or take a real one."""
        assert C._tier(np.array([-1]))[0] == "beyond"

    def test_rows_the_old_cap_pooled_are_told_apart(self):
        """The point of the column: Omaha and the Arctic were both `-1` under
        the retired `ring`, and are nowhere near each other in grids."""
        s = _space()
        out = C.score_method(_frame(s, [OMAHA, ARCTIC]), s)
        omaha, arctic = out[C.GRID_OFFSET].tolist()
        assert omaha > C.MAX_RING and arctic > C.MAX_RING
        assert omaha != arctic


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
        assert row["n_ring0_cell_correct"] == 1
        assert row["n_beyond_cell_wrong"] == 1  # Omaha
        # The Arctic joins the 400 km-east row: both beyond, both credited.
        assert row["n_beyond_cell_correct"] == 2
        assert row["n_cell_correct"] == 3
        assert row["accuracy_cell_correct"] == pytest.approx(0.6)
        # The three cell labels partition n_tgs, and `unanswered` is the same
        # rows the grid axis calls failed.
        parts = row["n_cell_correct"] + row["n_cell_wrong"] + row["n_cell_unanswered"]
        assert parts == row["n_tgs"]
        assert row["n_cell_unanswered"] == row["n_failed"] == 1

    def test_fallback_is_labelled_but_not_counted(self):
        s = _space()
        scored = C.score_method(_frame(s, [_offset(SEATTLE, north_km=2)], ["FALLBACK"]), s)
        assert scored["cell_label"].iloc[0] == "correct"
        row = C.summarize({"m": scored}, s.nside).iloc[0]
        assert row["n_cell_correct"] == 0 and row["n_failed"] == 1
        assert row["n_cell_unanswered"] == 1
        assert row["accuracy_ring0"] == 0.0

    def test_answered_without_a_coordinate_is_failed_not_beyond(self):
        """A BASELINE row with no eval-source match: answered, nothing to label."""
        row = self._summary([SEATTLE, None], ["BASELINE", "BASELINE"])
        assert row["n_failed"] == 1 and row["n_beyond"] == 0

    def test_grid_offset_percentiles_are_over_answered_rows(self):
        row = self._summary(
            [_offset(SEATTLE, north_km=2), OMAHA, None],
            ["SUCCESS", "SUCCESS", "ERROR"],
        )
        # Two answered rows, at 0 and tens of grids out. Were the unanswered
        # row's -1 included, it would sort first and pull p50 down to 0.
        biggest = row[f"{C.GRID_OFFSET}_max"]
        assert biggest > C.MAX_RING
        assert row[f"{C.GRID_OFFSET}_p50"] == pytest.approx(biggest / 2)

    def test_percentiles_are_none_when_nothing_was_answered(self):
        row = self._summary([None], ["ERROR"])
        assert row[f"{C.GRID_OFFSET}_p50"] is None
        assert row[f"{C.GRID_OFFSET}_max"] is None

    def test_a_frame_from_before_the_column_existed_is_refused(self):
        """`figure_outcome_bars` re-summarizes per-TG parquets off disk, so a
        stale artifact must say so rather than raise a bare KeyError."""
        s = _space()
        scored = C.score_method(_frame(s, [OMAHA]), s).drop(columns=[C.GRID_OFFSET])
        with pytest.raises(MissingArtifactError, match="re-run"):
            C.summarize({"m": scored}, s.nside)

    def test_guard_catches_a_broken_cross_tab(self):
        row = self._summary([OMAHA])
        bad = pd.DataFrame([row])
        bad["n_beyond_cell_wrong"] = 0
        with pytest.raises(ValueError, match="does not close"):
            C.guard_cross_tab(bad)

    def test_guard_catches_the_two_axes_disagreeing(self):
        """`n_cell_unanswered` and `n_failed` count the same rows along the
        two axes, so a drift between them is a real inconsistency."""
        row = self._summary([None], ["ERROR"])
        bad = pd.DataFrame([row])
        bad["n_cell_unanswered"] = 0
        bad["n_cell_wrong"] = 1
        with pytest.raises(ValueError, match="unanswered"):
            C.guard_cross_tab(bad)


class TestPopulationContract:
    def test_unknown_tg_is_refused(self):
        s = _space()
        f = _frame(s, [SEATTLE])
        f["tg_id"] = "nobody"
        with pytest.raises(ValueError, match="not in the nside=128 answer space"):
            C.score_method(f, s)
