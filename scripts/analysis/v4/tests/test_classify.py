"""Tests for v4's ring-graded laddered classifier.

`TestTheDefectThisFixes` is the reason the module exists: it reconstructs the
exact case v3 scored correct at 2,360 km and asserts the new rule refuses it,
while the retired rule still accepts it — so the fix and the evidence for it are
pinned together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v4.modules import answer_space as A
from scripts.analysis.v4.modules import classify as C
from scripts.analysis.v4.modules import healpix as H

SEATTLE = (47.4490, -122.3090)
OMAHA = (41.2565, -95.9345)
CHICAGO = (41.8781, -87.6298)
MIAMI = (25.7617, -80.1918)
#: Spotter's prediction for tg-e1a1545 — the Canadian Arctic.
ARCTIC = (63.7369, -97.4685)


def _space(coords=(SEATTLE, OMAHA, CHICAGO, MIAMI), nside=128):
    t = pd.DataFrame(
        {
            "target_id": [f"tg-{i}" for i in range(len(coords))],
            "target_lat": [c[0] for c in coords],
            "target_lon": [c[1] for c in coords],
        }
    )
    return A.build_answer_space(t, nside=nside)


def _frame(space, preds, statuses=None):
    """One row per target, with `preds` as `(lat, lon)` or None."""
    a = space.assignments
    statuses = statuses or ["SUCCESS"] * len(a)
    return pd.DataFrame(
        {
            "target_id": a["target_id"].to_numpy(),
            "target_lat": a["target_lat"].to_numpy(),
            "target_lon": a["target_lon"].to_numpy(),
            "pred_lat": [p[0] if p else None for p in preds],
            "pred_lon": [p[1] if p else None for p in preds],
            "status": statuses,
            "fold": 0,
        }
    )


class TestRingGrading:
    def test_an_exact_hit_is_ring_zero(self):
        space = _space()
        exact = list(zip(space.assignments["target_lat"], space.assignments["target_lon"]))
        scored = C.score_method(_frame(space, exact), space)
        assert (scored["ring"] == 0).all()
        assert (scored["error_km"] < 1e-6).all()

    def test_the_cell_centre_is_also_ring_zero(self):
        """The best any grid-quantised estimator can do, and it must count."""
        space = _space()
        seeds = space.seeds.set_index("seed_id")
        preds = [
            (seeds.loc[s, "seed_lat"], seeds.loc[s, "seed_lon"])
            for s in space.assignments["seed_id"]
        ]
        assert (C.score_method(_frame(space, preds), space)["ring"] == 0).all()

    def test_a_neighbouring_cell_is_ring_one(self):
        space = _space()
        preds = []
        for cell in space.assignments["cell_id"]:
            nb = H.neighbours([int(cell)], space.nside)[0]
            centre = H.pix2ang([int(nb[nb >= 0][0])], space.nside)[0]
            preds.append((centre[0], centre[1]))
        assert (C.score_method(_frame(space, preds), space)["ring"] == 1).all()

    def test_a_far_prediction_is_unplaced_not_a_large_ring(self):
        space = _space()
        preds = [ARCTIC] * len(space.assignments)
        assert (C.score_method(_frame(space, preds), space)["ring"] == -1).all()

    def test_a_missing_prediction_is_unplaced_and_stays_in_the_denominator(self):
        space = _space()
        preds = [None] * len(space.assignments)
        scored = C.score_method(
            _frame(space, preds, ["FALLBACK"] * len(space.assignments)), space
        )
        assert (scored["ring"] == -1).all()
        assert scored["pred_cell"].eq(-1).all()
        assert scored["error_km"].isna().all()
        summary = C.summarize({"m": scored}, space.nside)
        assert summary["n_targets"].iloc[0] == len(space.assignments)
        assert summary["accuracy_ring0"].iloc[0] == 0.0


class TestTheDefectThisFixes:
    """v3 credited a 2,360 km miss because nearest-seed is unbounded."""

    def test_the_arctic_prediction_is_refused_but_the_retired_rule_accepts_it(self):
        space = _space()
        # Only the Seattle target; predict the Arctic for it.
        frame = _frame(
            space,
            [ARCTIC if i == 0 else (c[0], c[1]) for i, c in enumerate(
                zip(space.assignments["target_lat"], space.assignments["target_lon"])
            )],
        )
        scored = C.score_method(frame, space)
        row = scored.iloc[0]
        assert row["ring"] == -1, "the new rule must refuse it"
        # And the retired rule credits it, which is the measurement that
        # justifies the change. Seattle is the nearest of these seeds to the
        # Arctic point, exactly as it was among as01's real 18.
        assert row["nearest_seed_id_retired"] == row["tg_seed_id"]
        assert row["error_km"] > 2000

    def test_the_gap_between_the_rules_is_reported(self):
        space = _space()
        preds = [ARCTIC] * len(space.assignments)
        summary = C.summarize({"m": C.score_method(_frame(space, preds), space)}, space.nside)
        assert summary["accuracy_ring0"].iloc[0] == 0.0
        assert summary["accuracy_nearest_seed_retired"].iloc[0] > 0.0


class TestCumulative:
    def test_ring_accuracies_are_cumulative(self):
        space = _space()
        seeds = space.seeds.set_index("seed_id")
        preds = []
        for i, (cell, seed) in enumerate(
            zip(space.assignments["cell_id"], space.assignments["seed_id"])
        ):
            if i == 0:  # exact
                preds.append((seeds.loc[seed, "seed_lat"], seeds.loc[seed, "seed_lon"]))
            elif i == 1:  # ring 1
                nb = H.neighbours([int(cell)], space.nside)[0]
                c = H.pix2ang([int(nb[nb >= 0][0])], space.nside)[0]
                preds.append((c[0], c[1]))
            else:
                preds.append(ARCTIC)
        s = C.summarize({"m": C.score_method(_frame(space, preds), space)}, space.nside)
        r0, r1, r2 = (s[f"accuracy_ring{k}"].iloc[0] for k in range(3))
        assert r0 <= r1 <= r2
        assert r0 == pytest.approx(0.25)
        assert r1 == pytest.approx(0.50)


class TestLadderMonotonicity:
    """The guarantee H3 could not give: coarser cells never score worse."""

    def test_accuracy_is_monotone_across_the_ladder(self):
        t = pd.DataFrame(
            {
                "target_id": [f"tg-{i}" for i in range(4)],
                "target_lat": [c[0] for c in (SEATTLE, OMAHA, CHICAGO, MIAMI)],
                "target_lon": [c[1] for c in (SEATTLE, OMAHA, CHICAGO, MIAMI)],
            }
        )
        # One fixed jitter, reused at every rung: the ladder must be monotone
        # for the SAME predictions, not for independently drawn ones.
        rng = np.random.default_rng(3)
        dlat, dlon = rng.normal(0, 1.5, 4), rng.normal(0, 1.5, 4)
        preds = [
            (lat + a, lon + b)
            for lat, lon, a, b in zip(t["target_lat"], t["target_lon"], dlat, dlon)
        ]
        rows = []
        for nside in H.NSIDE_LADDER:
            space = A.build_answer_space(t, nside=nside)
            rows.append(
                C.summarize({"m": C.score_method(_frame(space, preds), space)}, nside)
            )
        long = pd.concat(rows, ignore_index=True)
        assert C.monotonicity_violations(long).empty

    def test_a_violation_is_reported_rather_than_swallowed(self):
        """The detector itself must work, or the guarantee is unverified."""
        long = pd.DataFrame(
            {
                "method": ["m", "m"],
                "nside": [128, 64],
                "accuracy_ring0": [0.5, 0.4],  # coarser scored worse: impossible
            }
        )
        bad = C.monotonicity_violations(long)
        assert len(bad) == 1
        assert bad.iloc[0]["finer_nside"] == 128
        assert bad.iloc[0]["coarser_nside"] == 64


class TestPopulationContract:
    def test_a_target_outside_the_answer_space_is_refused(self):
        """Not dropped: the space is built from the same folds, so a mismatch
        means the two are out of step and any number over the intersection
        would be reported against the wrong denominator."""
        space = _space()
        frame = _frame(space, [(0.0, 0.0)] * len(space.assignments))
        frame.loc[0, "target_id"] = "tg-not-in-space"
        with pytest.raises(ValueError, match="not in the nside=128 answer space"):
            C.score_method(frame, space)

    def test_baseline_rows_count_as_solved(self):
        """shortest_ping writes BASELINE everywhere and has no fallback path;
        treating those as unsolved would divide its accuracy by zero answers."""
        space = _space()
        exact = list(zip(space.assignments["target_lat"], space.assignments["target_lon"]))
        frame = _frame(space, exact, ["BASELINE"] * len(space.assignments))
        assert C.solved_mask(frame).all()
        s = C.summarize({"sp": C.score_method(frame, space)}, space.nside)
        assert s["accuracy_ring0"].iloc[0] == 1.0

    def test_fallback_is_wrong_not_excluded(self):
        space = _space()
        exact = list(zip(space.assignments["target_lat"], space.assignments["target_lon"]))
        statuses = ["SUCCESS"] * len(space.assignments)
        statuses[0] = "FALLBACK"
        s = C.summarize(
            {"m": C.score_method(_frame(space, exact, statuses), space)}, space.nside
        )
        assert s["n_targets"].iloc[0] == 4
        assert s["n_fallback"].iloc[0] == 1
        assert s["accuracy_ring0"].iloc[0] == pytest.approx(0.75)

    def test_error_percentiles_exclude_unanswered_rows(self):
        space = _space()
        exact = list(zip(space.assignments["target_lat"], space.assignments["target_lon"]))
        preds = [None] + exact[1:]
        statuses = ["FALLBACK"] + ["SUCCESS"] * (len(exact) - 1)
        s = C.summarize(
            {"m": C.score_method(_frame(space, preds, statuses), space)}, space.nside
        )
        assert s["error_km_p50"].iloc[0] == pytest.approx(0.0, abs=1e-6)
