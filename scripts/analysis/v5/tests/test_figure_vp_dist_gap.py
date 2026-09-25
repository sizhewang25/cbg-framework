"""The gap at three scopes, and the two ways this figure could lie.

The load-bearing classes are `TestForcedZeroGap` and `TestScopesUseCohortFrame`.

`TestForcedZeroGap` pins an argument, not a number. A cohort whose bound sits
below the population's smallest positive gap *cannot* contain a nonzero gap,
so its 100% zero share is arithmetic rather than evidence. On as01-03 that is
exactly S-P's p5 (bound 1.55 km, floor 1.71 km). The flag has to flip when a
mesh puts the bound above the floor, because there the same 100% would be a
real finding and the paper should be free to claim it.

`TestScopesUseCohortFrame` guards the FALLBACK trap. A FALLBACK row carries a
real `pred_dist_to_tg_km`, since the fallback prediction is the shortest-ping
VP's own coordinate. Ranking without `solved_mask` therefore fills a "most
accurate" cohort with give-up rows, and the wrong number is the flattering
one. This module must never rank locally.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import figure_vp_dist_gap as D
from scripts.analysis.v5.modules.figure_vp_distance_cdf import GAP
from scripts.analysis.v5.modules.figure_vp_proximity import MEASURE_COLUMNS, RANK_COLUMN
from scripts.analysis.v5.modules.figure_vp_proximity import GEO as _GEO
from scripts.analysis.v5.modules.figure_vp_proximity import SPING as _SPING
from scripts.analysis.v5.modules.status import SHORTEST_PING

GEO_COL = MEASURE_COLUMNS[_GEO]
SPING_COL = MEASURE_COLUMNS[_SPING]

METHODS = (SHORTEST_PING, "m_other")

#: Twenty TGs, the smallest population where p5 asks for a whole TG:
#: k = round(.05*20) = 1 and round(.25*20) = 5. The gap is 0 on the ten with
#: the smallest d_sp, so the cohorts the reference selects are exactly the
#: zero-gap TGs, which is the shape the paper argues for.
_GEO_KM = [float(i + 1) for i in range(20)]
_SPING_KM = _GEO_KM[:10] + [g + 45.0 for g in _GEO_KM[10:]]


def _long(*, geo=None, sping=None, ref_err=None) -> pd.DataFrame:
    """Two methods over twenty TGs, one run. p5 -> k = 1, p25 -> k = 5.

    The reference's error tracks `d_sp`, as it does in the data: S-P returns
    the smallest-RTT VP's coordinate, so its error *is* that VP's distance.
    """
    geo = list(_GEO_KM if geo is None else geo)
    sping = list(_SPING_KM if sping is None else sping)
    n = len(geo)
    ref_err = list(sping if ref_err is None else ref_err)
    frame = pd.DataFrame(
        {
            "run_id": ["r"] * (n * len(METHODS)),
            "tg_id": [f"tg-{i}" for i in range(n)] * len(METHODS),
            "method": sum(([m] * n for m in METHODS), []),
            RANK_COLUMN: ref_err + [100.0] * n,
            "status": ["SUCCESS"] * (n * len(METHODS)),
            GEO_COL: geo * len(METHODS),
            SPING_COL: sping * len(METHODS),
        }
    )
    frame["solved"] = frame["status"] == "SUCCESS"
    return frame


class TestScopesUseCohortFrame:
    def test_population_scope_holds_every_tg_once(self):
        frames = D.scopes(_long())
        assert len(frames[D.ALL]) == 20
        assert set(frames[D.ALL].tg_id) == {f"tg-{i}" for i in range(20)}

    def test_cohorts_are_nested_inside_the_population(self):
        frames = D.scopes(_long())
        for cohort in ("p25", "p5"):
            assert set(frames[cohort].tg_id) <= set(frames[D.ALL].tg_id)

    def test_cohort_selects_on_the_reference_error(self):
        """k = 5 at p25, so the reference's five smallest errors."""
        frames = D.scopes(_long(), cohorts=("p25",))
        assert set(frames["p25"].tg_id) == {f"tg-{i}" for i in range(5)}

    def test_a_fallback_row_cannot_enter_a_cohort(self):
        """The trap: a give-up row carries a real, small prediction distance.

        `tg-19` is given the smallest error of all and a FALLBACK status. If
        this module ranked locally it would lead the cohort; `cohort_frame`
        drops it because `solved` is False.
        """
        long = _long()
        mask = (long.method == SHORTEST_PING) & (long.tg_id == "tg-19")
        long.loc[mask, RANK_COLUMN] = 0.01
        long.loc[mask, "status"] = "FALLBACK"
        long["solved"] = long["status"] == "SUCCESS"

        frames = D.scopes(long, cohorts=("p25",))
        assert "tg-19" not in set(frames["p25"].tg_id)

    def test_unknown_reference_raises_rather_than_emptying_the_cohort(self):
        with pytest.raises(ValueError, match="not scored"):
            D.scopes(_long(), reference="no_such_method")

    def test_an_empty_cohort_raises_rather_than_drawing_nothing(self):
        """k rounds to 0 below 10 TGs at p5. A silent empty scope draws as a
        missing curve, which reads as a missing method."""
        few = _long(geo=_GEO_KM[:8], sping=_SPING_KM[:8])
        with pytest.raises(ValueError, match="is empty"):
            D.scopes(few, cohorts=("p5",))

    def test_all_is_not_accepted_as_a_cohort(self):
        """`all` is already drawn as the population; taking it as a cohort
        would silently plot the same curve twice in two shades."""
        with pytest.raises(ValueError, match="percentile cohort"):
            D.scopes(_long(), cohorts=("all",))


class TestCohortBounds:
    def test_bound_is_the_references_worst_error_in_the_cohort(self):
        bounds = D.cohort_bounds(_long(), cohorts=("p25",))
        assert bounds["p25"] == pytest.approx(5.0)

    def test_bound_ignores_the_other_methods(self):
        """`m_other` sits at 100 km on every TG and must not set the bound."""
        bounds = D.cohort_bounds(_long(), cohorts=("p25", "p5"))
        assert max(bounds.values()) < 100.0


class TestForcedZeroGap:
    """A 100% zero share is only a finding when a nonzero gap could have fit."""

    def test_forced_when_the_bound_sits_below_the_population_floor(self):
        long = _long()
        frames = D.scopes(long, cohorts=("p25",))
        forced = D.forced_zero_gap(frames, D.cohort_bounds(long, cohorts=("p25",)))
        # Bound 5.0 km; smallest positive gap in the population is 45.0 km.
        assert forced["p25"]["cohort_bound_km"] == pytest.approx(5.0)
        assert forced["p25"]["population_min_positive_gap_km"] == pytest.approx(45.0)
        assert forced["p25"]["forced"] is True

    def test_not_forced_when_the_bound_clears_the_floor(self):
        """The flag must flip, or the module only ever says "forced" and the
        paper can never claim the result on a mesh where it is real."""
        # tg-2 keeps its small d_sp but its nearest VP moves 1 km closer, so
        # the population's smallest positive gap drops under the p25 bound.
        geo = list(_GEO_KM)
        geo[2] = 2.0
        long = _long(geo=geo)
        frames = D.scopes(long, cohorts=("p25",))
        forced = D.forced_zero_gap(frames, D.cohort_bounds(long, cohorts=("p25",)))
        assert forced["p25"]["population_min_positive_gap_km"] == pytest.approx(1.0)
        assert forced["p25"]["cohort_bound_km"] == pytest.approx(5.0)
        assert forced["p25"]["forced"] is False

    def test_verdict_reaches_the_manifest(self):
        long = _long()
        frames = D.scopes(long, cohorts=("p25",))
        stats = D.stats_table(frames)
        forced = D.forced_zero_gap(frames, D.cohort_bounds(long, cohorts=("p25",)))
        body = json.loads(D._manifest(
            {"run_ids": ["r"], "nside": 128, "n_tgs": 20, "reference": SHORTEST_PING},
            stats, forced,
        ))
        assert body["forced_zero_gap"]["p25"]["forced"] is True
        assert body["max_gap_km"]["p25"] == 0.0


class TestStats:
    def test_max_is_emitted_per_scope(self):
        """`max_km` is the number the prose quotes, so it must exist even
        where every gap is zero and the percentiles are uninformative."""
        stats = D.stats_table(D.scopes(_long())).set_index("scope")
        assert stats.loc[D.ALL, "max_km"] == pytest.approx(45.0)
        assert stats.loc["p25", "max_km"] == pytest.approx(0.0)

    def test_distinct_counts_travel_with_every_n(self):
        """317 TGs carrying 3 gap values is ~3 independent observations."""
        stats = D.stats_table(D.scopes(_long())).set_index("scope")
        assert stats.loc[D.ALL, "n"] == 20
        assert stats.loc[D.ALL, "n_distinct"] == 2   # 0 km and 45 km, nothing between
        assert stats.loc["p25", "n_distinct"] == 1

    def test_zero_share_is_still_reported(self):
        stats = D.stats_table(D.scopes(_long())).set_index("scope")
        assert stats.loc[D.ALL, "zero_share_pct"] == pytest.approx(50.0)
        assert stats.loc["p5", "zero_share_pct"] == pytest.approx(100.0)

    def test_min_positive_is_nan_when_no_gap_is_positive(self):
        stats = D.stats_table(D.scopes(_long())).set_index("scope")
        assert np.isnan(stats.loc["p5", "min_positive_km"])


class TestCurveReachesTheRightEdge:
    def test_an_all_zero_cohort_draws_as_a_line_not_a_point(self):
        x, y, zero_share = D._curve(np.zeros(5))
        assert zero_share == pytest.approx(100.0)
        assert len(x) == 2 and x[-1] == D.X_MAX_KM
        assert y.tolist() == pytest.approx([100.0, 100.0])

    def test_extension_does_not_invent_height(self):
        """The added point carries the final height, so no TG is implied
        beyond the largest observed gap."""
        gap = np.array([0.0, 10.0, 20.0])
        x, y, _ = D._curve(gap)
        assert y[-1] == pytest.approx(y[-2])
        assert x[-2] == pytest.approx(20.0)


class TestPrivacy:
    def test_no_emitted_column_carries_a_location(self):
        """Target-based analysis only: no coordinates, no place names."""
        forbidden = ("lat", "lon", "city", "country", "site", "region", "asn")
        for col in D.stats_table(D.scopes(_long())).columns:
            assert not any(f in col.lower() for f in forbidden), col


class TestPlot:
    def test_it_writes_a_png(self, tmp_path):
        frames = D.scopes(_long())
        png = D.plot(frames,
                     meta={"run_ids": ["r"], "n_tgs": 20, "reference": SHORTEST_PING},
                     out_png=tmp_path / "gap.png")
        assert png.exists() and png.stat().st_size > 0
