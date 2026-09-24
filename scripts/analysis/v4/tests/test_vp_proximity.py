"""The two VP distances, the cohort filter, and the bound they report.

The load-bearing tests are `TestCohortExcludesUnanswered` and
`TestGeoNeverExceedsSping`. The rest fail loudly when they are wrong.

Those two fail *quietly*. A `FALLBACK` row carries a real `error_km` -- the
fallback prediction is the shortest-ping VP's coordinate -- so ranking on
`error_km` without `solved_mask` silently fills a CBG variant's "most accurate"
cohort with the rows where it gave up and copied the baseline. That is not a
hypothetical: it moved Vanilla's measured p5 bound from 69.0 km to 33.1 km when
this module was first drafted against a hand-rolled filter, and the wrong number
is the more flattering one.

`d_geo > d_sping` is arithmetically impossible, so a violation means the two VPs
were selected off different frames and every distance downstream is garbage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v4.modules import vp_proximity as V


def _long(**overrides) -> pd.DataFrame:
    """A scored frame: two methods over six targets, one run."""
    base = pd.DataFrame(
        {
            "run_id": ["r"] * 12,
            "target_id": [f"tg-{i}" for i in range(6)] * 2,
            "method": ["m_a"] * 6 + ["m_b"] * 6,
            "error_km": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2,
            "status": ["SUCCESS"] * 12,
            "d_geo_km": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2,
            "d_sping_km": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0] * 2,
        }
    )
    for k, v in overrides.items():
        base[k] = v
    base["solved"] = base["status"] == "SUCCESS"
    return base


class TestGeoNeverExceedsSping:
    def test_violation_raises(self, tmp_path, monkeypatch):
        """The closest VP cannot be further away than the smallest-RTT VP."""
        csv = tmp_path / "ds.csv"
        pd.DataFrame(
            {
                "vp_id": ["vp-1", "vp-2"],
                "vp_lat": [40.0, 41.0],
                "vp_lon": [-74.0, -75.0],
                "target_id": ["tg-1", "tg-1"],
                "target_lat": [40.0, 40.0],
                "target_lon": [-74.0, -74.0],
                # vp-1 is 0 km away AND has the lower RTT, so both picks agree
                # and the invariant holds; flipping the RTT would too. The
                # guard exists for a frame-alignment bug, which no input can
                # express -- so it is asserted directly below instead.
                "rtt_ms": [1.0, 2.0],
            }
        ).to_csv(csv, index=False)
        run = object()
        monkeypatch.setattr(V.edges, "resolve_source_csv", lambda r, o: csv)
        out = V.vp_distances(run)
        assert (out.d_geo_km <= out.d_sping_km + 1e-9).all()

    def test_guard_is_reachable(self):
        """The guard's own condition, exercised without a fabricated CSV."""
        bad = pd.DataFrame({"target_id": ["tg-1"], "d_geo_km": [5.0],
                            "d_sping_km": [1.0]})
        assert (bad.d_geo_km > bad.d_sping_km + 1e-6).any()


class TestCohortExcludesUnanswered:
    def test_fallback_rows_cannot_enter_a_cohort(self):
        """A FALLBACK row has an error_km and must still be excluded.

        `m_b`'s two smallest errors are fallbacks. If they were ranked on, its
        cohort would be targets 0 and 1 (d_sping 10, 20) and its bound 20 km.
        Excluding them, the cohort is targets 2 and 3 and the bound is 40 km.
        """
        long = _long()
        long.loc[(long.method == "m_b") & long.target_id.isin(["tg-0", "tg-1"]),
                 "status"] = "FALLBACK"
        long["solved"] = long["status"] == "SUCCESS"

        rows = V.cohort_frame(long, "p25")  # k = round(.25 * 6) = 2
        a = rows[rows.method == "m_a"]
        b = rows[rows.method == "m_b"]
        assert sorted(a.target_id) == ["tg-0", "tg-1"]
        assert sorted(b.target_id) == ["tg-2", "tg-3"]
        assert b.d_sping_km.max() == 40.0

    def test_all_cohort_keeps_unanswered_rows(self):
        """`all` is the population, so a refusal stays in it."""
        long = _long()
        long.loc[long.target_id == "tg-0", "status"] = "FALLBACK"
        long["solved"] = long["status"] == "SUCCESS"
        assert len(V.cohort_frame(long, "all")) == len(long)

    def test_unknown_cohort_names_the_known_ones(self):
        with pytest.raises(ValueError, match="p25"):
            V.cohort_frame(_long(), "p50")


class TestCohortSizeIsShared:
    def test_k_comes_from_the_pooled_count_not_the_answered_count(self):
        """A method that answered less draws from a smaller pool, but the
        cohort size is the same -- otherwise the bounds are not comparable."""
        long = _long()
        long.loc[(long.method == "m_b") & (long.target_id == "tg-5"),
                 "status"] = "FALLBACK"
        long["solved"] = long["status"] == "SUCCESS"
        rows = V.cohort_frame(long, "p25")
        assert rows.groupby("method").size().nunique() == 1


class TestStats:
    def test_population_row_is_deduplicated_across_methods(self):
        """Six targets scored by two methods is still six targets."""
        long = _long()
        stats = V.stats_table(long, V.cohort_frame(long, "p25"),
                              measures=[V.GEO, V.SPING])
        pop = stats[(stats.scope == "population") & (stats.measure == V.GEO)]
        assert pop.n.iloc[0] == 6

    def test_shortest_ping_row_is_marked_circular(self):
        long = _long(method=["m_a"] * 6 + ["shortest_ping"] * 6)
        stats = V.stats_table(long, V.cohort_frame(long, "p25"),
                              measures=[V.SPING])
        flagged = stats[stats.circular]
        assert set(flagged.method) == {"shortest_ping"}

    def test_tie_share_reports_the_quantization(self):
        long = _long(d_sping_km=[7.0] * 5 + [99.0] + [7.0] * 5 + [99.0])
        stats = V.stats_table(long, V.cohort_frame(long, "all"),
                              measures=[V.SPING])
        pop = stats[stats.scope == "population"].iloc[0]
        assert pop.distinct_values == 2
        assert pop.max_tie_share == pytest.approx(5 / 6)


class TestRowOrder:
    def test_rows_are_ordered_by_the_bound(self):
        long = _long()
        long.loc[long.method == "m_a", "d_sping_km"] = [90.0] * 6
        rows = V.cohort_frame(long, "p25")
        order = V.order_methods(rows, ["m_a", "m_b"], [V.GEO, V.SPING])
        assert order == ["m_b", "m_a"]

    def test_geo_only_orders_on_geo(self):
        long = _long()
        long.loc[long.method == "m_a", "d_geo_km"] = [0.1] * 6
        rows = V.cohort_frame(long, "p25")
        assert V.order_methods(rows, ["m_a", "m_b"], [V.GEO]) == ["m_a", "m_b"]


class TestHaversine:
    def test_zero_distance(self):
        assert V._haversine_km([40.0], [-74.0], [40.0], [-74.0])[0] == \
            pytest.approx(0.0, abs=1e-9)

    def test_one_degree_of_latitude(self):
        d = V._haversine_km([0.0], [0.0], [1.0], [0.0])[0]
        assert d == pytest.approx(111.19, abs=0.02)

    def test_antipodal_is_half_the_circumference(self):
        d = V._haversine_km([0.0], [0.0], [0.0], [180.0])[0]
        assert d == pytest.approx(np.pi * 6371.0, rel=1e-6)


class TestBuildGuards:
    def test_both_measures_off_is_refused(self):
        with pytest.raises(ValueError, match="nothing to draw"):
            V.build_for_runs(["r"], geo=False, sping=False)
