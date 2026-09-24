"""The two VP distances, the cohort filter, and the bound they report. Ported from v4.

The load-bearing tests are `TestCohortExcludesUnanswered` and
`TestGeoNeverExceedsSping`. Both fail *quietly* when wrong: a FALLBACK row
carries a real `pred_dist_to_tg_km` (the S-P VP's coordinate), so ranking
without `solved_mask` fills a method's "most accurate" cohort with give-up rows
-- it moved VAN's p5 bound from 69.0 km to 33.1 km in v4, and the wrong number
is the flattering one. `geo > sping` is arithmetically impossible, so a
violation means the two VPs came off different frames.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import figure_vp_proximity as V
from scripts.analysis.v5.modules.geodesy import haversine_km
from scripts.analysis.v5.modules.paths import DEFAULT_OUTPUTS_ROOT, REPO_ROOT, resolve_run

GEO_COL = V.MEASURE_COLUMNS[V.GEO]
SPING_COL = V.MEASURE_COLUMNS[V.SPING]


def _long(**overrides) -> pd.DataFrame:
    """A scored frame: two methods over six TGs, one run."""
    base = pd.DataFrame(
        {
            "run_id": ["r"] * 12,
            "tg_id": [f"tg-{i}" for i in range(6)] * 2,
            "method": ["m_a"] * 6 + ["m_b"] * 6,
            V.RANK_COLUMN: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2,
            "status": ["SUCCESS"] * 12,
            GEO_COL: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2,
            SPING_COL: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0] * 2,
        }
    )
    for k, v in overrides.items():
        base[k] = v
    base["solved"] = base["status"] == "SUCCESS"
    return base


def _fallback(long: pd.DataFrame, method: str, tgs: list[str]) -> pd.DataFrame:
    long.loc[(long.method == method) & long.tg_id.isin(tgs), "status"] = "FALLBACK"
    long["solved"] = long["status"] == "SUCCESS"
    return long


class TestGeoNeverExceedsSping:
    def test_distances_off_a_canonical_csv(self, tmp_path, monkeypatch):
        """Benchmark names are mapped to tg_*, and geo <= sping holds."""
        csv = tmp_path / "ds.csv"
        pd.DataFrame(
            {
                "vp_id": ["vp-1", "vp-2", "vp-1"],
                "vp_lat": [40.0, 41.0, 40.0],
                "vp_lon": [-74.0, -75.0, -74.0],
                "target_id": ["tg-1", "tg-1", "tg-1"],
                "target_lat": [40.0] * 3,
                "target_lon": [-74.1] * 3,
                # vp-2 is further but answers faster; vp-1's min RTT is 3.0.
                "rtt_ms": [5.0, 2.0, 3.0],
            }
        ).to_csv(csv, index=False)
        monkeypatch.setattr(V.edges, "resolve_source_csv", lambda r, o: csv)
        out = V.vp_distances(object())
        row = out.iloc[0]
        assert row.tg_id == "tg-1" and row.n_vp == 2 and row.n_sping_vp_ties == 1
        assert row.geo_vp_rtt_ms == 3.0 and row.sping_vp_rtt_ms == 2.0
        assert row[GEO_COL] < row[SPING_COL]

    def test_guard_is_reachable(self):
        """The guard's own condition, exercised without a fabricated CSV."""
        bad = pd.DataFrame({"tg_id": ["tg-1"], GEO_COL: [5.0], SPING_COL: [1.0]})
        assert (bad[GEO_COL] > bad[SPING_COL] + 1e-6).any()


class TestCohortExcludesUnanswered:
    def test_fallback_rows_cannot_enter_a_cohort(self):
        """`m_b`'s two smallest distances are fallbacks. Ranked on, its cohort
        would be tg-0/1 and its bound 20 km; excluded, tg-2/3 and 40 km."""
        long = _fallback(_long(), "m_b", ["tg-0", "tg-1"])
        rows = V.cohort_frame(long, "p25")  # k = round(.25 * 6) = 2
        assert sorted(rows[rows.method == "m_a"].tg_id) == ["tg-0", "tg-1"]
        b = rows[rows.method == "m_b"]
        assert sorted(b.tg_id) == ["tg-2", "tg-3"]
        assert b[SPING_COL].max() == 40.0

    def test_baseline_rows_are_answered(self):
        """S-P's rows are all BASELINE; `status == SUCCESS` would empty it.
        `solved_mask` decides per frame, which is how `load` calls it."""
        long = _long(method=["m_a"] * 6 + [V.SHORTEST_PING] * 6)
        long.loc[long.method == V.SHORTEST_PING, "status"] = "BASELINE"
        long["solved"] = long.groupby("method", group_keys=False)[["status"]].apply(V.solved_mask)
        rows = V.cohort_frame(long, "p25")
        assert (rows.method == V.SHORTEST_PING).sum() == 2

    def test_all_cohort_keeps_unanswered_rows(self):
        long = _fallback(_long(), "m_a", ["tg-0"])
        assert len(V.cohort_frame(long, "all")) == len(long)

    def test_unknown_cohort_names_the_known_ones(self):
        with pytest.raises(ValueError, match="p25"):
            V.cohort_frame(_long(), "p50")


class TestP95IsNotAll:
    @staticmethod
    def _twenty() -> pd.DataFrame:
        """Twenty TGs, so `round(.95 * n)` actually drops one. From 1 km, not
        0: a 0 km distance is a log10(0) in the KDE."""
        err = [float(i) for i in range(1, 21)]
        base = pd.DataFrame(
            {
                "run_id": ["r"] * 40,
                "tg_id": [f"tg-{i}" for i in range(20)] * 2,
                "method": ["m_a"] * 20 + ["m_b"] * 20,
                V.RANK_COLUMN: err * 2,
                "status": ["SUCCESS"] * 40,
                GEO_COL: err * 2,
                SPING_COL: [e * 10 for e in err] * 2,
            }
        )
        base["solved"] = True
        return base

    def test_it_drops_each_methods_worst(self):
        rows = V.cohort_frame(self._twenty(), "p95")  # k = 19
        for method in ("m_a", "m_b"):
            kept = rows[rows.method == method]
            assert len(kept) == 19
            assert "tg-19" not in set(kept.tg_id)

    def test_it_excludes_unanswered_where_all_keeps_them(self):
        long = self._twenty()
        long.loc[long.tg_id == "tg-0", "status"] = "FALLBACK"
        long["solved"] = long["status"] == "SUCCESS"
        assert len(V.cohort_frame(long, "all")) == 40
        rows = V.cohort_frame(long, "p95")
        assert "tg-0" not in set(rows.tg_id)
        assert len(rows) == 38

    def test_a_method_that_cannot_fill_k_supplies_what_it_has(self):
        long = _fallback(self._twenty(), "m_b", [f"tg-{i}" for i in range(15, 20)])
        assert V.cohort_k(long, "p95") == 19
        sizes = V.cohort_frame(long, "p95").groupby("method").size()
        assert sizes["m_a"] == 19
        assert sizes["m_b"] == 15

    def test_the_title_gives_a_range_when_the_cohort_is_short(self, tmp_path):
        long = _fallback(self._twenty(), "m_b", [f"tg-{i}" for i in range(15, 20)])
        meta = {"run_ids": ["r"], "nside": 128, "methods": ["m_a", "m_b"],
                "n_tgs": 20, "n_vp_per_tg_median": 4.0}
        rows = V.cohort_frame(long, "p95")
        assert "15-19" in V.title_for("p95", rows, 20)

        png = V.plot(long, rows, cohort="p95", measures=[V.GEO, V.SPING],
                     meta=meta, out_png=tmp_path / "p95.png")
        assert png.exists()

        stats = V.stats_table(long, rows, measures=[V.GEO, V.SPING])
        body = json.loads(V._manifest(meta, "p95", [V.GEO, V.SPING], rows, stats))
        assert body["cohort_k_requested"] == 19
        assert body["n_cohort_short"] == {V.method_label("m_b"): 15}


class TestCohortSizeIsShared:
    def test_k_comes_from_the_pooled_count_not_the_answered_count(self):
        long = _fallback(_long(), "m_b", ["tg-5"])
        rows = V.cohort_frame(long, "p25")
        assert rows.groupby("method").size().nunique() == 1


class TestStats:
    def test_population_row_is_deduplicated_across_methods(self):
        long = _long()
        stats = V.stats_table(long, V.cohort_frame(long, "p25"), measures=[V.GEO, V.SPING])
        pop = stats[(stats.scope == "population") & (stats.measure == V.GEO)]
        assert pop.n.iloc[0] == 6

    def test_shortest_ping_row_is_marked_circular(self):
        long = _long(method=["m_a"] * 6 + [V.SHORTEST_PING] * 6)
        stats = V.stats_table(long, V.cohort_frame(long, "p25"), measures=[V.SPING])
        assert set(stats[stats.circular].method) == {V.SHORTEST_PING}

    def test_rows_carry_method_terms(self):
        long = _long(method=["octant_cbg_hull"] * 6 + [V.SHORTEST_PING] * 6)
        stats = V.stats_table(long, V.cohort_frame(long, "p25"), measures=[V.GEO])
        assert set(stats[stats.scope == "cohort"].method_label) == {"OCT-H", "S-P"}

    def test_tie_share_reports_the_quantization(self):
        long = _long(**{SPING_COL: [7.0] * 5 + [99.0] + [7.0] * 5 + [99.0]})
        stats = V.stats_table(long, V.cohort_frame(long, "all"), measures=[V.SPING])
        pop = stats[stats.scope == "population"].iloc[0]
        assert pop.distinct_values == 2
        assert pop.max_tie_share == pytest.approx(5 / 6)


class TestRowOrder:
    def test_rows_are_ordered_by_the_bound(self):
        long = _long()
        long.loc[long.method == "m_a", SPING_COL] = [90.0] * 6
        rows = V.cohort_frame(long, "p25")
        assert V.order_methods(rows, ["m_a", "m_b"], [V.GEO, V.SPING]) == ["m_b", "m_a"]

    def test_geo_only_orders_on_geo(self):
        long = _long()
        long.loc[long.method == "m_a", GEO_COL] = [0.1] * 6
        rows = V.cohort_frame(long, "p25")
        assert V.order_methods(rows, ["m_a", "m_b"], [V.GEO]) == ["m_a", "m_b"]


class TestHaversine:
    def test_zero_distance(self):
        assert haversine_km([40.0], [-74.0], [40.0], [-74.0])[0] == pytest.approx(0.0, abs=1e-9)

    def test_one_degree_of_latitude(self):
        assert haversine_km([0.0], [0.0], [1.0], [0.0])[0] == pytest.approx(111.19, abs=0.02)

    def test_antipodal_is_half_the_circumference(self):
        assert haversine_km([0.0], [0.0], [0.0], [180.0])[0] == pytest.approx(np.pi * 6371.0, rel=1e-6)


class TestBuildGuards:
    def test_both_measures_off_is_refused(self):
        with pytest.raises(ValueError, match="nothing to draw"):
            V.build_for_runs([], geo=False, sping=False)

    def test_unknown_cohort_is_refused_before_loading(self):
        with pytest.raises(ValueError, match="unknown cohort"):
            V.build_for_runs([], cohorts=["p50"])


# -- real runs --------------------------------------------------------------

RUN_IDS = [f"as0{i}-260728-260802-mesh" for i in (1, 2, 3)]
V4_DIR = (REPO_ROOT / "outputs" / "analysis" / "v4" / "_cross"
          / "as01+as02+as03@260728-260802-mesh" / "vp_proximity")
V5_CLASSIFY = REPO_ROOT / "outputs" / "analysis" / "v5"


def _available() -> bool:
    return all((DEFAULT_OUTPUTS_ROOT / r).exists()
               and (V5_CLASSIFY / r / "classify" / "healpix-128").exists() for r in RUN_IDS)


@pytest.fixture(scope="module")
def built():
    """Read-only: `load` writes nothing, so the real classify root is safe."""
    runs = [resolve_run(r) for r in RUN_IDS]
    return V.load(runs, analysis_root=V5_CLASSIFY)


@pytest.mark.skipif(not _available(), reason="as01-03 runs or their v5 classify absent")
class TestRealRuns:

    def test_population_and_invariant(self, built):
        long, meta = built
        assert meta["n_tgs"] == 1269
        dedup = long.drop_duplicates(["run_id", "tg_id"])
        assert (dedup[GEO_COL] <= dedup[SPING_COL] + 1e-9).all()

    def test_shortest_ping_distance_is_its_sping_vp(self, built):
        """The premise the `circular` flag rests on. Where they differ, the
        minimum RTT is tied and the benchmark broke the tie the other way."""
        long, _ = built
        sp = long[long.method == V.SHORTEST_PING]
        off = (sp[V.RANK_COLUMN] - sp[SPING_COL]).abs() > 0.01
        assert (sp.loc[off, "n_sping_vp_ties"] > 1).all()
        assert off.sum() <= 0.01 * len(sp)

    @pytest.mark.parametrize("cohort", ["p5", "p25", "all"])
    def test_matches_v4_cell_for_cell(self, built, cohort):
        path = V4_DIR / V.CSV_NAME.format(cohort=cohort)
        if not path.exists():
            pytest.skip(f"{path} absent")
        long, _ = built
        rows = V.cohort_frame(long, cohort)
        new = V.stats_table(long, rows, measures=[V.GEO, V.SPING]).fillna({"method": ""})
        old = pd.read_csv(path).fillna({"method": ""})
        keys = ["scope", "method", "measure"]
        merged = old.merge(new, on=keys, suffixes=("_v4", "_v5"), validate="1:1")
        assert len(merged) == len(old) == len(new)
        for col in [c for c in old.columns if c not in (*keys, "method_label")]:
            np.testing.assert_allclose(
                merged[f"{col}_v5"].astype(float), merged[f"{col}_v4"].astype(float),
                rtol=1e-9, atol=1e-9, err_msg=col,
            )
