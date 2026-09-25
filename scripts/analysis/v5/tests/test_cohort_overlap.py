"""Cohort set structure, behaviour on a reference cohort, and the two traps.

The load-bearing tests are `TestCohortsAreNeverRebuilt`, `TestMarginIsRequired`
and `TestPrivacy`. Each pins something that has already gone wrong once:

* a FALLBACK row carries a real `pred_dist_to_tg_km` (the S-P VP's coordinate),
  so a cohort rebuilt by ranking that column without `status.solved_mask`
  fills with refusals -- it moved VAN's p5 bound 69.0 -> 33.1 km in v4 and the
  wrong number was the flattering one;
* `baseline_margin` at `margin_km=0` scores S-P ~100% against itself, because
  S-P *predicts* the smallest-RTT VP's coordinate and the two sides differ by
  ~0.006 km of floating point;
* this analysis is target-based, and no coordinate, site id or place name may
  reach any emitted column.

The fixtures are hand-sized so that every expected number is arithmetic a
reader can redo in the docstring, and `TestRealRuns` re-derives the numbers
`evaluation.tex` prints straight off the as01-03 meshes.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from scripts.analysis.v5.modules import cohort_overlap as CO
from scripts.analysis.v5.modules import figure_vp_proximity as V
from scripts.analysis.v5.modules.paths import DEFAULT_OUTPUTS_ROOT, REPO_ROOT, resolve_run
from scripts.analysis.v5.modules.status import SHORTEST_PING

GEO_COL = CO.GEO_COL
SPING_COL = CO.SPING_COL
ERR = CO.ERROR_COLUMN

#: Six TGs, repeated onto every method. `geo <= sping` throughout.
#:
#: tg:        0     1     2      3     4     5
#: geo:     1.0   2.0   3.0    4.0   5.0   6.0
#: sping:   1.0   2.0   3.4   40.0  50.0  60.0
#: ties:      1     1     1      1     2     3
#:
#: So exact VP-ranking agreement holds on tg-0 and tg-1 (2/6 = 33.3%) while
#: within-1-km also picks up tg-2's 0.4 km gap (3/6 = 50%). Two of the six have
#: more than one VP at the minimum RTT (33.3%, max 3).
GEO = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
SPING = [1.0, 2.0, 3.4, 40.0, 50.0, 60.0]
TIES = [1, 1, 1, 1, 2, 3]

#: Per-method `pred_dist_to_tg_km`.
#:
#: S-P returns the smallest-RTT VP's coordinate, so its prediction distance is
#: `SPING` up to the ~0.003 km of floating point the real pipeline carries --
#: which is exactly what makes `margin_km=0` degenerate for it.
PRED = {
    SHORTEST_PING: [s + 0.003 for s in SPING],
    # Beats S-P on tg-0/tg-1, hopeless elsewhere.
    "m_a": [0.5, 1.5, 100.0, 200.0, 300.0, 400.0],
    # Best on tg-4/tg-5, so its cohort is disjoint from S-P's.
    "m_b": [500.0, 600.0, 700.0, 800.0, 1.0, 2.0],
    # Same cohort as S-P and 100 km off on it: overlap is not quality.
    "m_c": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
}
METHODS = list(PRED)


def _long(**overrides) -> pd.DataFrame:
    """Four methods over six TGs, one run. `p25` asks for k = round(.25*6) = 2.

    Cohorts: S-P and `m_a` and `m_c` take tg-0/tg-1, `m_b` takes tg-4/tg-5.
    """
    n = len(GEO)
    base = pd.DataFrame(
        {
            "run_id": ["r"] * (n * len(METHODS)),
            "tg_id": [f"tg-{i}" for i in range(n)] * len(METHODS),
            "method": sum(([m] * n for m in METHODS), []),
            ERR: sum((PRED[m] for m in METHODS), []),
            "status": ["SUCCESS"] * (n * len(METHODS)),
            GEO_COL: GEO * len(METHODS),
            SPING_COL: SPING * len(METHODS),
            "n_sping_vp_ties": TIES * len(METHODS),
            "n_vp": [4] * (n * len(METHODS)),
        }
    )
    for k, v in overrides.items():
        base[k] = v
    base["solved"] = base["status"] == "SUCCESS"
    return base


def _fallback(long: pd.DataFrame, method: str, tgs: list[str]) -> pd.DataFrame:
    long = long.copy()
    long.loc[(long.method == method) & long.tg_id.isin(tgs), "status"] = "FALLBACK"
    long["solved"] = long["status"] == "SUCCESS"
    return long


def _p25():
    long = _long()
    return long, V.cohort_frame(long, "p25")


def _row(frame: pd.DataFrame, method: str) -> pd.Series:
    return frame[frame.method == method].iloc[0]


class TestSetStructure:
    def test_union_and_degree_are_counted_over_the_union(self):
        """Cohorts {0,1}, {0,1}, {4,5}, {0,1}: union 4, tg-0/tg-1 at degree 3
        and tg-4/tg-5 at degree 1, so the histogram is {1: 2, 3: 2}."""
        _, rows = _p25()
        built = CO.set_structure(rows)
        s = built[CO.SUMMARY].iloc[0]
        assert s.n_methods == 4 and s.n_union == 4
        assert s.max_degree == 3 and s.n_singletons == 2
        assert s.n_all_methods == 0
        degree = built[CO.DEGREE].set_index("degree").n_tgs.to_dict()
        assert degree == {1: 2, 2: 0, 3: 2}

    def test_degree_shares_sum_to_one_hundred(self):
        _, rows = _p25()
        degree = CO.set_structure(rows)[CO.DEGREE]
        assert degree.share_of_union_pct.sum() == pytest.approx(100.0)

    def test_the_pairwise_diagonal_carries_the_cohort_size(self):
        _, rows = _p25()
        pw = CO.set_structure(rows)[CO.PAIRWISE]
        diag = pw[pw.method_a == pw.method_b].set_index("method_a")
        assert diag.n_shared.to_dict() == {m: 2 for m in METHODS}
        assert (diag.jaccard == 1.0).all()

    def test_disjoint_cohorts_score_zero(self):
        _, rows = _p25()
        pw = CO.set_structure(rows)[CO.PAIRWISE].set_index(["method_a", "method_b"])
        assert pw.loc[(SHORTEST_PING, "m_b")].n_shared == 0
        assert pw.loc[(SHORTEST_PING, "m_b")].jaccard == 0.0
        assert pw.loc[(SHORTEST_PING, "m_a")].shared_of_a_pct == 100.0

    def test_replica_ties_collapse_to_one_distinct_value(self):
        """Trap: ~20 IP replicas share a site and its VP geometry exactly, so
        four TGs at one `geo_vp_dist_to_tg_km` are one observation. Every count
        ships its distinct twin; here the union is 4 TGs but 1 distinct value."""
        long = _long()
        long[GEO_COL] = 7.0
        long[SPING_COL] = 99.0
        rows = V.cohort_frame(long, "p25")
        built = CO.set_structure(rows)
        s = built[CO.SUMMARY].iloc[0]
        assert s.n_union == 4 and s.n_union_distinct == 1
        assert set(built[CO.DEGREE].n_tgs_distinct) <= {0, 1}


class TestAgainstReference:
    def test_the_reference_row_is_emitted_and_first(self):
        """It is the scale the error column is read against, not an
        achievement: `m_c` shares 100% of the cohort at 100.5 km."""
        long, rows = _p25()
        out = CO.against_reference(long, rows)
        assert out.iloc[0].method == SHORTEST_PING
        assert bool(out.iloc[0].is_reference)
        assert out.is_reference.sum() == 1
        ref = _row(out, SHORTEST_PING)
        assert ref.shared_pct == 100.0
        # median(1.003, 2.003)
        assert ref.err_p50_km == pytest.approx(1.503)

    def test_overlap_does_not_track_quality(self):
        long, rows = _p25()
        out = CO.against_reference(long, rows).set_index("method")
        assert out.loc["m_c"].shared_pct == 100.0
        assert out.loc["m_c"].err_p50_km == pytest.approx(100.5)
        assert out.loc["m_b"].shared_pct == 0.0
        assert out.loc["m_b"].answered_not_best_pct == 100.0

    def test_the_three_shares_partition_the_reference_cohort(self):
        long = _fallback(_long(), "m_b", ["tg-0"])
        rows = V.cohort_frame(long, "p25")
        out = CO.against_reference(long, rows)
        total = out.shared_pct + out.answered_not_best_pct + out.refused_pct
        assert total.tolist() == pytest.approx([100.0] * len(out))
        m_b = _row(out, "m_b")
        assert m_b.n_refused == 1 and m_b.refused_pct == pytest.approx(50.0)

    def test_error_percentiles_use_the_answered_denominator(self):
        """`m_b` refuses tg-0, so its median on the 2-TG reference cohort rests
        on one row -- tg-1's 600 km -- and `n_solved` says 1, not 2."""
        long = _fallback(_long(), "m_b", ["tg-0"])
        out = CO.against_reference(long, V.cohort_frame(long, "p25"))
        m_b = _row(out, "m_b")
        assert m_b.n_reference_cohort == 2 and m_b.n_solved == 1
        assert m_b.err_p50_km == pytest.approx(600.0)

    def test_a_wholly_refusing_method_reports_no_error(self):
        long = _fallback(_long(), "m_b", ["tg-0", "tg-1"])
        out = CO.against_reference(long, V.cohort_frame(long, "p25"))
        m_b = _row(out, "m_b")
        assert m_b.n_solved == 0 and m_b.refused_pct == 100.0
        assert pd.isna(m_b.err_p50_km)

    def test_another_method_can_be_the_reference(self):
        long, rows = _p25()
        out = CO.against_reference(long, rows, reference="m_b")
        assert out.iloc[0].method == "m_b"
        assert (out.reference == "m_b").all()
        assert _row(out, SHORTEST_PING).shared_pct == 0.0

    def test_an_unscored_reference_names_the_scored_ones(self):
        long, rows = _p25()
        with pytest.raises(ValueError, match="m_a"):
            CO.against_reference(long, rows, reference="not_a_method")


class TestBeatsReference:
    def test_it_is_strictly_closer(self):
        """S-P against itself is a tie on every TG, and `<` scores that 0%.
        A `<=` would hand it 100% for drawing with itself."""
        long, rows = _p25()
        out = CO.beats_reference(long, rows).set_index("method")
        assert out.loc[SHORTEST_PING].beats_pct == 0.0
        assert out.loc[SHORTEST_PING].n_exact_ties == 2

    def test_a_method_that_wins_every_tg_scores_one_hundred(self):
        """`m_a` predicts 0.5/1.5 km where S-P predicts 1.003/2.003."""
        long, rows = _p25()
        out = CO.beats_reference(long, rows).set_index("method")
        assert out.loc["m_a"].n_beats == 2
        assert out.loc["m_a"].beats_pct == 100.0
        assert out.loc["m_a"].median_delta_km == pytest.approx(-0.503)
        assert out.loc["m_c"].beats_pct == 0.0

    def test_refusals_leave_the_denominator(self):
        """`m_a` refuses tg-0, so its win rate is 1 of 1, not 1 of 2."""
        long = _fallback(_long(), "m_a", ["tg-0"])
        out = CO.beats_reference(long, V.cohort_frame(long, "p25")).set_index("method")
        assert out.loc["m_a"].n_solved == 1 and out.loc["m_a"].beats_pct == 100.0

    def test_counts_ship_a_distinct_twin(self):
        long, rows = _p25()
        out = CO.beats_reference(long, rows).set_index("method")
        assert out.loc["m_a"].n_beats_distinct == 2  # geo 1.0 and 2.0
        long[GEO_COL] = 7.0
        out = CO.beats_reference(long, V.cohort_frame(long, "p25")).set_index("method")
        assert out.loc["m_a"].n_beats == 2 and out.loc["m_a"].n_beats_distinct == 1


class TestRankingAgreement:
    def test_exact_and_within_tolerance_are_reported_separately(self):
        """tg-0/tg-1 agree exactly; tg-2's VPs are 0.4 km apart, so exact is
        2/6 = 33.3% and within-1-km is 3/6 = 50%. They coincide on the real
        meshes -- see `TestRealRuns` -- and the columns stay separate so a run
        that breaks the tie is visible rather than silently rephrased."""
        long, rows = _p25()
        pop = CO.ranking_agreement(long, rows).iloc[0]
        assert pop.scope == "population"
        assert pop.exact_agree_pct == pytest.approx(100 * 2 / 6)
        assert pop.within_tolerance_pct == pytest.approx(100 * 3 / 6)
        assert pop.tolerance_km == CO.RANKING_TOLERANCE_KM

    def test_the_population_row_is_deduplicated_across_methods(self):
        """Four methods over six TGs is 24 rows and six TGs."""
        long, rows = _p25()
        pop = CO.ranking_agreement(long, rows).iloc[0]
        assert len(long) == 24
        assert pop.n == 6 and pop.method == ""

    def test_the_gap_is_sping_minus_geo(self):
        long, rows = _p25()
        pop = CO.ranking_agreement(long, rows).iloc[0]
        # sorted gaps: 0, 0, 0.4, 36, 45, 54 -> median (0.4 + 36) / 2
        assert pop.median_gap_km == pytest.approx(18.2)
        assert pop.max_gap_km == pytest.approx(54.0)

    def test_rtt_ties_are_reported_so_the_phrasing_stays_honest(self):
        """tg-4 and tg-5 have more than one VP at the minimum RTT, so "the
        smallest-RTT VP" is a tie-break on 2 of 6 TGs, at most 3 ways."""
        long, rows = _p25()
        pop = CO.ranking_agreement(long, rows).iloc[0]
        assert pop.n_rtt_tied == 2
        assert pop.rtt_tied_pct == pytest.approx(100 * 2 / 6)
        assert pop.max_rtt_ties == 3

    def test_each_cohort_gets_a_row(self):
        """S-P's cohort is tg-0/tg-1, where the two VPs coincide: 100%.
        `m_b`'s is tg-4/tg-5, where they never do: 0%."""
        long, rows = _p25()
        out = CO.ranking_agreement(long, rows)
        cohorts = out[out.scope == "cohort"].set_index("method")
        assert len(cohorts) == 4
        assert cohorts.loc[SHORTEST_PING].exact_agree_pct == 100.0
        assert cohorts.loc["m_b"].exact_agree_pct == 0.0


class TestMarginIsRequired:
    def test_zero_margin_scores_the_baseline_against_itself(self):
        """S-P predicts the smallest-RTT VP's own coordinate, so the two sides
        of the comparison are equal by construction and differ only by the
        0.003 km the pipeline carries. At margin 0 that reads as a 100% win for
        the baseline over itself; at the 1.0 km default it reads as 0%."""
        _, rows = _p25()
        loose = CO.baseline_margin(rows, margin_km=0.0).set_index("method")
        assert loose.loc[SHORTEST_PING].baseline_closer_pct == 100.0
        tight = CO.baseline_margin(rows, margin_km=CO.DEFAULT_MARGIN_KM).set_index("method")
        assert tight.loc[SHORTEST_PING].baseline_closer_pct == 0.0

    def test_the_default_is_one_kilometre(self):
        assert CO.DEFAULT_MARGIN_KM == 1.0

    def test_a_method_the_baseline_genuinely_beats(self):
        """`m_c` predicts 100/101 km on TGs whose smallest-RTT VP is 1/2 km
        away, so the baseline beats it on both at any sane margin."""
        _, rows = _p25()
        out = CO.baseline_margin(rows).set_index("method")
        assert out.loc["m_c"].n_baseline_closer == 2
        assert out.loc["m_c"].baseline_closer_pct == 100.0
        assert out.loc["m_a"].baseline_closer_pct == 0.0

    def test_it_runs_over_the_methods_own_cohort(self):
        """`m_b`'s cohort is tg-4/tg-5, not S-P's tg-0/tg-1."""
        _, rows = _p25()
        out = CO.baseline_margin(rows).set_index("method")
        assert out.loc["m_b"].n == 2
        # predictions of 1/2 km against sping 50/60: median(1-50, 2-60)
        assert out.loc["m_b"].median_delta_km == pytest.approx(-53.5)
        assert out.loc["m_b"].baseline_closer_pct == 0.0

    def test_counts_ship_a_distinct_twin(self):
        _, rows = _p25()
        assert _row(CO.baseline_margin(rows), "m_c").n_baseline_closer_distinct == 2

    def test_a_negative_margin_is_refused(self):
        _, rows = _p25()
        with pytest.raises(ValueError, match="margin_km"):
            CO.baseline_margin(rows, margin_km=-1.0)


class TestCohortsAreNeverRebuilt:
    def test_a_fallback_row_cannot_enter_a_cohort(self):
        """`m_a`'s two smallest prediction distances are refusals. Ranked on
        without the mask its cohort would be tg-0/tg-1 and it would share 100%
        of S-P's cohort; masked, it is tg-2/tg-3 and it shares none."""
        long = _fallback(_long(), "m_a", ["tg-0", "tg-1"])
        rows = V.cohort_frame(long, "p25")
        assert sorted(rows[rows.method == "m_a"].tg_id) == ["tg-2", "tg-3"]
        pw = CO.set_structure(rows)[CO.PAIRWISE].set_index(["method_a", "method_b"])
        assert pw.loc[("m_a", SHORTEST_PING)].n_shared == 0

    def test_a_fallback_row_cannot_be_shared_on_the_reference_cohort(self):
        long = _fallback(_long(), "m_a", ["tg-0", "tg-1"])
        out = CO.against_reference(long, V.cohort_frame(long, "p25"))
        m_a = _row(out, "m_a")
        assert m_a.n_shared == 0 and m_a.n_refused == 2
        assert m_a.refused_pct == 100.0

    def test_the_module_reuses_the_figures_cohort_filter(self):
        """Not a style point: a second implementation is a second chance to
        drop the mask."""
        assert CO.ERROR_COLUMN is V.RANK_COLUMN
        assert CO.GEO_COL == V.MEASURE_COLUMNS[V.GEO]
        assert CO.SPING_COL == V.MEASURE_COLUMNS[V.SPING]


# -- privacy ----------------------------------------------------------------

#: Anything resembling a location. A new column matching one of these is a
#: privacy regression, not a naming choice.
FORBIDDEN = (
    "lat", "lon", "coord", "city", "town", "place", "country", "region",
    "site", "address", "geohash", "postal", "venue", "facility", "name",
)

#: The full emitted schema. Deliberately pinned: adding a column must be a
#: deliberate act reviewed against `FORBIDDEN`, not something a test waves
#: through.
COLUMNS = {
    CO.SUMMARY: {
        "n_methods", "n_union", "n_union_distinct", "n_singletons",
        "n_all_methods", "max_degree", "mean_degree",
    },
    CO.DEGREE: {"degree", "n_tgs", "n_tgs_distinct", "share_of_union_pct"},
    CO.PAIRWISE: {
        "method_a", "method_label_a", "method_b", "method_label_b", "n_a",
        "n_a_distinct", "n_b", "n_b_distinct", "n_shared", "n_shared_distinct",
        "shared_of_a_pct", "jaccard",
    },
    "against_reference": {
        "method", "method_label", "is_reference", "reference",
        "n_reference_cohort", "n_reference_cohort_distinct", "n_shared",
        "n_shared_distinct", "shared_pct", "n_answered_not_best",
        "n_answered_not_best_distinct", "answered_not_best_pct", "n_refused",
        "n_refused_distinct", "refused_pct", "n_solved",
        "n_solved_distinct", "err_min_km", "err_max_km", "err_mean_km",
        "err_p25_km", "err_p50_km", "err_p75_km", "err_p90_km", "err_p95_km",
    },
    "beats_reference": {
        "method", "method_label", "is_reference", "reference",
        "n_reference_cohort", "n_solved", "n_solved_distinct", "n_beats",
        "n_beats_distinct", "beats_pct", "n_exact_ties", "median_delta_km",
    },
    "ranking_agreement": {
        "scope", "method", "method_label", "n", "n_distinct", "n_exact_agree",
        "n_exact_agree_distinct", "exact_agree_pct", "n_within_tolerance",
        "n_within_tolerance_distinct", "within_tolerance_pct",
        "tolerance_km", "median_gap_km", "p90_gap_km", "max_gap_km",
        "n_rtt_tied", "n_rtt_tied_distinct", "rtt_tied_pct", "max_rtt_ties",
    },
    "baseline_margin": {
        "method", "method_label", "margin_km", "n", "n_distinct",
        "n_baseline_closer", "n_baseline_closer_distinct",
        "baseline_closer_pct", "median_delta_km",
    },
}

#: Every string any table may emit: method ids, their terms, scope names and
#: the reference id. No free text, so no place name can ride in as a value.
ALLOWED_STRINGS = set(METHODS) | {
    V.method_label(m) for m in METHODS
} | {"population", "cohort", "all TGs", ""}


class TestDistinctCountsAccompanyEveryShare:
    """Trap: ~20 IP replicas per coordinate, so a raw numerator overstates
    independence. Six TGs at one distance are one observation."""

    @staticmethod
    def _built():
        long, rows = _p25()
        return CO.tables(long, rows)

    def test_every_emitted_share_declares_its_numerator(self):
        shares = {c for f in self._built().values() for c in f.columns if c.endswith("_pct")}
        assert shares == set(CO.PCT_NUMERATORS)

    def test_every_numerator_ships_a_distinct_twin(self):
        for name, frame in self._built().items():
            for pct in [c for c in frame.columns if c.endswith("_pct")]:
                count = CO.PCT_NUMERATORS[pct]
                assert count in frame.columns, f"{name}.{pct}"
                assert f"{count}_distinct" in frame.columns, f"{name}.{count}"

    def test_a_fully_tied_cluster_reports_one_distinct(self):
        long = _long()
        long[GEO_COL] = 7.0
        long[SPING_COL] = 99.0
        rows = V.cohort_frame(long, "p25")
        for name, frame in CO.tables(long, rows).items():
            for col in [c for c in frame.columns if c.endswith("_distinct")]:
                assert frame[col].max() <= 1, f"{name}.{col}"


class TestPrivacy:
    """Target-based analysis only: no coordinate, site id or place name."""

    @staticmethod
    def _built():
        long, rows = _p25()
        return CO.tables(long, rows)

    def test_every_table_emits_exactly_the_reviewed_schema(self):
        built = self._built()
        assert set(built) == set(CO.REPORTS)
        for name, frame in built.items():
            assert set(frame.columns) == COLUMNS[name], name

    def test_no_column_name_resembles_a_location(self):
        for name, frame in self._built().items():
            for col in frame.columns:
                bad = [f for f in FORBIDDEN if f in col.lower()]
                assert not bad, f"{name}.{col} contains {bad}"

    def test_no_string_value_is_free_text(self):
        for name, frame in self._built().items():
            for col in frame.columns:
                if frame[col].dtype != object:
                    continue
                unknown = set(frame[col].astype(str)) - ALLOWED_STRINGS
                assert not unknown, f"{name}.{col} emits {sorted(unknown)}"

    def test_no_value_is_shaped_like_a_coordinate_pair(self):
        decimal_pair = re.compile(r"-?\d+\.\d+\s*[,;]\s*-?\d+\.\d+")
        for name, frame in self._built().items():
            joined = "|".join(frame.astype(str).to_numpy().ravel())
            assert not decimal_pair.search(joined), name

    def test_the_manifest_carries_no_location_either(self):
        """Its data, that is. `policy` is documentation -- it says in prose
        *why* a FALLBACK row carries the S-P VP's coordinate -- so it is
        scanned for a coordinate-shaped value rather than for the word."""
        long, rows = _p25()
        built = CO.tables(long, rows)
        meta = {"run_ids": ["r"], "nside": 128, "methods": METHODS,
                "n_tgs": 6, "n_vp_per_tg_median": 4.0}
        body = json.loads(
            CO._manifest(meta, "p25", built, reference=SHORTEST_PING, margin_km=1.0)
        )
        policy = body.pop("policy")
        blob = json.dumps(body).lower()
        for token in FORBIDDEN:
            if token == "name":  # `full name` appears in the method term table
                continue
            assert token not in blob, token
        decimal_pair = re.compile(r"-?\d+\.\d+\s*[,;]\s*-?\d+\.\d+")
        assert not decimal_pair.search(json.dumps(policy))


class TestManifest:
    def test_it_records_k_and_the_headline_set_structure(self):
        long, rows = _p25()
        built = CO.tables(long, rows)
        meta = {"run_ids": ["r"], "nside": 128, "methods": METHODS,
                "n_tgs": 6, "n_vp_per_tg_median": 4.0}
        body = json.loads(
            CO._manifest(meta, "p25", built, reference=SHORTEST_PING, margin_km=1.0)
        )
        assert body["cohort_k_requested"] == 2
        assert body["reference"] == SHORTEST_PING
        assert body["margin_km"] == 1.0
        assert body["headline"]["n_union"] == 4
        assert body["headline"]["max_degree"] == 3
        assert body["headline"]["degree_histogram"] == {"1": 2, "2": 0, "3": 2}
        assert body["headline"]["reference_err_p50_km"] == pytest.approx(1.503)
        assert set(body["tables"]) == set(CO.REPORTS)

    def test_it_flags_a_short_cohort(self):
        long = _fallback(_long(), "m_b", ["tg-4", "tg-5", "tg-0", "tg-1", "tg-2"])
        rows = V.cohort_frame(long, "p25")
        built = CO.tables(long, rows)
        meta = {"run_ids": ["r"], "nside": 128, "methods": METHODS,
                "n_tgs": 6, "n_vp_per_tg_median": 4.0}
        body = json.loads(
            CO._manifest(meta, "p25", built, reference=SHORTEST_PING, margin_km=1.0)
        )
        assert body["n_cohort_short"] == {V.method_label("m_b"): 1}


class TestBuildGuards:
    def test_unknown_cohort_is_refused_before_loading(self):
        with pytest.raises(ValueError, match="unknown cohort"):
            CO.build_for_runs([], cohorts=["p50"])

    def test_negative_margin_is_refused_before_loading(self):
        with pytest.raises(ValueError, match="margin_km"):
            CO.build_for_runs([], margin_km=-0.5)

    def test_the_default_cohorts_are_the_two_the_paper_quotes(self):
        assert CO.DEFAULT_COHORTS == ("p5", "p25")
        assert all(c in V.COHORTS for c in CO.DEFAULT_COHORTS)


# -- real runs --------------------------------------------------------------

RUN_IDS = [f"as0{i}-260728-260802-mesh" for i in (1, 2, 3)]
V5_ANALYSIS = REPO_ROOT / "outputs" / "analysis" / "v5"


def _available() -> bool:
    return all((DEFAULT_OUTPUTS_ROOT / r).exists()
               and (V5_ANALYSIS / r / "classify" / "healpix-128").exists() for r in RUN_IDS)


@pytest.fixture(scope="module")
def built():
    """Read-only: `load` writes nothing, so the real classify root is safe."""
    runs = [resolve_run(r) for r in RUN_IDS]
    long, meta = V.load(runs, analysis_root=V5_ANALYSIS)
    return {
        c: (long, V.cohort_frame(long, c), meta) for c in ("p5", "p25")
    }


@pytest.mark.skipif(not _available(), reason="as01-03 runs or their v5 classify absent")
class TestRealRuns:
    """The numbers `evaluation.tex` prints, re-derived from the meshes.

    Each was measured independently during drafting; a mismatch here means the
    module changed, not that the paper was wrong.
    """

    @pytest.mark.parametrize(
        "cohort, k, union, histogram",
        [
            ("p5", 63, 254, {1: 154, 2: 78, 3: 20, 4: 2}),
            ("p25", 317, 686, {1: 147, 2: 191, 3: 157, 4: 88, 5: 68, 6: 35}),
        ],
    )
    def test_set_structure(self, built, cohort, k, union, histogram):
        long, rows, _ = built[cohort]
        assert V.cohort_k(long, cohort) == k
        out = CO.set_structure(rows)
        s = out[CO.SUMMARY].iloc[0]
        assert s.n_union == union
        assert s.max_degree == max(histogram)
        got = out[CO.DEGREE].set_index("degree").n_tgs.to_dict()
        assert got == histogram

    @pytest.mark.parametrize(
        "cohort, expected",
        [
            ("p5", {
                "shortest_ping": (100.0, 0.0, 0.46),
                "million_scale_cbg": (96.8, 0.0, 0.5),
                "octant_cbg_spl": (11.1, 0.0, 41.1),
                "octant_cbg_hull": (9.5, 0.0, 7.5),
                "spotter_cbg": (0.0, 0.0, 149.1),
                "vanilla_cbg": (0.0, 57.1, 85.2),
            }),
            ("p25", {
                "shortest_ping": (100.0, 0.0, 8.77),
                "million_scale_cbg": (94.0, 0.0, 13.8),
                "octant_cbg_spl": (47.0, 0.0, 20.8),
                "octant_cbg_hull": (42.6, 0.0, 20.8),
                "spotter_cbg": (43.8, 0.0, 152.9),
                "vanilla_cbg": (25.6, 59.3, 59.0),
            }),
        ],
    )
    def test_against_reference(self, built, cohort, expected):
        long, rows, _ = built[cohort]
        out = CO.against_reference(long, rows).set_index("method")
        assert out.index[0] == SHORTEST_PING
        for method, (shared, refused, err) in expected.items():
            row = out.loc[method]
            assert row.shared_pct == pytest.approx(shared, abs=0.05), method
            assert row.refused_pct == pytest.approx(refused, abs=0.05), method
            assert row.err_p50_km == pytest.approx(err, abs=0.05), method

    @pytest.mark.parametrize(
        "cohort, n_solved",
        [("p5", 27), ("p25", 129)],
    )
    def test_the_refusing_methods_denominator_is_its_answers(self, built, cohort, n_solved):
        """VAN's median is over 27 of 63 TGs at p5, not over 63."""
        long, rows, _ = built[cohort]
        row = CO.against_reference(long, rows).set_index("method").loc["vanilla_cbg"]
        assert row.n_solved == n_solved

    @pytest.mark.parametrize(
        "cohort, expected",
        [
            ("p5", {"octant_cbg_hull": 7.9, "octant_cbg_spl": 9.5,
                    "million_scale_cbg": 0.0, "vanilla_cbg": 0.0, "spotter_cbg": 0.0}),
            ("p25", {"octant_cbg_hull": 26.2, "octant_cbg_spl": 26.8,
                     "million_scale_cbg": 16.7, "vanilla_cbg": 3.9, "spotter_cbg": 0.0}),
        ],
    )
    def test_beats_reference(self, built, cohort, expected):
        long, rows, _ = built[cohort]
        out = CO.beats_reference(long, rows).set_index("method")
        assert out.loc[SHORTEST_PING].beats_pct == 0.0
        for method, pct in expected.items():
            assert out.loc[method].beats_pct == pytest.approx(pct, abs=0.05), method

    def test_ranking_agreement_population(self, built):
        long, rows, _ = built["p25"]
        pop = CO.ranking_agreement(long, rows).iloc[0]
        assert pop.n == 1269
        assert pop.exact_agree_pct == pytest.approx(19.1, abs=0.05)
        assert pop.median_gap_km == pytest.approx(113.83, abs=0.005)
        assert pop.rtt_tied_pct == pytest.approx(3.9, abs=0.05)
        assert pop.max_rtt_ties == 3

    @pytest.mark.parametrize("cohort, pct", [("p5", 100.0), ("p25", 67.5)])
    def test_ranking_agreement_on_the_baselines_cohort(self, built, cohort, pct):
        long, rows, _ = built[cohort]
        out = CO.ranking_agreement(long, rows)
        row = out[(out.scope == "cohort") & (out.method == SHORTEST_PING)].iloc[0]
        assert row.exact_agree_pct == pytest.approx(pct, abs=0.05)

    @pytest.mark.parametrize("cohort", ["p5", "p25"])
    def test_exact_and_within_tolerance_coincide_on_this_substrate(self, built, cohort):
        """The metric is bimodal here: the smallest-RTT VP is either the
        closest VP or a hundred kilometres from it. If this ever fails, the
        paper's phrasing needs a threshold and the columns already carry one."""
        long, rows, _ = built[cohort]
        out = CO.ranking_agreement(long, rows)
        assert (out.n_exact_agree == out.n_within_tolerance).all()

    @pytest.mark.parametrize(
        "cohort, expected",
        [
            ("p5", {"spotter_cbg": 87.3, "octant_cbg_hull": 0.0,
                    "octant_cbg_spl": 0.0, "shortest_ping": 0.0}),
            ("p25", {"spotter_cbg": 83.9, "vanilla_cbg": 51.4,
                     "octant_cbg_hull": 14.8, "octant_cbg_spl": 14.8,
                     "shortest_ping": 0.0}),
        ],
    )
    def test_baseline_margin(self, built, cohort, expected):
        _, rows, _ = built[cohort]
        out = CO.baseline_margin(rows).set_index("method")
        for method, pct in expected.items():
            assert out.loc[method].baseline_closer_pct == pytest.approx(pct, abs=0.05), method

    def test_zero_margin_is_degenerate_on_the_real_substrate(self, built):
        """Not a fixture artefact: at margin 0 the control scores 100% against
        itself on its own p5 cohort."""
        _, rows, _ = built["p5"]
        out = CO.baseline_margin(rows, margin_km=0.0).set_index("method")
        assert out.loc[SHORTEST_PING].baseline_closer_pct == 100.0
