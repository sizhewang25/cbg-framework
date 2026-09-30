"""The champion UpSet: who is within the tie of the lowest error, per TG.

Three invariants carry these tests. Unanswered rows never compete (the
`solved_mask` trap: a FALLBACK row carries S-P's coordinate). The exact
combinations partition the TGs, and each method's set size is the sum of the
combinations that contain it. Pooling is row-wise, so the pooled tables are
the per-run tables summed.
"""

from __future__ import annotations

import json

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import figure_champion_upset as U
from scripts.analysis.v5.modules import figure_error_cdf as E
from scripts.analysis.v5.modules.paths import RunPaths
from scripts.analysis.v5.modules.status import SHORTEST_PING

NSIDE = U.SOURCE_NSIDE


def _tgs(dists, *, status=None, first_id: int = 0, sites=None) -> pd.DataFrame:
    """A scored frame: one row per distance, SUCCESS unless `status` says not."""
    n = len(dists)
    status = list(status) if status is not None else ["SUCCESS"] * n
    site = list(sites) if sites is not None else list(range(n))
    return pd.DataFrame(
        {
            "tg_id": [f"tg-{first_id + i}" for i in range(n)],
            "tg_lat": [40.0 + s for s in site],
            "tg_lon": [-100.0] * n,
            "status": status,
            "pred_lat": [40.0] * n,
            "pred_lon": [-100.0] * n,
            "pred_dist_to_tg_km": [float(d) for d in dists],
        }
    )


def _baseline(dists, **kw) -> pd.DataFrame:
    df = _tgs(dists, **kw)
    df["status"] = "BASELINE"
    return df


def _write_run(root, run_id: str, frames: dict[str, pd.DataFrame]) -> RunPaths:
    run = RunPaths(run_id=run_id, root=root, source="s", setup="t")
    out = run.classify_dir(NSIDE, root=root)
    for method, df in frames.items():
        df.to_parquet(out / C.TGS_PARQUET.format(method=method), index=False)
    return run


def _mask(frame: dict[str, list[float]], tie_km: float = 1.0) -> pd.DataFrame:
    return U.champion_mask(pd.DataFrame(frame), tie_km)


class TestContest:
    def test_errors_within_the_tie_both_win(self):
        got = _mask({"a": [10.0], "b": [10.9], "c": [11.0 + 1e-6]})
        assert got.iloc[0].to_dict() == {"a": True, "b": True, "c": False}

    def test_the_tie_boundary_is_inclusive(self):
        assert _mask({"a": [10.0], "b": [11.0]}).iloc[0].all()

    def test_a_zero_tie_still_credits_exact_equality(self):
        got = _mask({"a": [5.0], "b": [5.0], "c": [5.001]}, tie_km=0.0)
        assert got.iloc[0].tolist() == [True, True, False]

    def test_nan_never_wins_and_an_all_nan_row_has_no_champion(self):
        got = _mask({"a": [np.nan, np.nan], "b": [3.0, np.nan]})
        assert got.iloc[0].tolist() == [False, True]
        assert not got.iloc[1].any()
        assert U.combination(got).iloc[1] == U.NONE_LABEL

    def test_a_negative_tie_is_refused(self):
        with pytest.raises(ValueError, match="tie_km"):
            _mask({"a": [1.0]}, tie_km=-1.0)


class TestRowPolicy:
    def test_a_fallback_never_wins_even_at_zero_km(self, tmp_path):
        run = _write_run(
            tmp_path, "as01-x-mesh",
            {
                "vanilla_cbg": _tgs([0.0, 0.0], status=["FALLBACK", "ERROR"]),
                SHORTEST_PING: _baseline([50.0, 60.0]),
            },
        )
        errors, _ = U.load_run(run, analysis_root=tmp_path)
        mask = U.champion_mask(errors)
        assert not mask["vanilla_cbg"].any()
        assert mask[SHORTEST_PING].all()

    def test_baseline_rows_are_answered(self, tmp_path):
        run = _write_run(tmp_path, "as01-x-mesh", {SHORTEST_PING: _baseline([1.0, 2.0])})
        errors, _ = E.error_matrix(run, analysis_root=tmp_path)
        assert errors[SHORTEST_PING].notna().all()

    def test_error_matrix_agrees_with_load_errors(self, tmp_path):
        run = _write_run(
            tmp_path, "as01-x-mesh",
            {
                "octant_cbg_hull": _tgs([3.0, 9.0, 4.0], status=["SUCCESS", "FALLBACK", "SUCCESS"]),
                SHORTEST_PING: _baseline([5.0, 6.0, 7.0]),
            },
        )
        keyed, _ = E.error_matrix(run, analysis_root=tmp_path)
        flat = E.load_errors(run, analysis_root=tmp_path)
        for method, entry in flat.items():
            assert sorted(keyed[method].dropna()) == sorted(entry["errors"])

    def test_methods_over_different_rosters_are_refused(self, tmp_path):
        run = _write_run(
            tmp_path, "as01-x-mesh",
            {"a": _tgs([1.0, 2.0]), "b": _tgs([1.0, 2.0, 3.0])},
        )
        with pytest.raises(ValueError, match="different TG sets"):
            E.error_matrix(run, analysis_root=tmp_path)


class TestTables:
    @pytest.fixture
    def mask(self):
        return _mask(
            {
                SHORTEST_PING: [5.0, 5.0, 50.0, 9.0, 30.0],
                "million_scale_cbg": [5.004, 20.0, 50.5, 9.5, 31.0],
                "octant_cbg_hull": [40.0, 1.0, 2.0, 9.9, 30.5],
                "spotter_cbg": [90.0, 90.0, 90.0, 90.0, 90.0],
            }
        )

    @pytest.fixture
    def sites(self, mask):
        return pd.Series(["s0", "s0", "s1", "s2", "s2"], index=mask.index)

    def test_intersections_partition_the_tgs(self, mask, sites):
        table = U.intersection_table(mask, sites)
        assert table["n_tgs"].sum() == len(mask)
        assert table["share"].sum() == pytest.approx(1.0, abs=1e-3)

    def test_set_size_is_the_sum_of_its_combinations(self, mask, sites):
        inter = U.intersection_table(mask, sites)
        sets = U.set_table(mask, sites).set_index("method_label")
        for label, row in sets.iterrows():
            has = inter["members"].str.split(U.MEMBER_SEP).apply(lambda ms: label in ms)
            assert inter.loc[has, "n_tgs"].sum() == row["n_champion"]

    def test_a_method_that_never_wins_keeps_its_row(self, mask, sites):
        sets = U.set_table(mask, sites).set_index("method")
        assert sets.loc["spotter_cbg", "n_champion"] == 0

    def test_sites_count_distinct_keys(self, mask, sites):
        sets = U.set_table(mask, sites).set_index("method")
        # S-P wins TG 0 (s0) and TGs 3, 4 (s2): three TGs, two sites.
        assert sets.loc[SHORTEST_PING, ["n_champion", "n_sites_champion"]].tolist() == [3, 2]

    def test_site_table_grades_any_majority_all(self, mask, sites):
        got = U.site_table(mask, sites).set_index("method")
        cols = ["n_sites_any", "n_sites_majority", "n_sites_all"]
        # S-P: s0 1/2 (any, not majority), s1 0/1, s2 2/2.
        assert got.loc[SHORTEST_PING, cols].tolist() == [2, 1, 1]
        # OCT-H: s0 1/2, s1 1/1, s2 2/2 (both s2 TGs are three-way ties).
        assert got.loc["octant_cbg_hull", cols].tolist() == [3, 2, 2]
        assert got.loc["spotter_cbg", cols].tolist() == [0, 0, 0]
        assert got["n_sites"].unique().tolist() == [3]

    def test_site_any_matches_the_set_tables_site_count(self, mask, sites):
        got = U.site_table(mask, sites).set_index("method")["n_sites_any"]
        want = U.set_table(mask, sites).set_index("method")["n_sites_champion"]
        pd.testing.assert_series_equal(got, want, check_names=False)

    def test_sole_counts_only_unshared_wins(self, mask, sites):
        sets = U.set_table(mask, sites).set_index("method")
        assert sets.loc["octant_cbg_hull", "n_sole"] == 2


def _three_method_run(root, run_id, first_id, dists):
    sp, soi, octh = dists
    return _write_run(
        root, run_id,
        {
            SHORTEST_PING: _baseline(sp, first_id=first_id),
            "million_scale_cbg": _tgs(soi, first_id=first_id),
            "octant_cbg_hull": _tgs(octh, first_id=first_id),
        },
    )


class TestPooling:
    @pytest.fixture
    def runs(self, tmp_path):
        a = _three_method_run(
            tmp_path, "as01-x-mesh", 0,
            ([5.0, 50.0, 8.0], [5.2, 60.0, 30.0], [40.0, 2.0, 9.0]),
        )
        b = _three_method_run(
            tmp_path, "as02-x-mesh", 100,
            ([7.0, 70.0], [7.0, 70.0], [1.0, 80.0]),
        )
        return [a, b]

    def test_pooled_tables_are_the_per_run_tables_summed(self, runs, tmp_path):
        pooled = U.build_for_runs(runs, layouts=(U.POOLED,), analysis_root=tmp_path)[0]
        per_run = U.build_for_runs(runs, layouts=(U.PER_RUN,), analysis_root=tmp_path)
        want = (
            pd.concat(pd.read_csv(w["intersections"]) for w in per_run)
            .groupby("members")["n_tgs"].sum()
        )
        got = pd.read_csv(pooled["intersections"]).set_index("members")["n_tgs"]
        pd.testing.assert_series_equal(got.sort_index(), want.sort_index(), check_names=False)
        sets = pd.read_csv(pooled["sets"]).set_index("method")
        assert sets["n_tgs"].unique().tolist() == [5]

    def test_a_method_missing_from_one_run_is_refused(self, runs, tmp_path):
        extra = runs[0].classify_dir(NSIDE, root=tmp_path) / C.TGS_PARQUET.format(
            method="vanilla_cbg"
        )
        _tgs([1.0, 2.0, 3.0]).to_parquet(extra, index=False)
        with pytest.raises(ValueError, match="cannot pool"):
            U.build_for_runs(runs, layouts=(U.POOLED,), analysis_root=tmp_path)

    def test_shared_tg_ids_are_refused(self, tmp_path):
        dists = ([1.0], [2.0], [3.0])
        runs = [
            _three_method_run(tmp_path, "as01-x-mesh", 0, dists),
            _three_method_run(tmp_path, "as02-x-mesh", 0, dists),
        ]
        with pytest.raises(ValueError, match="share"):
            U.build_for_runs(runs, layouts=(U.POOLED,), analysis_root=tmp_path)


class TestArtifacts:
    def test_names_carry_the_tie_and_the_pooled_infix(self):
        assert U.artifact_names(U.PER_RUN)["png"] == "champion_upset.tie-1km.png"
        assert U.artifact_names(U.POOLED, 0.5)["sets"] == "champion_upset.pooled.tie-0.5km.sets.csv"

    def test_every_artifact_is_written_and_the_manifest_reads_back(self, tmp_path):
        run = _three_method_run(
            tmp_path, "as01-x-mesh", 0,
            ([5.0, 50.0, 8.0], [5.2, 60.0, 30.0], [40.0, 2.0, 9.5]),
        )
        written = U.build_for_run(run, analysis_root=tmp_path)
        assert set(written) == set(U.KINDS)
        assert all(p.exists() for p in written.values())
        body = json.loads(written["manifest"].read_text())
        assert body["tie_km"] == 1.0 and body["n_tgs"] == 3
        assert body["n_champion_per_method"] == {
            SHORTEST_PING: 2, "million_scale_cbg": 1, "octant_cbg_hull": 1,
        }
        audit = pd.read_csv(written["membership"])
        assert audit["tg_id"].tolist() == ["tg-0", "tg-1", "tg-2"]
        assert audit["members"].tolist() == ["S-P+SOI", "OCT-H", "S-P"]
        assert mpimg.imread(written["png"]).shape[0] > 0

    def test_a_zero_champion_method_keeps_its_row_in_the_figure(self, tmp_path, monkeypatch):
        seen = {}
        real = U.plot_upset

        def spy(mask, out_path, **kw):
            seen["columns"] = list(mask.columns)
            return real(mask, out_path, **kw)

        monkeypatch.setattr(U, "plot_upset", spy)
        run = _write_run(
            tmp_path, "as01-x-mesh",
            {SHORTEST_PING: _baseline([1.0, 2.0]), "spotter_cbg": _tgs([500.0, 600.0])},
        )
        written = U.build_for_run(run, analysis_root=tmp_path)
        assert "spotter_cbg" in seen["columns"]
        sets = pd.read_csv(written["sets"]).set_index("method")
        assert sets.loc["spotter_cbg", "n_champion"] == 0
        assert json.loads(written["manifest"].read_text())["empty_sets"] == ["spotter_cbg"]


class TestBaselineInk:
    def test_the_baseline_takes_the_error_cdfs_colour(self):
        cdf = E._curve_style(SHORTEST_PING, {})["color"]
        assert U.upset_colors([SHORTEST_PING, "octant_cbg_hull"])[SHORTEST_PING] == cdf

    def test_the_baseline_is_not_the_tie_grey(self):
        assert U.BASELINE_INK != U.TIE_INK

    def test_other_methods_keep_their_hue(self):
        from scripts.analysis.v5.modules.methods import method_colors

        assert U.upset_colors(["octant_cbg_hull"]) == method_colors(["octant_cbg_hull"])
