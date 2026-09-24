"""The error CDF: which rows it draws, and which numbers it must agree with.

Ported from v4. Two invariants carry most of these tests: the population is
`status.solved_mask`, so `n_plotted` is `accuracy.csv`'s `n_solved`; and the
percentiles are the same pandas call `classify.summarize` makes, so the two
files agree digit for digit.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules import figure_error_cdf as E
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.paths import CLASSIFY_KIND, MissingArtifactError, RunPaths
from scripts.analysis.v5.modules.status import SHORTEST_PING


def _tgs(*, solved: int = 0, failed: int = 0, errors=None, first_id: int = 0) -> pd.DataFrame:
    """A scored frame: `solved` SUCCESS rows then `failed` FALLBACK rows.

    FALLBACK rows carry a finite distance, as the real scorer writes them.
    """
    n = solved + failed
    dist = list(errors) if errors is not None else [10.0 * (i + 1) for i in range(solved)]
    assert len(dist) == solved
    return pd.DataFrame(
        {
            "tg_id": [f"tg-{first_id + i}" for i in range(n)],
            "status": ["SUCCESS"] * solved + ["FALLBACK"] * failed,
            "pred_lat": [40.0] * n,
            "pred_lon": [-100.0] * n,
            "ring": [0] * n,
            "cell_label": ["true"] * n,
            "pred_dist_to_tg_km": [*dist, *([123.0] * failed)],
            "pred_dist_to_seed_km": [*dist, *([123.0] * failed)],
        }
    )


def _write_run(root, ds: str, frames: dict[str, pd.DataFrame], nside: int = 128) -> RunPaths:
    """A run whose `classify/healpix-<n>/` holds `frames` and their accuracy.csv."""
    run = RunPaths(run_id=f"{ds}-x-mesh", root=root, source="s", setup="t")
    out = run.classify_dir(nside, root=root)
    for method, df in frames.items():
        df.to_parquet(out / C.TGS_PARQUET.format(method=method), index=False)
    C.summarize(frames, nside).to_csv(out / C.ACCURACY_CSV, index=False)
    return run


def _acc(run, root, nside=128):
    return pd.read_csv(run.classify_dir(nside, root=root) / C.ACCURACY_CSV).set_index("method")


class TestPopulation:
    def test_unanswered_rows_are_excluded(self, tmp_path):
        run = _write_run(tmp_path, "as01", {"m": _tgs(solved=5, failed=5)})
        got = E.load_errors(run, analysis_root=tmp_path)["m"]
        assert (got["n_tgs"], got["n_solved"], got["n_failed"]) == (10, 5, 5)
        assert len(got["errors"]) == 5

    def test_a_nan_filter_alone_would_not_drop_fallbacks(self, tmp_path):
        """FALLBACK rows carry the VP's coordinate, so a finite distance."""
        run = _write_run(tmp_path, "as01", {"m": _tgs(solved=4, failed=4)})
        got = E.load_errors(run, analysis_root=tmp_path)["m"]
        assert got["n_solved"] == 4
        assert 123.0 not in set(got["errors"])

    def test_an_all_baseline_frame_is_wholly_included(self, tmp_path):
        """`status == "SUCCESS"` would empty the S-P curve."""
        df = _tgs(solved=6)
        df["status"] = "BASELINE"
        run = _write_run(tmp_path, "as01", {SHORTEST_PING: df})
        got = E.load_errors(run, analysis_root=tmp_path)[SHORTEST_PING]
        assert got["n_solved"] == 6 and len(got["errors"]) == 6

    def test_a_solved_row_without_a_distance_is_counted(self, tmp_path):
        df = _tgs(solved=4)
        df.loc[0, ["pred_lat", "pred_lon", "pred_dist_to_tg_km", "pred_dist_to_seed_km"]] = np.nan
        run = _write_run(tmp_path, "as01", {"m": df})
        got = E.load_errors(run, analysis_root=tmp_path)["m"]
        assert got["n_solved"] == 4 and got["n_no_distance"] == 1
        assert len(got["errors"]) == 3
        assert len(got["errors"]) == _acc(run, tmp_path).loc["m", "n_solved"]

    def test_tg_ids_are_the_whole_roster(self, tmp_path):
        run = _write_run(tmp_path, "as01", {"m": _tgs(solved=3, failed=3)})
        assert len(E.load_errors(run, analysis_root=tmp_path)["m"]["tg_ids"]) == 6

    def test_missing_parquet_names_the_command_that_writes_it(self, tmp_path):
        run = RunPaths(run_id="as01-x-mesh", root=tmp_path, source="s", setup="t")
        run.classify_dir(128, root=tmp_path)
        with pytest.raises(MissingArtifactError, match="classify --run-id"):
            E.load_errors(run, analysis_root=tmp_path)


class TestPercentiles:
    def test_p50_and_p90_match_the_accuracy_table_exactly(self, tmp_path):
        sp = _tgs(solved=8, errors=[2.0 * i for i in range(8)])
        sp["status"] = "BASELINE"
        run = _write_run(
            tmp_path, "as01",
            {"m": _tgs(solved=10, failed=4, errors=list(range(1, 11))), SHORTEST_PING: sp},
        )
        table = E.percentile_table(E.load_errors(run, analysis_root=tmp_path))
        acc = _acc(run, tmp_path)
        for _, row in table.iterrows():
            for p in (50, 90):
                assert row[E.pcol(p)] == acc.loc[row["method"], E.pcol(p)]
            assert row["n_plotted"] == acc.loc[row["method"], "n_solved"]

    def test_both_published_percentiles_are_reported(self):
        assert {50, 90} <= set(E.PERCENTILES)

    def test_a_method_that_answered_nothing_reports_no_percentile(self, tmp_path):
        run = _write_run(tmp_path, "as01", {"m": _tgs(failed=5)})
        table = E.percentile_table(E.load_errors(run, analysis_root=tmp_path))
        assert table.iloc[0]["n_plotted"] == 0
        assert np.isnan(table.iloc[0][E.pcol(50)])

    def test_labels_are_the_method_terms(self, tmp_path):
        run = _write_run(tmp_path, "as01", {"octant_cbg_hull": _tgs(solved=2)})
        table = E.percentile_table(E.load_errors(run, analysis_root=tmp_path))
        assert table.iloc[0]["method_label"] == "OCT-H"


class TestCurveOrder:
    def test_curves_rank_by_median_ascending(self, tmp_path):
        run = _write_run(tmp_path, "as01", {
            "far": _tgs(solved=4, errors=[300.0, 310, 320, 330]),
            "near": _tgs(solved=4, errors=[10.0, 11, 12, 13]),
            "mid": _tgs(solved=4, errors=[100.0, 101, 102, 103]),
        })
        table = E.percentile_table(E.load_errors(run, analysis_root=tmp_path))
        assert E.curve_order(table) == ["near", "mid", "far"]

    def test_a_p50_tie_is_broken_by_the_tail(self, tmp_path):
        run = _write_run(tmp_path, "as01", {
            "fat_tail": _tgs(solved=4, errors=[10.0, 10, 10, 9000]),
            "thin_tail": _tgs(solved=4, errors=[10.0, 10, 10, 11]),
        })
        table = E.percentile_table(E.load_errors(run, analysis_root=tmp_path))
        assert E.curve_order(table) == ["thin_tail", "fat_tail"]

    def test_the_order_is_total(self, tmp_path):
        run = _write_run(tmp_path, "as01", {
            "b_same": _tgs(solved=3, errors=[1.0, 2, 3]),
            "a_same": _tgs(solved=3, errors=[1.0, 2, 3]),
        })
        table = E.percentile_table(E.load_errors(run, analysis_root=tmp_path))
        assert E.curve_order(table) == ["a_same", "b_same"]


class TestPooling:
    def _three(self, tmp_path):
        return [
            _write_run(tmp_path, ds, {"m": _tgs(solved=len(errs), errors=errs, first_id=100 * i)})
            for i, (ds, errs) in enumerate(
                (("as01", [10.0, 20, 30, 40]), ("as02", [1000.0, 2000]), ("as03", [5.0]))
            )
        ]

    def test_the_pooled_curve_is_every_runs_rows_concatenated(self, tmp_path):
        pooled = E.pooled_errors(self._three(tmp_path), analysis_root=tmp_path)["m"]
        assert sorted(pooled["errors"]) == [5.0, 10, 20, 30, 40, 1000, 2000]
        assert pooled["n_tgs"] == 7 and pooled["n_solved"] == 7

    def test_pooled_percentiles_are_quantiles_not_a_mean_of_the_runs(self, tmp_path):
        table = E.percentile_table(E.pooled_errors(self._three(tmp_path), analysis_root=tmp_path))
        p50 = float(table.iloc[0][E.pcol(50)])
        assert p50 == round(float(pd.Series([5.0, 10, 20, 30, 40, 1000, 2000]).quantile(0.5)), 3)
        per_run = [pd.Series(e).quantile(0.5) for e in ([10.0, 20, 30, 40], [1000.0, 2000], [5.0])]
        assert p50 != pytest.approx(float(np.mean(per_run)))

    def test_a_method_missing_from_one_run_is_refused_with_this_figures_remedy(self, tmp_path):
        a = _write_run(tmp_path, "as01", {"m": _tgs(solved=2), "extra": _tgs(solved=2)})
        b = _write_run(tmp_path, "as02", {"m": _tgs(solved=2, first_id=50)})
        with pytest.raises(ValueError, match="not scored in every run") as err:
            E.pooled_errors([a, b], analysis_root=tmp_path)
        assert "--layout per-run" in str(err.value) and "compare" not in str(err.value)

    def test_overlapping_tg_ids_are_refused(self, tmp_path):
        a = _write_run(tmp_path, "as01", {"m": _tgs(solved=4)})
        b = _write_run(tmp_path, "as02", {"m": _tgs(solved=4)})
        with pytest.raises(ValueError, match="share 4 TG ids"):
            E.pooled_errors([a, b], analysis_root=tmp_path)


class TestBaselineEncoding:
    def test_the_baseline_is_not_drawn_in_the_unpublished_method_hue(self):
        from scripts.analysis.v5.modules.methods import OTHER_HUE

        baseline = E._curve_style(SHORTEST_PING, {})["color"]
        assert baseline != OTHER_HUE and baseline != E._MUTED

    def test_the_baseline_is_the_only_dashed_curve_and_on_top(self):
        colors = {"m": "#123456"}
        assert E._curve_style(SHORTEST_PING, colors)["linestyle"] == "--"
        assert E._curve_style("m", colors)["linestyle"] == "-"
        assert E._curve_style(SHORTEST_PING, {})["zorder"] > E._curve_style("m", colors)["zorder"]


class TestClamp:
    def test_the_floor_moves_the_drawn_curve_but_not_the_reported_percentile(self):
        values = np.array([0.001, 0.002, 5.0, 900.0])
        xs, _ = E._cdf(values, E.X_MIN_KM)
        assert xs.min() == E.X_MIN_KM
        table = E.percentile_table({"m": {"errors": values, **{k: 4 for k in E.COUNT_KEYS}}})
        assert table.iloc[0][E.pcol(5)] < E.X_MIN_KM

    def test_the_cdf_reaches_one(self):
        _, ys = E._cdf(np.array([1.0, 2.0, 3.0]))
        assert ys[-1] == 1.0


def test_no_artifact_name_carries_a_rung():
    assert not any("healpix" in n for triple in E.NAMES.values() for n in triple)


def test_an_unknown_layout_is_refused_by_name(tmp_path):
    run = _write_run(tmp_path, "as01", {"m": _tgs(solved=2)})
    with pytest.raises(ValueError, match="unknown layout"):
        E.build_for_runs([run], layouts=("compare",), analysis_root=tmp_path)


# --- real runs ---------------------------------------------------------------

MESH_RUNS = ("as01-260728-260802-mesh", "as02-260728-260802-mesh", "as03-260728-260802-mesh")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """as01-03 scored at the finest and coarsest rungs in a temp root."""
    from scripts.analysis.v5.modules import answer_space as A
    from scripts.analysis.v5.modules.paths import resolve_run

    root = tmp_path_factory.mktemp("v5cdf")
    rungs = (G.NSIDE_LADDER[0], G.NSIDE_LADDER[-1])
    runs = []
    for run_id in MESH_RUNS:
        try:
            run = resolve_run(run_id)
        except MissingArtifactError:
            pytest.skip(f"{run_id} not available")
        if not run.combo_ids:
            pytest.skip(f"{run_id} has no scored combo")
        A.build_for_run(run, nsides=rungs, analysis_root=root)
        C.score_for_run(run, nsides=rungs, analysis_root=root)
        runs.append(run)
    pngs = E.build_for_runs(runs, layouts=E.LAYOUTS, analysis_root=root)
    return runs, root, pngs


class TestRealRuns:
    def test_the_distance_is_identical_at_every_rung(self, built):
        """The premise the rung-free filenames rest on."""
        runs, root, _ = built
        for run in runs:
            fine = E.load_errors(run, G.NSIDE_LADDER[0], analysis_root=root)
            coarse = E.load_errors(run, G.NSIDE_LADDER[-1], analysis_root=root)
            assert set(fine) == set(coarse)
            for m in fine:
                assert np.array_equal(np.sort(fine[m]["errors"]), np.sort(coarse[m]["errors"]))

    def test_per_run_artifacts_sit_beside_the_rungs(self, built):
        runs, root, _ = built
        for run in runs:
            out = run.analysis_dir(CLASSIFY_KIND, root=root)
            for name in E.NAMES[E.PER_RUN]:
                assert (out / name).exists(), name
            assert (out / "healpix-128").is_dir()

    def test_percentiles_agree_with_every_runs_accuracy_table(self, built):
        runs, root, _ = built
        for run in runs:
            csv = pd.read_csv(
                run.analysis_dir(CLASSIFY_KIND, root=root) / E.NAMES[E.PER_RUN][1]
            ).set_index("method")
            acc = _acc(run, root)
            for method in csv.index:
                for p in (50, 90):
                    assert csv.loc[method, E.pcol(p)] == acc.loc[method, E.pcol(p)]
                assert csv.loc[method, "n_plotted"] == acc.loc[method, "n_solved"]

    def test_the_pooled_denominator_is_the_sum_of_the_runs(self, built):
        runs, root, _ = built
        out = cross.cross_dir([r.run_id for r in runs], analysis_root=root)
        pooled = pd.read_csv(out / E.NAMES[E.POOLED][1])
        per_run = [
            pd.read_csv(r.analysis_dir(CLASSIFY_KIND, root=root) / E.NAMES[E.PER_RUN][1])
            for r in runs
        ]
        for _, row in pooled.iterrows():
            expected = sum(
                int(t.loc[t["method"] == row["method"], "n_plotted"].iloc[0]) for t in per_run
            )
            assert int(row["n_plotted"]) == expected, row["method"]

    def test_nothing_is_clamped_and_the_manifest_carries_the_terms(self, built):
        runs, root, _ = built
        body = json.loads(
            (runs[0].analysis_dir(CLASSIFY_KIND, root=root) / E.NAMES[E.PER_RUN][2]).read_text()
        )
        assert body["x_axis"]["n_clamped_to_floor"] == {}
        assert body["source_rung"]["nside"] == E.SOURCE_NSIDE
        assert "S-P" in body["method_terms"]

    def test_one_png_per_run_plus_one_pooled(self, built):
        runs, _, pngs = built
        assert len(pngs) == len(runs) + 1
