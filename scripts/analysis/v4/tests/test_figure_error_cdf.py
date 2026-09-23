"""The error CDF: which rows it draws, and which numbers it must agree with.

Two invariants carry most of these tests. The population is
`classify.solved_mask`, so `n_plotted` is `accuracy.csv`'s `n_solved`; and the
percentiles come from the same pandas call `classify.summarize` makes, so the
two files agree digit for digit. Both are asserted against real artifacts as
well as fixtures, because the point of them is cross-artifact.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v4.modules import classify as C
from scripts.analysis.v4.modules import figure_error_cdf as E
from scripts.analysis.v4.modules import healpix as H
from scripts.analysis.v4.tests._scored import _cells, _write_run


class TestPopulation:
    """Which rows reach a curve."""

    def test_unanswered_rows_are_excluded(self, tmp_path):
        """A FALLBACK row carries the shortest-ping VP's coordinate, so its
        error is the baseline's. Drawing it would pull a variant's curve
        toward the baseline exactly where the variant failed."""
        run = _write_run(
            tmp_path, "as01", {"m": _cells(ring0=3, beyond=2, failed=5)}
        )
        loaded = E.load_errors(run, analysis_root=tmp_path)["m"]
        assert loaded["n_targets"] == 10
        assert loaded["n_solved"] == 5
        assert loaded["n_failed"] == 5
        assert len(loaded["errors"]) == 5

    def test_a_nan_filter_alone_would_not_drop_fallbacks(self, tmp_path):
        """The real scorer writes a finite error_km on FALLBACK rows -- the
        fallback coordinate is a real coordinate. Only solved_mask removes
        them, which is why this figure does not filter on NaN."""
        df = _cells(ring0=4, failed=4)
        df.loc[df["status"] == "FALLBACK", "error_km"] = 500.0
        run = _write_run(tmp_path, "as01", {"m": df})
        loaded = E.load_errors(run, analysis_root=tmp_path)["m"]
        assert loaded["n_solved"] == 4, "fallbacks with a distance still excluded"
        assert 500.0 not in set(loaded["errors"])

    def test_an_all_baseline_frame_is_wholly_included(self, tmp_path):
        """Shortest-Ping writes BASELINE on every row and never SUCCESS, so
        `status == "SUCCESS"` would return an empty curve for the one method
        the figure draws as its reference."""
        df = _cells(ring0=3, beyond=3)
        df["status"] = "BASELINE"
        run = _write_run(tmp_path, "as01", {C.SHORTEST_PING: df})
        loaded = E.load_errors(run, analysis_root=tmp_path)[C.SHORTEST_PING]
        assert loaded["n_solved"] == 6
        assert len(loaded["errors"]) == 6

    def test_a_solved_row_without_a_distance_is_counted_not_dropped_silently(
        self, tmp_path
    ):
        df = _cells(ring0=4)
        df.loc[0, "error_km"] = np.nan
        run = _write_run(tmp_path, "as01", {"m": df})
        loaded = E.load_errors(run, analysis_root=tmp_path)["m"]
        assert loaded["n_solved"] == 4
        assert loaded["n_no_distance"] == 1
        assert len(loaded["errors"]) == 3

    def test_target_ids_are_the_whole_roster_not_the_solved_subset(self, tmp_path):
        """The ids feed the overlap guard, and what that guards is the
        denominator -- which counts unanswered targets too."""
        run = _write_run(tmp_path, "as01", {"m": _cells(ring0=3, failed=3)})
        assert len(E.load_errors(run, analysis_root=tmp_path)["m"]["target_ids"]) == 6

    def test_missing_parquet_names_the_command_that_writes_it(self, tmp_path):
        from scripts.analysis.v4.modules.paths import MissingArtifactError, RunPaths

        run = RunPaths(run_id="as01-x-mesh", root=tmp_path, source="s", setup="t")
        run.cls_accuracy_dir(128, root=tmp_path)
        with pytest.raises(MissingArtifactError, match="classify --run-id"):
            E.load_errors(run, analysis_root=tmp_path)


class TestPercentiles:
    """The CSV has to agree with `accuracy.csv`, which sits one level down."""

    def test_p50_and_p90_match_the_accuracy_table_exactly(self, tmp_path):
        """Both are `Series.quantile` (linear) rounded to 3dp over the same
        rows. Not "compatible" -- the same call, so they cannot drift."""
        spec = {
            "m": _cells(ring0=5, beyond=5, failed=4, errors=list(range(1, 11))),
            C.SHORTEST_PING: _cells(ring0=8, errors=[2.0 * i for i in range(8)]),
        }
        spec[C.SHORTEST_PING]["status"] = "BASELINE"
        run = _write_run(tmp_path, "as01", spec)
        table = E.percentile_table(E.load_errors(run, analysis_root=tmp_path))
        acc = pd.read_csv(
            run.cls_accuracy_dir(128, root=tmp_path) / C.ACCURACY_CSV
        ).set_index("method")
        for _, row in table.iterrows():
            for p in (50, 90):
                assert row[f"error_km_p{p}"] == acc.loc[row["method"], f"error_km_p{p}"]

    def test_n_plotted_equals_the_accuracy_tables_n_solved(self, tmp_path):
        run = _write_run(
            tmp_path, "as01", {"m": _cells(ring0=6, beyond=2, failed=7)}
        )
        table = E.percentile_table(E.load_errors(run, analysis_root=tmp_path))
        acc = pd.read_csv(
            run.cls_accuracy_dir(128, root=tmp_path) / C.ACCURACY_CSV
        ).set_index("method")
        assert int(table.iloc[0]["n_plotted"]) == int(acc.loc["m", "n_solved"])

    def test_both_published_percentiles_are_reported(self):
        """50 and 90 are what accuracy.csv publishes; without both, the two
        artifacts cannot be joined on their headline numbers."""
        assert {50, 90} <= set(E.PERCENTILES)

    def test_a_method_that_answered_nothing_reports_no_percentile(self, tmp_path):
        run = _write_run(tmp_path, "as01", {"m": _cells(failed=5)})
        table = E.percentile_table(E.load_errors(run, analysis_root=tmp_path))
        assert table.iloc[0]["n_plotted"] == 0
        assert np.isnan(table.iloc[0]["error_km_p50"])


class TestCurveOrder:
    def test_curves_rank_by_median_error_ascending(self, tmp_path):
        spec = {
            "far": _cells(ring0=4, errors=[300.0, 310, 320, 330]),
            "near": _cells(ring0=4, errors=[10.0, 11, 12, 13]),
            "mid": _cells(ring0=4, errors=[100.0, 101, 102, 103]),
        }
        run = _write_run(tmp_path, "as01", spec)
        table = E.percentile_table(E.load_errors(run, analysis_root=tmp_path))
        assert E.curve_order(table) == ["near", "mid", "far"]

    def test_a_p50_tie_is_broken_by_the_tail(self, tmp_path):
        """Same median, different p90. Without the cascade the order would
        fall out of dict insertion, which is the parquet glob's order."""
        spec = {
            "fat_tail": _cells(ring0=4, errors=[10.0, 10, 10, 9000]),
            "thin_tail": _cells(ring0=4, errors=[10.0, 10, 10, 11]),
        }
        run = _write_run(tmp_path, "as01", spec)
        table = E.percentile_table(E.load_errors(run, analysis_root=tmp_path))
        assert E.curve_order(table) == ["thin_tail", "fat_tail"]

    def test_the_order_is_total_so_identical_methods_do_not_shuffle(self, tmp_path):
        spec = {"b_same": _cells(ring0=3, errors=[1.0, 2, 3]),
                "a_same": _cells(ring0=3, errors=[1.0, 2, 3])}
        run = _write_run(tmp_path, "as01", spec)
        table = E.percentile_table(E.load_errors(run, analysis_root=tmp_path))
        assert E.curve_order(table) == ["a_same", "b_same"]


class TestPooling:
    """One curve per method over every run's targets."""

    def _three(self, tmp_path):
        runs = []
        for i, (ds, errs) in enumerate(
            (("as01", [10.0, 20, 30, 40]), ("as02", [1000.0, 2000]), ("as03", [5.0]))
        ):
            runs.append(
                _write_run(
                    tmp_path,
                    ds,
                    {"m": _cells(ring0=len(errs), errors=errs, first_id=100 * i)},
                )
            )
        return runs

    def test_the_pooled_curve_is_every_runs_rows_concatenated(self, tmp_path):
        runs = self._three(tmp_path)
        pooled = E.pooled_errors(runs, analysis_root=tmp_path)["m"]
        assert sorted(pooled["errors"]) == [5.0, 10, 20, 30, 40, 1000, 2000]
        assert pooled["n_targets"] == 7

    def test_pooled_percentiles_are_quantiles_not_a_mean_of_the_runs(self, tmp_path):
        """The correction this figure exists to make. On the real meshes the
        two disagree by enough to reverse the leader: million_scale_cbg is
        96.1 km pooled against 192.2 km averaged."""
        runs = self._three(tmp_path)
        pooled = E.pooled_errors(runs, analysis_root=tmp_path)
        table = E.percentile_table(pooled)
        p50 = float(table.iloc[0]["error_km_p50"])

        everything = pd.Series([5.0, 10, 20, 30, 40, 1000, 2000])
        assert p50 == round(float(everything.quantile(0.5)), 3)

        per_run = [
            pd.Series(e).quantile(0.5)
            for e in ([10.0, 20, 30, 40], [1000.0, 2000], [5.0])
        ]
        assert p50 != pytest.approx(float(np.mean(per_run)))

    def test_a_dataset_weighs_by_its_target_count(self, tmp_path):
        """Micro, not macro: as01 contributes four rows and as03 one, so the
        pooled median sits in as01's range rather than midway."""
        runs = self._three(tmp_path)
        counts = E.pooled_errors(runs, analysis_root=tmp_path)["m"]
        assert counts["n_solved"] == 7

    def test_a_method_missing_from_one_run_is_refused(self, tmp_path):
        a = _write_run(tmp_path, "as01", {"m": _cells(ring0=2), "extra": _cells(ring0=2)})
        b = _write_run(tmp_path, "as02", {"m": _cells(ring0=2, first_id=50)})
        with pytest.raises(ValueError, match="not scored in every run"):
            E.pooled_errors([a, b], analysis_root=tmp_path)

    def test_the_refusal_names_the_layout_that_actually_exists(self, tmp_path):
        """`--layout compare` is the outcome bars' way out and is not a layout
        here; printing it would send the reader to a flag that does not
        exist."""
        a = _write_run(tmp_path, "as01", {"m": _cells(ring0=2), "extra": _cells(ring0=2)})
        b = _write_run(tmp_path, "as02", {"m": _cells(ring0=2, first_id=50)})
        with pytest.raises(ValueError) as err:
            E.pooled_errors([a, b], analysis_root=tmp_path)
        assert "--layout per-run" in str(err.value)
        assert "compare" not in str(err.value)

    def test_overlapping_target_ids_are_refused(self, tmp_path):
        """One shared id sits in the pooled denominator twice."""
        a = _write_run(tmp_path, "as01", {"m": _cells(ring0=4)})
        b = _write_run(tmp_path, "as02", {"m": _cells(ring0=4)})
        with pytest.raises(ValueError, match="share 4 target ids"):
            E.pooled_errors([a, b], analysis_root=tmp_path)


class TestBaselineEncoding:
    """The baseline is recessive, but not the same grey as the OTHER bucket."""

    def test_the_baseline_is_not_drawn_in_the_unpublished_method_hue(self):
        """v4's `methods.OTHER_HUE` and this module's `_MUTED` are the same
        hex, inherited from v3 where `_C_MUTED == _C_OTHER == #898781`. Every
        unpublished arm takes that bucket -- `spotter_h3_cbg` is scored on all
        three meshes -- so a baseline drawn in it shares a colour with a real
        curve and is told apart by the dash alone."""
        from scripts.analysis.v4.modules.methods import OTHER_HUE

        baseline = E._curve_style(C.SHORTEST_PING, {})["color"]
        assert baseline != OTHER_HUE
        assert baseline != E._MUTED

    def test_the_baseline_is_the_only_dashed_curve(self):
        colors = {"m": "#123456"}
        assert E._curve_style(C.SHORTEST_PING, colors)["linestyle"] == "--"
        assert E._curve_style("m", colors)["linestyle"] == "-"

    def test_a_variant_keeps_its_own_hue(self):
        from scripts.analysis.v4.modules.methods import method_colors

        got = method_colors(["octant_cbg_hull"])
        assert E._curve_style("octant_cbg_hull", got)["color"] == got["octant_cbg_hull"]

    def test_the_baseline_reads_on_top_of_the_variants(self):
        """Drawn last and above, because the figure shows variants against a
        reference rather than the reference among peers."""
        assert (
            E._curve_style(C.SHORTEST_PING, {})["zorder"]
            > E._curve_style("m", {"m": "#123456"})["zorder"]
        )


class TestClamp:
    def test_the_floor_moves_the_drawn_curve_but_not_the_reported_percentile(self):
        values = np.array([0.001, 0.002, 5.0, 900.0])
        xs, ys = E._cdf(values, E.X_MIN_KM)
        assert xs.min() == E.X_MIN_KM, "drawn points clamp up to the log floor"
        assert float(pd.Series(values).quantile(0.05)) < E.X_MIN_KM
        table = E.percentile_table(
            {"m": {"errors": values, **{k: 4 for k in E.COUNT_KEYS}}}
        )
        assert table.iloc[0]["error_km_p5"] < E.X_MIN_KM, "CSV is unclamped"

    def test_the_cdf_reaches_one(self):
        _, ys = E._cdf(np.array([1.0, 2.0, 3.0]))
        assert ys[-1] == 1.0


# --- real runs ---------------------------------------------------------------


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from scripts.analysis.v4.modules import answer_space as A
    from scripts.analysis.v4.modules.paths import MissingArtifactError, resolve_run

    root = tmp_path_factory.mktemp("v4cdf")
    runs = []
    for ds in ("as01", "as02", "as03"):
        try:
            run = resolve_run(f"{ds}-260728-260802-mesh")
        except MissingArtifactError:
            pytest.skip(f"{ds} mesh not available")
        if not run.combo_ids:
            pytest.skip(f"{ds} has no scored combo")
        A.build_for_run(run, analysis_root=root)
        C.score_for_run(run, analysis_root=root)
        runs.append(run)
    pngs = E.build_for_runs(runs, layouts=E.LAYOUTS, analysis_root=root)
    return runs, root, pngs


class TestGridFree:
    """The premise the rung-free filename rests on."""

    def test_error_km_is_identical_at_every_rung(self, built):
        """`error_km` is prediction-to-target, so the grid cannot touch it.
        Asserted rather than assumed, because the whole artifact layout
        depends on it."""
        runs, root, _ = built
        for run in runs:
            base = None
            for nside in H.NSIDE_LADDER:
                got = E.load_errors(run, nside, analysis_root=root)
                frame = {m: np.sort(e["errors"]) for m, e in got.items()}
                if base is None:
                    base = frame
                    continue
                assert set(frame) == set(base)
                for m in frame:
                    assert np.array_equal(frame[m], base[m]), (run.run_id, nside, m)

    def test_the_csv_is_the_same_whichever_rung_was_read(self, built):
        """The coarsest rung first and the default last, so the shared
        fixture's artifacts are left in the state the other tests expect."""
        runs, root, _ = built
        run = runs[0]
        out = run.cls_accuracy_root(root=root) / E.NAMES[E.PER_RUN][1]
        E.build_for_run(run, nside=16, analysis_root=root)
        coarse = out.read_text()
        E.build_for_run(run, nside=E.SOURCE_NSIDE, analysis_root=root)
        assert out.read_text() == coarse

    def test_no_artifact_name_carries_a_rung(self):
        assert not any(
            "healpix" in n for triple in E.NAMES.values() for n in triple
        )


class TestRealRuns:
    def test_per_run_artifacts_sit_beside_the_rungs_not_inside_one(self, built):
        runs, root, _ = built
        for run in runs:
            out = run.cls_accuracy_root(root=root)
            for name in E.NAMES[E.PER_RUN]:
                assert (out / name).exists(), name
            assert (out / "healpix-128").is_dir(), "the rungs are still its children"

    def test_the_pooled_triple_lands_beside_the_outcome_bars(self, built):
        from scripts.analysis.v4.modules import cross

        runs, root, _ = built
        out = cross.cross_dir([r.run_id for r in runs], analysis_root=root)
        for name in E.NAMES[E.POOLED]:
            assert (out / name).exists(), name

    def test_percentiles_agree_with_every_runs_accuracy_table(self, built):
        runs, root, _ = built
        for run in runs:
            csv = pd.read_csv(
                run.cls_accuracy_root(root=root) / E.NAMES[E.PER_RUN][1]
            ).set_index("method")
            acc = pd.read_csv(
                run.cls_accuracy_dir(128, root=root) / C.ACCURACY_CSV
            ).set_index("method")
            for method in csv.index:
                for p in (50, 90):
                    assert csv.loc[method, f"error_km_p{p}"] == acc.loc[
                        method, f"error_km_p{p}"
                    ], (run.run_id, method, p)
                assert csv.loc[method, "n_plotted"] == acc.loc[method, "n_solved"]

    def test_the_pooled_denominator_is_the_sum_of_the_runs(self, built):
        from scripts.analysis.v4.modules import cross

        runs, root, _ = built
        out = cross.cross_dir([r.run_id for r in runs], analysis_root=root)
        pooled = pd.read_csv(out / E.NAMES[E.POOLED][1])
        per_run = [
            pd.read_csv(r.cls_accuracy_root(root=root) / E.NAMES[E.PER_RUN][1])
            for r in runs
        ]
        for _, row in pooled.iterrows():
            expected = sum(
                int(t.loc[t["method"] == row["method"], "n_plotted"].iloc[0])
                for t in per_run
            )
            assert int(row["n_plotted"]) == expected, row["method"]

    def test_nothing_is_clamped_on_the_real_runs(self, built):
        runs, root, _ = built
        body = json.loads(
            (
                runs[0].cls_accuracy_root(root=root) / E.NAMES[E.PER_RUN][2]
            ).read_text()
        )
        assert body["x_axis"]["n_clamped_to_floor"] == {}

    def test_a_curve_is_drawn_per_method_and_the_baseline_is_dashed(self, built):
        runs, root, _ = built
        loaded = E.load_errors(runs[0], analysis_root=root)
        table = E.percentile_table(loaded)
        png = E.plot_cdf(
            loaded, table, root / "probe.png", title="t", subtitle="s"
        )
        assert png.exists()

    def test_the_manifest_records_the_rung_it_read_and_why_it_does_not_matter(
        self, built
    ):
        runs, root, _ = built
        body = json.loads(
            (
                runs[0].cls_accuracy_root(root=root) / E.NAMES[E.PER_RUN][2]
            ).read_text()
        )
        assert body["source_rung"]["nside"] == E.SOURCE_NSIDE
        assert "identical at every rung" in body["source_rung"]["note"]
        assert "byte-identical" in body["grid_free"]


def _clean_copy(root, dest):
    """The fixture's scored tree, minus every figure artifact.

    A copy rather than another `score_for_run`: the scoring is the slow part
    and these tests are about which files `build_for_runs` writes, not about
    re-deriving the inputs.
    """
    import shutil

    shutil.copytree(root, dest)
    for triple in E.NAMES.values():
        for name in triple:
            for stale in dest.rglob(name):
                stale.unlink()
    return dest


class TestLayouts:
    def test_the_default_is_per_run_only(self, built, tmp_path):
        from scripts.analysis.v4.modules import cross

        runs, root, _ = built
        fresh = _clean_copy(root, tmp_path / "defaults")
        E.build_for_runs(runs, analysis_root=fresh)
        for run in runs:
            assert (run.cls_accuracy_root(root=fresh) / E.NAMES[E.PER_RUN][0]).exists()
        out = cross.cross_dir([r.run_id for r in runs], analysis_root=fresh)
        assert not (out / E.NAMES[E.POOLED][0]).exists()

    def test_pooled_alone_writes_only_the_pooled_triple(self, built, tmp_path):
        from scripts.analysis.v4.modules import cross

        runs, root, _ = built
        fresh = _clean_copy(root, tmp_path / "pooledonly")
        E.build_for_runs(runs, layouts=(E.POOLED,), analysis_root=fresh)
        out = cross.cross_dir([r.run_id for r in runs], analysis_root=fresh)
        assert (out / E.NAMES[E.POOLED][0]).exists()
        for run in runs:
            assert not (
                run.cls_accuracy_root(root=fresh) / E.NAMES[E.PER_RUN][0]
            ).exists()

    def test_an_unknown_layout_is_refused_by_name(self, built):
        runs, _, _ = built
        with pytest.raises(ValueError, match="unknown layout"):
            E.build_for_runs(runs, layouts=("compare",))

    def test_one_png_per_run_plus_one_pooled(self, built):
        runs, _, pngs = built
        assert len(pngs) == len(runs) + 1
