"""The weight-scorer sweep: what it joins, what it refuses, and what it reports.

Three invariants carry most of these. Every column comes from ONE `-wsweep`
run -- no combo is pulled in from another run id, so the artifacts cannot
quietly mix configs. The cross-arm join is an INNER join, so a half-finished
sweep narrows the population and the manifest has to say so. And regression
statistics are median/max by construction, never a percentile, because those
cohorts are too small for one to mean anything.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from scripts.analysis.v5.modules import octant_finetuning as O
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths


def _combo(tmp_path, run_id, variant, tag, errors, *, folds=("fold_0",),
           lat_step=0.0, finished=True, first_id=0):
    """Write a combo's per-fold targets.parquet (+ run.json when finished).

    `first_id` offsets the target ids so two runs can be given disjoint
    rosters -- pooling refuses overlapping ones, and the guard needs both
    cases exercised.
    """
    combo = f"{variant}_{tag}" if tag else variant
    per = len(errors) // len(folds)
    for i, fold in enumerate(folds):
        d = tmp_path / run_id / "generic_csv" / "anchors_to_probes" / fold / combo
        d.mkdir(parents=True, exist_ok=True)
        chunk = errors[i * per:(i + 1) * per]
        pd.DataFrame({
            "target_id": [f"tg-{first_id + i * per + j}" for j in range(len(chunk))],
            "target_lat": [40.0 + lat_step * j for j in range(len(chunk))],
            "target_lon": [-70.0] * len(chunk),
            "error_km": chunk,
        }).to_parquet(d / "targets.parquet")
        if finished:
            (d / "run.json").write_text("{}")


@pytest.fixture
def tree(tmp_path):
    """One sweep run: four targets, the control arm and one scorer."""
    sweep = "asXX-wsweep"
    _combo(tmp_path, sweep, "octant_cbg_hull", "unw", [100.0, 200.0, 300.0, 400.0])
    _combo(tmp_path, sweep, "octant_cbg_hull", "ip1", [10.0, 200.0, 20.0, 500.0])
    return tmp_path, sweep


def _run(tree):
    root, sweep = tree
    return RunPaths(sweep, root, "generic_csv", "anchors_to_probes")


class TestScope:
    def test_every_column_comes_from_the_sweep_run(self, tree):
        """No combo is pulled in from another run id: the shipped tau=50 lives
        in the parent mesh run and must not appear here."""
        wide, present = O.assemble(_run(tree), "octant_cbg_hull")
        assert present == ["unw", "ip1"]
        assert "tau50" not in wide.columns

    def test_no_arms_scored_raises(self, tmp_path):
        run = RunPaths("empty-wsweep", tmp_path, "generic_csv", "anchors_to_probes")
        with pytest.raises(MissingArtifactError, match="no sweep arms"):
            O.assemble(run, "octant_cbg_hull")

    def test_baseline_default_is_the_control_arm(self):
        assert O.BASELINE_TAG == "unw"
        assert set(O.BASELINES) == set(O.ARM_TAGS)


class TestPartialRuns:
    def test_unfinished_fold_is_not_read(self, tmp_path):
        """targets.parquet is streamed a row group per target, so a mid-flight
        fold is on disk and short. Only run.json marks a fold finished."""
        _combo(tmp_path, "r", "octant_cbg_hull", "ip1", [1.0, 2.0], finished=False)
        run = RunPaths("r", tmp_path, "generic_csv", "anchors_to_probes")
        assert O.load_combo(run, "octant_cbg_hull_ip1") is None

    def test_manifest_declares_unequal_coverage(self, tmp_path):
        _combo(tmp_path, "p-wsweep", "octant_cbg_hull", "unw", [11.0, 21.0],
               folds=("fold_0", "fold_1"))
        _combo(tmp_path, "p-wsweep", "octant_cbg_hull", "ip1", [1.0])  # fold_0 only
        run = RunPaths("p-wsweep", tmp_path, "generic_csv", "anchors_to_probes")
        wide, present = O.assemble(run, "octant_cbg_hull")
        m = O._common_manifest([run], "octant_cbg_hull", "OCT-H", wide, present)
        assert m["complete"] is False
        assert "INNER JOIN" in m["partial_note"]
        assert m["fold_coverage"] == {"unw": 2, "ip1": 1}

    def test_inner_join_narrows_to_the_slowest_arm(self, tmp_path):
        _combo(tmp_path, "q-wsweep", "octant_cbg_hull", "unw", [11.0, 21.0],
               folds=("fold_0", "fold_1"))
        _combo(tmp_path, "q-wsweep", "octant_cbg_hull", "ip1", [1.0])
        run = RunPaths("q-wsweep", tmp_path, "generic_csv", "anchors_to_probes")
        wide, _ = O.assemble(run, "octant_cbg_hull")
        assert len(wide) == 1


class TestPairedTable:
    def test_better_worse_tie_partition_the_targets(self, tree):
        run = _run(tree)
        wide, present = O.assemble(run, "octant_cbg_hull")
        t = O.build_table(wide, present).set_index("arm_tag")
        # ip1 vs tau50: 10<100 better, 200==200 tie, 20<300 better, 500>400 worse
        r = t.loc["ip1"]
        assert (r.better_pct, r.worse_pct, r.tie_pct) == (50.0, 25.0, 25.0)
        assert r.better_pct + r.worse_pct + r.tie_pct == pytest.approx(100.0)

    def test_ties_are_their_own_column_not_folded_into_not_better(self, tree):
        run = _run(tree)
        wide, present = O.assemble(run, "octant_cbg_hull")
        t = O.build_table(wide, present).set_index("arm_tag")
        assert t.loc["ip1"].tie_pct == 25.0

    def test_gain_and_regression_are_conditional(self, tree):
        run = _run(tree)
        wide, present = O.assemble(run, "octant_cbg_hull")
        r = O.build_table(wide, present).set_index("arm_tag").loc["ip1"]
        # ip1 [10, 200, 20, 500] vs tau50 [100, 200, 300, 400]:
        # gains (100-10, 300-20) = (90, 280); one regression, 500-400.
        assert r.n_gain == 2 and r.n_reg == 1
        assert r.gain_p50 == pytest.approx(185.0)
        assert r.reg_p50 == pytest.approx(100.0)
        assert r.reg_max == pytest.approx(100.0)

    def test_baseline_is_excluded_from_its_own_table(self, tree):
        run = _run(tree)
        wide, present = O.assemble(run, "octant_cbg_hull")
        t = O.build_table(wide, present, baseline="unw")
        assert "unw" not in set(t.arm_tag)
        assert set(t.arm_tag) == {"ip1"}

    def test_unscored_baseline_raises(self, tree):
        run = _run(tree)
        wide, present = O.assemble(run, "octant_cbg_hull")
        with pytest.raises(ValueError, match="not among scored arms"):
            O.build_table(wide, present, baseline="tau1")

    def test_rows_are_ordered_by_example_ratio(self, tree):
        run = _run(tree)
        wide, present = O.assemble(run, "octant_cbg_hull")
        t = O.build_table(wide, present)
        assert list(t.example_ratio) == sorted(t.example_ratio)

    def test_gain_and_regression_carry_the_same_statistics(self, tree):
        """Symmetric by request: both sides read off one scale, with n beside
        each because the cohorts differ greatly in size."""
        run = _run(tree)
        wide, present = O.assemble(run, "octant_cbg_hull")
        cols = set(O.build_table(wide, present).columns)
        for stat in ("p50", "p95", "max"):
            assert f"gain_{stat}" in cols and f"reg_{stat}" in cols, stat
        assert {"n_gain", "n_reg"} <= cols


class TestLatex:
    def test_caret_labels_are_math_mode(self):
        """A bare `^` outside math mode is 'Missing $ inserted' at compile."""
        assert "^" not in O.TEX_LABEL["1/rtt^2"].replace(r"^{2}", "")
        assert O.TEX_LABEL["1/rtt^2"].startswith("$")

    def test_every_arm_label_has_a_tex_form(self):
        """The legend renders these as matplotlib mathtext and the table as
        LaTeX, so a missing entry degrades both at once."""
        for _, label, _, _ in O.ARMS:
            assert label in O.TEX_LABEL, label

    def test_large_ranges_are_scientific_not_e_notation(self):
        out = O.tex_range(3.83e22)
        assert "e+" not in out and r"\times10^{22}" in out

    def test_small_ranges_stay_plain(self):
        assert O.tex_range(1.0) == "1.0"
        assert O.tex_range(336.1) == "336"


class TestArtifacts:
    def test_baseline_is_in_the_table_stem_but_not_the_cdf_stem(self):
        """Two baselines are two different results and must not overwrite each
        other; the CDF does not depend on a baseline at all."""
        a = O.artifact_names("octant_cbg_hull", "unw", "table")
        b = O.artifact_names("octant_cbg_hull", "ip1", "table")
        assert a != b and all("vs-unw" in n for n in a)
        cdf = O.artifact_names("octant_cbg_hull", "unw", "cdf")
        assert all("vs-" not in n for n in cdf)

    def test_cdf_writes_the_png_csv_manifest_triple(self, tree, tmp_path):
        run = _run(tree)
        out = tmp_path / "analysis"
        O.write_cdf(run, "octant_cbg_hull", analysis_root=out, dpi=60)
        d = run.octant_finetuning_dir(root=out)
        for name in O.artifact_names("octant_cbg_hull", "unw", "cdf"):
            assert (d / name).exists(), name
        man = json.loads((d / "error_cdf.octant_cbg_hull.manifest.json").read_text())
        assert man["n_targets"] == 4
        assert man["arms_scored"] == ["unw", "ip1"]
        assert "unpaired_warning" in man
        # The composite encoding is the figure's whole legibility story; if it
        # is ever swapped for cycled hues the manifest must stop claiming it.
        assert "dash" in man["encoding"]["rule"]

    def test_table_writes_the_csv_tex_manifest_triple(self, tree, tmp_path):
        run = _run(tree)
        out = tmp_path / "analysis"
        O.write_table(run, "octant_cbg_hull", analysis_root=out)
        d = run.octant_finetuning_dir(root=out)
        for name in O.artifact_names("octant_cbg_hull", "unw", "table"):
            assert (d / name).exists(), name
        tex = (d / "paired.octant_cbg_hull.vs-unw.tex").read_text()
        assert tex.count(r"\\") >= 1 and r"\bottomrule" in tex

    def test_two_baselines_do_not_overwrite(self, tree, tmp_path):
        """Regression guard: Path.with_suffix() reads `.vs-unw` as the suffix
        and strips it, which made both baselines write one filename."""
        run = _run(tree)
        out = tmp_path / "analysis"
        for b in ("unw", "ip1"):
            O.write_table(run, "octant_cbg_hull", baseline=b, analysis_root=out)
        d = run.octant_finetuning_dir(root=out)
        assert len(list(d.glob("paired.octant_cbg_hull.vs-*.csv"))) == 2


class TestArmMetadata:
    def test_example_ratio_matches_its_definition(self):
        assert O.ARM_RATIO["unw"] == 1.0
        assert O.ARM_RATIO["ip2"] == pytest.approx(
            (O.REF_SLOW_MS / O.REF_FAST_MS) ** 2)
        assert O.ARM_RATIO["tau5"] == pytest.approx(
            math.exp((O.REF_SLOW_MS - O.REF_FAST_MS) / 5))

    def test_within_family_order_is_stable_under_the_reference_window(self):
        """The window reorders arms ACROSS families but must not reorder them
        within one, or the dash-by-steepness encoding would flip meaning."""
        for fam in ("inverse-power", "neg-exponential"):
            sibs = sorted((t for t in O.ARM_TAGS if O.ARM_FAMILY[t] == fam),
                          key=O.ARM_RATIO.get)
            assert [O.rank_within_family(t) for t in sibs] == list(range(len(sibs)))

    def test_pick_is_a_scored_arm(self):
        assert O.PICK in O.ARM_TAGS

    def test_every_arm_can_serve_as_a_baseline(self):
        assert O.BASELINE_TAG == "unw"
        assert set(O.BASELINES) == set(O.ARM_TAGS)


class TestPooling:
    """Micro-pooling across runs, and the two guards that make it legal."""

    def _two_runs(self, tmp_path, *, share_ids=False, drop_arm=False):
        for i, run in enumerate(("r1-wsweep", "r2-wsweep")):
            first = 0 if share_ids else i * 100
            _combo(tmp_path, run, "octant_cbg_hull", "unw",
                   [100.0, 200.0], first_id=first)
            if not (drop_arm and i == 1):
                _combo(tmp_path, run, "octant_cbg_hull", "ip1",
                       [10.0, 300.0], first_id=first)
        return [RunPaths(r, tmp_path, "generic_csv", "anchors_to_probes")
                for r in ("r1-wsweep", "r2-wsweep")]

    def test_rows_are_concatenated_not_averaged(self, tmp_path):
        """Micro-pool: a dataset weighs by its target count, so the pooled
        frame is the sum of the inputs' rows."""
        runs = self._two_runs(tmp_path)
        wide, present = O.assemble_pooled(runs, "octant_cbg_hull")
        assert len(wide) == 4
        assert present == ["unw", "ip1"]
        # No config declares a label for these, so each falls back to its run
        # id -- verbose, but never two runs sharing one name.
        assert set(wide.dataset) == {"r1-wsweep", "r2-wsweep"}

    def test_shared_target_ids_are_refused(self, tmp_path):
        """A shared id lands in the pooled denominator twice."""
        runs = self._two_runs(tmp_path, share_ids=True)
        with pytest.raises(ValueError, match="share"):
            O.assemble_pooled(runs, "octant_cbg_hull")

    def test_an_arm_missing_from_one_run_is_refused(self, tmp_path):
        """Pooling it would rest that column on a different denominator."""
        runs = self._two_runs(tmp_path, drop_arm=True)
        with pytest.raises(ValueError, match="not scored in every run"):
            O.assemble_pooled(runs, "octant_cbg_hull")

    def test_sites_are_namespaced_on_the_run_id(self, tmp_path):
        """A rounded coordinate is only unique within one run; unnamespaced,
        two runs' sites would merge and the clustered bootstrap would resample
        a site that does not exist.

        On the **run id**, via `sites.site_key`, never on the display label:
        the label is declared in a config, so two runs can share one, and the
        prefix would stop separating anything -- which is exactly what happened
        when it was parsed off the run id and every `pro-*` run read `pro`.
        """
        runs = self._two_runs(tmp_path)
        wide, _ = O.assemble_pooled(runs, "octant_cbg_hull")
        assert wide.site.nunique() == 2
        assert {s.split("|")[0] for s in wide.site} == {"r1-wsweep", "r2-wsweep"}

    def test_runs_sharing_a_declared_label_are_refused(self, tmp_path, monkeypatch):
        """Two runs may legitimately declare the same `dataset_label`; every
        per-dataset number is grouped on it, so they must not pool."""
        runs = self._two_runs(tmp_path)
        monkeypatch.setattr(O.cross, "short_dataset", lambda r, **kw: "as01")
        with pytest.raises(ValueError, match="declared by"):
            O.assemble_pooled(runs, "octant_cbg_hull")

    def test_pooled_manifest_carries_the_per_dataset_spread(self, tmp_path):
        """One pooled number hides a real spread, so the per-dataset p50s ride
        along in the manifest rather than only in the per-run artifacts."""
        runs = self._two_runs(tmp_path)
        wide, present = O.assemble_pooled(runs, "octant_cbg_hull")
        m = O._common_manifest(runs, "octant_cbg_hull", "OCT-H", wide, present)
        assert m["layout"] == O.POOLED
        assert m["pooling"]["n_per_dataset"] == {"r1-wsweep": 2, "r2-wsweep": 2}
        assert set(m["pooling"]["p50_km_per_dataset"]) == {"r1-wsweep", "r2-wsweep"}
        assert "lossy_note" in m["pooling"]

    def test_per_run_manifest_has_no_pooling_block(self, tree):
        run = _run(tree)
        wide, present = O.assemble(run, "octant_cbg_hull")
        m = O._common_manifest([run], "octant_cbg_hull", "OCT-H", wide, present)
        assert m["layout"] == O.PER_RUN
        assert "pooling" not in m

    def test_pooled_artifacts_land_under_cross(self, tmp_path):
        runs = self._two_runs(tmp_path)
        out = tmp_path / "analysis"
        O.write_cdf(runs, "octant_cbg_hull", analysis_root=out, dpi=60)
        O.write_table(runs, "octant_cbg_hull", analysis_root=out)
        d = out / "_cross" / "octant-finetuning" / O.cross.cross_name(
            [r.run_id for r in runs])
        assert (d / "error_cdf.octant_cbg_hull.png").exists()
        assert (d / "paired.octant_cbg_hull.vs-unw.csv").exists()
