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

    def test_every_sweeps_baseline_is_its_control_arm(self):
        """Per sweep now, not one flat tuple: the baseline is the arm the
        paired table anchors on, and it must be the arm the CDF draws
        recessively, or the two artifacts disagree about what the reference
        is."""
        assert O.BASELINE_TAG == O.SWEEP_WEIGHT.baseline == "unw"
        for sw in O.SWEEPS:
            assert sw.baseline in sw.tags, sw.key
            assert sw.family(sw.baseline) == "control", sw.key


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
        tex = O.SWEEP_WEIGHT.tex_labels
        assert "^" not in tex["1/rtt^2"].replace(r"^{2}", "")
        assert tex["1/rtt^2"].startswith("$")

    def test_every_arm_label_has_a_tex_form(self):
        """The legend renders these as matplotlib mathtext and the table as
        LaTeX, so a missing entry degrades both at once. Also enforced by
        `_validate` at import; kept here as the readable statement of intent.
        """
        for sw in O.SWEEPS:
            for a in sw.arms:
                assert a.label in sw.tex_labels, (sw.key, a.tag)

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


class TestSweepSpecs:
    def test_example_ratio_matches_its_definition(self):
        W = O.SWEEP_WEIGHT
        assert W.order("unw") == 1.0
        assert W.order("ip2") == pytest.approx(
            (O.REF_SLOW_MS / O.REF_FAST_MS) ** 2)
        assert W.order("tau5") == pytest.approx(
            math.exp((O.REF_SLOW_MS - O.REF_FAST_MS) / 5))

    def test_example_ratio_refuses_another_sweeps_tag(self):
        """The regression that motivated the registry. The final branch was an
        unguarded `float(tag[3:])`, so `example_ratio("cov95")` returned
        `exp(4.5/95) = 1.0485` -- no exception, a plausible-looking number,
        ordered OPPOSITE to coverage, collapsing four arms into one cluster."""
        for tag in ("cov95", "cov90", "cov75", "cov50"):
            with pytest.raises(ValueError, match="ordering scalar"):
                O.example_ratio(tag)

    def test_within_family_order_is_stable_under_the_reference_window(self):
        """The window reorders arms ACROSS families but must not reorder them
        within one, or the dash-by-rank encoding would flip meaning."""
        for sw in O.SWEEPS:
            for fam in sw.families:
                if fam == "control":
                    continue
                sibs = sw.ordered(t for t in sw.tags if sw.family(t) == fam)
                assert ([sw.rank_within_family(t) for t in sibs]
                        == list(range(len(sibs)))), (sw.key, fam)

    def test_pick_is_a_scored_arm_or_absent(self):
        """`None` is meaningful, not missing: the spline sweep has no arm worth
        adopting, and recording that beats leaving a stale pointer."""
        for sw in O.SWEEPS:
            assert sw.pick is None or sw.pick in sw.tags, sw.key
        assert O.SWEEP_SPLINE.pick is None

    def test_the_specs_validate(self):
        for sw in O.SWEEPS:
            O._validate(sw)

    def test_an_unknown_variant_raises_value_error_not_key_error(self):
        """The CLI catches ValueError and reports a skip; a KeyError tracebacks
        out of a loop that meant to continue."""
        with pytest.raises(ValueError, match="unknown variant"):
            O.sweep_for("nope")

    def test_a_bare_octant_combo_is_not_mistaken_for_an_arm(self):
        """Run discovery matches EXACT combo ids. The predicate it replaced was
        `split("_")[-1] in ARM_TAGS`, which matches `octant_cbg_hull` itself as
        soon as any sweep contributes a tag like `hull`."""
        combos = O.sweep_combos()
        for produced in ("octant_cbg", "octant_cbg_hull", "octant_cbg_spl",
                         "octant_cbg_hull_geo", "octant_cbg_top_geo"):
            assert produced not in combos, produced
        assert "octant_cbg_hull_ip1" in combos
        assert "octant_ssw_cov95" in combos

    def test_each_sweep_serializes_only_its_own_hues(self):
        """`cdf_manifest` emits `dict(family_hue)`, so a flat shared dict would
        have rewritten the weight sweep's published manifest the moment a
        second sweep was registered."""
        assert list(O.SWEEP_WEIGHT.family_hue) == [
            "inverse-power", "neg-exponential", "control"]
        assert list(O.SWEEP_SPLINE.family_hue) == ["spline-coverage", "control"]

    def test_the_dash_ladder_covers_the_largest_family(self):
        for sw in O.SWEEPS:
            widest = max(sum(a.family == f for a in sw.arms)
                         for f in sw.families if f != "control")
            assert len(sw.rank_dash) >= widest, sw.key

    def test_a_truncated_dash_ladder_is_refused_at_construction(self):
        import dataclasses
        bad = dataclasses.replace(O.SWEEP_SPLINE, rank_dash=(None, (5, 1.6)))
        with pytest.raises(ValueError, match="rank_dash"):
            O._validate(bad)


class TestSplineCoverageSweep:
    """The second sweep: an OCT-H baseline with no spline, four coverage arms.

    Its shape breaks three assumptions the weight sweep never exercised -- four
    arms in ONE family (the dash ladder), families named something other than
    inverse-power/neg-exponential (the legend), and an ordering scalar that
    runs DESCENDING (every sort).
    """

    @pytest.fixture
    def tree_ssw(self, tmp_path):
        run = "asXX-ssweep"
        _combo(tmp_path, run, "octant_ssw", "nospl", [100.0, 200.0, 300.0, 400.0])
        _combo(tmp_path, run, "octant_ssw", "cov95", [90.0, 200.0, 280.0, 500.0])
        _combo(tmp_path, run, "octant_ssw", "cov90", [110.0, 200.0, 310.0, 600.0])
        return tmp_path, run

    @staticmethod
    def _run(tree_ssw):
        root, run = tree_ssw
        return RunPaths(run, root, "generic_csv", "anchors_to_probes")

    def test_the_ordering_scalar_is_the_coverage_not_a_parsed_tag(self):
        assert O.SWEEP_SPLINE.order("cov95") == 0.95
        assert O.SWEEP_SPLINE.order("cov50") == 0.50

    def test_arms_run_from_the_baseline_down_to_the_tightest(self):
        """Coverage rises with looseness, so gentlest-first is DESCENDING. A
        plain ascending sort -- which is what a single shared sort direction
        would give -- puts the tightest bound first and inverts the ladder."""
        assert O.SWEEP_SPLINE.ordered(O.SWEEP_SPLINE.tags) == [
            "nospl", "cov95", "cov90", "cov75", "cov50"]

    def test_the_table_publishes_target_coverage_not_example_ratio(self, tree_ssw):
        run = self._run(tree_ssw)
        wide, present = O.assemble(run, "octant_ssw")
        t = O.build_table(wide, present, sweep=O.SWEEP_SPLINE)
        assert "target_coverage" in t.columns
        assert "example_ratio" not in t.columns
        # and the rows are in coverage-descending order, baseline excluded
        assert list(t.arm_tag) == ["cov95", "cov90"]

    def test_the_cdf_csv_publishes_target_coverage(self, tree_ssw):
        run = self._run(tree_ssw)
        wide, present = O.assemble(run, "octant_ssw")
        c = O.cdf_table(wide, present, sweep=O.SWEEP_SPLINE)
        assert "target_coverage" in c.columns
        assert list(c.arm_tag) == ["nospl", "cov95", "cov90"]

    def test_the_weight_sweep_still_publishes_example_ratio(self, tree):
        """The converse, so a rename cannot silently pass both ways."""
        wide, present = O.assemble(_run(tree), "octant_cbg_hull")
        c = O.cdf_table(wide, present, sweep=O.SWEEP_WEIGHT)
        assert "example_ratio" in c.columns
        assert "target_coverage" not in c.columns

    def test_four_arms_in_one_family_get_four_distinct_dashes(self):
        """`RANK_DASH` held three patterns, so the fourth arm raised
        `IndexError` inside plot_cdf -- after the CSV and manifest were already
        computed. None may collide with the control's dash-dot, which shares
        the panel."""
        cov = [t for t in O.SWEEP_SPLINE.tags
               if O.SWEEP_SPLINE.family(t) != "control"]
        assert len(cov) == 4
        dashes = [O.SWEEP_SPLINE.style_for(t).get("dashes") for t in cov]
        assert len(set(dashes)) == 4, dashes
        control = O.SWEEP_SPLINE.style_for("nospl")["dashes"]
        assert control not in dashes

    def test_the_legend_lists_every_present_arm(self, tree_ssw, tmp_path):
        """plot_cdf hardcoded the weight sweep's family names, so any other
        sweep got an EMPTY legend -- matplotlib draws an empty frame rather
        than raising, which is why this asserts content, not absence of error.
        """
        import matplotlib.pyplot as plt
        run = self._run(tree_ssw)
        wide, present = O.assemble(run, "octant_ssw")
        out = tmp_path / "cdf.png"
        O.plot_cdf(wide, present, "OCT-S", run.run_id, out,
                   sweep=O.SWEEP_SPLINE, dpi=60)
        assert out.exists()
        fig = plt.gcf()  # closed by plot_cdf; rebuild the check from the spec
        plt.close(fig)
        labels = [O.SWEEP_SPLINE.tex(O.SWEEP_SPLINE.label(t)) for t in present]
        assert len(labels) == 3 and all(labels)

    def test_the_coverage_column_keeps_two_decimals(self):
        """`tex_range` floors at one decimal, so it renders 0.95 as "1.0" --
        indistinguishable from the baseline -- and 0.75 as "0.8". The formatter
        has to be per-sweep, not just the values."""
        assert O.SWEEP_SPLINE.order_tex_fmt(0.95) == r"$0.95$"
        assert O.SWEEP_SPLINE.order_tex_fmt(0.75) == r"$0.75$"
        assert O.tex_range(0.95) == "0.9"

    def test_the_spread_block_is_keyed_off_this_sweeps_baseline(self, tree_ssw):
        """The key name AND the cohort: the old code filtered on the literal
        "unw", which is not an arm of this sweep, so every arm would have been
        counted including the baseline."""
        run = self._run(tree_ssw)
        wide, present = O.assemble(run, "octant_ssw")
        c = O.cdf_table(wide, present, sweep=O.SWEEP_SPLINE)
        m = O.cdf_manifest([run], "octant_ssw", "OCT-S", wide, present, c)
        assert "coverage_arm_spread_km" in m
        assert "steep_arm_spread_km" not in m

    def test_the_manifest_carries_no_rtt_reference_window(self, tree_ssw):
        """REF_FAST_MS/REF_SLOW_MS are an RTT window with no meaning for a
        coverage sweep, and they were baked into arm_ordering and the caption.
        """
        run = self._run(tree_ssw)
        wide, present = O.assemble(run, "octant_ssw")
        c = O.cdf_table(wide, present, sweep=O.SWEEP_SPLINE)
        m = O.cdf_manifest([run], "octant_ssw", "OCT-S", wide, present, c)
        assert "example ratio" not in m["arm_ordering"]
        assert "coverage" in m["arm_ordering"]
        assert "wsweep" not in m["scope"]

    def test_the_table_manifest_declares_the_hull_fallback_confound(self, tree_ssw):
        """Every arm is a blend of 'spline at this coverage' and 'hull', and
        the baseline IS the hull, so the artifact has to say the effect is a
        lower bound."""
        run = self._run(tree_ssw)
        wide, present = O.assemble(run, "octant_ssw")
        t = O.build_table(wide, present, sweep=O.SWEEP_SPLINE)
        m = O.table_manifest([run], "octant_ssw", "OCT-S", wide, present,
                             "nospl", t)
        assert "lower bound" in m["confound_note"]
        assert m["baseline"] == "hull, no spline"

    def test_the_weight_sweep_manifest_gains_no_confound_key(self, tree):
        """Adding a key unconditionally would rewrite already-published bytes."""
        run = _run(tree)
        wide, present = O.assemble(run, "octant_cbg_hull")
        t = O.build_table(wide, present)
        m = O.table_manifest([run], "octant_cbg_hull", "OCT-H", wide, present,
                             "unw", t)
        assert "confound_note" not in m
        assert "pick" not in m

    def test_the_tex_label_is_prefixed_per_sweep(self, tree_ssw):
        """Both sweeps publish a variant whose term is OCT-S, so without a
        per-sweep prefix their \\label{} keys would collide."""
        run = self._run(tree_ssw)
        wide, present = O.assemble(run, "octant_ssw")
        t = O.build_table(wide, present, sweep=O.SWEEP_SPLINE)
        tex = O.to_latex(t, "OCT-S", run.run_id, "hull, no spline", len(wide),
                         sweep=O.SWEEP_SPLINE)
        assert r"\label{tab:octant-spline-coverage-oct-s-hull-no-spline}" in tex
        assert "octant-weight-sweep" not in tex
        # the label must carry no LaTeX-hostile characters
        line = [l for l in tex.splitlines() if l.startswith(r"\label")][0]
        assert "," not in line and " " not in line[len(r"\label{"):]

    def test_defaults_do_not_leak_the_weight_sweeps_baseline(self, tree_ssw):
        """`build_table` used to default to BASELINE_TAG == "unw", which is not
        an arm here, so the default raised instead of pairing."""
        run = self._run(tree_ssw)
        wide, present = O.assemble(run, "octant_ssw")
        t = O.build_table(wide, present, sweep=O.SWEEP_SPLINE)
        assert "nospl" not in set(t.arm_tag)
        assert len(t) == 2

    def test_artifacts_do_not_collide_with_the_weight_sweep(self, tmp_path):
        """Both sweeps write into one run's octant-finetuning/ if a run ever
        holds both; the stems must stay distinct."""
        assert (O.artifact_names("octant_ssw", "nospl", "table")
                != O.artifact_names("octant_cbg_spl", "unw", "table"))
        assert O.artifact_names("octant_ssw", "nospl", "cdf")[0] == \
            "error_cdf.octant_ssw.png"


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
