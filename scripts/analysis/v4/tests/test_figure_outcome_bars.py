"""Tests for the outcome-composition bars.

`TestPalette` is the one worth keeping honest: the ramp's hexes were chosen by
running the dataviz validator, not by eye, and the properties it checked are
asserted here so a later "nicer green" cannot silently break them.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v4.modules import classify as C
from scripts.analysis.v4.modules import figure_outcome_bars as F
from scripts.analysis.v4.modules import healpix as H


def _row(method, n=100, ring0=40, ring1=20, ring2=10, beyond=20, failed=10, ds="as01"):
    return {
        "run_id": f"{ds}-260728-260802-mesh",
        "dataset": ds,
        "method": method,
        "nside": 128,
        "n_targets": n,
        "n_ring0": ring0,
        "n_ring1": ring1,
        "n_ring2": ring2,
        "n_beyond": beyond,
        "n_failed": failed,
        "accuracy_ring0": ring0 / n,
        "accuracy_nearest_seed_retired": 0.9,
        "error_km_p50": 50.0,
        "error_km_p90": 500.0,
    }


def _table(rows=None):
    rows = rows or [
        _row("octant_cbg_hull", ring0=40),
        _row("spotter_cbg", ring0=0, ring1=30, ring2=20, beyond=50, failed=0),
    ]
    t = pd.DataFrame(rows)
    for seg in F.SEGMENTS:
        t[f"share_{seg}"] = t[seg] / t["n_targets"]
    return t


def _luminance(hexstr: str) -> float:
    r, g, b = (int(hexstr[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


class TestSegments:
    def test_segments_are_the_classifier_partition(self):
        """The figure must not invent its own outcome set — it draws exactly the
        columns `classify` guarantees sum to n_targets."""
        assert F.SEGMENTS == C.OUTCOME_COUNTS

    def test_every_segment_has_a_label_and_an_ink(self):
        for seg in F.SEGMENTS:
            assert seg in F.SEGMENT_LABELS
            assert seg in F.SEGMENT_INK

    def test_reading_order_runs_precise_to_absent(self):
        assert F.SEGMENTS == (
            "n_ring0", "n_ring1", "n_ring2", "n_beyond", "n_failed",
        )


class TestPalette:
    """Properties the dataviz validator checked; asserted so they stay true."""

    def _greens(self):
        """The four placed segments — the single-hue part of the ramp."""
        return [F.SEGMENT_INK[s] for s in F.SEGMENTS if s != "n_failed"]

    def test_the_placed_segments_are_monotone_light_to_dark(self):
        """Lightness is the separator no colour-vision deficiency and no
        greyscale print can take away, so the four *ordered* outcomes carry it.
        Light = precise, darkening as the prediction lands further out."""
        lums = [_luminance(c) for c in self._greens()]
        assert lums == sorted(lums, reverse=True), f"not monotone: {lums}"

    def test_lighter_means_better(self):
        """The direction, pinned explicitly, over the ramp."""
        greens = self._greens()
        assert _luminance(greens[0]) == max(_luminance(c) for c in greens)
        assert _luminance(greens[-1]) == min(_luminance(c) for c in greens)

    def test_no_answer_sits_outside_the_ramp(self):
        """Deliberately not the ramp's next step. 'No answer' is not a worse
        *placement* — it is the absence of one — so giving it a rank on the
        precision ramp would claim an ordering it does not have."""
        grey = _luminance(F.SEGMENT_INK["n_failed"])
        assert grey > max(_luminance(c) for c in self._greens())

    def test_adjacent_ramp_steps_are_visibly_apart(self):
        """Every adjacent gap must be a real step; checked in relative
        luminance with a floor loose enough not to re-litigate the colour
        space."""
        lums = [_luminance(c) for c in self._greens()]
        gaps = -np.diff(lums)
        assert (gaps > 0.015).all(), f"a step is too close to its neighbour: {gaps}"

    def test_the_placed_segments_are_one_hue(self):
        """Four steps of one hue, not four hues. A multi-hue good-to-bad scheme
        was measured and rejected: red against the ramp's mid-green is dE 1.8
        under protanopia, so a protanope could not separate 'two rings out'
        from a total miss. The grey is excluded — it has no meaningful hue."""
        import colorsys

        hues = [
            colorsys.rgb_to_hls(
                *[int(c[i : i + 2], 16) / 255 for i in (1, 3, 5)]
            )[0]
            * 360
            for c in self._greens()
        ]
        assert max(hues) - min(hues) < 25, f"hue spread too wide: {hues}"

    def test_no_answer_is_an_achromatic_light_grey(self):
        """Grey, and specifically a LIGHT grey — the conventional reading for
        absent, legible because the ramp vacated the light end. A *mid* grey is
        where green lands under deuteranopia (dE 2.6-5.0 against the ramp), so
        the light end is the only safe place for it."""
        import colorsys

        hexstr = F.SEGMENT_INK["n_failed"]
        r, g, b = (int(hexstr[i : i + 2], 16) / 255 for i in (1, 3, 5))
        sat = colorsys.rgb_to_hls(r, g, b)[2]
        assert sat < 0.12, f"{hexstr} is not achromatic (S={sat:.3f})"
        assert _luminance(hexstr) > 0.6, "a mid grey collides with the ramp"

    def test_the_grey_slot_carries_an_edge_rather_than_a_stripe(self):
        """At 1.44:1 the fill alone does not delineate against the surface, and
        the hatch channel is reserved, so the definition comes from an edge."""
        assert F._FAILED_EDGE != F._SURFACE
        assert _luminance(F._FAILED_EDGE) < _luminance(F.SEGMENT_INK["n_failed"])

    def test_no_outcome_uses_the_hatch_channel(self):
        """Reserved for the traffic-weighted arm beside a mesh bar. Spending it
        on an outcome would leave that distinction with nowhere to go."""
        src = pathlib.Path(F.__file__).read_text()
        assert "WEIGHTED_HATCH" in src
        assert "hatch=" not in src, "an outcome is drawing with a hatch"

    def test_label_ink_flips_with_the_fill(self):
        """Text inside a fill picks white or ink by luminance, so it always
        clears contrast — the one place a label may sit on a colour."""
        assert F._label_ink(F.SEGMENT_INK["n_ring0"]) == F._INK
        assert F._label_ink(F.SEGMENT_INK["n_failed"]) == F._INK
        assert F._label_ink(F.SEGMENT_INK["n_beyond"]) == "#ffffff"


class TestTable:
    def test_shares_sum_to_one_per_method(self):
        t = _table()
        total = t[[f"share_{s}" for s in F.SEGMENTS]].sum(axis=1)
        assert np.allclose(total, 1.0)

    def test_a_broken_partition_is_refused(self):
        """The failure is invisible in the artifact: a stack summing to 0.98
        still looks like a stack."""
        bad = _table([_row("m", n=100, ring0=40, ring1=20, ring2=10, beyond=20, failed=5)])
        with pytest.raises(ValueError, match="do not partition"):
            C.guard_partition(bad)

    def test_order_is_best_in_the_cell_first(self):
        order = F.method_order(_table())
        assert order == ["octant_cbg_hull", "spotter_cbg"]

    def test_a_tie_at_the_current_ring_breaks_on_the_next(self):
        """The ladder IS the metric: level at 51 km and better at 102 km is
        genuinely better, so a tie must cascade outward rather than fall back
        to a name or a row order."""
        rows = [
            # Same in-cell, but `b` places more of the rest one ring out.
            _row("a", ring0=30, ring1=10, ring2=30, beyond=30, failed=0),
            _row("b", ring0=30, ring1=30, ring2=10, beyond=30, failed=0),
        ]
        assert F.panel_order(_table(rows), "as01") == ["b", "a"]

    def test_the_cascade_goes_past_the_first_ring(self):
        """The real case this was written for: as01's shortest_ping and
        million_scale_cbg tie at ring 0 (0.3008) AND at within-ring-1 (0.6366),
        separating only at within-ring-2. A one-level tiebreak ordered them
        arbitrarily."""
        rows = [
            _row("a", ring0=30, ring1=34, ring2=0, beyond=36, failed=0),
            _row("b", ring0=30, ring1=34, ring2=5, beyond=31, failed=0),
        ]
        assert F.panel_order(_table(rows), "as01") == ["b", "a"]

    def test_a_tie_all_the_way_out_falls_back_to_answering_at_all(self):
        rows = [
            _row("a", ring0=20, ring1=20, ring2=20, beyond=20, failed=20),
            _row("b", ring0=20, ring1=20, ring2=20, beyond=40, failed=0),
        ]
        assert F.panel_order(_table(rows), "as01") == ["b", "a"]

    def test_a_total_tie_is_still_deterministic(self):
        """Identical at every rung, so the order must come from the explicit
        tiebreak rather than from however the rows arrived."""
        rows = [_row("z"), _row("a")]
        forward = F.panel_order(_table(rows), "as01")
        backward = F.panel_order(_table(list(reversed(rows))), "as01")
        assert forward == backward

    def test_the_primary_key_is_still_the_current_ring(self):
        """The next ring only breaks ties — it must not outrank a worse in-cell
        score. as03 has exactly this shape: million_scale beats shortest_ping
        at within-ring-2 but loses at ring 0, so it ranks below."""
        rows = [
            _row("worse_in_cell", ring0=10, ring1=10, ring2=70, beyond=10, failed=0),
            _row("better_in_cell", ring0=20, ring1=10, ring2=0, beyond=70, failed=0),
        ]
        assert F.panel_order(_table(rows), "as01") == [
            "better_in_cell", "worse_in_cell",
        ]

    def test_dataset_slug_names_the_comparison(self):
        assert F.dataset_slug(
            ["as01-260728-260802-mesh", "as03-x", "as02-y"]
        ) == "as01+as02+as03"

    def test_slug_is_order_independent(self):
        a = F.dataset_slug(["as02-x", "as01-y"])
        b = F.dataset_slug(["as01-y", "as02-x"])
        assert a == b


class TestMethodLabels:
    def test_known_methods_get_their_published_names(self):
        assert F.method_label("octant_cbg_spl") == "Octant-Spline"
        assert F.method_label("million_scale_cbg") == "SoI"
        assert F.method_label("spotter_cbg") == "Spotter"

    def test_an_unknown_method_degrades_readably(self):
        assert F.method_label("some_new_arm") == "some new arm"


class TestRender:
    def test_it_writes_a_png(self, tmp_path):
        png = F.render(_table(), 128, tmp_path)
        assert png.exists() and png.stat().st_size > 5000

    def test_the_order_can_be_pinned_from_outside(self, tmp_path):
        """The override, for a caller that wants x position comparable across
        panels instead of each panel ranking itself."""
        pinned = ["spotter_cbg", "octant_cbg_hull"]
        png = F.render(_table(), 64, tmp_path, order=pinned)
        assert png.exists()

    def test_a_multi_dataset_table_renders_one_panel_each(self, tmp_path):
        rows = [
            _row("octant_cbg_hull", ds=d) for d in ("as01", "as02", "as03")
        ] + [_row("spotter_cbg", ring0=0, ring1=30, ring2=20, beyond=50, failed=0, ds=d)
             for d in ("as01", "as02", "as03")]
        png = F.render(_table(rows), 128, tmp_path)
        assert png.exists()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Answer space + scores + figures for all three meshes, into a temp root.

    Module-scoped and defined at module level: a class-scoped fixture written
    as an instance method is deprecated in pytest, and its attributes would not
    reach the test methods anyway.
    """
    from scripts.analysis.v4.modules import answer_space as A
    from scripts.analysis.v4.modules.paths import MissingArtifactError, resolve_run

    root = tmp_path_factory.mktemp("v4fig")
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
    pngs = F.build_for_runs(runs, analysis_root=root)
    return runs, root, pngs


class TestRealRuns:
    """The figure over the real scored runs, including the artifact contract."""

    def test_one_figure_per_rung(self, built):
        _, _, pngs = built
        assert len(pngs) == len(H.NSIDE_LADDER)
        assert {p.name for p in pngs} == {
            F.FIGURE_PNG.format(slug=f"healpix-{n}") for n in H.NSIDE_LADDER
        }

    def test_every_figure_has_a_csv_twin_and_a_manifest(self, built):
        runs, root, _ = built
        out = F.cross_dir([r.run_id for r in runs], analysis_root=root)
        for n in H.NSIDE_LADDER:
            slug = f"healpix-{n}"
            assert (out / F.FIGURE_CSV.format(slug=slug)).exists()
            assert (out / F.FIGURE_MANIFEST.format(slug=slug)).exists()

    def test_every_panel_is_ranked_by_the_declared_sorting_key(self, built):
        """The contract: `(in-cell, 1-ring, 2-ring, further-out)` DESC, each
        cumulative and rounded to `ACCURACY_DECIMALS`.

        Asserted on the **rounded** key rather than the exact share, because
        that is what the figure sorts on and what its labels show. Checking the
        exact share instead would fail on a legitimate case: as03 rounds
        `million_scale_cbg` and `shortest_ping` both to 0.17 in-cell, and the
        cascade then puts million_scale first on the 2-ring rung even though
        its exact in-cell share is 0.4 pp lower. A difference below the
        reported precision must not decide an order the reader cannot verify.
        """
        runs, root, _ = built
        out = F.cross_dir([r.run_id for r in runs], analysis_root=root)
        for n in H.NSIDE_LADDER:
            slug = f"healpix-{n}"
            m = json.loads((out / F.FIGURE_MANIFEST.format(slug=slug)).read_text())
            df = F._with_rank_keys(pd.read_csv(out / F.FIGURE_CSV.format(slug=slug)))
            keyed = df.set_index(["dataset", "method"])
            for ds, order in m["panel_order"].items():
                keys = [
                    tuple(float(keyed.loc[(ds, meth), k]) for k in F._RANK_KEYS)
                    for meth in order
                ]
                assert keys == sorted(keys, reverse=True), (
                    f"nside={n} {ds} not descending on the key: "
                    f"{list(zip(order, keys))}"
                )

    def test_a_cumulative_key_never_exceeds_one(self, built):
        """It is a share. Summing already-rounded shares put `within_beyond` at
        1.01; the keys are now accumulated on the exact counts and rounded
        once."""
        runs, root, _ = built
        out = F.cross_dir([r.run_id for r in runs], analysis_root=root)
        for n in H.NSIDE_LADDER:
            df = F._with_rank_keys(
                pd.read_csv(out / F.FIGURE_CSV.format(slug=f"healpix-{n}"))
            )
            for k in F._RANK_KEYS:
                assert (df[k] <= 1.0).all(), f"nside={n} {k} exceeded 1: {df[k].max()}"
                assert (df[k] >= 0.0).all()

    def test_the_drawn_shares_stay_exact_so_the_stack_closes(self, built):
        """Geometry is not rounded — a rounded share would leave the bar short
        of or past 100%. Rounding lives in the label and the sort key only."""
        runs, root, _ = built
        out = F.cross_dir([r.run_id for r in runs], analysis_root=root)
        for n in H.NSIDE_LADDER:
            df = pd.read_csv(out / F.FIGURE_CSV.format(slug=f"healpix-{n}"))
            total = df[[f"share_{s}" for s in F.SEGMENTS]].sum(axis=1)
            assert np.allclose(total, 1.0), f"nside={n} stack does not close"

    def test_the_panel_orders_genuinely_differ_between_datasets(self, built):
        """Which is why they rank themselves: a single pooled order would hide
        that Octant-Spline leads as01 while Octant-Hull leads as02 and as03."""
        runs, root, _ = built
        out = F.cross_dir([r.run_id for r in runs], analysis_root=root)
        m = json.loads(
            (out / F.FIGURE_MANIFEST.format(slug="healpix-128")).read_text()
        )
        orders = {tuple(v) for v in m["panel_order"].values()}
        assert len(orders) > 1, "no dataset disagreed; the per-panel rank is moot"

    def test_the_manifest_records_the_absent_weighted_arm(self, built):
        """Rather than drawing a placeholder. v3 filled that half from a
        hard-coded dict and the figure showed 99.3% bars that measured
        nothing."""
        runs, root, _ = built
        out = F.cross_dir([r.run_id for r in runs], analysis_root=root)
        m = json.loads(
            (out / F.FIGURE_MANIFEST.format(slug="healpix-128")).read_text()
        )
        assert "weighted_arm" in m
        assert "no traffic-weighted run exists" in m["weighted_arm"]

    def test_the_csv_counts_partition_every_row(self, built):
        runs, root, _ = built
        out = F.cross_dir([r.run_id for r in runs], analysis_root=root)
        df = pd.read_csv(out / F.FIGURE_CSV.format(slug="healpix-128"))
        assert (df[list(F.SEGMENTS)].sum(axis=1) == df["n_targets"]).all()

    def test_three_datasets_are_present(self, built):
        runs, root, _ = built
        out = F.cross_dir([r.run_id for r in runs], analysis_root=root)
        df = pd.read_csv(out / F.FIGURE_CSV.format(slug="healpix-128"))
        assert sorted(df["dataset"].unique()) == ["as01", "as02", "as03"]


class TestReportedPrecision:
    """Accuracy is reported and ranked at the same precision, on purpose."""

    def test_the_sorting_key_is_the_declared_tuple(self):
        assert F._RANK_KEYS == (
            "within_ring0", "within_ring1", "within_ring2", "within_beyond",
        )
        assert F.ACCURACY_DECIMALS == 2

    def test_a_sub_precision_lead_does_not_decide_the_order(self):
        """The case that prompted this. as01 at nside 16: million_scale_cbg is
        in-cell on 259 of 399 targets and octant_cbg_spl on 258 — a one-target
        lead, 0.6491 vs 0.6466, both printing as 65%. Ranking on the exact
        share put the *worse* method first with no way for a reader to see why;
        rounding first makes it a tie and the 1-ring rung resolves it visibly.
        """
        rows = [
            # 259/399 in-cell, weak one ring out.
            _row("sub_precision_leader", n=399, ring0=259, ring1=65, ring2=30,
                 beyond=45, failed=0),
            # 258/399 in-cell, much stronger one ring out.
            _row("better_next_ring", n=399, ring0=258, ring1=108, ring2=31,
                 beyond=2, failed=0),
        ]
        table = _table(rows)
        keyed = F._with_rank_keys(table).set_index("method")
        # Rounded to the reported precision they tie in-cell ...
        assert keyed.loc["sub_precision_leader", "within_ring0"] == keyed.loc[
            "better_next_ring", "within_ring0"
        ]
        # ... and separate on the next rung, which the labels also show.
        assert (
            keyed.loc["better_next_ring", "within_ring1"]
            > keyed.loc["sub_precision_leader", "within_ring1"]
        )
        assert F.panel_order(table, "as01")[0] == "better_next_ring"

    def test_keys_accumulate_on_counts_not_on_rounded_shares(self):
        """Summing rounded shares compounds: it put `within_beyond` at 1.01."""
        rows = [_row("m", n=3, ring0=1, ring1=1, ring2=1, beyond=0, failed=0)]
        keyed = F._with_rank_keys(_table(rows))
        assert keyed["within_beyond"].iloc[0] <= 1.0

    def test_the_geometry_is_not_rounded(self, tmp_path):
        """The drawn `share_*` must stay exact, or the bars stop closing at
        100%. Asserted on the values rather than by grepping the source, which
        matched the comment explaining the rule and not the code.
        """
        from scripts.analysis.v4.modules import classify as CC

        # 399 targets split so that no share is a round 2-decimal number.
        acc = pd.DataFrame(
            [
                {
                    "method": "m", "nside": 128, "cell_km": 50.9,
                    "n_targets": 399, "n_solved": 399, "n_fallback": 0,
                    "n_error": 0, "fallback_rate": 0.0,
                    "n_ring0": 259, "n_ring1": 65, "n_ring2": 30,
                    "n_beyond": 45, "n_failed": 0,
                    "accuracy_ring0": 0.649, "accuracy_nearest_seed_retired": 0.9,
                    "error_km_p50": 1.0, "error_km_p90": 2.0,
                }
            ]
        )
        d = tmp_path / "as01-x" / "target-cls-accuracy" / "healpix-128"
        d.mkdir(parents=True)
        acc.to_csv(d / CC.ACCURACY_CSV, index=False)

        from scripts.analysis.v4.modules.paths import RunPaths

        run = RunPaths(
            run_id="as01-x", root=tmp_path, source="generic_csv", setup="s"
        )
        table = F.build_table([run], 128, analysis_root=tmp_path)
        got = float(table["share_n_ring0"].iloc[0])
        assert got == pytest.approx(259 / 399, abs=1e-12)
        assert got != round(got, F.ACCURACY_DECIMALS), "the share was rounded"
