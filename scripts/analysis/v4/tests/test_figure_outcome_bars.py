"""Tests for the outcome-composition bars.

`TestPalette` is the one worth keeping honest: the ramp's hexes were chosen by
running the dataviz validator, not by eye, and the properties it checked are
asserted here so a later "nicer green" cannot silently break them.
"""

from __future__ import annotations

import json

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

    def _ramp(self):
        return [F.SEGMENT_INK[s] for s in F.SEGMENTS if F.SEGMENT_INK[s] != "none"]

    def test_the_placed_segments_are_a_monotone_lightness_ramp(self):
        """Monotone lightness is the separator no colour-vision deficiency and
        no greyscale print can take away. It is also the ordinal validator's
        first check."""
        lums = [_luminance(c) for c in self._ramp()]
        assert lums == sorted(lums), f"ramp is not monotone: {lums}"

    def test_adjacent_steps_are_visibly_apart(self):
        """The ordinal rule wants every adjacent gap >= 0.06 in perceptual
        lightness; checked here in relative luminance as a proxy, with a floor
        loose enough not to re-litigate the colour space."""
        lums = [_luminance(c) for c in self._ramp()]
        gaps = np.diff(lums)
        assert (gaps > 0.02).all(), f"a step is too close to its neighbour: {gaps}"

    def test_the_ramp_is_one_hue(self):
        """Four steps of one hue, not four hues. A multi-hue 'good to bad'
        scheme was measured and rejected: red against the ramp's mid-green is
        dE 1.8 under protanopia."""
        import colorsys

        hues = []
        for c in self._ramp():
            r, g, b = (int(c[i : i + 2], 16) / 255 for i in (1, 3, 5))
            hues.append(colorsys.rgb_to_hls(r, g, b)[0] * 360)
        assert max(hues) - min(hues) < 25, f"hue spread too wide: {hues}"

    def test_never_answered_carries_no_fill(self):
        """Not a fifth hue. Every grey tested collided with some step of the
        ramp under deuteranopia (dE 1.6-4.5), and an absence has nothing to
        colour anyway."""
        assert F.SEGMENT_INK["n_failed"] == "none"

    def test_label_ink_flips_with_the_fill(self):
        """Text inside a fill picks white or ink by luminance, so it always
        clears contrast — the one place a label may sit on a colour."""
        assert F._label_ink(F.SEGMENT_INK["n_ring0"]) == "#ffffff"
        assert F._label_ink(F.SEGMENT_INK["n_beyond"]) == F._INK
        assert F._label_ink("none") == F._INK_2


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
        """So every rung of a figure set shares one x order. Ranking per rung
        moved Spotter between the 4th and 5th slot from figure to figure."""
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

    def test_the_x_order_is_identical_across_rungs(self, built):
        runs, root, _ = built
        out = F.cross_dir([r.run_id for r in runs], analysis_root=root)
        orders = []
        for n in H.NSIDE_LADDER:
            m = json.loads(
                (out / F.FIGURE_MANIFEST.format(slug=f"healpix-{n}")).read_text()
            )
            orders.append(tuple(m["methods"]))
        assert len(set(orders)) == 1, f"order drifted across rungs: {orders}"

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
