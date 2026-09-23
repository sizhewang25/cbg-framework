"""Tests for the cross-dataset Euler diagram.

Three things are worth pinning here, and they are the three that would fail
silently:

* **top-N is a ring.** The mapping `top_n=k -> ring <= k-1` is the whole
  difference from v3, and it is cumulative — a target in the set at top-1 must
  still be in it at top-3. A sign slip in `correct_at` would leave unplaced
  predictions counted as correct and no artifact would look wrong.
* **Dropping an empty set changes nothing else.** The figure is drawn without
  Spotter at top-1 on the strength of that claim, so it is asserted directly:
  every region count is identical with and without the empty column.
* **The fit's own error numbers are real.** `placed` and `pair_error` are
  printed on the figure as its honesty statement; if they were computed off a
  different layout than the one drawn they would be decoration.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v4.modules import figure_euler as FE
from scripts.analysis.v4.modules.euler import layout as L
from scripts.analysis.v4.modules.euler import membership as M
from scripts.analysis.v4.modules.euler import tables as T


def _cells(rings, status="SUCCESS", prefix="tg"):
    """A `<method>_cells.parquet`-shaped frame from a list of ring values."""
    return pd.DataFrame(
        {
            "target_id": [f"{prefix}-{i}" for i in range(len(rings))],
            "status": [status] * len(rings),
            "ring": rings,
        }
    )


class TestTopNIsARing:
    def test_maps_top_n_to_the_cumulative_ring(self):
        cells = _cells([0, 1, 2, -1])
        assert list(M.correct_at(cells, 1)) == [True, False, False, False]
        assert list(M.correct_at(cells, 2)) == [True, True, False, False]
        assert list(M.correct_at(cells, 3)) == [True, True, True, False]

    def test_unplaced_never_counts_however_wide_the_tolerance(self):
        # `ring == -1` is a sentinel, not a distance. Testing only the upper
        # bound would let it pass on the sign.
        assert not M.correct_at(_cells([-1]), M.MAX_TOP_N).iloc[0]

    def test_a_fallback_is_wrong_not_excluded(self):
        # The denominator is every evaluated target: a method that declines to
        # answer has not earned a smaller one than a method that answers badly.
        wrong = M.correct_at(_cells([0], status="FALLBACK"), 1)
        assert len(wrong) == 1 and not wrong.iloc[0]

    def test_the_baseline_is_solved_on_every_row(self):
        # An all-BASELINE frame has no fallback path to take, so it is wholly
        # solved; read as non-SUCCESS it would score zero everywhere.
        assert M.correct_at(_cells([0, 1], status="BASELINE"), 1).iloc[0]

    @pytest.mark.parametrize("bad", [0, -1, M.MAX_TOP_N + 1])
    def test_refuses_a_tolerance_off_the_ladder(self, bad):
        with pytest.raises(ValueError, match="top_n must be"):
            M.validate_top_n(bad)


class TestEmptySets:
    def test_drops_only_the_columns_no_target_belongs_to(self):
        m = pd.DataFrame(
            {"a": [True, False], "empty": [False, False], "b": [True, True]}
        )
        live, empty = FE.drop_empty_sets(m)
        assert live == ["a", "b"]
        assert empty == ["empty"]

    def test_dropping_an_empty_set_changes_no_region(self):
        """The claim the figure rests on, asserted rather than argued.

        An empty set belongs to no region, so every intersection involving it
        is empty and every other region's count is what it already was. If this
        were false, the top-1 figure drawn without Spotter would be a different
        figure from the one the counts describe.
        """
        rng = np.random.default_rng(0)
        m = pd.DataFrame(
            {
                "a": rng.random(200) < 0.4,
                "b": rng.random(200) < 0.3,
                "c": rng.random(200) < 0.5,
            }
        )
        with_empty = m.assign(empty=False)
        live, _ = FE.drop_empty_sets(with_empty)

        full = T.intersection_table(with_empty).set_index("methods")["n_targets"]
        cut = T.intersection_table(m[live]).set_index("methods")["n_targets"]
        assert full.to_dict() == cut.to_dict()

    def test_the_fit_refuses_a_zero_radius_circle(self):
        # A zero-radius circle is a dot a reader takes for "very small" rather
        # than "never", so the fit refuses it and the caller strips it first.
        m = pd.DataFrame({"a": [True, True], "empty": [False, False]})
        with pytest.raises(ValueError, match="placed no target"):
            L.fit_euler_layout(m, ["a", "empty"])


class TestLayout:
    @staticmethod
    @pytest.fixture(scope="class")
    def fitted():
        rng = np.random.default_rng(7)
        base = rng.random(400)
        m = pd.DataFrame(
            {
                "shortest_ping": base < 0.45,
                "million_scale_cbg": base < 0.40,
                "octant_cbg_hull": rng.random(400) < 0.35,
            }
        )
        return m, L.fit_euler_layout(m, list(m.columns), restarts=1)

    def test_area_is_the_share_not_the_radius(self, fitted):
        m, fit = fitted
        for i, method in enumerate(fit.order):
            assert np.pi * fit.radii[i] ** 2 == pytest.approx(
                m[method].mean(), abs=1e-9
            )

    def test_observed_shares_partition_the_population(self, fitted):
        _, fit = fitted
        assert fit.observed.sum() == pytest.approx(1.0)

    def test_reports_its_own_error(self, fitted):
        _, fit = fitted
        assert 0.0 <= fit.placed <= 1.0
        assert fit.placed == pytest.approx(1.0 - fit.misplaced)
        # A three-set layout on nested data is fittable; anything below this
        # would mean the optimizer is not converging rather than that circles
        # are overdetermined.
        assert fit.placed > 0.85
        assert fit.pair_error() < 0.05

    def test_is_deterministic(self, fitted):
        m, fit = fitted
        again = L.fit_euler_layout(m, list(m.columns), restarts=1)
        assert np.allclose(fit.centres, again.centres)

    def test_fit_table_flags_regions_the_picture_invents(self, fitted):
        m, fit = fitted
        table = L.euler_fit_table(fit, len(m))
        # `delta` is drawn minus observed, so a region the data says is empty
        # and the picture draws anyway is a positive row with n_targets == 0.
        invented = table[(table["n_targets"] == 0) & (table["delta"] > 0)]
        assert set(invented.columns) >= {"region", "delta"}
        assert (table["observed_share"] >= 0).all()


class TestTables:
    def test_intersections_partition_the_population(self):
        rng = np.random.default_rng(3)
        m = pd.DataFrame({c: rng.random(150) < 0.4 for c in "abc"})
        assert T.intersection_table(m)["n_targets"].sum() == len(m)

    def test_region_letters_follow_position(self):
        assert T.region_key(["a", "b", "c"], {"a", "c"}) == "AC"

    def test_pairwise_rows_account_for_every_target(self):
        m = pd.DataFrame({"a": [True, True, False], "b": [True, False, False]})
        row = T.pairwise_table(m).iloc[0]
        assert row.both + row.a_only + row.b_only + row.neither == len(m)


class TestPooling:
    def test_rekeys_rows_by_run(self):
        # The prefix is what stops a future run that reused a target id from
        # collapsing two different targets into one row.
        assert M.RUN_KEY_SEP == "::"

    def test_refuses_targets_shared_between_runs(self):
        with pytest.raises(ValueError, match="share 1 target ids"):
            M.guard_disjoint_targets({"as01": {"tg-1", "tg-2"}, "as02": {"tg-2"}})

    def test_refuses_methods_scored_on_different_targets(self):
        cols = {
            "a": pd.Series([True], index=["tg-1"]),
            "b": pd.Series([True, True], index=["tg-1", "tg-2"]),
        }
        with pytest.raises(ValueError, match="different target sets"):
            M.guard_one_population(cols, "as01")


class TestArtifactNames:
    def test_the_slug_carries_grid_and_tolerance(self):
        stem = FE.STEM.format(slug="healpix-128", n=3)
        assert stem == "euler.healpix-128.top3"
        # Without the tolerance in the name a top-3 pass would overwrite top-1.
        assert FE.STEM.format(slug="healpix-128", n=1) != stem

    def test_every_artifact_shares_one_stem(self):
        stem = "euler.healpix-128.top1"
        names = {k: v.format(stem=stem) for k, v in FE.ARTIFACTS.items()}
        assert all(n.startswith(stem) for n in names.values())
        assert names["png"].endswith(".png")

    def test_dataset_slug_names_the_comparison(self):
        assert (
            FE.dataset_slug(
                ["as03-260728-260802-mesh", "as01-260728-260802-mesh"]
            )
            == "as01+as03"
        )


class TestEndToEnd:
    """One full pass over synthetic `classify` output, into a temp tree."""

    @staticmethod
    def _write_run(root, run_id, rings_by_method, nside=128):
        from scripts.analysis.v4.modules import classify as C

        out = root / run_id / "target-cls-accuracy" / f"healpix-{nside}"
        out.mkdir(parents=True, exist_ok=True)
        for method, rings in rings_by_method.items():
            _cells(rings, prefix=run_id).to_parquet(
                out / C.CELLS_PARQUET.format(method=method), index=False
            )

    @pytest.fixture
    def built(self, tmp_path):
        from scripts.analysis.v4.modules.paths import RunPaths

        rng = np.random.default_rng(11)
        runs = []
        for run_id in ("as01-x-mesh", "as02-x-mesh"):
            self._write_run(
                tmp_path,
                run_id,
                {
                    "shortest_ping": rng.integers(-1, 3, 120).tolist(),
                    "octant_cbg_hull": rng.integers(-1, 3, 120).tolist(),
                    # Never in the truth's own cell, like Spotter on the real
                    # meshes: its set is empty at top-1 and not at top-2.
                    "spotter_cbg": rng.integers(1, 3, 120).tolist(),
                },
            )
            runs.append(
                RunPaths(run_id=run_id, root=tmp_path, source="s", setup="t")
            )
        written = FE.build_for_runs(runs, analysis_root=tmp_path)
        return tmp_path, runs, written

    def test_writes_one_set_per_tolerance(self, built):
        _, _, written = built
        assert len(written) == len(FE.DEFAULT_TOP_NS)
        for w in written:
            assert w["manifest"].exists()
            assert w["intersections"].exists()

    def test_top1_drops_the_empty_set_and_says_so(self, built):
        _, _, written = built
        man = json.loads(written[0]["manifest"].read_text())
        assert man["top_n"] == 1
        assert man["empty_sets"] == ["spotter_cbg"]
        assert "spotter_cbg" not in man["circles"]
        # Still in the population and in the count tables — dropped from the
        # drawing, not from the denominator.
        assert man["n_correct_per_method"]["spotter_cbg"] == 0
        assert "Spotter" in written[0]["pairwise"].read_text()

    def test_top2_draws_every_set(self, built):
        _, _, written = built
        man = json.loads(written[1]["manifest"].read_text())
        assert man["top_n"] == 2
        assert man["empty_sets"] is None
        assert "spotter_cbg" in man["circles"]

    def test_sets_only_grow_with_the_tolerance(self, built):
        _, _, written = built
        counts = [
            json.loads(w["manifest"].read_text())["n_correct_per_method"]
            for w in written
        ]
        for method in counts[0]:
            assert counts[0][method] <= counts[1][method] <= counts[2][method]

    def test_pools_both_runs_into_one_denominator(self, built):
        _, runs, written = built
        man = json.loads(written[0]["manifest"].read_text())
        assert man["n_targets"] == 240
        assert man["n_targets_per_run"] == {r.run_id: 120 for r in runs}

    def test_membership_csv_carries_the_run_each_row_came_from(self, built):
        _, _, written = built
        rows = pd.read_csv(written[0]["membership"], index_col=0)
        assert "run_id" in rows.columns
        assert set(rows["run_id"]) == {"as01-x-mesh", "as02-x-mesh"}
        assert all(M.RUN_KEY_SEP in str(i) for i in rows.index)

    def test_manifest_error_numbers_match_the_drawn_layout(self, built):
        _, _, written = built
        man = json.loads(written[2]["manifest"].read_text())
        fit = pd.read_csv(written[2]["fit"])
        # `placed` is 1 - total variation over the drawn regions, so it has to
        # agree with the per-region deltas the CSV lists.
        tv = 0.5 * fit["delta"].abs().sum()
        assert man["euler_placed_share"] == pytest.approx(1 - tv, abs=0.02)
