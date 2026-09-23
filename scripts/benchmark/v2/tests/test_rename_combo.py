"""Tests for `rename-combo`.

`outputs/` is gitignored, so this command's effect is invisible to review and
to CI -- which is the whole reason it is a tested command rather than a shell
loop. The two properties worth the most here are the ones a loop gets wrong:
`combo_id` inside `run.json` moves with the directory, and derived artifacts are
matched by exact suffix under the renamed runs only.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.benchmark.v2.rename_combo import (
    DEFAULT_DERIVED_SUFFIXES,
    RenameRefused,
    apply,
    plan_rename,
)


def _combo(root: Path, run: str, fold: str, combo: str, mtl: str) -> Path:
    d = root / run / "generic_csv" / "anchors_to_probes" / fold / combo
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({
        "run_id": run, "combo_id": combo, "mtl": mtl, "ctr": "density_argmax",
    }))
    (d / "targets.parquet").write_bytes(b"")
    return d


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name) / "benchmark"
        self.derived = Path(self._tmp.name) / "analysis"

    def tearDown(self):
        self._tmp.cleanup()

    def plan(self, **kw):
        kw.setdefault("derived_roots", [self.derived])
        return plan_rename(self.root, "spotter_cbg", "spotter_h3_cbg", **kw)


class TestTheGate(Base):
    def test_only_the_matching_mtl_is_renamed(self):
        """The same combo id names different compositions across runs. On this
        repo `spotter_cbg` is the density arm in three runs and the
        Octant-geometry hybrid in six others."""
        _combo(self.root, "mesh", "fold_0", "spotter_cbg", "gaussian_density")
        _combo(self.root, "legacy", "fold_0", "spotter_cbg", "planar_annulus_weighted")
        plan = self.plan(require_mtl="gaussian_density")
        self.assertEqual(len(plan.combo_dirs), 1)
        self.assertEqual(plan.combo_dirs[0].parts[-2:], ("fold_0", "spotter_cbg"))
        self.assertIn("mesh", str(plan.combo_dirs[0]))
        self.assertEqual(len(plan.skipped), 1)
        self.assertIn("planar_annulus_weighted", plan.skipped[0][1])

    def test_without_a_gate_everything_matches(self):
        _combo(self.root, "a", "fold_0", "spotter_cbg", "gaussian_density")
        _combo(self.root, "b", "fold_0", "spotter_cbg", "planar_annulus_weighted")
        self.assertEqual(len(self.plan().combo_dirs), 2)

    def test_a_run_filter_narrows_further(self):
        _combo(self.root, "a", "fold_0", "spotter_cbg", "gaussian_density")
        _combo(self.root, "b", "fold_0", "spotter_cbg", "gaussian_density")
        plan = self.plan(run_ids=["a"])
        self.assertEqual(len(plan.combo_dirs), 1)
        self.assertIn("/a/", str(plan.combo_dirs[0]))

    def test_an_unreadable_run_json_is_skipped_with_a_reason(self):
        d = _combo(self.root, "a", "fold_0", "spotter_cbg", "gaussian_density")
        (d / "run.json").write_text("{not json")
        plan = self.plan(require_mtl="gaussian_density")
        self.assertEqual(plan.combo_dirs, [])
        self.assertIn("run.json", plan.skipped[0][1])


class TestTheMetadataMovesToo(Base):
    def test_combo_id_is_patched_inside_run_json(self):
        """`_summarize_combo` reads `meta["combo_id"]`, not the directory name,
        and does not check uniqueness -- so a directory renamed without its
        metadata gives two rows per slice both claiming the old id, and any
        `groupby("combo_id")` averages two implementations together."""
        _combo(self.root, "mesh", "fold_0", "spotter_cbg", "gaussian_density")
        apply(self.plan(require_mtl="gaussian_density"), "spotter_cbg", "spotter_h3_cbg")
        moved = (
            self.root / "mesh" / "generic_csv" / "anchors_to_probes"
            / "fold_0" / "spotter_h3_cbg"
        )
        self.assertTrue(moved.is_dir())
        self.assertFalse(moved.with_name("spotter_cbg").exists())
        self.assertEqual(
            json.loads((moved / "run.json").read_text())["combo_id"],
            "spotter_h3_cbg",
        )

    def test_the_rest_of_run_json_survives(self):
        _combo(self.root, "mesh", "fold_0", "spotter_cbg", "gaussian_density")
        apply(self.plan(require_mtl="gaussian_density"), "spotter_cbg", "spotter_h3_cbg")
        meta = json.loads((
            self.root / "mesh" / "generic_csv" / "anchors_to_probes"
            / "fold_0" / "spotter_h3_cbg" / "run.json"
        ).read_text())
        self.assertEqual(meta["mtl"], "gaussian_density")
        self.assertEqual(meta["ctr"], "density_argmax")

    def test_every_fold_moves(self):
        for f in ("fold_0", "fold_1", "fold_2"):
            _combo(self.root, "mesh", f, "spotter_cbg", "gaussian_density")
        plan = self.plan(require_mtl="gaussian_density")
        self.assertEqual(len(plan.combo_dirs), 3)
        apply(plan, "spotter_cbg", "spotter_h3_cbg")
        self.assertEqual(
            len(list(self.root.glob("*/*/*/fold_*/spotter_h3_cbg"))), 3
        )


class TestDerivedArtifacts(Base):
    def _artifact(self, run: str, name: str) -> Path:
        d = self.derived / run / "target-cls-accuracy" / "h3-4"
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        p.write_bytes(b"")
        return p

    def test_exact_suffixes_move(self):
        _combo(self.root, "mesh", "fold_0", "spotter_cbg", "gaussian_density")
        for suffix in DEFAULT_DERIVED_SUFFIXES:
            self._artifact("mesh", f"spotter_cbg{suffix}")
        plan = self.plan(require_mtl="gaussian_density")
        self.assertEqual(len(plan.derived), len(DEFAULT_DERIVED_SUFFIXES))
        apply(plan, "spotter_cbg", "spotter_h3_cbg")
        for suffix in DEFAULT_DERIVED_SUFFIXES:
            self.assertTrue(
                (self.derived / "mesh" / "target-cls-accuracy" / "h3-4"
                 / f"spotter_h3_cbg{suffix}").exists()
            )

    def test_a_sibling_combos_artifacts_are_left_alone(self):
        """The one that a prefix match gets wrong. `spotter_cbg_c80`,
        `spotter_cbg_c100` and `spotter_cbg_top` are *different combos* from a
        coverage sweep, and all three are prefixed by `spotter_cbg`."""
        _combo(self.root, "mesh", "fold_0", "spotter_cbg", "gaussian_density")
        keep = [
            self._artifact("mesh", "spotter_cbg_c80_seed_distances.parquet"),
            self._artifact("mesh", "spotter_cbg_c100_seed_distances.parquet"),
            self._artifact("mesh", "spotter_cbg_top_geo_seed_distances.parquet"),
        ]
        move = self._artifact("mesh", "spotter_cbg_seed_distances.parquet")
        plan = self.plan(require_mtl="gaussian_density")
        self.assertEqual(plan.derived, [move])
        apply(plan, "spotter_cbg", "spotter_h3_cbg")
        for p in keep:
            self.assertTrue(p.exists(), p.name)

    def test_artifacts_of_a_run_that_did_not_move_are_left_alone(self):
        """Scoping to the renamed runs, not tidiness: relabelling the hybrid's
        scorings as the density arm's would be worse than leaving them."""
        _combo(self.root, "mesh", "fold_0", "spotter_cbg", "gaussian_density")
        _combo(self.root, "legacy", "fold_0", "spotter_cbg", "planar_annulus_weighted")
        mine = self._artifact("mesh", "spotter_cbg_cells.parquet")
        theirs = self._artifact("legacy", "spotter_cbg_cells.parquet")
        plan = self.plan(require_mtl="gaussian_density")
        self.assertEqual(plan.derived, [mine])
        apply(plan, "spotter_cbg", "spotter_h3_cbg")
        self.assertTrue(theirs.exists())

    def test_a_directory_named_exactly_the_combo_moves(self):
        _combo(self.root, "mesh", "fold_0", "spotter_cbg", "gaussian_density")
        d = self.derived / "mesh" / "mtl-map" / "h3-4" / "regions" / "spotter_cbg"
        d.mkdir(parents=True)
        (d / "tg-0.json").write_text("{}")
        apply(self.plan(require_mtl="gaussian_density"), "spotter_cbg", "spotter_h3_cbg")
        self.assertTrue((d.with_name("spotter_h3_cbg") / "tg-0.json").exists())

    def test_a_missing_derived_root_is_not_an_error(self):
        _combo(self.root, "mesh", "fold_0", "spotter_cbg", "gaussian_density")
        plan = plan_rename(
            self.root, "spotter_cbg", "spotter_h3_cbg",
            derived_roots=[self.derived / "nope"],
        )
        self.assertEqual(plan.derived, [])


class TestRefusals(Base):
    def test_it_refuses_to_merge_two_combos(self):
        _combo(self.root, "mesh", "fold_0", "spotter_cbg", "gaussian_density")
        _combo(self.root, "mesh", "fold_0", "spotter_h3_cbg", "gaussian_density")
        with self.assertRaises(RenameRefused) as ctx:
            self.plan()
        self.assertIn("merge", str(ctx.exception))

    def test_it_refuses_to_overwrite_a_derived_artifact(self):
        _combo(self.root, "mesh", "fold_0", "spotter_cbg", "gaussian_density")
        d = self.derived / "mesh" / "x"
        d.mkdir(parents=True)
        (d / "spotter_cbg_cells.parquet").write_bytes(b"")
        (d / "spotter_h3_cbg_cells.parquet").write_bytes(b"")
        with self.assertRaises(RenameRefused) as ctx:
            self.plan()
        self.assertIn("re-run the analysis stage", str(ctx.exception))

    def test_renaming_to_itself_is_refused(self):
        with self.assertRaises(RenameRefused):
            plan_rename(self.root, "same", "same")


class TestIdempotence(Base):
    def test_a_second_run_is_a_noop(self):
        _combo(self.root, "mesh", "fold_0", "spotter_cbg", "gaussian_density")
        self._art = (self.derived / "mesh" / "x")
        self._art.mkdir(parents=True)
        (self._art / "spotter_cbg_cells.parquet").write_bytes(b"")

        apply(self.plan(require_mtl="gaussian_density"), "spotter_cbg", "spotter_h3_cbg")
        again = self.plan(require_mtl="gaussian_density")
        self.assertTrue(again.is_noop)
        self.assertEqual(len(again.already_done), 1)

    def test_a_half_finished_rename_can_be_completed(self):
        """A run interrupted between folds must be resumable, not wedged."""
        for f in ("fold_0", "fold_1"):
            _combo(self.root, "mesh", f, "spotter_cbg", "gaussian_density")
        first = self.plan(require_mtl="gaussian_density")
        first.combo_dirs = first.combo_dirs[:1]
        apply(first, "spotter_cbg", "spotter_h3_cbg")

        rest = self.plan(require_mtl="gaussian_density")
        self.assertEqual(len(rest.combo_dirs), 1)
        self.assertEqual(len(rest.already_done), 1)
        apply(rest, "spotter_cbg", "spotter_h3_cbg")
        self.assertEqual(
            len(list(self.root.glob("*/*/*/fold_*/spotter_h3_cbg"))), 2
        )
        self.assertEqual(list(self.root.glob("*/*/*/fold_*/spotter_cbg")), [])


if __name__ == "__main__":
    unittest.main()
