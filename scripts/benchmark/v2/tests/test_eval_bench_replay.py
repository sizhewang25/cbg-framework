"""One unreplayable combo must not take the whole run's eval with it.

`compute_bench_metrics` discovers combos by **globbing the output tree**, not by
reading a config, so any arm left on disk from an older code version lands in
its loop. It rebuilt each combo's MTL/CTR straight from the stored `run.json`
with no guard, so a single stale `mtl_kwargs` payload raised and the command
produced nothing at all -- including every combo sorting after it. Its Snakemake
rule is commented out, so nothing in the workflow would have surfaced that; the
failure lands on whoever next runs the command by hand.
"""

from __future__ import annotations

import json
import unittest
import unittest.mock
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.benchmark.v2 import schema as bench_schema
from scripts.framework.v2.registry import MTL_REGISTRY
from scripts.benchmark.v2.eval_bench_results import (
    UnreplayableCombo,
    _replay_methods,
    compute_bench_metrics,
)

#: A payload that is valid today.
LIVE = {
    "mtl": "planar_circle",
    "mtl_kwargs": {},
    "ctr": "geometric_centroid",
    "ctr_kwargs": {},
}


class TestReplayMethods(unittest.TestCase):
    def test_a_live_payload_builds_both_methods(self):
        mtl, ctr = _replay_methods(LIVE, "combo", "fold_0")
        self.assertEqual(type(mtl).__name__, "PlanarCircleMTL")
        self.assertEqual(type(ctr).__name__, "GeometricCentroidCTR")

    def test_a_retired_mtl_name_is_unreplayable(self):
        with self.assertRaises(UnreplayableCombo) as ctx:
            _replay_methods({**LIVE, "mtl": "no_such_mtl"}, "combo", "fold_1")
        self.assertIn("combo/fold_1", str(ctx.exception))

    def test_a_retired_ctr_name_is_unreplayable(self):
        with self.assertRaises(UnreplayableCombo):
            _replay_methods({**LIVE, "ctr": "density_mean"}, "combo", "fold_0")

    def test_a_kwarg_that_no_longer_exists_is_unreplayable(self):
        """The case that motivated this: a combo stored kwargs its class has
        since renamed, added to, or made required."""
        with self.assertRaises(UnreplayableCombo) as ctx:
            _replay_methods(
                {**LIVE, "mtl_kwargs": {"retired_knob": 4}}, "spotter_h3_cbg", "fold_2"
            )
        msg = str(ctx.exception)
        self.assertIn("spotter_h3_cbg/fold_2", msg)
        self.assertIn("TypeError", msg)

    def test_a_kwarg_outside_the_accepted_range_is_unreplayable(self):
        """The `ValueError` arm of the catch list: a kwarg whose accepted range
        narrowed since the run.

        Deliberately a stub rather than a real registered class. Every concrete
        constructor's validation is free to change -- that is the whole reason
        this guard exists -- so pinning the test to one class's current rules
        would make it fail for the very reason it is testing.
        """
        class _Narrowed:
            def __init__(self, **_):
                raise ValueError("top_k must be >= 1, got 0")

        with unittest.mock.patch.dict(
            MTL_REGISTRY, {"narrowed": _Narrowed}, clear=False
        ):
            with self.assertRaises(UnreplayableCombo) as ctx:
                _replay_methods(
                    {**LIVE, "mtl": "narrowed", "mtl_kwargs": {"top_k": 0}},
                    "combo", "fold_0",
                )
        self.assertIn("ValueError", str(ctx.exception))

    def test_the_message_says_what_to_do_about_it(self):
        with self.assertRaises(UnreplayableCombo) as ctx:
            _replay_methods({**LIVE, "mtl": "gone"}, "combo", "fold_0")
        self.assertIn("re-run the benchmark", str(ctx.exception))


def _write_combo(root: Path, combo_id: str, run_meta: dict) -> Path:
    """A combo directory with the two files discovery and replay need."""
    d = root / "generic_csv" / "anchors_to_probes" / "fold_0" / combo_id
    d.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([], schema=bench_schema.TARGETS_SCHEMA),
        d / "targets.parquet",
    )
    (d / "run.json").write_text(json.dumps({"combo_id": combo_id, **run_meta}))
    return d


class TestCombosAreSkippedNotFatal(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.run_dir = Path(self._tmp.name) / "a-run"
        self.run_dir.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_an_unreplayable_combo_is_reported_rather_than_raised(self):
        _write_combo(self.run_dir, "legacy_cbg", {**LIVE, "mtl_kwargs": {"gone": 1}})
        skipped: list[str] = []
        with self.assertLogs(
            "scripts.benchmark.v2.eval_bench_results", level="WARNING"
        ):
            out = compute_bench_metrics(
                self.run_dir, Path("unused"), unreplayable=skipped,
            )
        self.assertEqual(out, {})
        self.assertEqual(skipped, ["legacy_cbg"])

    def test_a_skipped_combo_is_not_half_computed(self):
        """Folds pool into one frame, so a combo that fails on any fold must be
        absent entirely -- a partial one is a table row silently computed over a
        subset of the population."""
        _write_combo(self.run_dir, "legacy_cbg", {**LIVE, "mtl": "gone"})
        skipped: list[str] = []
        with self.assertLogs(
            "scripts.benchmark.v2.eval_bench_results", level="WARNING"
        ):
            out = compute_bench_metrics(
                self.run_dir, Path("unused"), unreplayable=skipped,
            )
        self.assertNotIn("legacy_cbg", out)

    def test_the_sink_is_optional(self):
        """Callers that do not care must not have to pass one."""
        _write_combo(self.run_dir, "legacy_cbg", {**LIVE, "mtl": "gone"})
        with self.assertLogs(
            "scripts.benchmark.v2.eval_bench_results", level="WARNING"
        ):
            self.assertEqual(
                compute_bench_metrics(self.run_dir, Path("unused")), {}
            )

    def test_an_id_is_recorded_once_even_across_folds(self):
        for fold in ("fold_0", "fold_1"):
            d = self.run_dir / "generic_csv" / "anchors_to_probes" / fold / "legacy_cbg"
            d.mkdir(parents=True)
            pq.write_table(
                pa.Table.from_pylist([], schema=bench_schema.TARGETS_SCHEMA),
                d / "targets.parquet",
            )
            (d / "run.json").write_text(
                json.dumps({"combo_id": "legacy_cbg", **LIVE, "mtl": "gone"})
            )
        skipped: list[str] = []
        with self.assertLogs(
            "scripts.benchmark.v2.eval_bench_results", level="WARNING"
        ):
            compute_bench_metrics(self.run_dir, Path("unused"), unreplayable=skipped)
        self.assertEqual(skipped, ["legacy_cbg"])

    def test_no_combos_at_all_is_still_an_error(self):
        """Skipping a stale arm must not turn an empty tree into a silent pass."""
        with self.assertRaises(FileNotFoundError):
            compute_bench_metrics(self.run_dir, Path("unused"))


if __name__ == "__main__":
    unittest.main()
