"""End-to-end runner test: synthetic source → materialize → run_one_combo.

Asserts that the writer schema matches the reader schema, the checkpoint
sidecar is produced (stateless marker for SoI LTD), and per-target rows carry
populated stage timings.
"""

from __future__ import annotations

import json
import pickle
import tempfile
import textwrap
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from scripts.benchmark.v2.inputs import materialize_inputs, outputs_combo_dir
from scripts.benchmark.v2.runner import ComboSpec, run_one_combo
from scripts.benchmark.v2.sources.generic_csv import GenericCSVSource


# Canonical-schema synth CSV: vp_* = anchor side (acting as VP),
# target_* = probe side.
_SYNTH_CSV = textwrap.dedent("""
    vp_id,vp_lat,vp_lon,vp_asn,vp_country,target_id,target_lat,target_lon,target_asn,target_country,rtt_ms
    1.1.1.1,33.0,-84.0,20473,US,1001,33.5,-84.5,7922,US,5.0
    1.1.1.1,33.0,-84.0,20473,US,1002,32.5,-83.5,7922,US,6.0
    1.1.1.1,33.0,-84.0,20473,US,1003,33.5,-83.5,7922,US,7.0
    1.1.1.1,33.0,-84.0,20473,US,1004,32.5,-84.5,7922,US,5.5
""").strip() + "\n"


class TestRunOneCombo(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.csv_path = root / "synth.csv"
        self.csv_path.write_text(_SYNTH_CSV)
        # k=4 with 4 single-ASN probes deterministically places exactly one
        # probe per fold under DistGeo's per-bucket round-robin → fold_0 has
        # n_targets == 1, matching the original test's "one target" shape.
        src = GenericCSVSource(
            slice="fold_0", setup="anchors_to_probes",
            csv_path=self.csv_path, k=4,
        )
        self.src = src
        self.inputs_dir = materialize_inputs(src, root=root / "inputs", run_id="test-run")
        self.out_dir = outputs_combo_dir(root / "outputs", "test-run", src, "combo1")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_full_runner_writes_three_artifacts(self) -> None:
        spec = ComboSpec(
            combo_id="combo1",
            ltd="speed_of_internet", mtl="planar_circle", ctr="geometric_centroid",
            ltd_kwargs={}, mtl_kwargs={}, ctr_kwargs={},
        )
        run_one_combo(
            spec, inputs_dir=self.inputs_dir, out_dir=self.out_dir,
            run_id="test-run", source_name="generic_csv", slice_name="fold_0",
        )

        # 1. run.json populated
        run_meta = json.loads((self.out_dir / "run.json").read_text())
        self.assertEqual(run_meta["combo_id"], "combo1")
        self.assertEqual(run_meta["n_targets"], 1)
        self.assertGreater(run_meta["fit_ms"], 0.0)
        self.assertGreaterEqual(run_meta["fit_alloc_peak_bytes"], 0)
        self.assertGreaterEqual(run_meta["fit_rss_peak_bytes"], 0)
        self.assertIn("status_counts", run_meta)
        self.assertGreater(run_meta["run_peak_rss_bytes"], 1_000_000)
        self.assertGreater(run_meta["run_baseline_rss_bytes"], 1_000_000)
        # Baseline is captured before any combo work, true peak is captured
        # at run end → getrusage monotonicity makes the inequality strict-or-equal.
        self.assertGreaterEqual(
            run_meta["run_peak_rss_bytes"], run_meta["run_baseline_rss_bytes"]
        )

        # 2. .stateless marker (SoI LTD has no fitted state)
        self.assertTrue((self.out_dir / ".stateless").exists())
        self.assertFalse((self.out_dir / "fit_checkpoint.pkl").exists())

        # 3. targets.parquet has 1 row (k=4, 4 single-ASN probes → 1 per fold),
        # all timing fields populated.
        table = pq.read_table(self.out_dir / "targets.parquet")
        self.assertEqual(table.num_rows, 1)
        row = table.to_pylist()[0]
        # target_id is a prb_id from the CSV ("1001".."1004"), not an anchor IP.
        self.assertIn(row["target_id"], {"1001", "1002", "1003", "1004"})
        self.assertEqual(row["status"], "SUCCESS")
        self.assertIsNotNone(row["error_km"])
        self.assertGreater(row["ltd_ms"], 0.0)
        self.assertIsNotNone(row["mtl_ms"])
        self.assertIsNotNone(row["ctr_ms"])
        # Both memory channels present on every stage. alloc_peak is
        # tracemalloc (always >= 0); rss_peak is sampler delta (may be 0
        # for sub-5ms stages — assert presence, not magnitude).
        self.assertGreaterEqual(row["ltd_alloc_peak_bytes"], 0)
        self.assertGreaterEqual(row["ltd_rss_peak_bytes"], 0)
        self.assertGreaterEqual(row["mtl_alloc_peak_bytes"], 0)
        self.assertGreaterEqual(row["mtl_rss_peak_bytes"], 0)
        self.assertGreaterEqual(row["ctr_alloc_peak_bytes"], 0)
        self.assertGreaterEqual(row["ctr_rss_peak_bytes"], 0)
        # Under anchors_to_probes the lone anchor (1.1.1.1) is the single VP
        # observing each probe target → n_obs == 1.
        self.assertEqual(row["n_obs"], 1)
        self.assertEqual(len(row["ltd_predictions"]), 1)
        # Each per-VP prediction must have a stamped vp_id and a finite upper_km.
        for pred in row["ltd_predictions"]:
            self.assertTrue(pred["vp_id"])
            self.assertTrue(pred["success"])
            self.assertGreater(pred["upper_km"], 0)

    def test_seed_recorded_and_makes_stochastic_combo_deterministic(self) -> None:
        """Same base_seed → byte-identical predictions on a stochastic combo."""
        from scripts.benchmark.v2.sources.generic_csv import GenericCSVSource as _Src

        # Need a richer fixture: MonteCarloMedoidCTR over PlanarAnnulusMTL
        # requires an annular feasible region. Use the NormalDist LTD which
        # fits per-VP normals.
        spec = ComboSpec(
            combo_id="mc_combo",
            ltd="normal_dist", mtl="planar_annulus", ctr="monte_carlo_medoid",
            # 3 fit pairs (k=4 → 3 fit probes vs 1 eval); deg_mu=1/deg_sigma=0
            # keeps the polyfit well-determined at n_bins=2.
            ltd_kwargs={
                "cutoff_min_points": 1, "min_per_bin": 1, "n_bins": 2,
                "deg_mu": 1, "deg_sigma": 0,
            },
            mtl_kwargs={}, ctr_kwargs={"n_samples": 256},
            base_seed=42,
        )
        out_a = outputs_combo_dir(self.out_dir.parents[3], "test-run", self.src, "mc_a")
        out_b = outputs_combo_dir(self.out_dir.parents[3], "test-run", self.src, "mc_b")
        run_one_combo(
            spec, inputs_dir=self.inputs_dir, out_dir=out_a,
            run_id="seed-test", source_name="generic_csv", slice_name="fold_0",
        )
        run_one_combo(
            spec, inputs_dir=self.inputs_dir, out_dir=out_b,
            run_id="seed-test", source_name="generic_csv", slice_name="fold_0",
        )

        # seed column populated (one row per target).
        ta = pq.read_table(out_a / "targets.parquet").to_pylist()
        tb = pq.read_table(out_b / "targets.parquet").to_pylist()
        self.assertEqual(len(ta), 1)
        self.assertIsNotNone(ta[0]["seed"])
        # Determinism: identical seed → identical prediction (status, coord).
        self.assertEqual(ta[0]["status"], tb[0]["status"])
        if ta[0]["status"] == "SUCCESS":
            self.assertEqual(ta[0]["pred_lat"], tb[0]["pred_lat"])
            self.assertEqual(ta[0]["pred_lon"], tb[0]["pred_lon"])

        # base_seed echoed into run.json.
        meta_a = json.loads((out_a / "run.json").read_text())
        self.assertEqual(meta_a["base_seed"], 42)

    def test_seed_none_leaves_column_null(self) -> None:
        spec = ComboSpec(
            combo_id="mc_no_seed",
            ltd="speed_of_internet", mtl="planar_circle", ctr="geometric_centroid",
            ltd_kwargs={}, mtl_kwargs={}, ctr_kwargs={},
            base_seed=None,
        )
        out_dir = outputs_combo_dir(self.out_dir.parents[3], "no-seed-test", self.src, "mc_no_seed")
        run_one_combo(
            spec, inputs_dir=self.inputs_dir, out_dir=out_dir,
            run_id="no-seed-test", source_name="generic_csv", slice_name="fold_0",
        )
        rows = pq.read_table(out_dir / "targets.parquet").to_pylist()
        self.assertEqual(rows[0]["seed"], None)
        meta = json.loads((out_dir / "run.json").read_text())
        self.assertIsNone(meta["base_seed"])

    def test_stateful_ltd_writes_pickle_checkpoint(self) -> None:
        spec = ComboSpec(
            combo_id="combo_le",
            ltd="low_envelope", mtl="planar_circle", ctr="geometric_centroid",
            ltd_kwargs={}, mtl_kwargs={}, ctr_kwargs={},
        )
        out_dir = outputs_combo_dir(self.out_dir.parents[3], "test-run", self.src, "combo_le")
        run_one_combo(
            spec, inputs_dir=self.inputs_dir, out_dir=out_dir,
            run_id="test-run", source_name="generic_csv", slice_name="fold_0",
        )
        # LowEnvelopeLTD carries per-VP fitted state → pickle should exist.
        pickle_path = out_dir / "fit_checkpoint.pkl"
        self.assertTrue(pickle_path.exists())
        self.assertFalse((out_dir / ".stateless").exists())
        with open(pickle_path, "rb") as fh:
            ltd = pickle.load(fh)
        self.assertEqual(type(ltd).__name__, "LowEnvelopeLTD")


class TestPairWeightEval(unittest.TestCase):
    """pair_weight round-trip through materialize + the runner's
    --pair-weight-min traffic-weighted eval filter."""

    # Two targets in one city: 2001 has one heavy (8.0) and one light (2.0)
    # obs; 2002 has only light obs (1.0, 1.5). At threshold 5.0, 2001 keeps
    # a single obs and 2002 drops out of the eval set entirely.
    _CSV = textwrap.dedent("""
        vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,target_city,rtt_ms,pair_weight
        1.1.1.1,33.0,-84.0,2001,33.5,-84.5,atlanta,5.0,8.0
        2.2.2.2,34.0,-85.0,2001,33.5,-84.5,atlanta,6.0,2.0
        1.1.1.1,33.0,-84.0,2002,32.5,-83.5,atlanta,6.5,1.0
        2.2.2.2,34.0,-85.0,2002,32.5,-83.5,atlanta,7.0,1.5
    """).strip() + "\n"

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.csv_path = root / "weighted.csv"
        self.csv_path.write_text(self._CSV)
        self.src = GenericCSVSource(
            slice="all", setup="anchors_to_probes", csv_path=self.csv_path,
        )
        self.root = root
        self.inputs_dir = materialize_inputs(
            self.src, root=root / "inputs", run_id="pw-test",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _spec(self, combo_id: str, pair_weight_min=None) -> ComboSpec:
        return ComboSpec(
            combo_id=combo_id,
            ltd="speed_of_internet", mtl="planar_circle", ctr="geometric_centroid",
            ltd_kwargs={}, mtl_kwargs={}, ctr_kwargs={},
            pair_weight_min=pair_weight_min,
        )

    def test_pair_weight_round_trips_through_parquet(self) -> None:
        from scripts.benchmark.v2.inputs import load_eval_targets_parquet

        table = pq.read_table(self.inputs_dir / "eval_observations.parquet")
        self.assertIn("pair_weight", table.column_names)
        self.assertEqual(
            sorted(table.column("pair_weight").to_pylist()),
            [1.0, 1.5, 2.0, 8.0],
        )
        targets = load_eval_targets_parquet(
            self.inputs_dir / "eval_observations.parquet"
        )
        by_id = {t.target_id: t for t in targets}
        assert by_id["2001"].obs_weights is not None
        self.assertEqual(sorted(by_id["2001"].obs_weights), [2.0, 8.0])

    def test_unweighted_source_defaults_weight_to_one(self) -> None:
        csv_path = self.root / "no_weight.csv"
        csv_path.write_text(_SYNTH_CSV)
        src = GenericCSVSource(
            slice="all", setup="anchors_to_probes", csv_path=csv_path,
        )
        inputs_dir = materialize_inputs(src, root=self.root / "inputs2", run_id="pw-test")
        table = pq.read_table(inputs_dir / "eval_observations.parquet")
        self.assertEqual(set(table.column("pair_weight").to_pylist()), {1.0})

    def test_pair_weight_min_filters_obs_and_drops_targets(self) -> None:
        out_dir = outputs_combo_dir(
            self.root / "outputs", "pw-test", self.src, "pw_combo",
        )
        run_one_combo(
            self._spec("pw_combo", pair_weight_min=5.0),
            inputs_dir=self.inputs_dir, out_dir=out_dir,
            run_id="pw-test", source_name="generic_csv", slice_name="all",
        )
        meta = json.loads((out_dir / "run.json").read_text())
        self.assertEqual(meta["pair_weight_min"], 5.0)
        # 2002 has no obs >= 5.0 → excluded from the test set entirely.
        self.assertEqual(meta["n_targets"], 1)
        self.assertEqual(meta["n_targets_dropped_below_min_weight"], 1)

        rows = pq.read_table(out_dir / "targets.parquet").to_pylist()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target_id"], "2001")
        # Only the 8.0-weight obs survives the threshold.
        self.assertEqual(rows[0]["n_obs"], 1)
        self.assertEqual(len(rows[0]["ltd_predictions"]), 1)
        self.assertEqual(rows[0]["ltd_predictions"][0]["vp_id"], "1.1.1.1")

    def test_no_threshold_keeps_every_obs(self) -> None:
        out_dir = outputs_combo_dir(
            self.root / "outputs", "pw-test", self.src, "pw_all",
        )
        run_one_combo(
            self._spec("pw_all", pair_weight_min=None),
            inputs_dir=self.inputs_dir, out_dir=out_dir,
            run_id="pw-test", source_name="generic_csv", slice_name="all",
        )
        meta = json.loads((out_dir / "run.json").read_text())
        self.assertIsNone(meta["pair_weight_min"])
        self.assertIsNone(meta["n_targets_dropped_below_min_weight"])
        self.assertEqual(meta["n_targets"], 2)
        rows = pq.read_table(out_dir / "targets.parquet").to_pylist()
        self.assertEqual({r["n_obs"] for r in rows}, {2})

    def test_threshold_emptying_eval_set_raises(self) -> None:
        out_dir = outputs_combo_dir(
            self.root / "outputs", "pw-test", self.src, "pw_too_high",
        )
        with self.assertRaises(ValueError):
            run_one_combo(
                self._spec("pw_too_high", pair_weight_min=100.0),
                inputs_dir=self.inputs_dir, out_dir=out_dir,
                run_id="pw-test", source_name="generic_csv", slice_name="all",
            )

    def test_materialize_time_eval_weight_filter_bakes_into_parquet(self) -> None:
        """eval_pair_weight_min at materialize time: the parquet itself holds
        only clearing obs (2001's 8.0 row), dropped targets are absent, the
        manifest counts reflect the mask — and fit_samples stay full-mesh,
        identical to the unfiltered materialization."""
        src = GenericCSVSource(
            slice="all", setup="anchors_to_probes", csv_path=self.csv_path,
            eval_pair_weight_min=5.0,
        )
        inputs_dir = materialize_inputs(
            src, root=self.root / "inputs_masked", run_id="pw-test",
        )

        table = pq.read_table(inputs_dir / "eval_observations.parquet")
        self.assertEqual(table.column("target_id").to_pylist(), ["2001"])
        self.assertEqual(table.column("pair_weight").to_pylist(), [8.0])
        self.assertEqual(table.column("vp_id").to_pylist(), ["1.1.1.1"])

        manifest = json.loads((inputs_dir / "manifest.json").read_text())
        self.assertEqual(manifest["n_eval_targets"], 1)
        self.assertEqual(manifest["n_eval_observations"], 1)

        fit_masked = pq.read_table(inputs_dir / "fit_samples.parquet")
        fit_full = pq.read_table(self.inputs_dir / "fit_samples.parquet")
        self.assertTrue(fit_masked.equals(fit_full))


if __name__ == "__main__":
    unittest.main()
