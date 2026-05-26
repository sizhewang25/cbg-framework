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
from scripts.benchmark.v2.sources.vultr_csv import VultrCSVSource


_SYNTH_CSV = textwrap.dedent("""
    src_ip,dst_ip,prb_id,min_rtt,mean_rtt,sent,rcvd,msm_id,date,probe_asn,probe_country,probe_latitude,probe_longitude,anchor_asn,anchor_country,anchor_latitude,anchor_longitude,anchor_city
    10.0.0.1,1.1.1.1,1001,5.0,5.0,3,3,1,2023-05-01,7922,US,33.5,-84.5,20473,US,33.0,-84.0,Atlanta
    10.0.0.2,1.1.1.1,1002,6.0,6.0,3,3,2,2023-05-01,7922,US,32.5,-83.5,20473,US,33.0,-84.0,Atlanta
    10.0.0.3,1.1.1.1,1003,7.0,7.0,3,3,3,2023-05-01,7922,US,33.5,-83.5,20473,US,33.0,-84.0,Atlanta
    10.0.0.4,1.1.1.1,1004,5.5,5.5,3,3,4,2023-05-01,7922,US,32.5,-84.5,20473,US,33.0,-84.0,Atlanta
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
        src = VultrCSVSource(
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
            run_id="test-run", source_name="vultr_csv", slice_name="fold_0",
        )

        # 1. run.json populated
        run_meta = json.loads((self.out_dir / "run.json").read_text())
        self.assertEqual(run_meta["combo_id"], "combo1")
        self.assertEqual(run_meta["n_targets"], 1)
        self.assertGreater(run_meta["fit_ms"], 0.0)
        self.assertGreaterEqual(run_meta["fit_peak_bytes"], 0)
        self.assertIn("status_counts", run_meta)
        self.assertGreater(run_meta["run_peak_rss_bytes"], 1_000_000)

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
        from scripts.benchmark.v2.sources.vultr_csv import VultrCSVSource as _Src

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
            run_id="seed-test", source_name="vultr_csv", slice_name="fold_0",
        )
        run_one_combo(
            spec, inputs_dir=self.inputs_dir, out_dir=out_b,
            run_id="seed-test", source_name="vultr_csv", slice_name="fold_0",
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
            run_id="no-seed-test", source_name="vultr_csv", slice_name="fold_0",
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
            run_id="test-run", source_name="vultr_csv", slice_name="fold_0",
        )
        # LowEnvelopeLTD carries per-VP fitted state → pickle should exist.
        pickle_path = out_dir / "fit_checkpoint.pkl"
        self.assertTrue(pickle_path.exists())
        self.assertFalse((out_dir / ".stateless").exists())
        with open(pickle_path, "rb") as fh:
            ltd = pickle.load(fh)
        self.assertEqual(type(ltd).__name__, "LowEnvelopeLTD")


if __name__ == "__main__":
    unittest.main()
