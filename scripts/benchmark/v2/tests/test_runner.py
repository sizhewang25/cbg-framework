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

from scripts.benchmark.v2.inputs import materialize_inputs
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
        src = VultrCSVSource(slice="all_us", csv_path=self.csv_path)
        self.inputs_dir = materialize_inputs(src, root=root / "inputs")
        self.out_dir = root / "outputs" / "test-run" / src.name / src.slice_id() / "combo1"

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
            run_id="test-run", source_name="vultr_csv", slice_name="all_us",
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

        # 3. targets.parquet has 1 row, all timing fields populated.
        table = pq.read_table(self.out_dir / "targets.parquet")
        self.assertEqual(table.num_rows, 1)
        row = table.to_pylist()[0]
        self.assertEqual(row["target_id"], "1.1.1.1")
        self.assertEqual(row["status"], "SUCCESS")
        self.assertIsNotNone(row["error_km"])
        self.assertGreater(row["ltd_ms"], 0.0)
        self.assertIsNotNone(row["mtl_ms"])
        self.assertIsNotNone(row["ctr_ms"])
        self.assertEqual(row["n_obs"], 4)
        self.assertEqual(len(row["ltd_predictions"]), 4)
        # Each per-VP prediction must have a stamped vp_id and a finite upper_km.
        for pred in row["ltd_predictions"]:
            self.assertTrue(pred["vp_id"])
            self.assertTrue(pred["success"])
            self.assertGreater(pred["upper_km"], 0)

    def test_stateful_ltd_writes_pickle_checkpoint(self) -> None:
        spec = ComboSpec(
            combo_id="combo_le",
            ltd="low_envelope", mtl="planar_circle", ctr="geometric_centroid",
            ltd_kwargs={}, mtl_kwargs={}, ctr_kwargs={},
        )
        out_dir = self.out_dir.parent / "combo_le"
        run_one_combo(
            spec, inputs_dir=self.inputs_dir, out_dir=out_dir,
            run_id="test-run", source_name="vultr_csv", slice_name="all_us",
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
