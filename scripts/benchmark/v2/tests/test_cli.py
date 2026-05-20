"""CLI tests — use Typer's CliRunner to invoke commands in-process."""

from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

import pyarrow.parquet as pq
from typer.testing import CliRunner

from scripts.benchmark.v2.cli import app


_SYNTH_CSV = textwrap.dedent("""
    src_ip,dst_ip,prb_id,min_rtt,mean_rtt,sent,rcvd,msm_id,date,probe_asn,probe_country,probe_latitude,probe_longitude,anchor_asn,anchor_country,anchor_latitude,anchor_longitude,anchor_city
    10.0.0.1,1.1.1.1,1001,5.0,5.0,3,3,1,2023-05-01,7922,US,33.5,-84.5,20473,US,33.0,-84.0,Atlanta
    10.0.0.2,1.1.1.1,1002,6.0,6.0,3,3,2,2023-05-01,7922,US,32.5,-83.5,20473,US,33.0,-84.0,Atlanta
    10.0.0.3,1.1.1.1,1003,7.0,7.0,3,3,3,2023-05-01,7922,US,33.5,-83.5,20473,US,33.0,-84.0,Atlanta
    10.0.0.4,1.1.1.1,1004,5.5,5.5,3,3,4,2023-05-01,7922,US,32.5,-84.5,20473,US,33.0,-84.0,Atlanta
    10.0.0.5,2.2.2.2,1005,8.0,8.0,3,3,5,2023-05-01,7922,US,46.5,-122.5,40,US,47.0,-122.0,Seattle
    10.0.0.6,2.2.2.2,1006,9.0,9.0,3,3,6,2023-05-01,7922,US,47.5,-122.5,40,US,47.0,-122.0,Seattle
    10.0.0.7,2.2.2.2,1007,7.5,7.5,3,3,7,2023-05-01,7922,US,47.5,-121.5,40,US,47.0,-122.0,Seattle
""").strip() + "\n"


class TestCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.csv_path = root / "vultr.csv"
        self.csv_path.write_text(_SYNTH_CSV)
        self.inputs_root = root / "inputs"
        self.outputs_root = root / "outputs"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _materialize(self, source: str = "vultr_csv", slice: str = "all_us") -> Path:
        # Patch the default CSV path by setting the env variable on the source.
        # Simpler: call the source directly with our synth path via inputs.materialize_inputs.
        from scripts.benchmark.v2.inputs import materialize_inputs
        from scripts.benchmark.v2.sources.vultr_csv import VultrCSVSource

        src = VultrCSVSource(slice=slice, csv_path=self.csv_path)
        return materialize_inputs(src, root=self.inputs_root)

    def test_run_combo_command_writes_outputs(self) -> None:
        self._materialize()
        result = self.runner.invoke(app, [
            "run-combo",
            "--source", "vultr_csv", "--slice", "all_us",
            "--ltd", "speed_of_internet", "--mtl", "planar_circle", "--ctr", "geometric_centroid",
            "--run-id", "cli-test",
            "--inputs-root", str(self.inputs_root),
            "--outputs-root", str(self.outputs_root),
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        combo_dir = (
            self.outputs_root / "cli-test" / "vultr_csv" / "probes_to_anchors" / "all_us"
            / "speed_of_internet__planar_circle__geometric_centroid"
        )
        self.assertTrue((combo_dir / "run.json").exists())
        self.assertTrue((combo_dir / "targets.parquet").exists())

    def test_run_combo_fails_without_materialized_inputs(self) -> None:
        result = self.runner.invoke(app, [
            "run-combo",
            "--source", "vultr_csv", "--slice", "missing",
            "--ltd", "speed_of_internet", "--mtl", "planar_circle", "--ctr", "geometric_centroid",
            "--run-id", "cli-test",
            "--inputs-root", str(self.inputs_root),
            "--outputs-root", str(self.outputs_root),
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("materialize-inputs", result.output)

    def test_summarize_aggregates_combos(self) -> None:
        self._materialize()
        # Run two combos under one run id.
        for ltd_name in ("speed_of_internet", "low_envelope"):
            r = self.runner.invoke(app, [
                "run-combo",
                "--source", "vultr_csv", "--slice", "all_us",
                "--ltd", ltd_name, "--mtl", "planar_circle", "--ctr", "geometric_centroid",
                "--run-id", "sum-test",
                "--inputs-root", str(self.inputs_root),
                "--outputs-root", str(self.outputs_root),
            ])
            self.assertEqual(r.exit_code, 0, msg=r.output)

        result = self.runner.invoke(app, [
            "summarize",
            "--run-id", "sum-test",
            "--outputs-root", str(self.outputs_root),
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        summary_path = self.outputs_root / "sum-test" / "summary.parquet"
        self.assertTrue(summary_path.exists())
        table = pq.read_table(summary_path)
        self.assertEqual(table.num_rows, 2)
        combos = set(table.column("ltd").to_pylist())
        self.assertEqual(combos, {"speed_of_internet", "low_envelope"})


if __name__ == "__main__":
    unittest.main()
