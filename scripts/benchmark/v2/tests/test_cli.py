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


# Canonical-schema synth CSV: vp_* = anchor side (acting as VP),
# target_* = probe side (the entity being geolocated).
_SYNTH_CSV = textwrap.dedent("""
    vp_id,vp_lat,vp_lon,vp_asn,vp_country,target_id,target_lat,target_lon,target_asn,target_country,rtt_ms
    1.1.1.1,33.0,-84.0,20473,US,1001,33.5,-84.5,7922,US,5.0
    1.1.1.1,33.0,-84.0,20473,US,1002,32.5,-83.5,7922,US,6.0
    1.1.1.1,33.0,-84.0,20473,US,1003,33.5,-83.5,7922,US,7.0
    1.1.1.1,33.0,-84.0,20473,US,1004,32.5,-84.5,7922,US,5.5
    2.2.2.2,47.0,-122.0,40,US,1005,46.5,-122.5,7922,US,8.0
    2.2.2.2,47.0,-122.0,40,US,1006,47.5,-122.5,7922,US,9.0
    2.2.2.2,47.0,-122.0,40,US,1007,47.5,-121.5,7922,US,7.5
""").strip() + "\n"


class TestCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.csv_path = root / "canonical.csv"
        self.csv_path.write_text(_SYNTH_CSV)
        self.inputs_root = root / "inputs"
        self.outputs_root = root / "outputs"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _materialize(
        self, source: str = "generic_csv", slice: str = "fold_0", run_id: str = "cli-test",
    ) -> Path:
        # Call the source directly with our synth path via inputs.materialize_inputs.
        from scripts.benchmark.v2.inputs import materialize_inputs
        from scripts.benchmark.v2.sources.generic_csv import GenericCSVSource

        src = GenericCSVSource(
            slice=slice, setup="anchors_to_probes",
            csv_path=self.csv_path, k=4,
        )
        return materialize_inputs(src, root=self.inputs_root, run_id=run_id)

    def test_run_combo_command_writes_outputs(self) -> None:
        self._materialize()
        result = self.runner.invoke(app, [
            "run-combo",
            "--source", "generic_csv", "--slice", "fold_0",
            "--setup", "anchors_to_probes",
            "--ltd", "speed_of_internet", "--mtl", "planar_circle", "--ctr", "geometric_centroid",
            "--run-id", "cli-test",
            "--inputs-root", str(self.inputs_root),
            "--outputs-root", str(self.outputs_root),
            "--source-kwargs", json.dumps({"csv_path": str(self.csv_path), "k": 4}),
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        combo_dir = (
            self.outputs_root / "cli-test" / "generic_csv" / "anchors_to_probes" / "fold_0"
            / "speed_of_internet__planar_circle__geometric_centroid"
        )
        self.assertTrue((combo_dir / "run.json").exists())
        self.assertTrue((combo_dir / "targets.parquet").exists())

    def test_run_combo_fails_without_materialized_inputs(self) -> None:
        result = self.runner.invoke(app, [
            "run-combo",
            "--source", "generic_csv", "--slice", "fold_2",
            "--setup", "anchors_to_probes",
            "--ltd", "speed_of_internet", "--mtl", "planar_circle", "--ctr", "geometric_centroid",
            "--run-id", "cli-test",
            "--inputs-root", str(self.inputs_root),
            "--outputs-root", str(self.outputs_root),
            "--source-kwargs", json.dumps({"csv_path": str(self.csv_path), "k": 4}),
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("materialize-inputs", result.output)

    def test_materialize_forwards_source_kwargs(self) -> None:
        """`--source-kwargs` JSON is parsed and forwarded as **kwargs to the
        source constructor — exercised here with a generic_csv override that
        points at a temp CSV."""
        alt_csv = Path(self.tmp.name) / "alt.csv"
        alt_csv.write_text(_SYNTH_CSV)
        # Path the source through --source-kwargs rather than positional args.
        result = self.runner.invoke(app, [
            "materialize-inputs",
            "--source", "generic_csv", "--slice", "fold_0",
            "--setup", "anchors_to_probes",
            "--run-id", "kw-test",
            "--inputs-root", str(self.inputs_root),
            "--source-kwargs", json.dumps({"csv_path": str(alt_csv), "k": 4}),
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        manifest = (
            self.inputs_root / "generic_csv" / "kw-test" / "anchors_to_probes" / "fold_0"
            / "manifest.json"
        )
        self.assertTrue(manifest.exists())

    def test_materialize_rejects_invalid_source_kwargs_json(self) -> None:
        result = self.runner.invoke(app, [
            "materialize-inputs",
            "--source", "generic_csv", "--slice", "fold_0",
            "--setup", "anchors_to_probes",
            "--run-id", "kw-test",
            "--inputs-root", str(self.inputs_root),
            "--source-kwargs", "not-json",
        ])
        self.assertNotEqual(result.exit_code, 0)

    def test_summarize_aggregates_combos(self) -> None:
        self._materialize(run_id="sum-test")
        # Run two combos under one run id.
        for ltd_name in ("speed_of_internet", "low_envelope"):
            r = self.runner.invoke(app, [
                "run-combo",
                "--source", "generic_csv", "--slice", "fold_0",
                "--setup", "anchors_to_probes",
                "--ltd", ltd_name, "--mtl", "planar_circle", "--ctr", "geometric_centroid",
                "--run-id", "sum-test",
                "--inputs-root", str(self.inputs_root),
                "--outputs-root", str(self.outputs_root),
                "--source-kwargs", json.dumps({"csv_path": str(self.csv_path), "k": 4}),
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

    def _write_tiny_airports(self) -> Path:
        """A hermetic 3-airport reference parquet so the test doesn't depend on
        the (uncommitted, regenerated) full OurAirports artifact."""
        import pandas as pd
        path = Path(self.tmp.name) / "airports.parquet"
        pd.DataFrame({
            "iata_code": ["ATL", "SEA", "LHR"],
            "latitude_deg": [33.6407, 47.4502, 51.4700],
            "longitude_deg": [-84.4277, -122.3088, -0.4543],
            "municipality": ["Atlanta", "Seattle", "London"],
        }).to_parquet(path, index=False)
        return path

    def test_airport_eval_annotates_targets_and_writes_summary(self) -> None:
        from scripts.benchmark.v2.airport_eval import AIRPORT_COLUMNS

        airports = self._write_tiny_airports()
        self._materialize(run_id="ap-test")
        r = self.runner.invoke(app, [
            "run-combo",
            "--source", "generic_csv", "--slice", "fold_0",
            "--setup", "anchors_to_probes",
            "--ltd", "speed_of_internet", "--mtl", "planar_circle", "--ctr", "geometric_centroid",
            "--run-id", "ap-test",
            "--inputs-root", str(self.inputs_root),
            "--outputs-root", str(self.outputs_root),
            "--source-kwargs", json.dumps({"csv_path": str(self.csv_path), "k": 4}),
        ])
        self.assertEqual(r.exit_code, 0, msg=r.output)

        result = self.runner.invoke(app, [
            "airport-eval",
            "--run-id", "ap-test",
            "--outputs-root", str(self.outputs_root),
            "--airports", str(airports),
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)

        combo_dir = (
            self.outputs_root / "ap-test" / "generic_csv" / "anchors_to_probes" / "fold_0"
            / "speed_of_internet__planar_circle__geometric_centroid"
        )
        cols = set(pq.read_table(combo_dir / "targets.parquet").column_names)
        for col in AIRPORT_COLUMNS:
            self.assertIn(col, cols)

        summary_path = self.outputs_root / "ap-test" / "airport_summary.parquet"
        self.assertTrue(summary_path.exists())
        summ = pq.read_table(summary_path)
        self.assertEqual(summ.num_rows, 1)
        self.assertIn("airport_match_rate", summ.column_names)


if __name__ == "__main__":
    unittest.main()
