from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from typer.testing import CliRunner

from scripts.analysis.benchmark.cli import app
from scripts.analysis.benchmark.runner import parse_combo_ids
from scripts.analysis.benchmark.tests.test_dataset import _sample_df


class BenchmarkCliTests(unittest.TestCase):
    def test_list_datasets_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp) / "input.csv"
            _sample_df().to_csv(input_csv, index=False)

            result = CliRunner().invoke(
                app,
                [
                    "list-datasets",
                    "--input-csv",
                    str(input_csv),
                    "--max-top-k",
                    "3",
                    "--json",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            rows = json.loads(result.output)
            self.assertEqual([row["dataset_id"] for row in rows], ["top1", "top2", "top3", "all_us"])

    def test_materialize_dataset_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_csv = tmp_path / "input.csv"
            output_csv = tmp_path / "top1.csv"
            manifest = tmp_path / "top1.json"
            _sample_df().to_csv(input_csv, index=False)

            result = CliRunner().invoke(
                app,
                [
                    "materialize-dataset",
                    "top1",
                    "--input-csv",
                    str(input_csv),
                    "--output",
                    str(output_csv),
                    "--manifest-output",
                    str(manifest),
                    "--max-top-k",
                    "3",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(output_csv.exists())
            self.assertTrue(manifest.exists())
            self.assertEqual(json.loads(result.output)["dataset_id"], "top1")

    def test_parse_combo_ids_rejects_unknown_ids(self):
        with self.assertRaises(ValueError):
            parse_combo_ids("S1,NOPE")


if __name__ == "__main__":
    unittest.main()
