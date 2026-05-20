from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts.benchmark.v1.dataset import CSVDataLoader
from scripts.benchmark.v1.runner import (
    collect_summaries,
    default_run_output_dir,
    make_run_id,
    _write_combo_checkpoint,
    _write_incremental_outputs,
)
from scripts.libs.core.benchmarking import BenchmarkRecorder
from scripts.libs.core.combinations import SPECS_BY_ID
from scripts.libs.core.evaluate import ProbeResult, SettingEvaluation


class BenchmarkRunnerCheckpointTests(unittest.TestCase):
    def test_make_run_id_uses_utc_microsecond_timestamp(self):
        now = datetime(2026, 5, 1, 12, 34, 56, 789012, tzinfo=timezone.utc)

        self.assertEqual(make_run_id(now), "20260501T123456789012Z")

    def test_default_run_output_dir_uses_run_id_root(self):
        output_dir = default_run_output_dir(
            "top1",
            output_root=Path("/tmp/benchmark"),
            run_id="smoke",
        )

        self.assertEqual(output_dir, Path("/tmp/benchmark/runs/smoke/top1"))

    def test_write_combo_checkpoint_persists_probe_results_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            artifact = _artifact("S1")

            _write_combo_checkpoint(
                output_dir,
                artifact,
                {"dataset_id": "top1", "n_probes": 1},
                run_id="smoke",
                run_started_at_utc="2026-05-01T00:00:00+00:00",
            )

            results_path = output_dir / "checkpoints" / "S1_probe_results.csv"
            checkpoint_path = output_dir / "checkpoints" / "S1_checkpoint.json"
            self.assertTrue(results_path.exists())
            self.assertTrue(checkpoint_path.exists())

            with open(results_path, newline="") as f:
                rows = list(csv.DictReader(f))
            checkpoint = json.loads(checkpoint_path.read_text())
            self.assertEqual(rows[0]["probe_ip"], "probe-a")
            self.assertEqual(checkpoint["combo_id"], "S1")
            self.assertEqual(checkpoint["run_id"], "smoke")
            self.assertEqual(checkpoint["n_probes"], 1)
            self.assertEqual(checkpoint["intersection_count"], 1)

    def test_incremental_outputs_write_partial_summary_and_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            input_csv = output_dir / "input.csv"
            input_csv.write_text("placeholder\n")
            loader = CSVDataLoader(input_csv, "top1", preselected=True)
            artifact = _artifact("S1")
            all_results = {"S1": artifact.results}
            artifacts_by_combo = {"S1": artifact}
            completed_specs = [SPECS_BY_ID["S1"]]
            requested_specs = [SPECS_BY_ID["S1"], SPECS_BY_ID["S2"]]

            _write_incremental_outputs(
                output_dir=output_dir,
                input_csv=input_csv,
                data_loader=loader,
                benchmark_recorder=BenchmarkRecorder(),
                all_results=all_results,
                artifacts_by_combo=artifacts_by_combo,
                completed_specs=completed_specs,
                requested_specs=requested_specs,
                is_complete=False,
                run_id="smoke",
                run_started_at_utc="2026-05-01T00:00:00+00:00",
            )

            summary = json.loads((output_dir / "evaluation_summary.json").read_text())
            progress = json.loads((output_dir / "checkpoints" / "progress.json").read_text())
            self.assertEqual(list(summary["combinations"]), ["S1"])
            self.assertEqual(summary["dataset_metadata"]["run_id"], "smoke")
            self.assertFalse(summary["dataset_metadata"]["checkpoint"]["is_complete"])
            self.assertEqual(progress["run_id"], "smoke")
            self.assertEqual(progress["completed_combo_ids"], ["S1"])
            self.assertEqual(progress["requested_combo_ids"], ["S1", "S2"])
            self.assertFalse(progress["is_complete"])
            self.assertTrue((output_dir / "benchmark_phase_raw.csv").exists())
            self.assertTrue((output_dir / "benchmark_phase_summary.json").exists())

    def test_collect_summaries_finds_nested_run_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_summary(
                root / "top1" / "evaluation_summary.json",
                dataset_id="top1",
                run_id=None,
            )
            _write_summary(
                root / "runs" / "run-a" / "top2" / "evaluation_summary.json",
                dataset_id="top2",
                run_id="run-a",
            )

            rows = collect_summaries(root)

            self.assertEqual([row["dataset_id"] for row in rows], ["top1", "top2"])
            self.assertIsNone(rows[0]["run_id"])
            self.assertEqual(rows[1]["run_id"], "run-a")
            self.assertEqual(rows[1]["combo_ids"], "S1,S2")


def _artifact(combo_id: str) -> SettingEvaluation:
    result = ProbeResult(
        probe_ip="probe-a",
        true_lat=40.0,
        true_lon=-75.0,
        estimated_lat=40.1,
        estimated_lon=-75.1,
        error_km=10.0,
        n_circles=2,
        min_rtt_ms=5.0,
        did_intersect=True,
        fallback_used=False,
        fallback_reason=None,
    )
    return SettingEvaluation(
        spec=SPECS_BY_ID[combo_id],
        results=[result],
        anchor_coords={"anchor-a": (39.0, -76.0)},
        probe_targets={
            "probe-a": {
                "measurements": {"anchor-a": 5.0},
                "true_lat": 40.0,
                "true_lon": -75.0,
            }
        },
        lp_models={},
        octant_models={},
        octant_delta=None,
        data_fingerprint="abc123",
        benchmark_ms={"setting_total_ms": 1.0},
    )


def _write_summary(path: Path, dataset_id: str, run_id: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset_metadata": {
                    "dataset_id": dataset_id,
                    "run_id": run_id,
                    "run_started_at_utc": "2026-05-01T00:00:00+00:00",
                    "run_output_dir": str(path.parent),
                    "n_rows": 10,
                    "n_probes": 2,
                    "n_anchors": 7,
                },
                "n_combinations": 2,
                "combinations": {"S1": {}, "S2": {}},
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    unittest.main()
