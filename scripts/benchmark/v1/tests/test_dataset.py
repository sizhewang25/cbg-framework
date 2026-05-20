from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.benchmark.v1.dataset import (
    CSVDataLoader,
    build_dataset_specs,
    materialize_dataset,
    rank_top_asns,
    select_dataset,
)
from scripts.libs.core.combinations import SPECS_BY_ID


class DatasetSelectionTests(unittest.TestCase):
    def test_rank_top_asns_uses_probe_count_then_row_count(self):
        df = _sample_df()

        self.assertEqual(rank_top_asns(df, top_n=3), [100, 200, 300])

    def test_select_dataset_is_cumulative_top_k(self):
        df = _sample_df()

        selected, spec = select_dataset(df, "top2", max_top_k=3)

        self.assertEqual(spec.selected_asns, (100, 200))
        self.assertEqual(spec.n_probes, 4)
        self.assertEqual(spec.n_anchors, 2)
        self.assertIn("distance_km", selected.columns)
        self.assertEqual(set(selected["probe_asn"]), {100.0, 200.0})

    def test_all_us_keeps_all_rows(self):
        df = _sample_df()

        selected, spec = select_dataset(df, "all_us", max_top_k=3)

        self.assertIsNone(spec.selected_asns)
        self.assertEqual(spec.n_rows, len(df))
        self.assertEqual(spec.n_probes, df["src_ip"].nunique())
        self.assertEqual(len(selected), len(df))

    def test_materialize_dataset_writes_csv_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_csv = tmp_path / "input.csv"
            output_csv = tmp_path / "selected.csv"
            manifest = tmp_path / "selected.json"
            _sample_df().to_csv(input_csv, index=False)

            spec = materialize_dataset("top1", input_csv, output_csv, manifest, max_top_k=3)

            materialized = pd.read_csv(output_csv)
            manifest_data = json.loads(manifest.read_text())
            self.assertEqual(spec.selected_asns, (100,))
            self.assertIn("distance_km", materialized.columns)
            self.assertEqual(manifest_data["dataset_id"], "top1")
            self.assertEqual(manifest_data["n_anchors"], 2)

    def test_csv_data_loader_preselected_builds_framework_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp) / "selected.csv"
            selected, _ = select_dataset(_sample_df(), "top1", max_top_k=3)
            selected.to_csv(input_csv, index=False)
            loader = CSVDataLoader(input_csv, "top1", preselected=True)

            prepared = loader(SPECS_BY_ID["S1"], None, {})

            self.assertEqual(len(prepared.anchor_coords), 2)
            self.assertEqual(len(prepared.probe_targets), 2)
            self.assertTrue(prepared.data_fingerprint)
            self.assertEqual(loader.manifest()["n_probes"], 2)

    def test_build_dataset_specs_does_not_require_materialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp) / "input.csv"
            _sample_df().to_csv(input_csv, index=False)

            specs = build_dataset_specs(input_csv, max_top_k=3)

            self.assertEqual([spec.dataset_id for spec in specs], ["top1", "top2", "top3", "all_us"])
            self.assertEqual(specs[-1].n_probes, 5)


def _sample_df() -> pd.DataFrame:
    rows = [
        _row("p1", "a1", 100, 10.0),
        _row("p1", "a2", 100, 11.0),
        _row("p2", "a1", 100, 12.0),
        _row("p3", "a1", 200, 13.0),
        _row("p3", "a2", 200, 14.0),
        _row("p4", "a1", 200, 15.0),
        _row("p5", "a2", 300, 16.0),
    ]
    return pd.DataFrame(rows)


def _row(src_ip: str, dst_ip: str, asn: int, rtt: float) -> dict:
    anchor_coords = {
        "a1": (47.6, -122.3, "Seattle"),
        "a2": (40.7, -74.0, "New York"),
    }
    anchor_lat, anchor_lon, city = anchor_coords[dst_ip]
    return {
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "prb_id": int(src_ip.removeprefix("p")),
        "min_rtt": rtt,
        "mean_rtt": rtt + 1,
        "probe_asn": float(asn),
        "probe_country": "US",
        "probe_latitude": 39.0 + int(src_ip.removeprefix("p")),
        "probe_longitude": -95.0,
        "anchor_asn": 20473,
        "anchor_country": "US",
        "anchor_latitude": anchor_lat,
        "anchor_longitude": anchor_lon,
        "anchor_city": city,
    }


if __name__ == "__main__":
    unittest.main()

