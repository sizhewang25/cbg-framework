"""Tests for JSON-driven benchmark summary plots."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.cbg_evaluation.plot_benchmark_summary import (
    extract_end_to_end_seconds,
    extract_intersection_rates,
    extract_max_rss_mb,
    extract_memory_mb,
    extract_phase_memory_mb,
    extract_phase_latency_seconds,
    load_per_ip_e2e_latency_ms,
    ordered_combo_ids,
    ordered_combo_ids_from_keys,
    rank_by_value_desc,
    rank_samples_by_median_desc,
)


class TestBenchmarkSummaryPlots(unittest.TestCase):
    def test_extract_end_to_end_latency_uses_seconds(self):
        summary = {
            "combinations": {"S1": {}, "B1": {}},
            "setting_benchmark_ms": {
                "S1": {"setting_total_ms": 1500.0},
                "B1": {"setting_total_ms": 32123.0},
            },
        }

        combo_ids = ordered_combo_ids(summary)
        self.assertEqual(combo_ids, ["S1", "B1"])
        self.assertEqual(
            extract_end_to_end_seconds(summary, combo_ids),
            [1.5, 32.123],
        )

    def test_extract_intersection_rate_computes_from_counts(self):
        summary = {
            "combinations": {
                "S1": {"intersection_count": 64, "n_probes": 266},
                "B1": {"intersection_count": 200, "n_probes": 250},
            },
        }

        rates = extract_intersection_rates(summary, ["S1", "B1"])

        self.assertAlmostEqual(rates[0], 64 / 266 * 100.0)
        self.assertEqual(rates[1], 80.0)

    def test_extract_phase_latency_uses_stackable_phase_totals(self):
        summary = {
            "combinations": {
                "S1": {
                    "phases": {
                        "load_data": {"total_ms": 100.0},
                        "distance_estimation": {"total_ms": 25.0},
                        "total_geolocate": {"total_ms": 999.0},
                        "setting_total": {"total_ms": 1000.0},
                    }
                }
            }
        }

        phase_seconds = extract_phase_latency_seconds(summary, ["S1"])

        self.assertEqual(phase_seconds["load_data"], [0.1])
        self.assertEqual(phase_seconds["distance_estimation"], [0.025])
        self.assertNotIn("total_geolocate", phase_seconds)
        self.assertNotIn("setting_total", phase_seconds)

    def test_extract_memory_uses_max_across_phases(self):
        summary = {
            "combinations": {
                "S1": {
                    "phases": {
                        "load_data": {
                            "max_tracemalloc_peak_mb": 10.0,
                            "max_rss_after_mb": 200.0,
                        },
                        "centroid": {
                            "max_tracemalloc_peak_mb": 14.0,
                            "max_rss_after_mb": 190.0,
                        },
                    }
                },
                "B1": {
                    "phases": {
                        "centroid": {
                            "max_tracemalloc_peak_mb": None,
                            "max_rss_after_mb": 300.0,
                        },
                    }
                },
            }
        }

        tracemalloc_mb, rss_after_mb = extract_memory_mb(summary, ["S1", "B1"])

        self.assertEqual(tracemalloc_mb, [14.0, 0.0])
        self.assertEqual(rss_after_mb, [200.0, 300.0])

    def test_extract_phase_memory_uses_phase_local_peak_delta(self):
        summary = {
            "combinations": {
                "S1": {
                    "phases": {
                        "load_data": {
                            "max_tracemalloc_phase_peak_delta_mb": 2.0,
                            "max_tracemalloc_peak_mb": 20.0,
                            "max_rss_after_mb": 100.0,
                        },
                        "centroid": {
                            "max_tracemalloc_phase_peak_delta_mb": 3.5,
                            "max_rss_after_mb": 120.0,
                        },
                    }
                },
                "B1": {
                    "phases": {
                        "load_data": {
                            "max_tracemalloc_peak_mb": 7.0,
                            "max_rss_after_mb": 300.0,
                        },
                    }
                },
            }
        }

        phase_memory = extract_phase_memory_mb(summary, ["S1", "B1"])
        rss_mb = extract_max_rss_mb(summary, ["S1", "B1"])

        self.assertEqual(phase_memory["load_data"], [2.0, 7.0])
        self.assertEqual(phase_memory["centroid"], [3.5, 0.0])
        self.assertEqual(rss_mb, [120.0, 300.0])

    def test_load_per_ip_e2e_latency_uses_total_geolocate_rows_in_ms(self):
        fieldnames = [
            "combo_id",
            "probe_ip",
            "phase",
            "elapsed_ms",
            "fallback_used",
        ]
        rows = [
            {
                "combo_id": "S1",
                "probe_ip": "probe-a",
                "phase": "total_geolocate",
                "elapsed_ms": "10.0",
                "fallback_used": "False",
            },
            {
                "combo_id": "S1",
                "probe_ip": "probe-a",
                "phase": "distance_estimation",
                "elapsed_ms": "1.0",
                "fallback_used": "",
            },
            {
                "combo_id": "S1",
                "probe_ip": "probe-b",
                "phase": "total_geolocate",
                "elapsed_ms": "20.0",
                "fallback_used": "True",
            },
            {
                "combo_id": "B1",
                "probe_ip": "probe-b",
                "phase": "total_geolocate",
                "elapsed_ms": "2500.0",
                "fallback_used": "False",
            },
            {
                "combo_id": "B1",
                "probe_ip": "",
                "phase": "setting_total",
                "elapsed_ms": "9999.0",
                "fallback_used": "",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "benchmark_phase_raw.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            latencies = load_per_ip_e2e_latency_ms(path)
            all_latencies = load_per_ip_e2e_latency_ms(
                path,
                include_fallback=True,
            )

        self.assertEqual(latencies["S1"], [10.0])
        self.assertEqual(latencies["B1"], [2500.0])
        self.assertEqual(all_latencies["S1"], [10.0, 20.0])

    def test_order_combo_ids_from_keys_uses_registry_order(self):
        self.assertEqual(
            ordered_combo_ids_from_keys(["B1", "S1", "custom"]),
            ["S1", "B1", "custom"],
        )

    def test_rank_by_value_desc_preserves_tie_order(self):
        combo_ids, values = rank_by_value_desc(
            ["S1", "B1", "L1", "custom"],
            [2.0, 5.0, 5.0, 1.0],
        )

        self.assertEqual(combo_ids, ["B1", "L1", "S1", "custom"])
        self.assertEqual(values, [5.0, 5.0, 2.0, 1.0])

    def test_rank_samples_by_median_desc(self):
        combo_ids, samples = rank_samples_by_median_desc(
            ["S1", "B1", "L1"],
            [[1.0, 2.0, 3.0], [10.0, 20.0], [5.0, 6.0, 7.0]],
        )

        self.assertEqual(combo_ids, ["B1", "L1", "S1"])
        self.assertEqual(samples, [[10.0, 20.0], [5.0, 6.0, 7.0], [1.0, 2.0, 3.0]])


if __name__ == "__main__":
    unittest.main()
