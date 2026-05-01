"""Tests for CBG evaluation benchmark instrumentation."""

from __future__ import annotations

import unittest

from scripts.analysis.cbg_evaluation.benchmarking import (
    BenchmarkRecord,
    BenchmarkRecorder,
    summarize_records,
)
from scripts.analysis.cbg_evaluation.combinations import PipelineSpec
from scripts.analysis.cbg_evaluation.evaluate import (
    DistanceModelCache,
    evaluate_combination,
)
from scripts.framework.pipeline import CBGPipeline
from scripts.framework.types import CircleConstraint, MultilatResult


class FakeDistance:
    name = "fake_distance"

    def estimate(self, measurements, anchor_coords):
        return [
            CircleConstraint(
                vp_lat=0.0,
                vp_lon=0.0,
                vp_ip="anchor-a",
                rtt_ms=1.0,
                radius_km=100.0,
            )
        ]


class FakeFilter:
    name = "fake_filter"

    def filter(self, circles):
        return list(circles)


class FakeMultilateration:
    name = "fake_multilateration"

    def __init__(self, success=True):
        self.success = success

    def multilaterate(self, circles):
        return MultilatResult(
            vertices=[(1.0, 1.0)] if self.success else None,
            circles_used=circles,
            success=self.success,
        )


class FakeCentroid:
    name = "fake_centroid"

    def __init__(self, location=(1.0, 1.0)):
        self.location = location

    def select(self, multilat_result):
        return self.location


def fake_spec() -> PipelineSpec:
    return PipelineSpec(
        "T1",
        "test combo",
        "speed_of_internet",
        "none",
        "spherical_circle",
        "boundary_vertex_mean",
        "#000000",
        "-",
    )


def fake_lp_spec() -> PipelineSpec:
    return PipelineSpec(
        "L1",
        "lp combo",
        "low_envelope",
        "none",
        "spherical_circle",
        "boundary_vertex_mean",
        "#000000",
        "-",
        needs_lp_fit=True,
    )


def fake_octant_spec() -> PipelineSpec:
    return PipelineSpec(
        "B1",
        "octant combo",
        "bounded_spline",
        "none",
        "planar_annulus",
        "geometric_centroid",
        "#000000",
        "-",
        needs_octant_fit=True,
    )


def fake_targets():
    return {
        "probe-a": {
            "measurements": {"anchor-a": 1.0},
            "true_lat": 1.0,
            "true_lon": 1.0,
        }
    }


class TestBenchmarking(unittest.TestCase):
    def test_summarize_records_uses_millisecond_fields(self):
        summary = summarize_records([
            BenchmarkRecord(
                combo_id="A",
                probe_ip="p1",
                phase="distance_estimation",
                elapsed_ms=1.0,
                tracemalloc_peak_bytes=1_000_000,
                tracemalloc_peak_delta_bytes=500_000,
                rss_delta_bytes=2_000_000,
                rss_after_bytes=10_000_000,
                rss_high_water_delta_bytes=2_000_000,
            ),
            BenchmarkRecord(
                combo_id="A",
                probe_ip="p2",
                phase="distance_estimation",
                elapsed_ms=3.0,
                tracemalloc_peak_bytes=3_000_000,
                tracemalloc_peak_delta_bytes=1_500_000,
                rss_delta_bytes=4_000_000,
                rss_after_bytes=20_000_000,
                rss_high_water_delta_bytes=10_000_000,
            ),
        ])

        phase = summary["combinations"]["A"]["phases"]["distance_estimation"]
        self.assertEqual(summary["time_unit"], "ms")
        self.assertEqual(phase["count"], 2)
        self.assertEqual(phase["total_ms"], 4.0)
        self.assertEqual(phase["mean_ms"], 2.0)
        self.assertEqual(phase["median_ms"], 2.0)
        self.assertEqual(phase["mean_tracemalloc_peak_mb"], 2.0)
        self.assertEqual(phase["mean_tracemalloc_phase_peak_delta_mb"], 1.0)
        self.assertEqual(phase["max_tracemalloc_phase_peak_delta_mb"], 1.5)
        self.assertEqual(phase["mean_rss_delta_mb"], 3.0)
        self.assertEqual(phase["max_rss_after_mb"], 20.0)
        self.assertEqual(phase["mean_rss_high_water_delta_mb"], 6.0)
        self.assertEqual(phase["max_rss_high_water_delta_mb"], 10.0)

    def test_instrumented_evaluation_preserves_successful_result(self):
        pipe = CBGPipeline(
            FakeDistance(),
            FakeFilter(),
            FakeMultilateration(success=True),
            FakeCentroid(location=(1.0, 1.0)),
        )
        recorder = BenchmarkRecorder()

        results = evaluate_combination(
            fake_spec(),
            pipe,
            {"anchor-a": (0.0, 0.0)},
            fake_targets(),
            benchmark_recorder=recorder,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].error_km, 0.0)
        self.assertFalse(results[0].fallback_used)
        phases = {record.phase for record in recorder.records}
        self.assertEqual(
            phases,
            {
                "distance_estimation",
                "filtering",
                "multilateration",
                "centroid",
                "total_geolocate",
                "pipeline_overhead",
            },
        )
        self.assertTrue(all(record.elapsed_ms >= 0 for record in recorder.records))

    def test_instrumented_evaluation_records_fallback_metadata(self):
        pipe = CBGPipeline(
            FakeDistance(),
            FakeFilter(),
            FakeMultilateration(success=False),
            FakeCentroid(location=None),
        )
        recorder = BenchmarkRecorder()

        results = evaluate_combination(
            fake_spec(),
            pipe,
            {"anchor-a": (0.0, 0.0)},
            fake_targets(),
            benchmark_recorder=recorder,
        )

        self.assertTrue(results[0].fallback_used)
        self.assertEqual(results[0].fallback_reason, "multilateration_failed")
        total = next(r for r in recorder.records if r.phase == "total_geolocate")
        overhead = next(r for r in recorder.records if r.phase == "pipeline_overhead")
        self.assertTrue(total.fallback_used)
        self.assertTrue(overhead.fallback_used)
        self.assertEqual(overhead.fallback_reason, "multilateration_failed")

    def test_model_cache_reuses_lp_models_by_data_fingerprint(self):
        calls = []

        def fit_lp(df_asn):
            calls.append(df_asn)
            return {"anchor-a": object()}

        cache = DistanceModelCache(fit_lp_fn=fit_lp)
        recorder = BenchmarkRecorder()
        benchmark_a = {}
        benchmark_b = {}

        first = cache.get_for_spec(
            fake_lp_spec(),
            df_asn="same-data",
            data_fingerprint="fingerprint-a",
            benchmark_recorder=recorder,
            benchmark_ms=benchmark_a,
        )
        second = cache.get_for_spec(
            fake_lp_spec(),
            df_asn="same-data",
            data_fingerprint="fingerprint-a",
            benchmark_recorder=recorder,
            benchmark_ms=benchmark_b,
        )

        self.assertIs(first.lp_models, second.lp_models)
        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(benchmark_a["fit_lp_model_ms"], 0.0)
        self.assertEqual(benchmark_b["fit_lp_model_ms"], 0.0)
        lookup_rows = [
            r for r in recorder.records
            if r.phase == "model_cache_lookup" and r.model_family == "low_envelope"
        ]
        self.assertEqual([r.cache_hit for r in lookup_rows], [False, True])

    def test_model_cache_reuses_octant_models_by_data_fingerprint(self):
        calls = []

        def fit_octant(df_asn, target_coverage):
            calls.append((df_asn, target_coverage))
            return {"anchor-a": object()}, 10.0

        cache = DistanceModelCache(fit_octant_fn=fit_octant)
        recorder = BenchmarkRecorder()
        benchmark_a = {}
        benchmark_b = {}

        first = cache.get_for_spec(
            fake_octant_spec(),
            df_asn="same-data",
            data_fingerprint="fingerprint-a",
            benchmark_recorder=recorder,
            benchmark_ms=benchmark_a,
        )
        second = cache.get_for_spec(
            fake_octant_spec(),
            df_asn="same-data",
            data_fingerprint="fingerprint-a",
            benchmark_recorder=recorder,
            benchmark_ms=benchmark_b,
        )

        self.assertIs(first.octant_models, second.octant_models)
        self.assertEqual(first.octant_delta, second.octant_delta)
        self.assertEqual(calls, [("same-data", 0.80)])
        self.assertGreaterEqual(benchmark_a["fit_octant_model_ms"], 0.0)
        self.assertEqual(benchmark_b["fit_octant_model_ms"], 0.0)
        lookup_rows = [
            r for r in recorder.records
            if r.phase == "model_cache_lookup" and r.model_family == "bounded_spline"
        ]
        self.assertEqual([r.cache_hit for r in lookup_rows], [False, True])


if __name__ == "__main__":
    unittest.main()
