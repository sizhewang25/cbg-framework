"""Tests for evaluation reporting semantics."""

from __future__ import annotations

import unittest
from typing import Optional

from scripts.analysis.cbg_evaluation.combinations import PipelineSpec
from scripts.analysis.cbg_evaluation.evaluate import ProbeResult
from scripts.analysis.cbg_evaluation.reporting import (
    count_fitted_anchors,
    count_result_outcomes,
    format_intersection_fallback_total,
)


class DummyModel:
    def __init__(self, fitted: bool):
        self.fitted = fitted


def spec(distance: str) -> PipelineSpec:
    return PipelineSpec(
        "T",
        "test",
        distance,
        "none",
        "spherical_circle",
        "boundary_vertex_mean",
        "#000000",
        "-",
    )


def result(
    *,
    error_km: Optional[float],
    did_intersect: bool,
    fallback_used: bool,
) -> ProbeResult:
    return ProbeResult(
        probe_ip="probe",
        true_lat=0.0,
        true_lon=0.0,
        estimated_lat=0.0 if error_km is not None else None,
        estimated_lon=0.0 if error_km is not None else None,
        error_km=error_km,
        n_circles=1,
        min_rtt_ms=1.0,
        did_intersect=did_intersect,
        fallback_used=fallback_used,
    )


class TestReporting(unittest.TestCase):
    def test_result_counts_are_disjoint_for_cdf_reporting(self):
        results = [
            result(error_km=10.0, did_intersect=True, fallback_used=False),
            result(error_km=20.0, did_intersect=False, fallback_used=True),
            result(error_km=30.0, did_intersect=True, fallback_used=True),
            result(error_km=None, did_intersect=False, fallback_used=False),
        ]

        counts = count_result_outcomes(results)

        self.assertEqual(counts.total_probes, 4)
        self.assertEqual(counts.estimated_count, 3)
        self.assertEqual(counts.intersection_count, 1)
        self.assertEqual(counts.fallback_count, 2)
        self.assertEqual(counts.no_estimate_count, 1)
        self.assertEqual(counts.multilateration_success_count, 2)

    def test_format_intersection_fallback_total(self):
        results = [
            result(error_km=10.0, did_intersect=True, fallback_used=False),
            result(error_km=20.0, did_intersect=False, fallback_used=True),
            result(error_km=30.0, did_intersect=False, fallback_used=True),
        ]

        self.assertEqual(
            format_intersection_fallback_total(results),
            "1 I + 2 F = 3",
        )

    def test_format_includes_no_estimate_when_total_would_not_balance(self):
        results = [
            result(error_km=10.0, did_intersect=True, fallback_used=False),
            result(error_km=20.0, did_intersect=False, fallback_used=True),
            result(error_km=None, did_intersect=False, fallback_used=False),
        ]

        self.assertEqual(
            format_intersection_fallback_total(results),
            "1 I + 1 F + 1 N = 3",
        )

    def test_count_fitted_anchors_by_distance_model(self):
        anchor_coords = {
            "a": (0.0, 0.0),
            "b": (1.0, 1.0),
            "c": (2.0, 2.0),
        }
        models = {
            "a": DummyModel(True),
            "b": DummyModel(False),
            "extra": DummyModel(True),
        }

        self.assertEqual(
            count_fitted_anchors(spec("speed_of_internet"), anchor_coords),
            3,
        )
        self.assertEqual(
            count_fitted_anchors(
                spec("low_envelope"),
                anchor_coords,
                lp_models=models,
            ),
            1,
        )
        self.assertEqual(
            count_fitted_anchors(
                spec("bounded_spline"),
                anchor_coords,
                octant_models=models,
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
