"""Tests for PlanarAnnulusWeightedMTL.

Ports scripts/framework/multilateration/tests/test_planar_annulus_weighted.py.

The wrapper computes per-VP weights from `latency` (`exp(-rtt/tau)`). Until
LTDResult.latency lands, the weighted variant returns INSUFFICIENT_DATA on any
plain LTDResult — that path is covered explicitly. The happy-path test uses a
duck-typed namespace from helpers.ltd_result_with_latency.
"""

from __future__ import annotations

import math
import unittest

from shapely.geometry import Point

from scripts.framework.v2.mtl.base import AnnulusMTLMethod
from scripts.framework.v2.mtl.planar_annulus_weighted import (
    _RTT_FLOOR_MS,
    PlanarAnnulusWeightedMTL,
)
from scripts.framework.v2.mtl.tests.helpers import (
    ltd_result,
    ltd_result_with_latency,
)
from scripts.framework.v2.registry import MTL_REGISTRY
from scripts.framework.v2.types import Error


class TestPlanarAnnulusWeightedMTL(unittest.TestCase):
    def test_empty_input_fails_with_insufficient_data(self):
        result = PlanarAnnulusWeightedMTL().multilaterate([])

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)
        self.assertIsNone(result.intersection)
        self.assertEqual(result.method, "PlanarAnnulusWeightedMTL")

    def test_missing_latency_fails_with_insufficient_data(self):
        """Plain LTDResult has no latency field; the wrapper must bail out."""
        result = PlanarAnnulusWeightedMTL().multilaterate([
            ltd_result("a", lat=0.0, lon=0.0, upper_km=111.0),
        ])

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)

    def test_zero_threshold_returns_full_disk_face(self):
        """Single annulus → one face (the whole disk) carrying all the weight.

        latency=0 → weight=exp(0)=1.0. Σw=1.0; threshold 0.0 → target 0.
        The top-and-only face clears 0 immediately, so the full disk is
        returned. With the legacy grid algorithm this used to assert a 4-cell
        grid union; the face-decomposition algorithm returns the disk polygon.
        """
        result = PlanarAnnulusWeightedMTL(weight_threshold=0.0).multilaterate([
            ltd_result_with_latency(
                "a", lat=0.0, lon=0.0, upper_km=111.0, lower_km=0.0, latency=0.0
            ),
        ])

        self.assertTrue(result.success)
        self.assertEqual(result.intersection.geom_type, "Polygon")
        # 64-vertex polygon inscribed in the unit circle at the equator.
        self.assertEqual(result.intersection.bounds, (-1.0, -1.0, 1.0, 1.0))
        # Inscribed 64-gon area: (n/2)·sin(2π/n) ≈ 3.1365 for n=64.
        self.assertAlmostEqual(result.intersection.area, 3.1365, places=3)

    def test_registered_in_mtl_registry(self):
        self.assertIn("planar_annulus_weighted", MTL_REGISTRY)
        self.assertIs(
            MTL_REGISTRY["planar_annulus_weighted"], PlanarAnnulusWeightedMTL
        )

    def test_is_annulus_family(self):
        self.assertTrue(issubclass(PlanarAnnulusWeightedMTL, AnnulusMTLMethod))

    def test_highest_weight_only_collapses_disconnected_union(self):
        """Two far-apart VP pairs would union to a MultiPolygon under the
        legacy path. Pair (A,B) gets latency 0 (weight≈1.0) and pair (C,D)
        gets latency 50 ms (weight≈0.368), so the heavy pair's face is
        unambiguously face #1. With highest_weight_only=True the wrapper
        returns just that face — Polygon, contains the heavy-pair centroid,
        excludes the light-pair centroid."""
        results = [
            ltd_result_with_latency("A", 0.0,  0.0, upper_km=222.0, lower_km=0.0, latency=0.0),
            ltd_result_with_latency("B", 0.0, -1.0, upper_km=222.0, lower_km=0.0, latency=0.0),
            ltd_result_with_latency("C", 0.0,  5.0, upper_km=222.0, lower_km=0.0, latency=50.0),
            ltd_result_with_latency("D", 0.0,  6.0, upper_km=222.0, lower_km=0.0, latency=50.0),
        ]
        mtl = PlanarAnnulusWeightedMTL(
            highest_weight_only=True,
            enable_circle_filter=False,
        )
        result = mtl.multilaterate(results)
        self.assertTrue(result.success)
        self.assertEqual(result.intersection.geom_type, "Polygon")
        self.assertTrue(result.intersection.contains(Point(-0.5, 0.0)))   # heavy pair
        self.assertFalse(result.intersection.contains(Point(5.5, 0.0)))   # light pair excluded


class TestWeightModes(unittest.TestCase):
    """The three `weight_mode` families, exercised through `_weight` directly.

    Going through `multilaterate` would drag in the planar arrangement, which
    is both the expensive part and entirely weight-independent — so it would
    make these slow without making them test anything more.
    """

    def test_default_is_exp_tau_50(self):
        """Regression guard. The published as0* mesh configs pass
        `weight_tau_ms: 50.0` and no mode, so this is the behaviour their
        committed results were produced under."""
        mtl = PlanarAnnulusWeightedMTL()
        self.assertEqual(mtl.weight_mode, "exp")
        self.assertEqual(mtl.weight_tau_ms, 50.0)
        for rtt in (0.0, 2.1945, 35.2, 71.8):
            self.assertAlmostEqual(mtl._weight(rtt), math.exp(-rtt / 50.0), places=15)

    def test_exp_honours_tau(self):
        for tau in (1.0, 5.0, 10.0):
            mtl = PlanarAnnulusWeightedMTL(weight_mode="exp", weight_tau_ms=tau)
            for rtt in (3.0, 55.0):
                self.assertAlmostEqual(
                    mtl._weight(rtt), math.exp(-rtt / tau), places=15
                )

    def test_inv_power_honours_k(self):
        for k in (1.0, 2.0, 3.0):
            mtl = PlanarAnnulusWeightedMTL(weight_mode="inv_power", weight_k=k)
            for rtt in (3.0, 55.0):
                self.assertAlmostEqual(mtl._weight(rtt), rtt ** -k, places=15)

    def test_inv_power_is_scale_free(self):
        """`rtt -> c*rtt` scales every weight by the same constant, which the
        downstream face argmax discards. This is the property that makes the
        inverse-power family transfer across VP fleets; the exponential does
        not have it."""
        mtl = PlanarAnnulusWeightedMTL(weight_mode="inv_power", weight_k=2.0)
        c = 2.5
        ratio_1x = mtl._weight(3.0) / mtl._weight(55.0)
        ratio_cx = mtl._weight(c * 3.0) / mtl._weight(c * 55.0)
        self.assertAlmostEqual(ratio_1x, ratio_cx, places=12)

    def test_uniform_ignores_rtt(self):
        mtl = PlanarAnnulusWeightedMTL(weight_mode="uniform")
        for rtt in (0.0, 3.0, 55.0, 1e6):
            self.assertEqual(mtl._weight(rtt), 1.0)

    def test_inv_power_floors_zero_rtt(self):
        """A 0.0 ms sample must not become an infinitely heavy constraint."""
        mtl = PlanarAnnulusWeightedMTL(weight_mode="inv_power", weight_k=2.0)
        w = mtl._weight(0.0)
        self.assertTrue(math.isfinite(w))
        self.assertEqual(w, _RTT_FLOOR_MS ** -2.0)

    def test_inv_power_floor_does_not_bind_on_production_rtts(self):
        """Pins the 0.1-vs-1.0 floor choice to a test rather than a comment:
        0.584 ms is the smallest RTT in any of the three as0* datasets, and it
        must pass through unmodified."""
        mtl = PlanarAnnulusWeightedMTL(weight_mode="inv_power", weight_k=2.0)
        self.assertEqual(mtl._weight(0.584), 0.584 ** -2.0)

    def test_exp_does_not_floor_rtt(self):
        """Guard against a refactor hoisting the floor out of the inv_power
        branch — that would silently change every published as0* number."""
        mtl = PlanarAnnulusWeightedMTL()
        self.assertEqual(mtl._weight(0.0), 1.0)
        self.assertEqual(mtl._weight(0.5), math.exp(-0.01))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError) as ctx:
            PlanarAnnulusWeightedMTL(weight_mode="bogus")
        msg = str(ctx.exception)
        self.assertIn("bogus", msg)
        for mode in ("exp", "inv_power", "uniform"):
            self.assertIn(mode, msg)

    def test_mode_is_not_normalised(self):
        """Documents that we do not strip/lowercase: a near-miss must fail."""
        for bad in ("Exp", " exp", "EXP"):
            with self.assertRaises(ValueError):
                PlanarAnnulusWeightedMTL(weight_mode=bad)

    def test_nonpositive_tau_raises(self):
        for tau in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                PlanarAnnulusWeightedMTL(weight_mode="exp", weight_tau_ms=tau)

    def test_inv_power_requires_k(self):
        with self.assertRaises(ValueError) as ctx:
            PlanarAnnulusWeightedMTL(weight_mode="inv_power")
        self.assertIn("weight_k", str(ctx.exception))

    def test_nonpositive_k_raises(self):
        for k in (0.0, -2.0):
            with self.assertRaises(ValueError):
                PlanarAnnulusWeightedMTL(weight_mode="inv_power", weight_k=k)

    def test_tau_rejected_outside_exp_mode(self):
        """The highest-value validation test: this is the exact shape produced
        by copy-pasting a shipped as0* combo (which carries weight_tau_ms:
        50.0) and editing only weight_mode."""
        with self.assertRaises(ValueError) as ctx:
            PlanarAnnulusWeightedMTL(weight_mode="uniform", weight_tau_ms=50.0)
        self.assertIn("weight_tau_ms", str(ctx.exception))
        with self.assertRaises(ValueError):
            PlanarAnnulusWeightedMTL(
                weight_mode="inv_power", weight_k=2.0, weight_tau_ms=50.0
            )

    def test_k_rejected_outside_inv_power_mode(self):
        for mode in ("exp", "uniform"):
            with self.assertRaises(ValueError) as ctx:
                PlanarAnnulusWeightedMTL(weight_mode=mode, weight_k=2.0)
            self.assertIn("weight_k", str(ctx.exception))

    def test_tau1_absorbs_far_constraints(self):
        """Documented float64 behaviour: at tau=1 a constraint more than
        ~37 ms above the nearest one is absorbed into the running sum and
        contributes exactly nothing. Asserted on the arithmetic rather than
        through geometry, so it cannot go flaky."""
        m1 = PlanarAnnulusWeightedMTL(weight_tau_ms=1.0)
        near, far = m1._weight(0.6), m1._weight(50.6)
        self.assertGreater(far, 0.0)          # not literally zero
        self.assertEqual(near + far, near)    # but absorbed
        m10 = PlanarAnnulusWeightedMTL(weight_tau_ms=10.0)
        self.assertNotEqual(
            m10._weight(0.6) + m10._weight(50.6), m10._weight(0.6)
        )

    def test_uniform_and_exp_select_different_faces(self):
        """The one test proving the knob reaches the geometry rather than
        stopping at `_weight`.

        Three mutually-overlapping VPs at 60 ms around lon 0, two at 0 ms
        around lon 5. Uniform scores the clusters 3 vs 2 and takes the first;
        exp at tau=50 scores 3*exp(-1.2)=0.90 vs 2*1.0=2.0 and takes the
        second. Spacing reuses the lon 0/-1 vs 5/6 separation already proven
        disjoint by test_highest_weight_only_collapses_disconnected_union.
        """
        results = [
            ltd_result_with_latency("A", 0.0,  0.0, upper_km=222.0, lower_km=0.0, latency=60.0),
            ltd_result_with_latency("B", 0.0, -0.5, upper_km=222.0, lower_km=0.0, latency=60.0),
            ltd_result_with_latency("C", 0.0, -1.0, upper_km=222.0, lower_km=0.0, latency=60.0),
            ltd_result_with_latency("D", 0.0,  5.0, upper_km=222.0, lower_km=0.0, latency=0.0),
            ltd_result_with_latency("E", 0.0,  6.0, upper_km=222.0, lower_km=0.0, latency=0.0),
        ]
        kw = dict(highest_weight_only=True, enable_circle_filter=False)
        uni = PlanarAnnulusWeightedMTL(weight_mode="uniform", **kw).multilaterate(results)
        exp = PlanarAnnulusWeightedMTL(weight_tau_ms=50.0, **kw).multilaterate(results)

        self.assertTrue(uni.success)
        self.assertTrue(exp.success)
        self.assertFalse(uni.intersection.equals(exp.intersection))
        # Uniform follows the count -> the slow three-VP cluster.
        self.assertTrue(uni.intersection.contains(Point(-0.5, 0.0)))
        self.assertFalse(uni.intersection.contains(Point(5.5, 0.0)))
        # Exponential follows the weight -> the fast two-VP cluster.
        self.assertTrue(exp.intersection.contains(Point(5.5, 0.0)))
        self.assertFalse(exp.intersection.contains(Point(-0.5, 0.0)))

    def test_uniform_still_requires_latency(self):
        """The control arm must keep the same participant set as the arms it
        controls for, so it short-circuits on missing latency like the rest."""
        result = PlanarAnnulusWeightedMTL(weight_mode="uniform").multilaterate([
            ltd_result("a", lat=0.0, lon=0.0, upper_km=111.0),
        ])
        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)


if __name__ == "__main__":
    unittest.main()
