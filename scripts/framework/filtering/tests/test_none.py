"""Tests for the no-op filtering variant."""

from __future__ import annotations

import unittest

from scripts.framework.filtering.none import NoFilter
from scripts.framework.filtering.tests.helpers import circle


class TestNoFilter(unittest.TestCase):
    def test_filter_returns_all_constraints_in_order(self):
        circles = [
            circle("a", rtt_ms=10.0, radius_km=1000.0),
            circle("b", lat=1.0, lon=1.0, rtt_ms=20.0, radius_km=2000.0),
        ]

        filtered = NoFilter().filter(circles)

        self.assertEqual([c.vp_ip for c in filtered], ["a", "b"])
        self.assertIs(filtered[0], circles[0])
        self.assertIs(filtered[1], circles[1])

    def test_filter_returns_new_list(self):
        circles = [circle("a")]

        filtered = NoFilter().filter(circles)

        self.assertEqual(filtered, circles)
        self.assertIsNot(filtered, circles)


if __name__ == "__main__":
    unittest.main()
