"""Invariants of the synthetic random-weight fixture generator.

The generator is only a test fixture, but two of its properties are relied on
elsewhere: the draw must be reproducible from the seed (so a materialized run
can be regenerated), and it must refuse to overwrite real traffic volume.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.processing.source.assign_random_weights import (
    _default_output,
    assign_random_weights,
    describe_cut,
)
from pathlib import Path


def _mesh(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame({
        "vp_id": [f"v{i % 7}" for i in range(n)],
        "target_id": [f"t{i % 13}" for i in range(n)],
        "rtt_ms": np.linspace(1.0, 50.0, n),
    })


class TestAssignRandomWeights(unittest.TestCase):
    def test_deterministic_in_seed(self) -> None:
        a = assign_random_weights(_mesh(), seed=42)
        b = assign_random_weights(_mesh(), seed=42)
        pd.testing.assert_series_equal(a["weight"], b["weight"])

    def test_different_seeds_differ(self) -> None:
        a = assign_random_weights(_mesh(), seed=42)
        b = assign_random_weights(_mesh(), seed=43)
        self.assertFalse(a["weight"].equals(b["weight"]))

    def test_respects_range_and_decimals(self) -> None:
        out = assign_random_weights(_mesh(), seed=1, low=0.0, high=1.0, decimals=3)
        w = out["weight"].to_numpy()
        self.assertTrue(((w >= 0.0) & (w <= 1.0)).all())
        np.testing.assert_array_equal(w, np.round(w, 3))

    def test_does_not_mutate_input_or_drop_columns(self) -> None:
        df = _mesh()
        out = assign_random_weights(df, seed=1)
        self.assertNotIn("weight", df.columns)
        self.assertEqual(list(out.columns), list(df.columns) + ["weight"])
        self.assertEqual(len(out), len(df))

    def test_invalid_parameters_raise(self) -> None:
        for kw in ({"low": 1.0, "high": 1.0}, {"low": -0.5}, {"decimals": -1}):
            with self.assertRaises(ValueError):
                assign_random_weights(_mesh(), seed=1, **kw)

    def test_default_output_keeps_multi_dot_stage_tags(self) -> None:
        out = _default_output(Path("d/as01.mainland.sanitized.csv"))
        self.assertEqual(out.name, "as01.mainland.sanitized.randweight.csv")


class TestDescribeCut(unittest.TestCase):
    """describe_cut must agree with filter_weighted_flows' own kernel."""

    def test_matches_filter_weighted_flows(self) -> None:
        from scripts.processing.source.filter_weighted_flows import (
            derive_flow_weight_threshold,
        )

        weights = assign_random_weights(_mesh(500), seed=7)["weight"].to_numpy(float)
        for frac in (0.5, 0.8, 0.95, 1.0):
            expected, _ = derive_flow_weight_threshold(weights, frac)
            self.assertEqual(describe_cut(weights, frac)["threshold"], expected,
                             msg=f"disagreement at frac={frac}")

    def test_uniform_draw_keeps_the_analytic_flow_share(self) -> None:
        """Uniform[0,1] at fraction f cuts at sqrt(1-f) and keeps 1-sqrt(1-f)
        of flows. This is why the fixture exercises no node elimination."""
        rng = np.random.default_rng(0)
        weights = rng.uniform(0.0, 1.0, size=200_000)
        cut = describe_cut(weights, 0.95)
        self.assertAlmostEqual(cut["threshold"], np.sqrt(0.05), places=2)
        self.assertAlmostEqual(cut["kept_share_of_flows"], 1 - np.sqrt(0.05), places=2)


if __name__ == "__main__":
    unittest.main()
