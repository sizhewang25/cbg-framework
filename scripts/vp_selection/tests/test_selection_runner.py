"""Tests for scripts.vp_selection.selection_runner.

Focuses on the pure compute helpers (build_k_grid, run_one_sampling,
run_one_sequence) that don't need ClickHouse.
"""

from __future__ import annotations

import unittest

from scripts.vp_selection.selection_runner import (
    build_k_grid,
    run_one_sampling,
    run_one_sequence,
)
from scripts.vp_selection.strategies import VpMeta
from scripts.vp_selection.pair_distances import compute_geodesic_distances


def _build_pool(n: int) -> dict[str, VpMeta]:
    pool: dict[str, VpMeta] = {}
    for i in range(n):
        pool[f"vp{i:03d}"] = VpMeta(
            lat=float(i % 90),
            lon=float((i * 7) % 180),
            asn=100 + (i % 4),
            city=f"city{i % 3}",
            country=f"C{i % 5}",
        )
    return pool


class TestBuildKGrid(unittest.TestCase):

    def test_descending_step(self):
        grid = build_k_grid(pool_size=1000, k_min=100, k_step=100)
        self.assertEqual(grid[0], 1000)
        # Step 100 down to k_min=100 → {1000, 900, ..., 100}
        self.assertEqual(grid, list(range(1000, 99, -100)))

    def test_includes_pool_size(self):
        grid = build_k_grid(pool_size=950, k_min=100, k_step=100)
        self.assertIn(950, grid)
        self.assertEqual(grid[0], 950)

    def test_handles_pool_smaller_than_k_min(self):
        grid = build_k_grid(pool_size=50, k_min=100, k_step=100)
        # No K satisfies pool_size >= k_min; grid is just [pool_size]
        self.assertEqual(grid, [50])

    def test_k_min_clamped_to_at_least_1(self):
        grid = build_k_grid(pool_size=10, k_min=0, k_step=5)
        # k_min defaults to >=1; grid down to 1 (or k_min, whichever)
        self.assertGreaterEqual(min(grid), 1)


class TestRunOneSampling(unittest.TestCase):

    def test_emits_K_rows_per_k_value(self):
        pool = _build_pool(20)
        k_grid = [20, 10, 5]
        rows = run_one_sampling(
            pool=pool,
            strategy="random",
            seed=0,
            k_grid=k_grid,
        )
        # Total rows = 20 + 10 + 5 = 35
        self.assertEqual(len(rows), 35)

    def test_row_schema(self):
        pool = _build_pool(10)
        rows = run_one_sampling(
            pool=pool,
            strategy="random",
            seed=0,
            k_grid=[5],
        )
        for r in rows:
            self.assertIn("strategy", r)
            self.assertIn("seed", r)
            self.assertIn("k", r)
            self.assertIn("vp_id", r)
            self.assertEqual(r["strategy"], "random")
            self.assertEqual(r["seed"], 0)
            self.assertEqual(r["k"], 5)

    def test_each_k_subset_has_exactly_k_distinct_vps(self):
        pool = _build_pool(20)
        rows = run_one_sampling(
            pool=pool,
            strategy="cluster_as",
            seed=0,
            k_grid=[12, 8],
        )
        # Group by k
        from collections import defaultdict
        by_k = defaultdict(list)
        for r in rows:
            by_k[r["k"]].append(r["vp_id"])
        self.assertEqual(len(by_k[12]), 12)
        self.assertEqual(len(set(by_k[12])), 12)
        self.assertEqual(len(by_k[8]), 8)
        self.assertEqual(len(set(by_k[8])), 8)


class TestRunOneSequence(unittest.TestCase):

    def test_emits_N_rows_with_positions(self):
        pool = _build_pool(12)
        distances = compute_geodesic_distances(
            {vp: (m.lat, m.lon) for vp, m in pool.items()}
        )
        rows = run_one_sequence(
            pool=pool,
            distances=distances,
            strategy="h1_as",
            seed=0,
        )
        self.assertEqual(len(rows), 12)
        positions = sorted(r["position"] for r in rows)
        self.assertEqual(positions, list(range(1, 13)))

    def test_row_schema(self):
        pool = _build_pool(8)
        distances = compute_geodesic_distances(
            {vp: (m.lat, m.lon) for vp, m in pool.items()}
        )
        rows = run_one_sequence(
            pool=pool,
            distances=distances,
            strategy="h1_as",
            seed=7,
        )
        for r in rows:
            self.assertIn("strategy", r)
            self.assertIn("seed", r)
            self.assertIn("position", r)
            self.assertIn("vp_id", r)
            self.assertEqual(r["strategy"], "h1_as")
            self.assertEqual(r["seed"], 7)


if __name__ == "__main__":
    unittest.main()
