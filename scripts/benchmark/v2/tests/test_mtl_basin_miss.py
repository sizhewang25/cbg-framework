"""Tests for the basin-miss harness.

The harness is how `top_k` and `neighbor_ring` get chosen, so the property that
matters most is that it can **fail**. A sweep that reports zero misses at every
setting, including the deliberately-starved one, is measuring nothing -- and
that is exactly what a subtly broken reference or a too-generous threshold looks
like. `TestItCanDetectAMiss` pins the discrimination.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from scripts.benchmark.v2 import mtl_basin_miss as basin

#: Forwarded verbatim to the MTL. The sweep is about pruning, not
#: about which grid is being pruned.
GRID = {"grid": "healpix"}

#: nside 8 is 768 cells -- cheap to pass over globally, and coarse
#: enough to leave room to descend. Not 1 or 2, where a ring-1 disk
#: is 58% and 19% of the globe and the pruning stops being pruning.
COARSE = 8

#: Spread out enough that the surface is not trivially unimodal, and real
#: enough to be a plausible VP fleet.
VPS = pd.DataFrame(
    {
        "vp_id": ["a", "b", "c", "d", "e", "f"],
        "vp_lat": [47.61, 34.05, 41.88, 29.76, 40.71, 33.75],
        "vp_lon": [-122.33, -118.24, -87.63, -95.37, -74.01, -84.39],
    }
)
TARGETS = pd.DataFrame(
    {
        "target_id": [f"t{i}" for i in range(6)],
        "target_lat": [39.74, 45.52, 32.78, 38.63, 44.98, 35.23],
        "target_lon": [-104.99, -122.68, -96.80, -90.20, -93.27, -80.84],
    }
)


class TestConstraints(unittest.TestCase):
    def test_every_vp_yields_a_distribution_carrying_result(self):
        rng = np.random.default_rng(0)
        cons = basin.constraints_for(39.74, -104.99, VPS, rng)
        self.assertEqual(len(cons), len(VPS))
        for c in cons:
            self.assertTrue(c.success)
            self.assertTrue(c.tg_distance.has_distribution)
            self.assertGreater(c.tg_distance.sigma_km, 0.0)

    def test_the_inflation_is_one_sided(self):
        """RTT error inflates delay; nothing deflates it below the speed of
        light. A symmetric noise model would make the harness look easier than
        it is, because it is the long tail that creates the far-away spurious
        modes pruning can lose."""
        from scripts.framework.geometry import haversine

        rng = np.random.default_rng(1)
        for _ in range(20):
            cons = basin.constraints_for(39.74, -104.99, VPS, rng)
            for c in cons:
                truth = haversine(
                    (c.vp_coord.lat, c.vp_coord.lon), (39.74, -104.99)
                )
                self.assertGreaterEqual(c.tg_distance.mu_km, truth - 1e-6)

    def test_sigma_stays_inside_its_clip(self):
        rng = np.random.default_rng(2)
        lo, hi = basin._SIGMA_CLIP_KM
        for lat, lon in [(0.0, 0.0), (39.74, -104.99), (-40.0, 170.0)]:
            for c in basin.constraints_for(lat, lon, VPS, rng):
                self.assertGreaterEqual(c.tg_distance.sigma_km, lo)
                self.assertLessEqual(c.tg_distance.sigma_km, hi)


class TestSweep(unittest.TestCase):
    """Run at a tiny resolution so the global reference is cheap. The grid
    identity is not what these assert -- `TestItCanDetectAMiss` does that."""

    def test_the_shipped_setting_is_a_zero_miss_row(self):
        rows = basin.sweep(
            VPS, TARGETS,
            resolution=COARSE * 8, coarse_resolution=COARSE,
            grid_kwargs=GRID, settings=((8, 1),),
            n_targets=6, seed=7,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].n_misses, 0)
        self.assertEqual(rows[0].n_targets, 6)

    def test_a_global_pass_cannot_miss_itself(self):
        """`coarse == resolution` is the no-pruning mode, so it must agree with
        the reference exactly. A non-zero row here would mean the reference and
        the subject disagree for a reason other than pruning."""
        rows = basin.sweep(
            VPS, TARGETS,
            resolution=COARSE, coarse_resolution=COARSE,
            grid_kwargs=GRID, settings=((1, 0),),
            n_targets=6, seed=7,
        )
        self.assertEqual(rows[0].max_km, 0.0)

    def test_the_same_surfaces_are_reused_across_settings(self):
        """Settings must be compared on identical constraint sets, or a
        difference in the draw reads as a difference in pruning."""
        a = basin.sweep(
            VPS, TARGETS, resolution=COARSE * 8, coarse_resolution=COARSE,
            grid_kwargs=GRID, settings=((8, 1), (64, 1)), n_targets=6, seed=11,
        )
        b = basin.sweep(
            VPS, TARGETS, resolution=COARSE * 8, coarse_resolution=COARSE,
            grid_kwargs=GRID, settings=((64, 1), (8, 1)), n_targets=6, seed=11,
        )
        self.assertEqual(
            {(r.top_k, r.max_km) for r in a}, {(r.top_k, r.max_km) for r in b}
        )

    def test_it_samples_rather_than_demanding_the_whole_population(self):
        rows = basin.sweep(
            VPS, TARGETS, resolution=COARSE * 2, coarse_resolution=COARSE,
            grid_kwargs=GRID, settings=((8, 1),), n_targets=3, seed=5,
        )
        self.assertEqual(rows[0].n_targets, 3)

    def test_it_records_the_settings_it_ran(self):
        rows = basin.sweep(
            VPS, TARGETS, resolution=COARSE * 2, coarse_resolution=COARSE,
            grid_kwargs=GRID, settings=((4, 0),), n_targets=6, seed=5,
        )
        r = rows[0]
        self.assertEqual((r.resolution, r.coarse_resolution), (COARSE * 2, COARSE))
        self.assertEqual((r.top_k, r.neighbor_ring), (4, 0))
        self.assertEqual(r.grid, "healpix")


class TestItCanDetectAMiss(unittest.TestCase):
    def test_starving_the_descent_produces_misses(self):
        """The discrimination check. `top_k=1, ring=0` carries one cell per
        level, so it must lose modes on at least one of a spread-out target
        set. If this ever reports zero, the harness has stopped measuring and
        every other row it prints is worthless."""
        rows = basin.sweep(
            VPS, TARGETS,
            resolution=COARSE * 8, coarse_resolution=COARSE,
            grid_kwargs=GRID, settings=((1, 0), (64, 1)),
            n_targets=6, seed=3, miss_km=50.0,
        )
        starved = next(r for r in rows if r.top_k == 1)
        generous = next(r for r in rows if r.top_k == 64)
        self.assertGreater(
            starved.max_km, generous.max_km,
            "a one-cell descent did not do worse than a 64-cell one; the "
            "harness is not discriminating",
        )

    def test_the_threshold_is_what_turns_a_gap_into_a_miss(self):
        rows = basin.sweep(
            VPS, TARGETS, resolution=COARSE * 8, coarse_resolution=COARSE,
            grid_kwargs=GRID, settings=((1, 0),), n_targets=6, seed=3, miss_km=1e-9,
        )
        self.assertGreater(rows[0].n_misses, 0)
        lenient = basin.sweep(
            VPS, TARGETS, resolution=COARSE * 8, coarse_resolution=COARSE,
            grid_kwargs=GRID, settings=((1, 0),), n_targets=6, seed=3, miss_km=1e9,
        )
        self.assertEqual(lenient[0].n_misses, 0)


class TestReporting(unittest.TestCase):
    def test_the_table_has_a_row_per_setting(self):
        rows = basin.sweep(
            VPS, TARGETS, resolution=COARSE * 2, coarse_resolution=COARSE,
            grid_kwargs=GRID, settings=((1, 0), (8, 1)), n_targets=6, seed=5,
        )
        text = basin.format_table(rows)
        self.assertIn("top_k", text)
        self.assertEqual(len(text.splitlines()), 3)

    def test_the_json_round_trips(self):
        rows = basin.sweep(
            VPS, TARGETS, resolution=COARSE * 2, coarse_resolution=COARSE,
            grid_kwargs=GRID, settings=((8, 1),), n_targets=6, seed=5,
        )
        with TemporaryDirectory() as tmp:
            path = basin.write_report(rows, Path(tmp) / "nested" / "out.json")
            got = json.loads(path.read_text())
        self.assertEqual(got[0]["top_k"], 8)
        self.assertEqual(got[0]["n_targets"], 6)


_REAL_RUN = (
    Path(__file__).resolve().parents[4]
    / "outputs" / "benchmark" / "v2"
    / "as01-260728-260802-mesh" / "generic_csv" / "anchors_to_probes"
)


@pytest.mark.skipif(
    not (_REAL_RUN / "vps.csv").exists(),
    reason="as01 mesh output tree not present on this machine",
)
class TestAgainstARealRun(unittest.TestCase):
    def test_it_loads_the_run_geometry(self):
        vps, targets = basin.load_geometry(_REAL_RUN)
        self.assertGreater(len(vps), 50)
        self.assertGreater(len(targets), 10)
        self.assertEqual(
            set(targets.columns), {"target_id", "target_lat", "target_lon"}
        )

    def test_any_combo_supplies_the_same_population(self):
        """The loader defaults to whichever combo sorts first, precisely so a
        renamed arm does not break it. That is only safe if every combo in a
        fold scores the same targets."""
        fold = _REAL_RUN / "fold_0"
        combos = sorted(
            d.name for d in fold.glob("*") if (d / "targets.parquet").exists()
        )
        first = basin.load_geometry(_REAL_RUN, combo=combos[0])[1]
        last = basin.load_geometry(_REAL_RUN, combo=combos[-1])[1]
        self.assertEqual(
            sorted(first["target_id"]), sorted(last["target_id"])
        )

    def test_a_missing_combo_is_refused_with_the_available_names(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            basin.load_geometry(_REAL_RUN, combo="no_such_combo")
        self.assertIn("no_such_combo", str(ctx.exception))
        self.assertIn("vanilla_cbg", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
