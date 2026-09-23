"""Tests for TG/VP co-quantization.

`test_both_sides_share_one_nside_by_construction` is the load-bearing one: if
the two sides could be binned separately they could be binned differently, and
"the VP and the target are in the same cell" would quietly stop meaning
anything.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v4.modules import bipartite as B
from scripts.analysis.v4.modules import healpix as H

DENVER = (39.7392, -104.9903)
SEATTLE = (47.4490, -122.3090)
CHICAGO = (41.8781, -87.6298)
MIAMI = (25.7617, -80.1918)


def _tg(coords=(DENVER, SEATTLE, CHICAGO)):
    return pd.DataFrame(
        {
            "target_id": [f"tg-{i}" for i in range(len(coords))],
            "target_lat": [c[0] for c in coords],
            "target_lon": [c[1] for c in coords],
        }
    )


def _vp(coords):
    return pd.DataFrame(
        {
            "vp_id": [f"vp-{i}" for i in range(len(coords))],
            "vp_lat": [c[0] for c in coords],
            "vp_lon": [c[1] for c in coords],
        }
    )


class TestOneGridForBothSides:
    def test_both_sides_share_one_nside_by_construction(self):
        """There is no code path that bins one side without the other — the API
        takes both frames together, so they cannot be given different nsides."""
        assert list(inspect.signature(B.co_quantize).parameters) == [
            "targets", "vps", "nside",
        ]
        assert not [
            n for n in dir(B) if n.startswith("quantize_")
        ], "a single-side quantizer would let the two drift apart"

    def test_a_vp_at_a_target_lands_in_that_target_cell(self):
        q = B.co_quantize(_tg(), _vp([DENVER]), 128)
        tg_cell = q.targets.loc[q.targets.target_id == "tg-0", "cell_id"].iloc[0]
        assert q.vps["cell_id"].iloc[0] == tg_cell
        assert q.meta["n_shared_cells"] == 1

    def test_cells_agree_with_a_direct_ang2pix(self):
        q = B.co_quantize(_tg(), _vp([DENVER, CHICAGO]), 64)
        assert np.array_equal(
            q.targets["cell_id"].to_numpy(),
            H.ang2pix(q.targets["target_lat"], q.targets["target_lon"], 64),
        )
        assert np.array_equal(
            q.vps["cell_id"].to_numpy(),
            H.ang2pix(q.vps["vp_lat"], q.vps["vp_lon"], 64),
        )

    def test_missing_vp_columns_are_named(self):
        bad = pd.DataFrame({"vp_id": ["a"], "lat": [0.0], "lon": [0.0]})
        with pytest.raises(KeyError):
            B.co_quantize(_tg(), bad, 128)


class TestOccupancy:
    def test_occupancy_covers_the_union_of_both_sides(self):
        q = B.co_quantize(_tg(), _vp([MIAMI]), 128)
        assert len(q.occupancy) == 4  # 3 target cells + 1 VP-only cell
        assert q.occupancy["n_targets"].sum() == 3
        assert q.occupancy["n_vps"].sum() == 1

    def test_shared_flag_marks_only_cells_holding_both(self):
        q = B.co_quantize(_tg(), _vp([DENVER, MIAMI]), 128)
        shared = q.occupancy[q.occupancy["shared"]]
        assert len(shared) == 1
        assert shared["n_targets"].iloc[0] >= 1 and shared["n_vps"].iloc[0] >= 1

    def test_occupancy_carries_cell_centres_for_drawing(self):
        q = B.co_quantize(_tg(), _vp([DENVER]), 64)
        assert np.array_equal(
            H.ang2pix(q.occupancy["cell_lat"], q.occupancy["cell_lon"], 64),
            q.occupancy["cell_id"].to_numpy(),
        )

    def test_share_of_target_cells_with_a_vp(self):
        q = B.co_quantize(_tg(), _vp([DENVER, SEATTLE]), 128)
        # Rounded to 4dp at source, so compare at that precision.
        assert q.meta["share_of_target_cells_with_a_vp"] == pytest.approx(
            2 / 3, abs=5e-5
        )


class TestDispersion:
    def test_uniform_occupancy_equals_the_raw_count(self):
        assert B.dispersion([1, 1, 1, 1]) == pytest.approx(4.0)

    def test_concentration_pulls_it_toward_one(self):
        """The gap from the raw count is the read on how lopsided the geometry
        is — 18 cells with 17 points in one is not 18 places."""
        assert B.dispersion([17, 1]) < 1.5
        assert B.dispersion([17, 1]) > 1.0

    def test_empty_and_single(self):
        assert B.dispersion([]) == 0.0
        assert B.dispersion([0, 0]) == 0.0
        assert B.dispersion([5]) == pytest.approx(1.0)

    def test_never_exceeds_the_occupied_count(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            c = rng.integers(1, 40, rng.integers(1, 12))
            assert B.dispersion(c) <= len(c) + 1e-9


class TestLadderBehaviour:
    def test_cell_counts_never_grow_as_cells_grow(self):
        tg = _tg((DENVER, SEATTLE, CHICAGO, MIAMI))
        vps = _vp((DENVER, SEATTLE, CHICAGO, MIAMI, (40.0, -105.0)))
        tg_cells, vp_cells = [], []
        for nside in H.NSIDE_LADDER:
            q = B.co_quantize(tg, vps, nside)
            tg_cells.append(q.meta["n_target_cells"])
            vp_cells.append(q.meta["n_vp_cells"])
        assert tg_cells == sorted(tg_cells, reverse=True)
        assert vp_cells == sorted(vp_cells, reverse=True)

    def test_occupancy_row_carries_the_curve_fields(self):
        q = B.co_quantize(_tg(), _vp([DENVER]), 32)
        row = B.occupancy_row(q)
        assert row["grid"] == "healpix"
        assert row["nside"] == 32
        assert row["cell_km"] == pytest.approx(203.7, abs=0.1)
        for key in (
            "n_target_cells", "n_vp_cells", "n_shared_cells",
            "target_dispersion", "vp_dispersion",
        ):
            assert key in row


class TestRoundTrip:
    def test_write_emits_the_four_artifacts(self, tmp_path):
        q = B.co_quantize(_tg(), _vp([DENVER]), 128)
        q.write(tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "cell_occupancy.csv", "meta.json", "target_cells.csv", "vp_cells.csv",
        ]
