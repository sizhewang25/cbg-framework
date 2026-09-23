"""Tests for the v4 answer space.

The load-bearing ones are `TestNesting` and `TestSeedsAreCellCentres`: exact
nesting is what the laddered metric rests on, and seeding at the cell centre
rather than the target centroid is what keeps a class from moving when the
target set changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v4.modules import answer_space as A
from scripts.analysis.v4.modules import healpix as H

DENVER = (39.7392, -104.9903)
SEATTLE = (47.4490, -122.3090)
CHICAGO = (41.8781, -87.6298)
MIAMI = (25.7617, -80.1918)


def _targets(coords=(DENVER, SEATTLE, CHICAGO, MIAMI), per=3) -> pd.DataFrame:
    """`per` distinct IPs at each coordinate — the real shape of these datasets,
    where ~20 replicas share one facility."""
    rows = []
    for i, (lat, lon) in enumerate(coords):
        for j in range(per):
            rows.append({"target_id": f"tg-{i}-{j}", "target_lat": lat, "target_lon": lon})
    return pd.DataFrame(rows)


class TestInputContract:
    def test_missing_columns_are_named(self):
        with pytest.raises(ValueError, match="target_lat"):
            A.build_answer_space(pd.DataFrame({"target_id": ["a"]}))

    def test_empty_targets_is_refused(self):
        empty = pd.DataFrame(columns=["target_id", "target_lat", "target_lon"])
        with pytest.raises(ValueError, match="empty"):
            A.build_answer_space(empty)

    def test_duplicate_target_id_is_an_error(self):
        t = _targets()
        t.loc[1, "target_id"] = t.loc[0, "target_id"]
        with pytest.raises(ValueError, match="duplicate target_id"):
            A.build_answer_space(t)

    def test_duplicate_coordinates_collapse_into_one_class(self):
        """Expected, not an error: distinct IPs at one facility are one place."""
        space = A.build_answer_space(_targets(per=5))
        assert space.n_seeds == 4
        assert space.meta["n_targets"] == 20
        assert space.meta["n_unique_target_coords"] == 4


class TestSeedsAreCellCentres:
    def test_seed_is_the_cell_centre_not_the_target_centroid(self):
        """Two target sets occupying the same cells must give identical seeds.

        This is what makes a class stable: add a target and the seed does not
        move, so an accuracy number stays comparable across runs.
        """
        a = A.build_answer_space(_targets(per=2))
        b = A.build_answer_space(_targets(per=7))
        assert np.allclose(a.seeds["seed_lat"], b.seeds["seed_lat"])
        assert np.allclose(a.seeds["seed_lon"], b.seeds["seed_lon"])

    def test_seed_coordinates_round_trip_to_their_cell(self):
        space = A.build_answer_space(_targets())
        assert np.array_equal(
            H.ang2pix(space.seeds["seed_lat"], space.seeds["seed_lon"], space.nside),
            space.seeds["cell_id"].to_numpy(),
        )

    def test_seed_ids_are_ordered_by_cell_id_not_row_order(self):
        """So seed_id is a function of the occupied cell set alone — a shuffled
        input must not renumber the classes."""
        t = _targets()
        a = A.build_answer_space(t)
        b = A.build_answer_space(t.sample(frac=1.0, random_state=7).reset_index(drop=True))
        assert a.seeds["cell_id"].tolist() == b.seeds["cell_id"].tolist()
        assert a.seeds["cell_id"].is_monotonic_increasing

    def test_cell_offset_is_the_quantization_floor(self):
        """No estimator scores better than this against the seed, so it belongs
        in the metadata as the bound it is."""
        space = A.build_answer_space(_targets())
        offs = space.assignments["cell_offset_km"]
        assert (offs >= 0).all()
        # Within a cell, so bounded by the cell's own scale.
        assert offs.max() < H.nominal_cell_km(space.nside)
        assert space.meta["cell_offset_km"]["max"] == pytest.approx(offs.max(), abs=1e-3)


class TestNesting:
    """A coarse class is exactly the union of the fine classes inside it."""

    def test_every_target_keeps_its_parent_cell(self):
        t = _targets(per=2)
        fine = A.build_answer_space(t, nside=128)
        for nside in (64, 32, 16):
            coarse = A.build_answer_space(t, nside=nside)
            expected = H.degrade(fine.assignments["cell_id"].to_numpy(), 128, nside)
            got = coarse.assignments.set_index("target_id").loc[
                fine.assignments["target_id"]
            ]["cell_id"].to_numpy()
            assert np.array_equal(expected, got), nside

    def test_class_count_never_grows_as_cells_grow(self):
        t = _targets(per=2)
        counts = [A.build_answer_space(t, nside=n).n_seeds for n in H.NSIDE_LADDER]
        assert counts == sorted(counts, reverse=True)

    def test_occupied_by_nside_is_downward_only(self):
        space = A.build_answer_space(_targets(), nside=32)
        assert sorted(int(k) for k in space.meta["occupied_cells_by_nside"]) == [16, 32]


class TestAdjacency:
    def test_adjacency_is_exact_ring_one_membership(self):
        """v3 approximated this by sampling the great-circle path between seeds
        and counting boundary crossings, and undercounted by ~40% by its own
        measurement. Here it is the neighbour set, so it is exact."""
        # Two targets in adjacent cells: step one cell north of Denver.
        a = H.ang2pix([DENVER[0]], [DENVER[1]], 128)[0]
        nb = H.neighbours([a], 128)[0]
        b = int(nb[nb >= 0][0])
        centres = H.pix2ang([a, b], 128)
        t = pd.DataFrame(
            {
                "target_id": ["x", "y"],
                "target_lat": centres[:, 0],
                "target_lon": centres[:, 1],
            }
        )
        space = A.build_answer_space(t, nside=128)
        assert space.n_seeds == 2
        assert space.seeds["class_adjacency_degree"].tolist() == [1, 1]

    def test_distant_classes_have_degree_zero(self):
        space = A.build_answer_space(_targets(), nside=128)
        assert (space.seeds["class_adjacency_degree"] == 0).all()


class TestGeometryColumns:
    def test_margin_is_half_the_nearest_gap(self):
        space = A.build_answer_space(_targets())
        near = space.seeds["nearest_seed_km"].to_numpy()
        assert np.allclose(space.seeds["margin_km"].to_numpy(), near / 2.0, atol=1e-3)

    def test_mesh_is_symmetric_with_a_zero_diagonal(self):
        space = A.build_answer_space(_targets())
        m = space.seed_mesh_km.to_numpy(dtype=float)
        assert np.allclose(m, m.T)
        assert np.allclose(np.diag(m), 0.0)

    def test_a_single_class_has_no_nearest_neighbour(self):
        """One place means no gap to measure — NaN, not zero, which would read
        as 'another class is right here'."""
        t = _targets(coords=(DENVER,), per=3)
        space = A.build_answer_space(t)
        assert space.n_seeds == 1
        assert np.isnan(space.seeds["nearest_seed_km"].iloc[0])


class TestRoundTrip:
    def test_write_then_load_preserves_everything(self, tmp_path):
        space = A.build_answer_space(_targets(), source_label="unit")
        space.write(tmp_path)
        back = A.load_answer_space(tmp_path)
        assert back.nside == space.nside
        assert back.n_seeds == space.n_seeds
        assert back.meta["source"] == "unit"
        pd.testing.assert_frame_equal(
            back.seeds.drop(columns=["grid_scheme"]),
            space.seeds.drop(columns=["grid_scheme"]),
            check_dtype=False,
        )

    def test_cell_ids_survive_as_integers(self):
        """A CSV round trip hands back floats, and a float cell id silently
        fails every equality test the classifier makes."""
        import tempfile, pathlib

        with tempfile.TemporaryDirectory() as d:
            space = A.build_answer_space(_targets())
            space.write(pathlib.Path(d))
            back = A.load_answer_space(pathlib.Path(d))
            assert back.seeds["cell_id"].dtype == np.int64
            assert back.assignments["cell_id"].dtype == np.int64


class TestSweepRow:
    def test_row_carries_the_curve_fields(self):
        row = A.sweep_row(A.build_answer_space(_targets(), nside=64))
        assert row["grid"] == "healpix"
        assert row["nside"] == 64
        assert row["n_classes"] == 4
        assert row["cell_km"] == pytest.approx(101.9, abs=0.1)
        for key in ("cell_offset_km_p50", "nearest_seed_km_p50", "mean_adjacency_degree"):
            assert key in row
