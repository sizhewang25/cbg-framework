"""Tests for the answer-space map and its cartopy primitives.

Two of these are load-bearing rather than smoke:

`TestFrameCells` pins the property the lattice depends on — that the cells
selected for the frame **cover** the frame. Selection is by cell *centre*, so a
cell straddling the border has its centre outside and is only kept by the
padding; get that wrong and the lattice stops short of the edge in a ragged
fringe that reads as a feature of the data.

`TestPalette` pins the hexes against the validator run that chose them, and
against the manifest string that reports it, so a later "nicer orange" cannot
leave the figure claiming a validation it no longer passes.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v4.modules import healpix as H
from scripts.analysis.v4.modules import map_answer_space as MAS
from scripts.analysis.v4.modules import mapping as M
from scripts.analysis.v4.modules.paths import MissingArtifactError, RunPaths

US = M.US_MAINLAND_EXTENT


class TestAutoExtent:
    def test_pads_by_frac(self):
        lon_min, lon_max, lat_min, lat_max = M.auto_extent([10.0, 20.0], [0.0, 100.0])
        assert lon_min == pytest.approx(-8.0)
        assert lon_max == pytest.approx(108.0)
        assert lat_min == pytest.approx(9.2)
        assert lat_max == pytest.approx(20.8)

    def test_degenerate_span_gets_a_floor(self):
        """Every target in one metro must not give a zero-width frame."""
        lon_min, lon_max, lat_min, lat_max = M.auto_extent([40.0, 40.0], [-74.0, -74.0])
        assert lon_max - lon_min == pytest.approx(1.0)
        assert lat_max - lat_min == pytest.approx(1.0)

    def test_clips_to_valid_lonlat(self):
        assert M.auto_extent([-89.0, 89.0], [-179.0, 179.0]) == (
            -180.0,
            180.0,
            -90.0,
            90.0,
        )


class TestAngularRadius:
    def test_covers_every_corner(self):
        """The cone must reach the frame's corners, which the lon/lat diagonal
        does not measure: a degree of longitude is not a degree of distance."""
        r = M._angular_radius_deg(US)
        lon_min, lon_max, lat_min, lat_max = US
        c_lat, c_lon = np.radians((lat_min + lat_max) / 2), np.radians(
            (lon_min + lon_max) / 2
        )
        for lon in (lon_min, lon_max):
            for lat in (lat_min, lat_max):
                p_lat, p_lon = np.radians(lat), np.radians(lon)
                sep = np.degrees(
                    np.arccos(
                        np.sin(c_lat) * np.sin(p_lat)
                        + np.cos(c_lat) * np.cos(p_lat) * np.cos(p_lon - c_lon)
                    )
                )
                assert sep <= r + 1e-9

    def test_whole_sphere_is_a_half_turn(self):
        assert M._angular_radius_deg((-180.0, 180.0, -90.0, 90.0)) == pytest.approx(
            180.0, abs=1.0
        )


class TestFrameCells:
    @pytest.mark.parametrize("nside", H.NSIDE_LADDER)
    def test_covers_the_frame(self, nside):
        """Every cell any point of the frame falls in is selected.

        The real correctness property. Sampled rather than proved, but sampled
        densely enough (a lattice finer than the cell) that a missing border
        cell cannot hide.
        """
        got = set(int(c) for c in M.frame_cells(nside, US))
        lon_min, lon_max, lat_min, lat_max = US
        step = H.nominal_cell_km(nside) / 111.19 / 3.0
        lons = np.arange(lon_min, lon_max, step)
        lats = np.arange(lat_min, lat_max, step)
        grid_lon, grid_lat = np.meshgrid(lons, lats)
        needed = set(
            int(c) for c in H.ang2pix(grid_lat.ravel(), grid_lon.ravel(), nside)
        )
        assert needed <= got

    @pytest.mark.parametrize("nside", H.NSIDE_LADDER)
    def test_stays_near_the_frame(self, nside):
        """Padding is bounded: the lattice must not spill far off-screen and
        pay for cells the axes will clip anyway."""
        cells = M.frame_cells(nside, US)
        centres = H.pix2ang(cells, nside)
        pad = 2.0 * H.nominal_cell_km(nside) / 111.19
        assert centres[:, 1].min() >= US[0] - pad - 1e-9
        assert centres[:, 1].max() <= US[1] + pad + 1e-9
        assert centres[:, 0].min() >= US[2] - pad - 1e-9
        assert centres[:, 0].max() <= US[3] + pad + 1e-9

    def test_count_is_drawable(self):
        """The whole point of drawing the empty grid is that a continental frame
        holds few enough cells to. If a rung blows past this the figure needs a
        different strategy, not a bigger number here."""
        assert len(M.frame_cells(128, US)) < 12_000
        assert len(M.frame_cells(16, US)) < 200

    def test_finer_rungs_hold_more(self):
        counts = [len(M.frame_cells(n, US)) for n in H.NSIDE_LADDER]
        assert counts == sorted(counts, reverse=True)

    def test_refuses_an_antimeridian_frame(self):
        """The mask is a plain interval test, so a wrapped frame would select
        nothing rather than fail."""
        with pytest.raises(ValueError, match="antimeridian"):
            M.frame_cells(16, (170.0, -170.0, -10.0, 10.0))

    def test_global_frame_takes_every_pixel(self):
        assert len(M.frame_cells(16, (-180.0, 180.0, -90.0, 90.0))) == H.npix(16)


class TestPalette:
    def test_hexes_are_the_validated_pair(self):
        """Chosen by running the dataviz validator, not by eye."""
        assert M.TARGET_FILL == "#eb6834"
        assert M.VP_EDGE == "#2a78d6"

    def test_manifest_reports_the_hexes_it_validated(self, tmp_path):
        """The validation note names the pair, so the two cannot drift apart."""
        run = _write_run(tmp_path)
        MAS.build_for_run(run, analysis_root=tmp_path)
        body = json.loads(
            (
                run.analysis_dir("bipartite-graph", root=tmp_path)
                / MAS.FIGURE_MANIFEST
            ).read_text()
        )
        assert M.TARGET_FILL in body["palette_validated"]
        assert M.VP_EDGE in body["palette_validated"]

    def test_one_marker_size_for_both_node_kinds(self):
        """Target site and VP are two categories, not two magnitudes."""
        assert isinstance(M.MARKER_AREA, float)


class TestSites:
    def test_distinct_sites_collapses_replicas(self):
        """~20 IP replicas share a coordinate; a per-target scatter is 20 marks
        on one pixel."""
        targets = pd.DataFrame(
            {
                "target_id": [f"t{i}" for i in range(5)],
                "target_lat": [40.0, 40.0, 40.0, 41.0, 41.0],
                "target_lon": [-74.0, -74.0, -74.0, -75.0, -75.0],
                "cell_id": [1, 1, 1, 2, 2],
            }
        )
        assert len(MAS.distinct_sites(targets)) == 2

    def test_merged_cells_counts_cells_not_sites(self):
        targets = pd.DataFrame(
            {
                "target_id": list("abcd"),
                "target_lat": [40.0, 41.0, 42.0, 43.0],
                "target_lon": [-74.0, -75.0, -76.0, -77.0],
                # cell 1 holds two distinct sites; cell 2 and 3 hold one each
                "cell_id": [1, 1, 2, 3],
            }
        )
        assert MAS.count_merged_cells(targets) == 1

    def test_replicas_alone_do_not_count_as_a_merge(self):
        """20 targets at one coordinate are one site, not a quantization cost."""
        targets = pd.DataFrame(
            {
                "target_id": [f"t{i}" for i in range(20)],
                "target_lat": [40.0] * 20,
                "target_lon": [-74.0] * 20,
                "cell_id": [1] * 20,
            }
        )
        assert MAS.count_merged_cells(targets) == 0

    def test_empty_is_zero(self):
        empty = pd.DataFrame(
            columns=["target_id", "target_lat", "target_lon", "cell_id"]
        )
        assert MAS.count_merged_cells(empty) == 0


# ---- a run on disk -----------------------------------------------------------


def _write_run(root, *, n_sites=4, replicas=3, n_vps=6):
    """A run carrying only what `plot-answer-space` reads: the bipartite rungs.

    Deliberately not an answer-space build. The command must work without one —
    an occupied target cell IS a class — and a fixture that supplied both would
    let a stray dependency pass unnoticed.
    """
    run = RunPaths(run_id="as99-260728-260802-mesh", root=root, source="gen", setup="s")
    lats = np.linspace(33.0, 45.0, n_sites)
    lons = np.linspace(-118.0, -74.0, n_sites)
    targets = pd.DataFrame(
        {
            "target_id": [f"tg-{i}-{r}" for i in range(n_sites) for r in range(replicas)],
            "target_lat": np.repeat(lats, replicas),
            "target_lon": np.repeat(lons, replicas),
        }
    )
    vps = pd.DataFrame(
        {
            "vp_id": [f"vp-{i}" for i in range(n_vps)],
            "vp_lat": np.linspace(30.0, 47.0, n_vps),
            "vp_lon": np.linspace(-122.0, -71.0, n_vps),
        }
    )
    from scripts.analysis.v4.modules.bipartite import co_quantize

    for nside in H.NSIDE_LADDER:
        co_quantize(targets, vps, nside).write(run.bipartite_dir(nside, root=root))
    return run


class TestRungCounts:
    def test_reports_both_sides_and_the_merge(self, tmp_path):
        run = _write_run(tmp_path)
        counts = MAS.rung_counts(run, 128, analysis_root=tmp_path)
        assert counts["nside"] == 128
        assert counts["cell_km"] == pytest.approx(50.9, abs=0.1)
        assert counts["n_sites"] == 4
        assert counts["n_vps"] == 6
        assert counts["n_shared_cells"] <= counts["n_target_cells"]
        assert counts["n_shared_cells"] <= counts["n_vp_cells"]

    def test_merges_are_monotone_down_the_ladder(self, tmp_path):
        """Cells only ever merge as the grid coarsens — exact nesting."""
        run = _write_run(tmp_path)
        cells = [
            MAS.rung_counts(run, n, analysis_root=tmp_path)["n_target_cells"]
            for n in H.NSIDE_LADDER
        ]
        assert cells == sorted(cells, reverse=True)

    def test_missing_bipartite_names_the_remedy(self, tmp_path):
        run = RunPaths(run_id="nope", root=tmp_path, source="gen", setup="s")
        with pytest.raises(MissingArtifactError, match="build-bipartite"):
            MAS.rung_counts(run, 128, analysis_root=tmp_path)


class TestBuild:
    def test_writes_figure_and_manifest_beside_the_curve(self, tmp_path):
        run = _write_run(tmp_path)
        png = MAS.build_for_run(run, analysis_root=tmp_path)
        out = run.analysis_dir("bipartite-graph", root=tmp_path)
        assert png == out / MAS.FIGURE_PNG
        assert png.stat().st_size > 10_000
        assert (out / MAS.FIGURE_MANIFEST).exists()
        # One level above the rung directories, not inside one.
        assert not (out / "healpix-128" / MAS.FIGURE_PNG).exists()

    def test_manifest_carries_every_rung(self, tmp_path):
        run = _write_run(tmp_path)
        MAS.build_for_run(run, analysis_root=tmp_path)
        body = json.loads(
            (
                run.analysis_dir("bipartite-graph", root=tmp_path)
                / MAS.FIGURE_MANIFEST
            ).read_text()
        )
        assert [r["nside"] for r in body["rungs"]] == list(H.NSIDE_LADDER)
        assert body["run"] == run.run_id
        assert body["extent"]["lon_min"] == US[0]

    def test_manifest_points_at_the_existing_csv(self, tmp_path):
        """No CSV twin is written; the merged curve already holds the numbers."""
        run = _write_run(tmp_path)
        MAS.build_for_run(run, analysis_root=tmp_path)
        out = run.analysis_dir("bipartite-graph", root=tmp_path)
        body = json.loads((out / MAS.FIGURE_MANIFEST).read_text())
        assert body["csv"].startswith("../")
        assert not list(out.glob("answer_space_map*.csv"))

    def test_auto_extent_frames_the_run(self, tmp_path):
        run = _write_run(tmp_path)
        lon_min, lon_max, lat_min, lat_max = MAS.auto_extent_for_run(
            run, analysis_root=tmp_path
        )
        # VPs reach wider than the targets on both axes, so they must be inside.
        assert lon_min < -122.0 and lon_max > -71.0
        assert lat_min < 30.0 and lat_max > 47.0
