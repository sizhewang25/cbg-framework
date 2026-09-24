"""The answer-space map: frame, counts, and a render over a synthetic run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import answer_space as A
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules import map_answer_space as MA
from scripts.analysis.v5.modules import mapping as M
from scripts.analysis.v5.modules.paths import MissingArtifactError, grid_slug

EWR = (40.6895, -74.1745)
JFK = (40.6413, -73.7781)
SEATTLE = (47.449, -122.309)
OMAHA = (41.2565, -95.9345)


@dataclass(frozen=True)
class _Run:
    """The slice of `RunPaths` the map touches."""

    run_id: str
    root: Path
    source: str = "src"
    setup: str = "setup"

    def analysis_dir(self, kind, *, root=None):
        d = (root or self.root) / self.run_id / kind
        d.mkdir(parents=True, exist_ok=True)
        return d

    def answer_space_dir(self, nside, *, root=None):
        d = self.analysis_dir(A.ANSWER_SPACE_KIND, root=root) / grid_slug(nside)
        d.mkdir(parents=True, exist_ok=True)
        return d


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("v5-map")
    r = _Run("syn-000000-000000-mesh", root)
    tgs = pd.DataFrame(
        [
            {"tg_id": f"tg-{i}-{k}", "tg_lat": lat, "tg_lon": lon}
            for i, (lat, lon) in enumerate((EWR, JFK, SEATTLE, OMAHA))
            for k in range(2)
        ]
    )
    for nside in G.NSIDE_LADDER:
        A.build_answer_space(tgs, nside=nside, run_id=r.run_id).write(r.answer_space_dir(nside))
    return r


def test_the_lattice_covers_the_frame_at_every_rung():
    for nside in G.NSIDE_LADDER:
        ids = M.frame_grids(nside, M.US_MAINLAND_EXTENT)
        assert ids.size > 0 and np.unique(ids).size == ids.size
    assert M.frame_grids(16, M.US_MAINLAND_EXTENT).size < M.frame_grids(128, M.US_MAINLAND_EXTENT).size


def test_an_antimeridian_frame_is_refused():
    with pytest.raises(ValueError, match="antimeridian"):
        M.frame_grids(64, (170.0, -170.0, 0.0, 10.0))


def test_a_missing_rung_names_the_command(tmp_path):
    with pytest.raises(MissingArtifactError, match="build-answer-space"):
        MA.load_rung(_Run("nobody", tmp_path), 128)


def test_render_writes_the_figure_and_a_manifest_per_rung(run):
    png = MA.build_for_run(run)
    assert png.exists() and png.stat().st_size > 10_000
    manifest = json.loads((png.parent / MA.FIGURE_MANIFEST).read_text())
    rungs = {r["nside"]: r for r in manifest["rungs"]}
    assert sorted(rungs) == sorted(G.NSIDE_LADDER)
    # EWR and JFK: two grids, one seed at the finest rung.
    assert rungs[128]["n_sites"] == 4 and rungs[128]["n_seeds"] == 3
    for r in rungs.values():
        assert r["n_cells_drawn"] == r["n_seeds"]
        assert r["cell_polygon_agreement"] > 0.98
        assert r["landmass_buffer_km"] == pytest.approx(G.grid_km(r["nside"]), abs=1e-3)
