"""Answer-space construction invariants (paper §7.3/§7.4).

Almost everything here is parametrized over **both grids**. That is the point of
the `grid.Grid` abstraction: the grid supplies cell membership and cell centre,
and everything the construction builds on top of those (nearest-seed labelling,
seed mesh, margins) is pure spherical geometry — so every invariant has to hold
whichever tessellation did the merging. A test that passes on one grid and fails
on the other marks a place where grid detail leaked past the two methods that
are allowed to know about it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules.answer_space import (
    build_answer_space,
    load_answer_space,
    pairwise_km,
)
from scripts.analysis.v3.modules.grid import get_grid
from scripts.analysis.v3.modules.paths import MissingArtifactError


@pytest.fixture(params=("h3", "healpix"))
def grid(request):
    """Every construction invariant, on both tessellations."""
    return request.param


def _targets(coords: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_id": [f"tg-{i}" for i in range(len(coords))],
            "target_lat": [c[0] for c in coords],
            "target_lon": [c[1] for c in coords],
        }
    )


def test_colocated_targets_collapse_to_one_seed(grid):
    """The grid's whole job: merge points close enough to count as one place."""
    space = build_answer_space(_targets([(41.9742, -87.9073)] * 5), grid=grid)
    assert space.n_seeds == 1
    assert space.seeds.loc[0, "n_targets"] == 5


def test_distant_targets_get_distinct_seeds(grid):
    space = build_answer_space(
        _targets([(41.97, -87.90), (37.46, -121.92), (40.71, -74.01)]), grid=grid
    )
    assert space.n_seeds == 3
    assert set(space.assignments["seed_id"]) == {0, 1, 2}


def test_seed_is_the_cell_centre_not_the_targets(grid):
    """The seed is the grid's, not the data's — that is the whole point."""
    coords = [(41.90, -87.90), (41.91, -87.91)]
    space = build_answer_space(_targets(coords), grid=grid)
    assert space.n_seeds == 1
    g = get_grid(grid)
    res = int(space.seeds.loc[0, "grid_resolution"])
    centre = g.cell_centers([space.seeds.loc[0, "cell_id"]], res)[0]
    assert space.seeds.loc[0, "seed_lat"] == pytest.approx(centre[0])
    assert space.seeds.loc[0, "seed_lon"] == pytest.approx(centre[1])
    # The negative half is what pins the change: a centroid would have landed
    # between the two targets, and the cell centre lands on neither.
    for lat, _ in coords:
        assert space.seeds.loc[0, "seed_lat"] != pytest.approx(lat, abs=1e-4)


def test_seed_does_not_depend_on_which_targets_landed_in_the_cell(grid):
    """Target-independence: the same cell yields the same seed in every run."""
    a = build_answer_space(_targets([(41.90, -87.90)]), grid=grid)
    b = build_answer_space(_targets([(41.99, -87.99), (41.95, -87.95)]), grid=grid)
    assert a.seeds.loc[0, "cell_id"] == b.seeds.loc[0, "cell_id"]
    assert a.seeds.loc[0, "seed_lat"] == b.seeds.loc[0, "seed_lat"]
    assert a.seeds.loc[0, "seed_lon"] == b.seeds.loc[0, "seed_lon"]


def test_cell_offset_is_bounded_by_the_cell(grid):
    """The quantization cost the grid + resolution declare up front."""
    coords = [(25 + i * 0.7, -120 + i * 1.3) for i in range(40)]
    space = build_answer_space(_targets(coords), grid=grid)
    g = get_grid(grid)
    res = int(space.seeds.loc[0, "grid_resolution"])
    assert space.assignments["cell_offset_km"].max() < g.nominal_cell_km(res)


def test_every_target_is_assigned_exactly_once(grid):
    coords = [(25 + i * 0.7, -120 + i * 1.3) for i in range(40)]
    space = build_answer_space(_targets(coords), grid=grid)
    assert len(space.assignments) == 40
    assert space.assignments["target_id"].is_unique
    assert space.assignments["seed_id"].isin(space.seeds["seed_id"]).all()


def test_seed_ids_are_contiguous_and_order_independent(grid):
    """seed_id must not depend on input row order, or scoring is unstable."""
    coords = [(41.97, -87.90), (37.46, -121.92), (40.71, -74.01)]
    a = build_answer_space(_targets(coords), grid=grid)
    shuffled = _targets(coords).iloc[::-1].reset_index(drop=True)
    b = build_answer_space(shuffled, grid=grid)
    assert list(a.seeds["seed_id"]) == list(range(a.n_seeds))
    pd.testing.assert_frame_equal(
        a.seeds.drop(columns=["seed_id"]).reset_index(drop=True),
        b.seeds.drop(columns=["seed_id"]).reset_index(drop=True),
    )


def test_margin_is_half_the_nearest_seed_distance(grid):
    """§7.4: the class boundary bisects the geodesic joining two seeds."""
    space = build_answer_space(_targets([(41.97, -87.90), (40.71, -74.01)]), grid=grid)
    assert space.seeds["margin_km"].to_numpy() == pytest.approx(
        space.seeds["nearest_seed_km"].to_numpy() / 2.0, rel=1e-6
    )


def test_seed_separation_is_a_grid_distance_not_a_data_one(grid):
    """`nearest_seed_km` / `margin_km` now describe the grid, not the targets.

    This is the property the mistaken-cell-closeness analysis rests on: two runs
    whose targets sit anywhere inside the same two cells must report the same
    separation. Under target-centroid seeds it varied with the data — as7018
    `h3-4` had a `margin_km` of 7.7 km between two cells 45 km apart, because
    both centroids hugged their shared edge.

    Kept loose against the pitch on purpose: H3 defines `nominal_cell_km` as true
    centre-to-centre, HEALPix as `sqrt(area)`, so only the order of magnitude is
    comparable across the two.
    """
    g = get_grid(grid)
    res = g.DEFAULT_RESOLUTION
    base = (41.90, -87.90)
    first = g.cell_ids([base[0]], [base[1]], res)[0]
    step = next(
        d
        for d in np.arange(0.05, 2.0, 0.01)
        if g.cell_ids([base[0]], [base[1] + d], res)[0] != first
    )
    a = build_answer_space(_targets([base, (base[0], base[1] + step)]), grid=grid)
    # Same two cells, targets nudged to different corners of them.
    b = build_answer_space(
        _targets([(base[0] + 0.02, base[1] - 0.02), (base[0], base[1] + step + 0.02)]),
        grid=grid,
    )
    assert a.n_seeds == b.n_seeds == 2
    assert list(a.seeds["cell_id"]) == list(b.seeds["cell_id"])
    assert list(a.seeds["nearest_seed_km"]) == list(b.seeds["nearest_seed_km"])
    assert list(a.seeds["margin_km"]) == list(b.seeds["margin_km"])
    pitch = g.nominal_cell_km(res)
    assert pitch / 2 < a.seeds["nearest_seed_km"].max() < pitch * 2


def test_duplicate_target_id_is_rejected(grid):
    t = _targets([(41.97, -87.90), (40.71, -74.01)])
    t.loc[1, "target_id"] = t.loc[0, "target_id"]
    with pytest.raises(ValueError, match="duplicate target_id"):
        build_answer_space(t, grid=grid)


def test_empty_targets_is_rejected(grid):
    with pytest.raises(ValueError, match="empty"):
        build_answer_space(_targets([]), grid=grid)


@pytest.mark.parametrize(
    "grid_name,fine,coarse", [("h3", 5, 2), ("healpix", 128, 16)]
)
def test_finer_resolution_cannot_reduce_seed_count(grid_name, fine, coarse):
    coords = [(30 + i * 0.4, -100 + i * 0.4) for i in range(30)]
    t = _targets(coords)
    n_coarse = build_answer_space(t, grid=grid_name, resolution=coarse).n_seeds
    n_fine = build_answer_space(t, grid=grid_name, resolution=fine).n_seeds
    assert n_fine >= n_coarse


def test_resolution_defaults_to_the_grids_own(grid):
    """A grid and a resolution from a *different* grid must never be pairable.

    `nside=128` and `res=4` are both "the default", but only one is valid per
    grid, so the default has to come from the grid rather than from a constant.
    """
    from scripts.analysis.v3.modules.grid import get_grid

    space = build_answer_space(_targets([(41.97, -87.90)]), grid=grid)
    assert space.meta["grid"]["resolution"] == get_grid(grid).DEFAULT_RESOLUTION
    assert space.meta["grid"]["scheme"] == grid


def test_roundtrip_through_disk(tmp_path, grid):
    """`cell_id` must survive CSV untyped storage on both grids.

    HEALPix ids are int64 and H3's are hex strings; pandas infers per column, so
    without `Grid.coerce_cell_ids` one of the two comes back as the wrong dtype
    and stops matching `assignments`.
    """
    space = build_answer_space(
        _targets([(41.97, -87.90), (37.46, -121.92), (40.71, -74.01)]), grid=grid
    )
    space.write(tmp_path)
    back = load_answer_space(tmp_path)
    assert back.n_seeds == space.n_seeds
    pd.testing.assert_frame_equal(back.seeds, space.seeds)
    pd.testing.assert_frame_equal(back.assignments, space.assignments)
    assert back.meta["grid"]["resolution"] == space.meta["grid"]["resolution"]
    assert back.meta["grid"]["scheme"] == space.meta["grid"]["scheme"]


def test_mesh_is_symmetric_with_zero_diagonal(grid):
    space = build_answer_space(
        _targets([(41.97, -87.90), (37.46, -121.92), (40.71, -74.01)]), grid=grid
    )
    m = space.seed_mesh_km.to_numpy()
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 0.0)


def test_hierarchy_reports_only_the_grid_built_and_coarser(grid):
    """Regression: the diagnostic used to describe grids that were never built.

    `occupied_cells_by_resolution` walked a fixed hierarchy regardless of the
    resolution in use, so at the coarsest rung every count came from *finer*
    grids than the answer space itself.
    """
    from scripts.analysis.v3.modules.grid import get_grid

    g = get_grid(grid)
    coarsest = g.HIERARCHY[-1]
    space = build_answer_space(
        _targets([(41.97, -87.90), (37.46, -121.92)]), grid=grid, resolution=coarsest
    )
    reported = [int(k) for k in space.meta["occupied_cells_by_resolution"]]
    assert reported == [coarsest]


def test_loading_a_target_centroid_answer_space_is_rejected(tmp_path, grid):
    """Pre-cell-centre artifacts must fail loudly, not half-load."""
    space = build_answer_space(_targets([(41.97, -87.90), (40.71, -74.01)]), grid=grid)
    space.write(tmp_path)
    stale = pd.read_csv(tmp_path / "seeds.csv").rename(
        columns={"seed_lat": "centroid_lat", "seed_lon": "centroid_lon"}
    )
    stale.to_csv(tmp_path / "seeds.csv", index=False)
    with pytest.raises(MissingArtifactError, match="Rebuild"):
        load_answer_space(tmp_path)


# ---- the traffic-weighted class set ----------------------------------------
#
# A weighted run's `targets.csv` is POST-filter, so building the answer space
# from it drops classes and inflates every method's accuracy. `build_for_run`
# therefore rebuilds a weighted run's classes from the config's pre-filter
# `mesh_csv_path`. These pin that it happens, that it is refused rather than
# silently defaulted when the mesh cannot be found, and that a non-weighted run
# is untouched.


def _canonical_csv(path, coords, *, ids=None, rows_per_target=3):
    """A canonical (VP, target) CSV: several rows per target, as the real ones are."""
    ids = ids or [f"tg-{i}" for i in range(len(coords))]
    rec = []
    for tid, (lat, lon) in zip(ids, coords):
        for v in range(rows_per_target):
            rec.append(
                {"vp_id": f"vp-{v}", "vp_lat": 0.0, "vp_lon": 0.0,
                 "target_id": tid, "target_lat": lat, "target_lon": lon,
                 "rtt_ms": 10.0, "weight": 1.0}
            )
    pd.DataFrame(rec).to_csv(path, index=False)


def _weighted_run(tmp_path, monkeypatch, *, mesh_coords, kept, cfg_body=None):
    """A minimal `traffic_weighted_csv` run plus its config, both under tmp_path.

    Returns `(RunPaths, mesh_csv)`. `kept` indexes into `mesh_coords` and becomes
    the run's post-filter `targets.csv`.
    """
    import yaml
    from scripts.analysis.v3.modules import config as config_mod
    from scripts.analysis.v3.modules.paths import RunPaths

    mesh_csv = tmp_path / "mesh.csv"
    _canonical_csv(mesh_csv, mesh_coords)

    setup = tmp_path / "outputs" / "wrun" / "traffic_weighted_csv" / "anchors_to_probes"
    (setup / "fold_0").mkdir(parents=True)
    pd.DataFrame(
        {
            "target_id": [f"tg-{i}" for i in kept],
            "target_lat": [mesh_coords[i][0] for i in kept],
            "target_lon": [mesh_coords[i][1] for i in kept],
        }
    ).to_csv(setup / "targets.csv", index=False)

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    body = cfg_body if cfg_body is not None else {
        "run_id": "wrun",
        "benchmark": {"source_kwargs": {"mesh_csv_path": str(mesh_csv)}},
        "analysis": {},
    }
    (cfg_dir / "wrun.yaml").write_text(yaml.safe_dump(body))
    # `config_path_for_run` is anchored on REPO_ROOT, so redirect it rather than
    # writing a config into the real repo from a test.
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)

    run = RunPaths(
        run_id="wrun", root=tmp_path / "outputs",
        source="traffic_weighted_csv", setup="anchors_to_probes",
    )
    return run, mesh_csv


#: Five coordinates, each in its own cell on both grids, and far enough apart
#: that dropping some strictly reduces the class count.
_SPREAD = [(41.9742, -87.9073), (34.0489, -118.2570), (40.7178, -74.0090),
           (29.7520, -95.3660), (47.6146, -122.3390)]


def test_weighted_run_keeps_the_pre_filter_class_set(tmp_path, monkeypatch):
    """The whole point: filtered targets, full mesh classes.

    Scoring the survivors against their own cells would leave 2 classes; the
    mesh universe keeps 5. That difference is the accuracy inflation the fix
    exists to prevent.
    """
    from scripts.analysis.v3.modules.answer_space import build_for_run

    run, _ = _weighted_run(tmp_path, monkeypatch, mesh_coords=_SPREAD, kept=[0, 1])
    space = build_for_run(run, grid="h3", resolution=4)

    assert space.n_seeds == 5
    assert space.meta["n_targets"] == 5
    prov = space.meta["targets_provenance"]
    assert prov["n_targets_scored_by_run"] == 2
    assert prov["n_targets_in_space"] == 5
    assert prov["n_seeds_if_built_from_run_targets"] == 2
    assert prov["n_seeds_with_no_scored_target"] == 3
    # Every scored target must still be present, or `classify` would refuse it.
    assert {"tg-0", "tg-1"} <= set(space.assignments["target_id"])


def test_generic_run_is_untouched(tmp_path, monkeypatch):
    """A non-weighted run still reads targets.csv and consults no config."""
    from scripts.analysis.v3.modules.answer_space import build_for_run
    from scripts.analysis.v3.modules.paths import RunPaths

    run, _ = _weighted_run(tmp_path, monkeypatch, mesh_coords=_SPREAD, kept=[0, 1])
    generic = RunPaths(
        run_id="wrun", root=run.root, source="generic_csv", setup="anchors_to_probes",
    )
    (run.root / "wrun" / "generic_csv" / "anchors_to_probes").mkdir(parents=True)
    (generic.setup_dir / "fold_0").mkdir()
    pd.read_csv(run.targets_csv).to_csv(generic.targets_csv, index=False)

    space = build_for_run(generic, grid="h3", resolution=4)
    assert space.n_seeds == 2
    assert "targets_provenance" not in space.meta


def test_weighted_run_refuses_a_missing_mesh_path(tmp_path, monkeypatch):
    """Absent `mesh_csv_path` must raise, never fall back to targets.csv.

    The fallback is the bug: it would score 2 classes and look normal.
    """
    import typer
    from scripts.analysis.v3.modules.answer_space import build_for_run

    run, _ = _weighted_run(
        tmp_path, monkeypatch, mesh_coords=_SPREAD, kept=[0, 1],
        cfg_body={"run_id": "wrun", "benchmark": {"source_kwargs": {"k": 5}},
                  "analysis": {}},
    )
    with pytest.raises(typer.BadParameter, match="mesh_csv_path"):
        build_for_run(run, grid="h3", resolution=4)


def test_weighted_run_refuses_a_nonexistent_mesh_csv(tmp_path, monkeypatch):
    from scripts.analysis.v3.modules.answer_space import build_for_run

    run, _ = _weighted_run(
        tmp_path, monkeypatch, mesh_coords=_SPREAD, kept=[0, 1],
        cfg_body={"run_id": "wrun",
                  "benchmark": {"source_kwargs": {"mesh_csv_path": "nope/absent.csv"}},
                  "analysis": {}},
    )
    with pytest.raises(MissingArtifactError, match="does not exist"):
        build_for_run(run, grid="h3", resolution=4)


def test_weighted_run_refuses_a_mesh_missing_a_scored_target(tmp_path, monkeypatch):
    """The subset guard: a config pointing at the wrong mesh fails loudly."""
    from scripts.analysis.v3.modules.answer_space import build_for_run

    run, mesh_csv = _weighted_run(tmp_path, monkeypatch, mesh_coords=_SPREAD, kept=[0, 1])
    # Rewrite the mesh without tg-1, which the run still scores.
    _canonical_csv(mesh_csv, _SPREAD[2:], ids=["tg-2", "tg-3", "tg-4"])
    with pytest.raises(ValueError, match="absent from"):
        build_for_run(run, grid="h3", resolution=4)


def test_weighted_mesh_roster_is_deduplicated(tmp_path, monkeypatch):
    """A canonical CSV has many rows per target; the space takes one each."""
    from scripts.analysis.v3.modules.answer_space import build_for_run

    run, _ = _weighted_run(tmp_path, monkeypatch, mesh_coords=_SPREAD, kept=[0])
    space = build_for_run(run, grid="h3", resolution=4)
    assert space.meta["n_targets"] == len(_SPREAD)  # not 3x that
    assert not space.assignments["target_id"].duplicated().any()
