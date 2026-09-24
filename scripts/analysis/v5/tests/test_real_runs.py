"""v5 against the real benchmark output, not fixtures.

Skipped when the benchmark output is absent, so a clean checkout still passes.
The v4 comparison rebuilds v4 in a temp root, so it never reads a stale tree.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.analysis.v5.modules import answer_space as A
from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.landmass import load_landmass
from scripts.analysis.v5.modules.paths import MissingArtifactError, resolve_run

MESH_RUNS = (
    "as01-260728-260802-mesh",
    "as02-260728-260802-mesh",
    "as03-260728-260802-mesh",
)


def _run(run_id):
    try:
        run = resolve_run(run_id)
    except MissingArtifactError as exc:
        pytest.skip(f"{run_id} not available: {exc}")
    if not run.combo_ids:
        pytest.skip(f"{run_id} has no scored combo")
    return run


@pytest.fixture(scope="module", params=MESH_RUNS)
def scored(request, tmp_path_factory):
    run = _run(request.param)
    root = tmp_path_factory.mktemp("v5")
    spaces = A.build_for_run(run, analysis_root=root)
    long = C.score_for_run(run, analysis_root=root)
    return run, root, spaces, long


def test_every_site_is_inland_at_every_rung(scored):
    _, _, spaces, _ = scored
    for s in spaces:
        assert load_landmass(s.grid_km).contains(s.sites["site_lat"], s.sites["site_lon"]).all()


def test_seeds_never_outnumber_sites_and_only_merge_as_grids_grow(scored):
    _, _, spaces, _ = scored
    by_nside = {s.nside: s for s in spaces}
    counts = [by_nside[n].n_seeds for n in G.NSIDE_LADDER]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] <= by_nside[128].meta["n_sites"]


def test_the_grid_axis_is_monotone(scored):
    assert C.monotonicity_violations(scored[3]).empty


def test_both_guards_hold_on_every_rung(scored):
    long = scored[3]
    C.guard_partition(long)
    C.guard_cross_tab(long)


@pytest.mark.parametrize("nside", G.NSIDE_LADDER)
def test_the_grid_axis_matches_v4_row_for_row(scored, nside, tmp_path):
    """`ring` and `pred_dist_to_tg_km` are v4's `ring` and `error_km`, ported."""
    from scripts.analysis.v4.modules import answer_space as A4
    from scripts.analysis.v4.modules import classify as C4
    from scripts.analysis.v4.modules.paths import resolve_run as resolve_v4

    run, root, _, _ = scored
    run4 = resolve_v4(run.run_id)
    A4.build_for_run(run4, nsides=(nside,), analysis_root=tmp_path)
    C4.score_rung(run4, nside, analysis_root=tmp_path)
    v4_dir = run4.cls_accuracy_dir(nside, root=tmp_path)
    v5_dir = run.classify_dir(nside, root=root)
    for p5 in sorted(v5_dir.glob("*_tgs.parquet")):
        method = p5.name.removesuffix("_tgs.parquet")
        a = pd.read_parquet(p5).set_index("tg_id").sort_index()
        b = pd.read_parquet(v4_dir / f"{method}_cells.parquet").set_index("target_id").sort_index()
        assert a.index.equals(b.index), method
        assert (a["ring"].to_numpy() == b["ring"].to_numpy()).all(), method
        pd.testing.assert_series_equal(
            a["pred_dist_to_tg_km"], b["error_km"], check_names=False, check_index=False
        )
