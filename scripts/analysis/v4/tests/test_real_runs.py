"""v4 against the real benchmark output, not fixtures.

Fixture tests can only show the metric is self-consistent. These show it holds
on the data that exposed the defect — and the monotonicity guarantee is exactly
the kind of claim that is easy to satisfy on four hand-placed points and hard on
1,269 real predictions, which is where H3 failed it 705 times.

Skipped when the benchmark output is absent, so a clean checkout still passes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.analysis.v4.modules import answer_space as A
from scripts.analysis.v4.modules import bipartite as B
from scripts.analysis.v4.modules import classify as C
from scripts.analysis.v4.modules import healpix as H
from scripts.analysis.v4.modules.paths import MissingArtifactError, resolve_run

MESH_RUNS = (
    "as01-260728-260802-mesh",
    "as02-260728-260802-mesh",
    "as03-260728-260802-mesh",
)

#: The case that motivated v4. as01 fold_4: truth Seattle, Spotter predicted the
#: Canadian Arctic 2,360 km away, and v3's nearest-seed rule scored it CORRECT.
REGRESSION_TARGET = "tg-e1a1545"


def _h3_density_arm(run) -> str:
    """Whichever combo id holds the H3-grid density predictions.

    The Arctic prediction belongs to the density MTL **on its old H3 grid**, and
    permanently: when that MTL moved to HEALPix nside 128, `cli.py rename-combo`
    preserved the H3 results as `spotter_h3_cbg` and `spotter_cbg` became the
    HEALPix arm. The HEALPix arm is under no obligation to reproduce the same
    prediction for the same target, so the evidence has to be read from the arm
    that produced it.

    Resolved at runtime rather than pinned, because `outputs/` is gitignored:
    the rename is a filesystem mutation git cannot see, so on one machine the
    backup exists and on another it does not. Pinning either name would make
    this suite pass or fail on a fact about the checkout.
    """
    return "spotter_h3_cbg" if "spotter_h3_cbg" in run.combo_ids else "spotter_cbg"


def _run(run_id):
    try:
        run = resolve_run(run_id)
    except MissingArtifactError as exc:
        pytest.skip(f"{run_id} not available: {exc}")
    if not run.combo_ids:
        pytest.skip(f"{run_id} has no scored combo")
    return run


@pytest.fixture(scope="module")
def as01(tmp_path_factory):
    """Answer space + scores for as01, into a temp root so a test run never
    touches the real artifacts."""
    run = _run(MESH_RUNS[0])
    root = tmp_path_factory.mktemp("v4")
    A.build_for_run(run, analysis_root=root)
    long = C.score_for_run(run, analysis_root=root)
    return run, root, long


class TestMonotonicityOnRealPredictions:
    """The guarantee, on the data where H3 broke it 705 times."""

    @pytest.mark.parametrize("run_id", MESH_RUNS)
    def test_no_violations_on_any_mesh(self, run_id, tmp_path):
        run = _run(run_id)
        A.build_for_run(run, analysis_root=tmp_path)
        long = C.score_for_run(run, analysis_root=tmp_path)
        bad = C.monotonicity_violations(long)
        assert bad.empty, f"{run_id}:\n{bad.to_string(index=False)}"

    def test_ring0_rises_as_cells_grow(self, as01):
        """Not merely non-decreasing — on real data the tolerance dial should
        visibly do something, or the ladder carries no information."""
        _, _, long = as01
        for method, g in long.groupby("method"):
            g = g.sort_values("nside", ascending=False)
            acc = g["accuracy_ring0"].tolist()
            assert acc == sorted(acc), f"{method}: {acc}"
        coarsest = long[long.nside == long.nside.min()]["accuracy_ring0"].mean()
        finest = long[long.nside == long.nside.max()]["accuracy_ring0"].mean()
        assert coarsest > finest


class TestTheRegressionCase:
    def test_the_arctic_prediction_is_never_credited(self, as01):
        run, root, _ = as01
        method = _h3_density_arm(run)
        for nside in H.NSIDE_LADDER:
            cells = pd.read_parquet(
                run.cls_accuracy_dir(nside, root=root)
                / C.CELLS_PARQUET.format(method=method)
            )
            row = cells[cells.target_id == REGRESSION_TARGET]
            if row.empty:
                pytest.skip(f"{REGRESSION_TARGET} absent from this run")
            assert row.iloc[0]["ring"] == -1, f"placed at nside={nside}"
            assert row.iloc[0]["error_km"] > 2000

    def test_the_retired_rule_still_shows_why_this_changed(self, as01):
        """The evidence has to survive alongside the fix, or the change looks
        arbitrary in six months."""
        run, root, _ = as01
        cells = pd.read_parquet(
            run.cls_accuracy_dir(128, root=root)
            / C.CELLS_PARQUET.format(method=_h3_density_arm(run))
        )
        row = cells[cells.target_id == REGRESSION_TARGET]
        if row.empty:
            pytest.skip(f"{REGRESSION_TARGET} absent from this run")
        assert row.iloc[0]["nearest_seed_id_retired"] == row.iloc[0]["tg_seed_id"]


class TestAnswerSpaceOnRealData:
    def test_as01_reproduces_the_known_class_ladder(self, as01):
        """v3's HEALPix answer space emitted {128: 18, 64: 18, 32: 17, 16: 15}
        for as01. v4 is a reimplementation, so it must agree."""
        run, root, _ = as01
        space = A.load_answer_space(run.answer_space_dir(128, root=root))
        assert space.meta["occupied_cells_by_nside"] == {
            "128": 18, "64": 18, "32": 17, "16": 15
        }

    def test_seeds_round_trip_at_every_rung(self, as01):
        import numpy as np

        run, root, _ = as01
        for nside in H.NSIDE_LADDER:
            s = A.load_answer_space(run.answer_space_dir(nside, root=root))
            assert np.array_equal(
                H.ang2pix(s.seeds["seed_lat"], s.seeds["seed_lon"], nside),
                s.seeds["cell_id"].to_numpy(),
            )

    def test_rungs_are_exactly_nested(self, as01):
        """Every target's coarse cell is the bit-shifted fine one — the
        property the whole ladder rests on."""
        import numpy as np

        run, root, _ = as01
        fine = A.load_answer_space(run.answer_space_dir(128, root=root))
        fine_map = fine.assignments.set_index("target_id")["cell_id"]
        for nside in (64, 32, 16):
            coarse = A.load_answer_space(run.answer_space_dir(nside, root=root))
            got = coarse.assignments.set_index("target_id")["cell_id"].reindex(
                fine_map.index
            )
            assert np.array_equal(
                H.degrade(fine_map.to_numpy(), 128, nside), got.to_numpy()
            )


class TestBipartiteOnRealData:
    def test_vp_and_target_cells_both_shrink_up_the_ladder(self, tmp_path):
        run = _run(MESH_RUNS[0])
        qs = B.build_for_run(run, analysis_root=tmp_path)
        tg = [q.meta["n_target_cells"] for q in qs]
        vp = [q.meta["n_vp_cells"] for q in qs]
        assert tg == sorted(tg, reverse=True)
        assert vp == sorted(vp, reverse=True)

    def test_every_target_cell_eventually_holds_a_vp(self, tmp_path):
        """A property of this dataset worth pinning: by the coarsest rung the
        VPs cover the whole answer space, so a failure there is the latency
        model's and not the geometry's."""
        run = _run(MESH_RUNS[0])
        qs = B.build_for_run(run, analysis_root=tmp_path)
        coarsest = min(qs, key=lambda q: q.nside)
        assert coarsest.meta["share_of_target_cells_with_a_vp"] == 1.0


class TestArtifactsLandWhereExpected:
    def test_merged_curves_sit_above_the_rung_dirs(self, as01):
        run, root, _ = as01
        assert (
            run.analysis_dir("target-cls-accuracy", root=root) / C.BY_RESOLUTION_CSV
        ).exists()
        assert (
            run.analysis_dir("target-answer-space", root=root) / A.SWEEP_CSV
        ).exists()

    def test_every_rung_has_a_complete_self_describing_artifact_set(self, as01):
        run, root, _ = as01
        for nside in H.NSIDE_LADDER:
            space_dir = run.answer_space_dir(nside, root=root)
            assert {p.name for p in space_dir.iterdir()} == {
                "seeds.csv", "assignments.csv", "seed_mesh_km.csv", "meta.json",
            }
            acc_dir = run.cls_accuracy_dir(nside, root=root)
            assert (acc_dir / C.ACCURACY_CSV).exists()
            assert (acc_dir / C.MANIFEST_JSON).exists()
