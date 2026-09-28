"""What the LTD viewer must not get wrong about a fit it did not perform.

The scatter is reconstructed for the operator runs rather than read back, and
the band is evaluated against a pickled model -- so the invariants worth pinning
are the ones that would otherwise fail *silently*: a scatter that is not the set
the model saw, a declined RTT rendered as a real prediction, and a page whose JS
never runs.

Neither v5 partition appears here, and that is the point: an LTD fit is upstream
of both, so there is no answer space to build and no nside to sweep.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import typer

from scripts.analysis.v5.cli import app
from scripts.analysis.v5.modules import edges as E
from scripts.analysis.v5.modules import figure_ltd_model as mod
from scripts.analysis.v5.modules.paths import (
    MissingArtifactError,
    RunPaths,
    resolve_run,
)

#: The three meshes the sweep script defaults to.
MESH_RUNS = (
    "as01-260728-260802-mesh",
    "as02-260728-260802-mesh",
    "as03-260728-260802-mesh",
)


# ---- fixtures ----------------------------------------------------------------


def _make_run(
    tmp_path: Path,
    *,
    method: str = "vanilla_cbg",
    folds: tuple[str, ...] = ("fold_0", "fold_1"),
    n_fit_samples: int | None = None,
    stateless: bool = False,
) -> RunPaths:
    """A run holding only what this command reads: run.json + targets.parquet.

    `target_*` and `n_targets` are the benchmark's own column and key names, so
    the fixture writes them under those names -- the `tg` rename happens on the
    way into the payload, not on disk.
    """
    run = RunPaths(run_id="r", root=tmp_path / "out", source="generic_csv", setup="s")
    for fold_id in folds:
        d = run.combo_dir(method, fold_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.json").write_text(
            json.dumps(
                {
                    "combo_id": method,
                    "ltd": "speed_of_internet" if stateless else "low_envelope",
                    "ltd_kwargs": {"speed_ratio": 0.6667} if stateless else {},
                    "mtl": "planar_circle",
                    "ctr": "geometric_centroid",
                    "n_fit_samples": n_fit_samples,
                    "n_targets": 1,
                    "status_counts": {"SUCCESS": 1},
                }
            )
        )
        pq.write_table(
            pa.table(
                {
                    "target_id": ["tg-eval"],
                    "target_lat": [41.0],
                    "target_lon": [-80.0],
                    "pred_lat": [41.1],
                    "pred_lon": [-80.1],
                    "status": ["SUCCESS"],
                    "error_km": [12.0],
                    "mtl_participants": [
                        [
                            {
                                "vp_id": "vp-0",
                                "rtt_ms": 20.0,
                                "echoed_upper_km": 1200.0,
                                "echoed_lower_km": 0.0,
                                "vp_lat": 40.0,
                                "vp_lon": -75.0,
                            }
                        ]
                    ],
                }
            ),
            d / "targets.parquet",
        )
        if stateless:
            (d / ".stateless").write_text("SpeedOfInternetLTD has no fitted state.\n")
    return run


def _write_dataset(tmp_path: Path, assignments: dict[str, int]) -> Path:
    """A canonical CSV plus the stratification sidecar that pins its folds."""
    csv = tmp_path / "ds.csv"
    rows = []
    for tg in assignments:
        # Two VPs per TG, so a fold's fit set is not just its TG count and a
        # miscount cannot hide behind a round number.
        for i, vp in enumerate(("vp-0", "vp-1")):
            rows.append((vp, 40.0 + i, -75.0 - i, tg, 41.0, -80.0, 20.0 + i))
    csv.write_text(
        "vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,rtt_ms\n"
        + "\n".join(",".join(str(c) for c in r) for r in rows)
        + "\n"
    )
    (tmp_path / "ds.stratification.json").write_text(
        json.dumps({"fold_assignments": assignments, "fold_sizes": [1, 1]})
    )
    return csv


class _FakeLTD:
    """Stands in for a fitted model, declining a configurable RTT range.

    Real pickles cannot be built in a unit test without a real fit, and the two
    behaviours that matter here -- a declined prediction and a circle-family zero
    lower bound -- are interface-level, not variant-specific.
    """

    def __init__(self, decline_below: float = 0.0, decline_above: float = 1e9):
        self._decline_below = decline_below
        self._decline_above = decline_above

    def predict(self, vp_id, vp_coord, latency):
        from scripts.framework.v2.ltd.base import LTDResult
        from scripts.framework.v2.types import Distance

        r = float(latency)
        if r < self._decline_below or r > self._decline_above:
            return LTDResult(success=False, error="out of domain")
        return LTDResult(
            success=True,
            tg_distance=Distance(upper_km=r * 100.0, lower_km=0.0),
        )


def _load(run, fold_id, monkey, expect_n=None):
    """Drive `load_fit_samples` with `resolve_source_csv` pinned to a fixture."""
    orig = E.resolve_source_csv
    E.resolve_source_csv = lambda r, override=None: monkey["csv"]
    try:
        return mod.load_fit_samples(
            run,
            fold_id,
            inputs_root=Path("/nonexistent"),
            expect_n=expect_n,
        )
    finally:
        E.resolve_source_csv = orig


# ---- fit-sample resolution ---------------------------------------------------


class TestTheScatterIsTheSetTheModelSaw:
    """Route 2 reconstructs the fit set. A wrong set is worse than no set."""

    def test_the_rebuild_is_every_row_outside_the_evaluated_fold(self, tmp_path):
        run = _make_run(tmp_path)
        csv = _write_dataset(tmp_path, {"tg-a": 0, "tg-b": 1})

        # fold_0 evaluates tg-a, so the fit set is tg-b's two rows, and vice versa.
        for fold_id, evaluated in (("fold_0", "tg-b"), ("fold_1", "tg-a")):
            df, prov = _load(run, fold_id, {"csv": csv})
            assert len(df) == 2
            assert prov["route"] == "rebuilt_from_dataset_csv"
            assert set(df["vp_id"]) == {"vp-0", "vp-1"}
            # The evaluated fold's own rows must be absent: including them would
            # train the drawn fit on data the model was scored against.
            assert evaluated not in set(df.get("target_id", []))

    def test_a_rebuild_that_misses_the_recorded_count_is_refused(self, tmp_path):
        """The assertion that makes reconstruction trustworthy, not plausible.

        `run.json` records how many samples the fit consumed. A rebuild that
        lands anywhere else is drawing a different set, and a scatter that is
        quietly the wrong set is worse than no scatter at all.
        """
        run = _make_run(tmp_path, n_fit_samples=999)
        csv = _write_dataset(tmp_path, {"tg-a": 0, "tg-b": 1})
        with pytest.raises(ValueError, match="n_fit_samples=999"):
            _load(run, "fold_0", {"csv": csv}, expect_n=999)

    def test_a_tg_missing_from_the_stratification_is_not_a_fit_sample(self, tmp_path):
        """Unassigned is not "in some other fold".

        A TG the stratification does not mention has no fold, so it cannot be
        known to be outside the evaluated one. Counting it in would inflate the
        scatter with rows the model may well have been scored on.
        """
        run = _make_run(tmp_path)
        csv = _write_dataset(tmp_path, {"tg-a": 0, "tg-b": 1})
        # Re-write the CSV with a third TG the sidecar never mentions.
        csv.write_text(csv.read_text() + "vp-0,40.0,-75.0,tg-ghost,41.0,-80.0,22.0\n")
        df, _ = _load(run, "fold_0", {"csv": csv})
        assert len(df) == 2, "the unassigned TG must not become a fit sample"

    def test_materialized_inputs_win_over_the_rebuild(self, tmp_path):
        """Route 1 is exact; route 2 is a reconstruction. Prefer the exact one."""
        run = _make_run(tmp_path)
        inputs_root = tmp_path / "in"
        d = inputs_root / run.source / run.run_id / run.setup / "fold_0"
        d.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "vp_id": ["vp-9"],
                    "vp_lat": [10.0],
                    "vp_lon": [10.0],
                    "probe_lat": [11.0],
                    "probe_lon": [11.0],
                    "latency_ms": [5.0],
                }
            ),
            d / "fit_samples.parquet",
        )
        df, prov = mod.load_fit_samples(run, "fold_0", inputs_root=inputs_root)
        assert prov["route"] == "materialized_inputs"
        assert list(df["vp_id"]) == ["vp-9"]
        assert {"tg_lat", "tg_lon", "rtt_ms"} <= set(df.columns)

    def test_a_fold_with_neither_route_names_both_of_them(self, tmp_path):
        run = _make_run(tmp_path)
        with pytest.raises(MissingArtifactError) as exc:
            _load(run, "fold_0", {"csv": tmp_path / "absent.csv"})
        msg = str(exc.value)
        assert "fit_samples.parquet" in msg and "stratification" in msg


# ---- the band ----------------------------------------------------------------


class TestTheBandIsWhatTheModelClaims:
    def test_a_declined_rtt_becomes_a_gap_and_never_a_zero(self):
        """The invariant the whole viewer turns on.

        Spotter declines every RTT below its fitted minimum -- on as01/fold_4
        that is 1 to 4 grid points on 413 of 670 VP panels. Rendering those as
        `lower=0, upper=0` would draw a band the model does not claim, exactly
        at the short RTTs the accuracy story turns on.
        """
        model = _FakeLTD(decline_below=10.0)
        band = mod.band_for_vp(model, "vp-0", 40.0, -75.0, np.array([5.0, 8.0, 20.0]))
        assert [b[1] for b in band] == [None, None, 0.0]
        assert [b[2] for b in band] == [None, None, 2000.0]

    def test_band_geometry_never_leaks_back_into_the_js(self):
        """The band arrives as a polyline; the page may not reconstruct one.

        The legacy viewer's whole failure mode was owning the fit in JS: it read
        `slope`/`intercept` off the submodel and drew the line itself, so it
        rendered one LTD family and raised on the rest.
        """
        js = Path(mod._JS_TEMPLATE_PATH).read_text()
        assert "bandSegments" in js, "the JS still owns the null-splitting"
        # Comments stripped: the file's own header names the legacy design it is
        # replacing, so a naive substring search matches its explanation of the
        # bug rather than the bug.
        code = "\n".join(
            line
            for line in js.splitlines()
            if not line.lstrip().startswith(("//", "*", "/*"))
        )
        # `theoretical_slope` is exempt and is the one slope the page may own:
        # 2/3 c is a physical constant, not a fitted parameter, and the baseline
        # it draws is the reference every variant is read against.
        code = code.replace("theoretical_slope", "")
        for leaked in ("slope", "intercept", "predict_distance", "_submodels"):
            assert leaked not in code, f"{leaked!r} must stay on the Python side"

    def test_the_grid_is_inset_so_a_closed_boundary_is_not_read_as_a_gap(self):
        """Octant's hull is degenerate at exactly its first and last RTT.

        Sampling the closed boundary is this module's choice, not the model's
        domain: uncorrected it put a spurious null on both ends of all 670 as01
        octant panels. The inset must be small enough to be invisible.
        """
        rtts = np.array([21.1, 50.0, 80.1])
        grid = mod._rtt_grid(rtts)
        assert grid[0] > 21.1 and grid[-1] < 80.1
        span = 80.1 - 21.1
        assert (grid[0] - 21.1) < span * 1e-3, "inset must be visually negligible"

    def test_a_stateless_method_still_yields_a_band(self, tmp_path):
        """`million_scale_cbg` writes a `.stateless` marker, never a pickle.

        That is not a missing model: the class plus run.json's kwargs
        reconstructs an equivalent instance. A viewer that treated it as absent
        would silently drop SOI, the one calibration-free baseline every other
        method is read against.
        """
        run = _make_run(tmp_path, method="million_scale_cbg", stateless=True)
        d = run.combo_dir("million_scale_cbg", "fold_0")
        model, rebuilt = mod.load_model(d, json.loads((d / "run.json").read_text()))
        assert rebuilt is True
        band = mod.band_for_vp(model, "vp-0", 40.0, -75.0, np.array([20.0]))
        assert band[0][2] is not None and band[0][2] > 0

    def test_a_model_with_no_centre_line_reports_none_rather_than_a_flat_zero(self):
        """OCT-H has no spline, and must not borrow OCT-S's.

        They are the same class with `fit_spline: false`, and their bands
        coincide -- the centre is the entire visible difference between the two
        pages, so inventing one would erase the distinction.
        """
        assert mod.center_for_vp(_FakeLTD(), "vp-0", np.array([20.0])) == []


# ---- payload and page --------------------------------------------------------


class TestThePayload:
    def test_every_number_reaching_the_page_survives_strict_json(self):
        """`render_html` serializes with `allow_nan=False`.

        A NaN error_km and an absent echoed bound are both ordinary in this
        data, so they must already be `None` by the time the payload is built --
        otherwise the build dies on a real run rather than in this test.
        """
        assert mod._num(float("nan")) is None
        assert mod._num(float("inf")) is None
        assert mod._num(None) is None
        assert mod._num("not a number") is None
        assert mod._num(3.14159) == 3.142

    def test_eval_tgs_carry_the_true_distance_for_each_participating_vp(self, tmp_path):
        run = _make_run(tmp_path)
        tgs = mod.eval_tgs(run.combo_dir("vanilla_cbg", "fold_0"))
        assert set(tgs) == {"tg-eval"}
        t = tgs["tg-eval"]
        assert t["status"] == "SUCCESS" and t["n_participants"] == 1
        rtt, km, lo, hi = t["vps"]["vp-0"]
        assert rtt == 20.0 and hi == 1200.0 and lo == 0.0
        # 40,-75 to 41,-80 is a bit over 400 km; the point of the assertion is
        # that the distance is computed, not echoed from the parquet.
        assert 380 < km < 460

    def test_the_payload_speaks_v5(self, tmp_path):
        """`tg`, and the method's term -- not `target`, not `Vanilla CBG`.

        The benchmark's own names stay on disk (`target_id` keys the overlay
        dict, because that is the parquet's column); everything this layer
        chooses is v5's.
        """
        run = _make_run(tmp_path, folds=("fold_0",))
        csv = _write_dataset(tmp_path, {"tg-a": 0, "tg-eval": 1})
        orig, orig_load = E.resolve_source_csv, mod.load_model
        E.resolve_source_csv = lambda r, override=None: csv
        mod.load_model = lambda d, cfg: (_FakeLTD(), False)
        try:
            payload = mod.build_payload(
                run,
                "vanilla_cbg",
                fold_ids=["fold_0"],
                inputs_root=Path("/nonexistent"),
                preselect_tgs=["tg-eval"],
            )
        finally:
            E.resolve_source_csv, mod.load_model = orig, orig_load

        assert payload["method"] == "vanilla_cbg"
        assert payload["method_label"] == "VAN"
        assert payload["preselect_tgs"] == ["tg-eval"]
        fold = payload["folds"]["fold_0"]
        assert "tgs" in fold and "n_tgs" in fold
        assert not {"targets", "n_targets"} & set(fold)
        assert "preselect_targets" not in payload and "combo_id" not in payload

    def test_the_scatter_is_capped_deterministically(self):
        import pandas as pd

        samples = pd.DataFrame(
            {
                "vp_id": ["vp-0"] * 100,
                "vp_lat": [40.0] * 100,
                "vp_lon": [-75.0] * 100,
                "tg_lat": np.linspace(41.0, 42.0, 100),
                "tg_lon": [-80.0] * 100,
                "rtt_ms": np.linspace(5.0, 50.0, 100),
            }
        )
        a = mod.scatter_by_vp(samples, max_points_per_vp=10)
        b = mod.scatter_by_vp(samples, max_points_per_vp=10)
        assert a == b, "two builds of one run must produce the same page"
        assert len(a["vp-0"]) <= 10


class TestTheCommand:
    def test_it_carries_the_flags_the_sweep_script_passes(self):
        """`create_ltd_modeling_html.sh` builds `--method` / `--fold` by hand.

        v5 has no config layer to validate these against, so this is the only
        thing standing between a renamed flag and a sweep that fails per run.
        """
        cmd = typer.main.get_command(app).commands["plot-ltd-model"]
        names = {p.name for p in cmd.params}
        assert {
            "run_id",
            "all_runs",
            "method",
            "fold",
            "tg",
            "max_points_per_vp",
            "inputs_root",
            "outputs_root",
            "analysis_root",
        } <= names

    def test_a_run_with_no_method_still_leaves_a_manifest(self, tmp_path):
        """The sweep script points at `manifest.json` when it finds no HTML.

        A run whose methods are all skipped exits 0 having written nothing else,
        so the manifest is the only record of why -- and it has to exist for
        that message to be worth printing.
        """
        run = _make_run(tmp_path, folds=("fold_0",))
        written, skipped = mod.build_for_run(
            run,
            methods=["not_a_method"],
            analysis_root=tmp_path / "analysis",
            inputs_root=Path("/nonexistent"),
        )
        assert written == [] and "not_a_method" in skipped
        manifest = json.loads(
            (run.ltd_model_dir(root=tmp_path / "analysis") / "manifest.json").read_text()
        )
        assert manifest["methods"] == {}
        assert "not a combo of this run" in manifest["skipped_methods"]["not_a_method"]


# ---- real runs ---------------------------------------------------------------


def _run(run_id: str) -> RunPaths:
    try:
        run = resolve_run(run_id)
    except MissingArtifactError as exc:
        pytest.skip(f"{run_id} not available: {exc}")
    if not run.combo_ids:
        pytest.skip(f"{run_id} has no scored combo")
    return run


class TestOnRealRuns:
    """The meshes the sweep script defaults to, on the route they actually take."""

    @pytest.mark.parametrize("run_id", MESH_RUNS)
    def test_every_fold_resolves_to_the_set_its_run_json_records(self, run_id):
        """The three meshes materialized their inputs, so route 1 covers them.

        `load_fit_samples` asserts the row count against `n_fit_samples` itself,
        so reaching the end of this loop IS the assertion -- what is checked here
        is that the exact route was taken rather than the reconstruction.
        """
        from scripts.benchmark.v2.inputs import DEFAULT_INPUTS_ROOT

        run = _run(run_id)
        method = run.combo_ids[0]
        assert run.fold_ids, "a scored run has folds"
        for fold_id in run.fold_ids:
            cfg = json.loads(
                (run.combo_dir(method, fold_id) / "run.json").read_text()
            )
            df, prov = mod.load_fit_samples(
                run,
                fold_id,
                inputs_root=DEFAULT_INPUTS_ROOT,
                expect_n=cfg.get("n_fit_samples"),
            )
            assert prov["route"] == "materialized_inputs"
            assert len(df) == cfg["n_fit_samples"] > 0

    def test_one_fold_renders_a_page_that_survives_strict_json(self, tmp_path):
        """End to end on real checkpoints, one fold so it stays seconds.

        The unit tests drive `_FakeLTD`; this is the only place a real pickled
        model reaches `band_for_vp`, which is where a variant that broke the
        uniform `predict` contract would show up.
        """
        from scripts.benchmark.v2.inputs import DEFAULT_INPUTS_ROOT

        run = _run(MESH_RUNS[0])
        method = run.combo_ids[0]
        written, skipped = mod.build_for_run(
            run,
            methods=[method],
            fold_ids=[run.fold_ids[0]],
            analysis_root=tmp_path,
            inputs_root=DEFAULT_INPUTS_ROOT,
        )
        assert skipped == {} and len(written) == 1
        name, path, payload = written[0]
        assert name == method and path.name == f"ltd_model.{method}.html"
        assert payload["folds"][run.fold_ids[0]]["vps"], "no VP panel was built"
        # `render_html` already ran with allow_nan=False inside build_for_run;
        # re-serializing here is what names the payload if it ever regresses.
        json.dumps(payload, allow_nan=False)
        assert "__PAYLOAD__" not in path.read_text(encoding="utf-8")


# ---- the page --------------------------------------------------------------


class TestTheViewerExecutes:
    def test_the_viewer_javascript_actually_runs(self, tmp_path):
        """A typo in draw() renders a blank page and passes every test above."""
        node = shutil.which("node")
        if node is None:
            pytest.skip("node is not available")

        run = _make_run(tmp_path, folds=("fold_0",))
        csv = _write_dataset(tmp_path, {"tg-a": 0, "tg-eval": 1})

        orig, orig_load = E.resolve_source_csv, mod.load_model
        E.resolve_source_csv = lambda r, override=None: csv
        mod.load_model = lambda d, cfg: (_FakeLTD(decline_below=20.5), False)
        try:
            payload = mod.build_payload(
                run,
                "vanilla_cbg",
                fold_ids=["fold_0"],
                inputs_root=Path("/nonexistent"),
            )
        finally:
            E.resolve_source_csv, mod.load_model = orig, orig_load

        out = tmp_path / "page.html"
        out.write_text(mod.render_html(payload), encoding="utf-8")

        harness = Path(__file__).with_name("test_ltd_model_viewer.js")
        proc = subprocess.run(
            [node, str(harness), str(out)], capture_output=True, text=True, timeout=120
        )
        assert proc.returncode == 0, proc.stderr
        report = json.loads(proc.stdout)

        assert report["folds"] == 1
        assert report["reactCalls"] > 0
        layers = " | ".join(report["firstLayers"])
        for expected in ("fit samples", "fitted band", "2/3 c"):
            assert expected in layers, f"{expected!r} missing from the default view"
        # The declined range must have been segmented away rather than handed to
        # Plotly, which would bridge it inside the filled polygon.
        assert report["nullBearing"] == [], report["nullBearing"]
        # Interior-gap segmentation, asserted against the real JS function. No
        # run produces that shape -- every observed gap is at one end of the axis
        # -- so it is unreachable through a payload and the harness calls it
        # directly.
        assert report["interiorShape"] == [1, 2], report["interiorShape"]
        assert report["edgeShape"] == [1], report["edgeShape"]
        # Every layer off means nothing is drawn -- proof the toggles reach draw().
        assert report["tracesAllOff"] == 0
        assert "eval TG (true)" in " | ".join(report["overlayLayers"])
