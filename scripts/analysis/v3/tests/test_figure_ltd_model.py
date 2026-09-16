"""What the LTD viewer must not get wrong about a fit it did not perform.

The scatter is reconstructed for the operator runs rather than read back, and
the band is evaluated against a pickled model — so the invariants worth pinning
are the ones that would otherwise fail *silently*: a scatter that is not the set
the model saw, a declined RTT rendered as a real prediction, and a page whose JS
never runs.
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

from scripts.analysis.v3.cli import app
from scripts.analysis.v3.modules import figure_ltd_model as mod
from scripts.analysis.v3.modules.paths import MissingArtifactError, RunPaths


# ---- fixtures ----------------------------------------------------------------


def _make_run(
    tmp_path: Path,
    *,
    combo: str = "vanilla_cbg",
    folds: tuple[str, ...] = ("fold_0", "fold_1"),
    n_fit_samples: int | None = None,
    stateless: bool = False,
) -> RunPaths:
    """A run holding only what this command reads: run.json + targets.parquet."""
    run = RunPaths(run_id="r", root=tmp_path / "out", source="generic_csv", setup="s")
    for fold_id in folds:
        d = run.combo_dir(combo, fold_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.json").write_text(
            json.dumps(
                {
                    "combo_id": combo,
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
        for i, vp in enumerate(("vp-0", "vp-1")):
            rows.append((vp, 40.0 + i, -75.0 - i, tg, 41.0, -80.0, 20.0 + i))
        # A VP-target pair the model saw twice, so a fold's fit set is not just
        # its target count and a miscount cannot hide behind a round number.
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
    behaviours that matter here — a declined prediction and a circle-family zero
    lower bound — are interface-level, not variant-specific.
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


# ---- fit-sample resolution ---------------------------------------------------


def test_the_rebuilt_scatter_is_every_row_outside_the_evaluated_fold(tmp_path):
    run = _make_run(tmp_path)
    csv = _write_dataset(tmp_path, {"tg-a": 0, "tg-b": 1})
    monkey = {"csv": csv}

    # fold_0 evaluates tg-a, so the fit set is tg-b's two rows, and vice versa.
    for fold_id, expect_target in (("fold_0", "tg-b"), ("fold_1", "tg-a")):
        df, prov = _load(run, fold_id, monkey)
        assert len(df) == 2
        assert prov["route"] == "rebuilt_from_dataset_csv"
        assert set(df["vp_id"]) == {"vp-0", "vp-1"}
        # The evaluated fold's own rows must be absent: including them would
        # train the drawn fit on data the model was scored against.
        assert expect_target not in set(df.get("target_id", []))


def test_a_rebuilt_scatter_that_misses_the_recorded_count_is_refused(tmp_path):
    """The assertion that makes reconstruction trustworthy rather than plausible.

    `run.json` records how many samples the fit consumed. A rebuild that lands
    anywhere else is drawing a different set, and a scatter that is quietly the
    wrong set is worse than no scatter at all.
    """
    run = _make_run(tmp_path, n_fit_samples=999)
    csv = _write_dataset(tmp_path, {"tg-a": 0, "tg-b": 1})
    with pytest.raises(ValueError, match="n_fit_samples=999"):
        _load(run, "fold_0", {"csv": csv}, expect_n=999)


def test_a_target_missing_from_the_stratification_is_not_a_fit_sample(tmp_path):
    """Unassigned is not "in some other fold".

    A target the stratification does not mention has no fold, so it cannot be
    known to be outside the evaluated one. Counting it in would inflate the
    scatter with rows the model may well have been scored on.
    """
    run = _make_run(tmp_path)
    csv = _write_dataset(tmp_path, {"tg-a": 0, "tg-b": 1})
    # Re-write the CSV with a third target the sidecar never mentions.
    csv.write_text(
        csv.read_text() + "vp-0,40.0,-75.0,tg-ghost,41.0,-80.0,22.0\n"
    )
    df, _ = _load(run, "fold_0", {"csv": csv})
    assert len(df) == 2, "the unassigned target must not become a fit sample"


def test_materialized_inputs_win_over_the_rebuild(tmp_path):
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


def test_a_fold_with_neither_route_names_both_of_them(tmp_path):
    run = _make_run(tmp_path)
    csv = tmp_path / "absent.csv"
    with pytest.raises(MissingArtifactError) as exc:
        _load(run, "fold_0", {"csv": csv})
    msg = str(exc.value)
    assert "fit_samples.parquet" in msg and "stratification" in msg


def _load(run, fold_id, monkey, expect_n=None):
    """Drive `load_fit_samples` with `resolve_source_csv` pinned to a fixture."""
    import scripts.analysis.v3.modules.bipartite as bipartite

    orig = bipartite.resolve_source_csv
    bipartite.resolve_source_csv = lambda r, override=None: monkey["csv"]
    try:
        return mod.load_fit_samples(
            run,
            fold_id,
            inputs_root=Path("/nonexistent"),
            expect_n=expect_n,
        )
    finally:
        bipartite.resolve_source_csv = orig


# ---- the band ----------------------------------------------------------------


def test_a_declined_rtt_becomes_a_gap_and_never_a_zero(tmp_path):
    """The invariant the whole viewer turns on.

    Spotter declines every RTT below its fitted minimum — on as01/fold_4 that is
    1 to 4 grid points on 413 of 670 VP panels. Rendering those as `lower=0,
    upper=0` would draw a band the model does not claim, exactly at the short
    RTTs the accuracy story turns on.
    """
    model = _FakeLTD(decline_below=10.0)
    band = mod.band_for_vp(model, "vp-0", 40.0, -75.0, np.array([5.0, 8.0, 20.0]))
    assert [b[1] for b in band] == [None, None, 0.0]
    assert [b[2] for b in band] == [None, None, 2000.0]


def test_the_band_is_the_only_thing_the_page_needs_the_python_side_for():
    """Band geometry must not leak back into the JS.

    The legacy viewer's whole failure mode was owning the fit in JS: it read
    `slope`/`intercept` off the submodel and drew the line itself, so it rendered
    one LTD family and raised on the rest. The band arrives as a polyline, and
    nothing in the template may reconstruct one.
    """
    js = Path(mod._JS_TEMPLATE_PATH).read_text()
    assert "bandSegments" in js, "the JS still owns the null-splitting"
    # Comments stripped: the file's own header names the legacy design it is
    # replacing, so a naive substring search matches its explanation of the bug
    # rather than the bug.
    code = "\n".join(
        line
        for line in js.splitlines()
        if not line.lstrip().startswith(("//", "*", "/*"))
    )
    # `theoretical_slope` is exempt and is the one slope the page may own: 2/3 c
    # is a physical constant, not a fitted parameter, and the baseline it draws
    # is the reference every variant is read against rather than any model's
    # output.
    code = code.replace("theoretical_slope", "")
    for leaked in ("slope", "intercept", "predict_distance", "_submodels"):
        assert leaked not in code, f"{leaked!r} must stay on the Python side"


def test_the_grid_is_inset_so_a_closed_boundary_is_not_read_as_a_gap():
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


def test_a_stateless_combo_still_yields_a_band(tmp_path):
    """`million_scale_cbg` writes a `.stateless` marker, never a pickle.

    That is not a missing model: the class plus run.json's kwargs reconstructs an
    equivalent instance. A viewer that treated it as absent would silently drop
    the one calibration-free baseline every other method is read against.
    """
    run = _make_run(tmp_path, combo="million_scale_cbg", stateless=True)
    cfg = json.loads(
        (run.combo_dir("million_scale_cbg", "fold_0") / "run.json").read_text()
    )
    model, rebuilt = mod.load_model(run.combo_dir("million_scale_cbg", "fold_0"), cfg)
    assert rebuilt is True
    band = mod.band_for_vp(model, "vp-0", 40.0, -75.0, np.array([20.0]))
    assert band[0][2] is not None and band[0][2] > 0


def test_a_model_with_no_centre_line_reports_none_rather_than_a_flat_zero(tmp_path):
    """`octant_cbg_hull` has no spline, and must not borrow one.

    It is the same class as `octant_cbg_spl` with `fit_spline: false`, and their
    bands coincide — the centre is the entire visible difference between the two
    pages, so inventing one would erase the distinction.
    """
    assert mod.center_for_vp(_FakeLTD(), "vp-0", np.array([20.0])) == []


# ---- payload and page --------------------------------------------------------


def test_every_number_reaching_the_page_survives_strict_json(tmp_path):
    """`render_html` serializes with `allow_nan=False`.

    A NaN error_km and an absent echoed bound are both ordinary in this data, so
    they must already be `None` by the time the payload is built — otherwise the
    build dies on a real run rather than in this test.
    """
    assert mod._num(float("nan")) is None
    assert mod._num(float("inf")) is None
    assert mod._num(None) is None
    assert mod._num("not a number") is None
    assert mod._num(3.14159) == 3.142


def test_eval_targets_carry_the_true_distance_for_each_participating_vp(tmp_path):
    run = _make_run(tmp_path)
    targets = mod.eval_targets(run.combo_dir("vanilla_cbg", "fold_0"))
    assert set(targets) == {"tg-eval"}
    t = targets["tg-eval"]
    assert t["status"] == "SUCCESS" and t["n_participants"] == 1
    rtt, km, lo, hi = t["vps"]["vp-0"]
    assert rtt == 20.0 and hi == 1200.0 and lo == 0.0
    # 40,-75 to 41,-80 is a bit over 400 km; the point of the assertion is that
    # the distance is computed, not echoed from the parquet.
    assert 380 < km < 460


def test_the_command_is_registered_with_the_flags_the_config_layer_validates():
    """`tests/test_config.py` validates every shipped config against these names."""
    cmd = typer.main.get_command(app).commands["plot-ltd-model"]
    names = {p.name for p in cmd.params}
    assert {"run_id", "all_runs", "method", "fold", "target",
            "max_points_per_vp", "inputs_root", "analysis_root"} <= names


def test_the_scatter_is_capped_deterministically(tmp_path):
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


def test_the_viewer_javascript_actually_runs(tmp_path):
    """A typo in draw() renders a blank page and passes every test above."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")

    run = _make_run(tmp_path, folds=("fold_0",))
    csv = _write_dataset(tmp_path, {"tg-a": 0, "tg-eval": 1})

    import scripts.analysis.v3.modules.bipartite as bipartite

    orig = bipartite.resolve_source_csv
    bipartite.resolve_source_csv = lambda r, override=None: csv
    orig_load = mod.load_model
    mod.load_model = lambda combo_dir, cfg: (_FakeLTD(decline_below=20.5), False)
    try:
        payload = mod.build_payload(
            run,
            "vanilla_cbg",
            fold_ids=["fold_0"],
            inputs_root=Path("/nonexistent"),
        )
    finally:
        bipartite.resolve_source_csv = orig
        mod.load_model = orig_load

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
    # Interior-gap segmentation, asserted against the real JS function. No run
    # produces that shape — every observed gap is at one end of the axis — so it
    # is unreachable through a payload and the harness calls it directly.
    assert report["interiorShape"] == [1, 2], report["interiorShape"]
    assert report["edgeShape"] == [1], report["edgeShape"]
    # Every layer off means nothing is drawn -- proof the toggles reach draw().
    assert report["tracesAllOff"] == 0
    assert "eval target (true)" in " | ".join(report["overlayLayers"])
