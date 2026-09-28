"""Interactive per-fold LTD fit viewer -- what the latency-to-distance stage learned.

Every other v5 command scores a *prediction*. This one looks at the stage that
produces it: for each fold and each VP, the RTT-vs-distance training scatter, the
2/3*c baseline, and the band the fitted model actually returns. It is the figure
to open when a run's accuracy moves and the question is whether the LTD fit is
the cause.

One self-contained HTML per method, because the method *is* the model -- each page
answers "how does this LTD behave?". That also matches `map_mtl`'s
`mtl_map.{method}.html` and keeps each file near the ~4 MB the existing viewer
already weighs.

## Neither partition enters this page

v5 grades a prediction on two axes, the HEALPix grid and the Voronoi cell. An LTD
fit is upstream of both: it maps an RTT to a distance, with no answer space in
sight. So this module reads no `build-answer-space` output, takes no `--nside`,
and writes to a rung-free directory (`paths.ltd_model_dir`). It is the one v5
command that needs nothing from `classify`.

## Three things make this work

  * **The model is named, not guessed.** `run.json` records `ltd` + `ltd_kwargs`
    per combo. The combo id cannot be pattern-matched: `octant_cbg_spl` and
    `octant_cbg_hull` are both `bounded_spline`, differing only in `fit_spline`.

  * **One uniform prediction API.** The band comes from
    `LTDModel.predict(vp_id, vp_coord, latency).tg_distance`, which every variant
    implements. Nothing here touches a private attribute except the optional
    centre line, which is guarded.

  * **The scatter is found, not assumed.** It is not in the output tree at all
    (see `load_fit_samples`).

## Benchmark names stay benchmark names

`targets.parquet`'s columns (`target_id`, `target_lat`, `target_lon`) and
`run.json`'s keys (`n_targets`, `n_fit_samples`) are written by the benchmark
layer. They are read under those names and emitted into the payload as `tg_*` --
the same boundary `answer_space.BENCHMARK_TG_COLUMNS` draws, rather than a rename
applied halfway up the stack.

Command:
    python -m scripts.analysis.v5.cli plot-ltd-model --run-id as01-260728-260802-mesh
Writes:
    outputs/analysis/v5/<run_id>/ltd-model/ltd_model.<method>.html + manifest.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.analysis.v5.modules import edges as E
from scripts.analysis.v5.modules.geodesy import elementwise_km
from scripts.analysis.v5.modules.methods import method_label, method_term_table
from scripts.analysis.v5.modules.paths import (
    MissingArtifactError,
    REPO_ROOT,
    RunPaths,
)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_HTML_TEMPLATE_PATH = _TEMPLATE_DIR / "ltd_model.html"
_JS_TEMPLATE_PATH = _TEMPLATE_DIR / "ltd_model.js"

#: `<method>` is a filename infix, not a directory: one run's viewers all share
#: the `ltd-model/` tree, and a method that overwrote its neighbour would make the
#: directory unreadable. Mirrors `map_mtl.MAP_HTML`.
VIEWER_HTML = "ltd_model.{method}.html"

MANIFEST_JSON = "manifest.json"

#: 2/3 c as ms-per-km of *one-way* propagation, doubled for a round trip:
#: rtt_ms = 0.01 * km. The reference line every LTD is read against, and the
#: same constant the static primitives draw (`libs/cbg/rtt_model.THEORETICAL_SLOPE`).
THEORETICAL_SLOPE = 0.01

#: Band sample count across each VP's observed RTT range. 48 is enough to render
#: a spline smoothly at plot resolution while keeping the payload a few hundred
#: KB per combo; the bands are monotone and near-piecewise-linear, so more points
#: buy nothing visible.
BAND_POINTS = 48

#: Per-VP scatter cap. Today's operator runs carry ~319 fit samples per VP, so
#: this is inert; it exists so a denser campaign degrades the page's weight
#: rather than its usability. The subsample is a deterministic stride, not an
#: RNG draw, so two builds of one run agree.
DEFAULT_MAX_POINTS_PER_VP = 400


# ---- fit samples -------------------------------------------------------------
#
# The scatter is the one input that is NOT in the fold directory. `run.json` and
# `fit_checkpoint.pkl` describe the fit; the data it was fit on lives elsewhere,
# and the two routes below are both load-bearing on today's runs:
# `as7018-ripe-mesh` has only a materialized inputs tree, and as01/02/03 have
# only a pinned stratification.


def _inputs_fold_dir(run: RunPaths, fold_id: str, inputs_root: Path) -> Path:
    """`<inputs_root>/<source>/<run_id>/<setup>/<fold>/`.

    Built from `RunPaths` rather than `benchmark.v2.inputs.inputs_dir_for`, which
    derives the same path from a live `DataSource` instance — constructing one
    here would mean re-deriving the source kwargs this layer deliberately does
    not carry.
    """
    return inputs_root / run.source / run.run_id / run.setup / fold_id


def _fold_index(fold_id: str) -> int:
    """`"fold_3"` -> `3`. The same split `RunPaths.fold_ids` sorts on."""
    return int(fold_id.split("_")[1])


def _stratification_path(csv_path: Path) -> Path:
    """`<stem>.csv` -> `<stem>.stratification.json`, the pinned fold assignment.

    A sidecar of the dataset, not of the run: the operator CSVs were
    reconstructed from run outputs and the folds were pinned at that point, which
    is why the run's own config carries `benchmark: {}`.
    """
    return csv_path.with_name(csv_path.name[: -len(".csv")] + ".stratification.json")


def load_fit_samples(
    run: RunPaths,
    fold_id: str,
    *,
    inputs_root: Path,
    expect_n: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The `(vp_id, vp_lat, vp_lon, tg_lat, tg_lon, rtt_ms)` rows a fold was fit on.

    Returns the frame and a provenance dict, which the page displays: the scatter
    is *reconstructed* for the operator runs and a reader has to be able to tell
    that from a materialized one.

    Two routes, tried in order:

      1. `inputs/benchmark/v2/<source>/<run_id>/<setup>/<fold>/fit_samples.parquet`
         — the exact artifact the runner consumed. No recomputation.
      2. The dataset CSV (`edges.resolve_source_csv`) filtered by its pinned
         `.stratification.json`: fit = every row whose TG is in one of the
         other K-1 folds.

    `expect_n` is `run.json`'s `n_fit_samples`. When given it is **asserted**, not
    logged: that equality is the whole difference between reconstructing route 2
    and guessing at it, and it costs nothing to check. Verified to hold exactly
    on all five as01 folds (fold_4: 42704).
    """
    parquet = _inputs_fold_dir(run, fold_id, inputs_root) / "fit_samples.parquet"
    if parquet.exists():
        df = pq.read_table(
            parquet,
            columns=["vp_id", "vp_lat", "vp_lon", "probe_lat", "probe_lon", "latency_ms"],
        ).to_pandas()
        df = df.rename(
            columns={
                "probe_lat": "tg_lat",
                "probe_lon": "tg_lon",
                "latency_ms": "rtt_ms",
            }
        )
        prov = {
            "route": "materialized_inputs",
            "path": str(parquet.relative_to(REPO_ROOT))
            if parquet.is_relative_to(REPO_ROOT)
            else str(parquet),
        }
    else:
        # `edges.resolve_source_csv` raises `MeshSupersetError` on a weighted arm,
        # where no file holds the pruned flows. That subclasses
        # `MissingArtifactError`, so it arrives at `build_payload`'s handler as an
        # ordinary per-fold skip carrying its own "derive a weighted CSV" hint --
        # which is the right outcome. A weighted run either materializes its
        # inputs and takes route 1, or has no scatter to draw.
        csv_path = Path(E.resolve_source_csv(run))
        strat_path = _stratification_path(csv_path)
        if not strat_path.exists():
            raise MissingArtifactError(
                f"{run.run_id}/{fold_id}: no fit samples. Neither {parquet} "
                f"(materialize with `benchmark.v2.cli materialize-inputs`) nor "
                f"{strat_path} (the dataset's pinned fold assignment) exists."
            )
        assignments = json.loads(strat_path.read_text()).get("fold_assignments") or {}
        if not assignments:
            raise MissingArtifactError(f"{strat_path} carries no fold_assignments")

        df = pd.read_csv(
            csv_path,
            usecols=[
                "vp_id",
                "vp_lat",
                "vp_lon",
                "target_id",
                "target_lat",
                "target_lon",
                "rtt_ms",
            ],
        )
        want = _fold_index(fold_id)
        assigned = df["target_id"].map(assignments)
        # A TG absent from the stratification is not "in another fold" and
        # must not silently become a fit sample -- that would inflate the scatter
        # with rows the model never saw. Dropping them here is also what makes
        # the `expect_n` assertion below meaningful.
        df = df[assigned.notna() & (assigned != want)]
        df = df.rename(columns={"target_lat": "tg_lat", "target_lon": "tg_lon"})
        df = df.drop(columns=["target_id"])
        prov = {
            "route": "rebuilt_from_dataset_csv",
            "csv": str(csv_path.relative_to(REPO_ROOT))
            if csv_path.is_relative_to(REPO_ROOT)
            else str(csv_path),
            "stratification": str(strat_path.relative_to(REPO_ROOT))
            if strat_path.is_relative_to(REPO_ROOT)
            else str(strat_path),
        }

    if expect_n is not None and len(df) != int(expect_n):
        raise ValueError(
            f"{run.run_id}/{fold_id}: rebuilt {len(df)} fit samples but run.json "
            f"records n_fit_samples={expect_n}. The scatter would not be the one "
            f"the model was fit on. Provenance: {prov}"
        )
    prov["n_rows"] = int(len(df))
    return df.reset_index(drop=True), prov


def scatter_by_vp(
    samples: pd.DataFrame, *, max_points_per_vp: int
) -> dict[str, list[list[float]]]:
    """`vp_id -> [[rtt_ms, distance_km], ...]`, distance recomputed per row.

    Deduped per (fold, VP) by the caller: the fit set is identical across combos
    within a fold, so holding it per combo would multiply the payload by the
    number of methods for no new information.
    """
    if samples.empty:
        return {}
    km = elementwise_km(
        samples["vp_lat"].to_numpy(),
        samples["vp_lon"].to_numpy(),
        samples["tg_lat"].to_numpy(),
        samples["tg_lon"].to_numpy(),
    )
    frame = pd.DataFrame(
        {
            "vp_id": samples["vp_id"].astype(str).to_numpy(),
            "rtt": np.round(samples["rtt_ms"].to_numpy(dtype=float), 2),
            "km": np.round(km, 1),
        }
    )
    out: dict[str, list[list[float]]] = {}
    for vp_id, group in frame.groupby("vp_id", sort=True):
        if len(group) > max_points_per_vp:
            # Deterministic stride, so a rebuild of one run reproduces the page.
            step = math.ceil(len(group) / max_points_per_vp)
            group = group.iloc[::step]
        out[str(vp_id)] = [
            [float(r), float(k)] for r, k in zip(group["rtt"], group["km"])
        ]
    return out


# ---- the fitted model --------------------------------------------------------


def load_model(combo_dir: Path, run_cfg: dict[str, Any]):
    """The fitted LTD for one (fold, combo), reconstructing the stateless case.

    `checkpoint.load_ltd_checkpoint` returns `None` when the combo wrote only a
    `.stateless` marker — true of `million_scale_cbg`, whose `SpeedOfInternetLTD`
    has no post-fit state. That is not an error and not an absent model: the
    class plus `run.json`'s `ltd_kwargs` reconstructs an equivalent instance,
    which predicts identically.
    """
    from scripts.benchmark.v2.checkpoint import load_ltd_checkpoint
    from scripts.framework.v2.registry import LTD_REGISTRY

    model = load_ltd_checkpoint(combo_dir)
    if model is not None:
        return model, False
    ltd_name = run_cfg.get("ltd")
    if ltd_name not in LTD_REGISTRY:
        raise MissingArtifactError(
            f"{combo_dir}: stateless checkpoint naming unknown ltd {ltd_name!r}"
        )
    return LTD_REGISTRY[ltd_name](**(run_cfg.get("ltd_kwargs") or {})), True


def band_for_vp(
    model, vp_id: str, vp_lat: float, vp_lon: float, rtts: np.ndarray
) -> list[list[float | None]]:
    """`[[rtt, lower_km, upper_km], ...]` from the uniform `predict` API.

    `None` marks an RTT the model declines. That is real rather than defensive:
    `normal_dist` (Spotter) is pooled and fails below its fitted `rtt_min`, so
    the honest rendering is a gap. Substituting 0 there would draw a band the
    model does not claim, at exactly the short RTTs the accuracy story turns on.

    Circle-family models (`low_envelope`, `speed_of_internet`) return
    `lower_km == 0` by construction, which the page draws as a one-sided band.
    """
    from scripts.framework.v2.types import Coord, Latency, VpId

    coord = Coord(lat=float(vp_lat), lon=float(vp_lon))
    vid = VpId(str(vp_id))
    out: list[list[float | None]] = []
    for rtt in rtts:
        r = float(rtt)
        # `predict` raises on non-positive latency by contract (base.py), and a
        # submodel can raise on its own domain, so both are treated as "no claim
        # here" rather than allowed to abort a 134-VP page.
        try:
            res = model.predict(vid, coord, Latency(r))
        except Exception:
            out.append([round(r, 3), None, None])
            continue
        if not res.success or res.tg_distance is None:
            out.append([round(r, 3), None, None])
            continue
        d = res.tg_distance
        lower = getattr(d, "lower_km", 0.0) or 0.0
        out.append([round(r, 3), round(float(lower), 2), round(float(d.upper_km), 2)])
    return out


def center_for_vp(model, vp_id: str, rtts: np.ndarray) -> list[list[float | None]]:
    """Optional centre line, `[[rtt, km], ...]`, or `[]` when the model has none.

    This is what visually separates `octant_cbg_spl` from `octant_cbg_hull`. The
    two are the same class and their `predict` bands coincide, so without the
    centre the two pages would be indistinguishable and the spline — the entire
    difference between the variants — would be invisible.

    Reaching into `_submodels` is the one private access here, and it is guarded
    on every step: pooled models (`normal_dist`) keep `_model` instead, stateless
    ones keep nothing, and `predict_distance` is a library-model method rather
    than part of the LTD interface.
    """
    sub = None
    submodels = getattr(model, "_submodels", None)
    if isinstance(submodels, dict):
        sub = submodels.get(vp_id)
    if sub is None:
        sub = getattr(model, "_model", None)
    predict_distance = getattr(sub, "predict_distance", None)
    if sub is None or predict_distance is None or not getattr(sub, "fitted", True):
        return []
    out: list[list[float | None]] = []
    any_value = False
    for rtt in rtts:
        r = float(rtt)
        try:
            km = predict_distance(r)
        except Exception:
            out.append([round(r, 3), None])
            continue
        if km is None or not math.isfinite(float(km)):
            out.append([round(r, 3), None])
            continue
        any_value = True
        out.append([round(r, 3), round(float(km), 2)])
    return out if any_value else []


#: Fraction of the observed RTT span to pull the grid's endpoints inward by.
#:
#: The Octant hull is built *through* a VP's extreme observations, so at exactly
#: its first and last RTT the upper and lower bounds coincide and `predict`
#: returns DEGENERATE_REGION — a zero-width annulus. Sampling the closed boundary
#: is this module's choice, not the model's domain, and left uncorrected it put a
#: spurious null on both ends of 670 of 670 as01 octant panels, each one raising
#: the page's "declines that RTT range" warning over what is really one degenerate
#: point. An inset of 1e-4 of the span is ~0.006 ms on a 59 ms range: invisible in
#: the drawn curve, and enough to land strictly inside the hull.
#:
#: This does NOT paper over a genuine gap. Spotter's leading nulls survive it
#: (0-4 points, varying per VP), because those come from its fitted `rtt_min`
#: rather than from the sample being exactly on a boundary.
GRID_INSET = 1e-4


def _rtt_grid(rtts: np.ndarray) -> np.ndarray:
    """`BAND_POINTS` samples spanning a VP's observed RTT range, ends inset.

    Anchored to what the VP actually measured rather than to a fixed window, so
    a short-RTT VP is not drawn mostly empty and a long-RTT one is not clipped.
    The floor keeps the first sample strictly positive, which `predict` requires.
    """
    finite = rtts[np.isfinite(rtts) & (rtts > 0)]
    if finite.size == 0:
        return np.array([], dtype=float)
    lo = max(float(finite.min()), 0.01)
    hi = float(finite.max())
    if hi <= lo:
        hi = lo * 1.5 + 0.01
    inset = (hi - lo) * GRID_INSET
    return np.linspace(lo + inset, hi - inset, BAND_POINTS)


# ---- eval TGs ---------------------------------------------------------------


def eval_tgs(combo_dir: Path) -> dict[str, Any]:
    """Per-TG overlays from `targets.parquet`, keyed by `target_id`.

    Reads the benchmark's column names and emits v5's: the file is written by
    `benchmark.v2`, so `target_*` is what is on disk, and `tg` is what the page
    and the rest of this layer say.

    `mtl_participants` carries `(vp_id, rtt_ms, vp_lat, vp_lon)` plus the echoed
    LTD bounds, and the row carries the TG's true coordinates — so each
    TG contributes one `(rtt, true distance)` point per participating VP,
    placeable on that VP's axes with no extra input.

    These are **post-filter** participants: the MTL drops a disk that fully
    contains another before intersecting, so a TG shows fewer VPs here than
    the fit had (27 of 134 on a sampled as01 row). That is the right set — it is
    what actually formed the region — but it is why this is an overlay on the fit
    rather than a substitute for it.
    """
    path = combo_dir / "targets.parquet"
    if not path.exists():
        return {}
    table = pq.read_table(
        path,
        columns=[
            "target_id",
            "target_lat",
            "target_lon",
            "pred_lat",
            "pred_lon",
            "status",
            "error_km",
            "mtl_participants",
        ],
    )
    out: dict[str, Any] = {}
    for row in table.to_pylist():
        tg_lat, tg_lon = row.get("target_lat"), row.get("target_lon")
        if tg_lat is None or tg_lon is None:
            continue
        points: dict[str, list[float | None]] = {}
        for p in row.get("mtl_participants") or []:
            vp_lat, vp_lon, rtt = p.get("vp_lat"), p.get("vp_lon"), p.get("rtt_ms")
            if vp_lat is None or vp_lon is None or rtt is None:
                continue
            km = float(
                elementwise_km(
                    np.array([vp_lat]),
                    np.array([vp_lon]),
                    np.array([tg_lat]),
                    np.array([tg_lon]),
                )[0]
            )
            points[str(p["vp_id"])] = [
                round(float(rtt), 2),
                round(km, 1),
                _num(p.get("echoed_lower_km")),
                _num(p.get("echoed_upper_km")),
            ]
        out[str(row["target_id"])] = {
            "status": row.get("status"),
            "error_km": _num(row.get("error_km")),
            "n_participants": len(points),
            "vps": points,
        }
    return out


def _num(x: Any) -> float | None:
    """Finite float, or `None`.

    Every number reaching the payload passes through here, because the page is
    serialized with `allow_nan=False`: a NaN error_km or an absent echoed bound
    is ordinary in this data and must become `null` rather than abort the build.
    """
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, 3) if math.isfinite(v) else None


# ---- payload -----------------------------------------------------------------


def build_payload(
    run: RunPaths,
    method: str,
    *,
    fold_ids: list[str],
    inputs_root: Path,
    max_points_per_vp: int = DEFAULT_MAX_POINTS_PER_VP,
    preselect_tgs: list[str] | None = None,
    sample_cache: dict[str, tuple[pd.DataFrame, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Everything one method's page draws. The browser computes nothing.

    Band and centre lines are evaluated here, in Python, against the real model
    objects. That is the design decision the legacy viewer got wrong by shipping
    slope/intercept and drawing the line in JS: doing it here means a new LTD
    variant renders with no JS change, and no variant's geometry is duplicated in
    two languages where the copies can drift.

    `sample_cache` is keyed by fold and shared across combos by `build_for_run`.
    Every combo of a fold was fit on the *same* samples — which is why the
    scatter is deduped per (fold, VP) in the first place — so without it a
    five-combo run re-reads and re-filters one 53k-row CSV 25 times.
    """
    folds: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    skipped: dict[str, str] = {}

    for fold_id in fold_ids:
        combo_dir = run.combo_dir(method, fold_id)
        cfg_path = combo_dir / "run.json"
        if not cfg_path.exists():
            skipped[fold_id] = f"no run.json under {combo_dir}"
            continue
        run_cfg = json.loads(cfg_path.read_text())

        try:
            if sample_cache is not None and fold_id in sample_cache:
                samples, prov = sample_cache[fold_id]
                # Still checked against THIS method's run.json. The cache is
                # keyed on the fold, and the equality it has to hold is per
                # method -- a method whose fit set differs from its fold-mates'
                # would otherwise inherit their scatter unnoticed.
                expect = run_cfg.get("n_fit_samples")
                if expect is not None and len(samples) != int(expect):
                    raise ValueError(
                        f"{run.run_id}/{fold_id}/{method}: cached fit set has "
                        f"{len(samples)} rows but run.json records "
                        f"n_fit_samples={expect}"
                    )
            else:
                samples, prov = load_fit_samples(
                    run,
                    fold_id,
                    inputs_root=inputs_root,
                    expect_n=run_cfg.get("n_fit_samples"),
                )
                if sample_cache is not None:
                    sample_cache[fold_id] = (samples, prov)
        except MissingArtifactError as exc:
            skipped[fold_id] = str(exc)
            continue

        provenance[fold_id] = prov
        model, rebuilt = load_model(combo_dir, run_cfg)

        vp_coords = (
            samples.groupby(samples["vp_id"].astype(str))[["vp_lat", "vp_lon"]]
            .first()
            .to_dict("index")
        )
        observed = {
            str(vp): g["rtt_ms"].to_numpy(dtype=float)
            for vp, g in samples.groupby(samples["vp_id"].astype(str))
        }
        scatter = scatter_by_vp(samples, max_points_per_vp=max_points_per_vp)

        vps: dict[str, Any] = {}
        for vp_id in sorted(scatter):
            grid = _rtt_grid(observed[vp_id])
            coord = vp_coords[vp_id]
            sub = (getattr(model, "_submodels", None) or {}).get(vp_id)
            vps[vp_id] = {
                "samples": scatter[vp_id],
                "n_samples": len(observed[vp_id]),
                "band": band_for_vp(
                    model, vp_id, coord["vp_lat"], coord["vp_lon"], grid
                ),
                "center": center_for_vp(model, vp_id, grid),
                "fitted": bool(getattr(sub, "fitted", True)) if sub is not None else None,
                "fit_message": getattr(sub, "fit_message", None) if sub is not None else None,
                "n_measurements": _num(getattr(sub, "n_measurements", None)),
                "lat": _num(coord["vp_lat"]),
                "lon": _num(coord["vp_lon"]),
            }

        folds[fold_id] = {
            "vps": vps,
            "tgs": eval_tgs(combo_dir),
            "n_fit_samples": run_cfg.get("n_fit_samples"),
            "n_tgs": run_cfg.get("n_targets"),
            "status_counts": run_cfg.get("status_counts") or {},
            "model_rebuilt_stateless": rebuilt,
        }

    if not folds:
        raise MissingArtifactError(
            f"{run.run_id}/{method}: no fold could be built. " + "; ".join(
                f"{k}: {v}" for k, v in skipped.items()
            )
        )

    first_cfg = json.loads(
        (run.combo_dir(method, sorted(folds)[0]) / "run.json").read_text()
    )
    return {
        "run_id": run.run_id,
        "source": run.source,
        "setup": run.setup,
        "method": method,
        "method_label": method_label(method),
        "ltd": first_cfg.get("ltd"),
        "ltd_kwargs": first_cfg.get("ltd_kwargs") or {},
        "mtl": first_cfg.get("mtl"),
        "ctr": first_cfg.get("ctr"),
        "theoretical_slope": THEORETICAL_SLOPE,
        "fold_ids": sorted(folds, key=_fold_index),
        "folds": folds,
        "provenance": provenance,
        "skipped_folds": skipped,
        "preselect_tgs": list(preselect_tgs or []),
        "max_points_per_vp": int(max_points_per_vp),
    }


def render_html(payload: dict[str, Any]) -> str:
    """Assemble the standalone page from the HTML shell + viewer JS.

    Substitution order is load-bearing, and is `map_mtl.render_html`'s: the JS
    goes in first (it contains neither of the other tokens), then the title, then
    the payload last. The blob escapes `</` so an embedded `</script>` cannot
    close the data block early, and `allow_nan=False` makes a stray NaN fail here
    rather than produce JSON the browser silently rejects.
    """
    title = (
        f"LTD fit — {payload['run_id']} · {payload['method_label']} "
        f"({payload['ltd']})"
    )
    html = _HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    js = _JS_TEMPLATE_PATH.read_text(encoding="utf-8")
    html = html.replace("__SCRIPT__", js).replace("__TITLE__", title)
    blob = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    return html.replace("__PAYLOAD__", blob)


def build_for_run(
    run: RunPaths,
    *,
    methods: list[str] | None = None,
    fold_ids: list[str] | None = None,
    analysis_root: Path | None = None,
    inputs_root: Path,
    max_points_per_vp: int = DEFAULT_MAX_POINTS_PER_VP,
    preselect_tgs: list[str] | None = None,
) -> tuple[list[tuple[str, Path, dict[str, Any]]], dict[str, str]]:
    """Render every requested method for one run. Returns `(written, skipped)`.

    Writes `manifest.json` beside the pages, always -- including when nothing
    rendered. That file is the only record of *why* a method produced no viewer,
    and `create_ltd_modeling_html.sh` points at it when it finds an empty
    directory, so an empty run has to leave one behind.

    `methods` defaults to `run.combo_ids`, the same default `plot-mtl-map` takes.
    `shortest_ping` is NOT appended the way that command appends it: it has no LTD
    stage and so no fit to draw, and under v3's published-variant default it was
    skipped on every run for exactly that reason.

    A named method absent from this run is skipped rather than raised on, so
    `METHODS="vanilla_cbg octant_cbg_spl"` over a mixed sweep reports per run
    instead of aborting on the first that lacks one.
    """
    available = set(run.combo_ids)
    wanted = list(methods) if methods else list(run.combo_ids)
    folds = list(fold_ids) if fold_ids else run.fold_ids
    if not folds:
        raise MissingArtifactError(f"{run.run_id}: no fold_* dirs under {run.setup_dir}")

    out_dir = run.ltd_model_dir(root=analysis_root)
    written: list[tuple[str, Path, dict[str, Any]]] = []
    skipped: dict[str, str] = {}
    #: Fit samples are per (fold), not per (fold, method) -- shared here so a
    #: five-method run loads each fold's scatter once instead of five times.
    sample_cache: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}

    for method in wanted:
        if method not in available:
            skipped[method] = "not a combo of this run (no targets.parquet)"
            continue
        try:
            payload = build_payload(
                run,
                method,
                fold_ids=folds,
                inputs_root=inputs_root,
                max_points_per_vp=max_points_per_vp,
                preselect_tgs=preselect_tgs,
                sample_cache=sample_cache,
            )
        except MissingArtifactError as exc:
            skipped[method] = str(exc)
            continue
        path = out_dir / VIEWER_HTML.format(method=method)
        path.write_text(render_html(payload), encoding="utf-8")
        written.append((method, path, payload))

    write_manifest(
        out_dir,
        run,
        written,
        skipped,
        inputs_root=inputs_root,
        max_points_per_vp=max_points_per_vp,
    )
    return written, skipped


def write_manifest(
    out_dir: Path,
    run: RunPaths,
    written: list[tuple[str, Path, dict[str, Any]]],
    skipped: dict[str, str],
    *,
    inputs_root: Path,
    max_points_per_vp: int,
) -> Path:
    """`manifest.json`: what rendered, from which scatter, and what did not.

    `fit_sample_provenance` is the load-bearing field. The scatter is
    *reconstructed* for the runs that never materialized their inputs, and a
    reader comparing two pages has to be able to tell that from a materialized
    one without opening either.
    """
    manifest = {
        "run_id": run.run_id,
        "source": run.source,
        "setup": run.setup,
        "inputs_root": str(inputs_root),
        "max_points_per_vp": int(max_points_per_vp),
        "method_terms": method_term_table(m for m, _, _ in written),
        "methods": {
            method: {
                "html": path.name,
                "term": payload["method_label"],
                "ltd": payload["ltd"],
                "ltd_kwargs": payload["ltd_kwargs"],
                "folds": payload["fold_ids"],
                "n_vps": {
                    f: len(payload["folds"][f]["vps"]) for f in payload["fold_ids"]
                },
                "fit_sample_provenance": payload["provenance"],
                "skipped_folds": payload["skipped_folds"],
            }
            for method, path, payload in written
        },
        "skipped_methods": skipped,
    }
    path = out_dir / MANIFEST_JSON
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
