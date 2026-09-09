"""Interactive per-target map of one method's MTL output against the answer space.

The v3 figures are static: they show a distribution, not a case. This one is the
case viewer — pick a target and read, on one map, every quantity the top-1
classification verdict is made of:

  - the **seed regions**, drawn as the H3 cells they actually are, with the
    prediction's top-1 / top-2 / top-3 candidates coloured red / orange / yellow;
  - the **margin**, half the geodesic between the top-1 and top-2 seed — the
    radius inside which any coordinate snaps to top-1, so it is the scale at
    which prediction error becomes a wrong label (§8.1);
  - the **nearest-seed Voronoi partition** over the continental US, which is the
    classifier's own decision boundary;
  - every VP's **LTD constraint** (disk, or annulus when the LTD emits a lower
    bound) and the **MTL feasible region** those constraints intersect to;
  - the **prediction** and the **truth**, joined by the error.

Outcome vocabulary is deliberately flat — `correct` / `wrong` / `failed` from the
top-1 verdict alone. The v2 viewer's failure-mechanism taxonomy
(`NO_PROXIMITY` / `ERRONEOUS_CONTAINMENT` / `RTT_INFLATION`) is not carried: it
was keyed to a retired radius-50km answer space, and §8.2's replacement strata
are derivable from the proximity flags this map already shows.

**Nothing here reads a v3 artifact.** The answer space, the seed scoring, the
crossing matrix and the proximity labels are all rebuilt in-process by calling
the same functions that write them (`answer_space.build_for_run`,
`classify.score_combo`, `answer_space.seed_crossing_matrix`,
`proximity.build_proximity`), which takes ~0.2 s on a 400-target run. So the map
runs on a bare benchmark run, cannot disagree with the CSVs, and cannot go stale
against them.

The MTL feasible region is likewise recomputed rather than read: it is never
serialized by the benchmark. `mtl_participants[]` carries `vp_id`, `rtt_ms`,
`echoed_upper_km`, `echoed_lower_km`, `vp_lat`, `vp_lon` inline — exactly an
`LTDResult` — so replaying `MTL_REGISTRY[run.json["mtl"]](**mtl_kwargs)` over the
participants reproduces the bench-time region.

Output is a **single self-contained HTML** per method: Plotly from CDN, payload
inlined, no sibling JSON. The v2 viewers lazy-fetched per-target polygons and so
needed a local web server; at this scale the whole payload is a few MB, so
`file://` works.

Command: `plot-mtl-map`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules import io
from scripts.analysis.v3.modules.answer_space import (
    AnswerSpace,
    build_for_run as build_answer_space_for_run,
    elementwise_km,
    seed_crossing_matrix,
)
from scripts.analysis.v3.modules.grid import (
    DEFAULT_GRID,
    GRID_HELP,
    RESOLUTION_HELP,
    get_grid,
    resolve_cli_grid,
)
from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    RunPaths,
    resolve_run,
)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_HTML_TEMPLATE_PATH = _TEMPLATE_DIR / "mtl_map.html"
_JS_TEMPLATE_PATH = _TEMPLATE_DIR / "mtl_map.js"

#: `<method>` is a filename infix, not a directory: one run's maps all share the
#: same (grid, resolution) tree, and a method that overwrote its neighbour would
#: make the directory unreadable.
MAP_HTML = "mtl_map.{method}.html"

#: Replayed feasible regions, cached per (method, target). Rebuilding them is
#: the dominant cost of this command, and the cache is what makes a second
#: render — after a template edit, or after an interrupt — effectively free.
REGION_CACHE_DIR = "regions"

#: The Shortest-Ping baseline has no `targets.parquet`, so it is scored through
#: `classify.score_shortest_ping` and draws no constraint or region layer.
SHORTEST_PING = "shortest_ping"

#: How many candidate seeds the popup and the red/orange/yellow ramp cover.
TOP_K = 3

#: `seed_crossing_matrix` is O(K^2) and samples each geodesic; above this the
#: crossing count is dropped rather than paid for. K is 18-22 on today's runs.
MAX_SEEDS_FOR_CROSSING = 256

#: Region replay is CPU-bound and embarrassingly parallel. Capped below the
#: core count because each worker holds its own shapely arrangement.
DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))


# ---- derivation helpers -----------------------------------------------------


def _safe_float(x: Any) -> float | None:
    """NaN/inf/None -> None, so `json.dumps(..., allow_nan=False)` can be used.

    Serializing NaN produces a bare `NaN` token that is not valid JSON; the
    browser's `JSON.parse` rejects the whole payload. Failing in Python instead
    is the point of `allow_nan=False`, and this is what keeps it from firing.
    """
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _nested(value: Any) -> list:
    """Coerce a parquet list-of-struct cell to a plain list.

    pyarrow hands these back as numpy object arrays, so the idiomatic
    `value or []` raises "truth value of an array is ambiguous" — quietly, only
    on rows that happen to be non-empty.
    """
    if value is None:
        return []
    return list(value)


def _status_of(solved: bool, tg_seed_rank: Any) -> str:
    """Top-1 verdict: `correct` | `wrong` | `failed`.

    `solved` must come from `io.solved_mask`, not from an inline
    `status == "SUCCESS"`: the Shortest-Ping frame is all-`BASELINE`, and the
    inline test would score the baseline as universally failed.
    """
    if not solved:
        return "failed"
    rank = _safe_float(tg_seed_rank)
    return "correct" if rank is not None and int(rank) == 0 else "wrong"


def _proximity_label(row: Any) -> str:
    """Collapse the four proximity flags onto the §8.1 three-level ladder.

    `has_discriminative_vp => has_proximate_vp` is one of
    `proximity.IMPLICATIONS`, so the ladder is well-formed. The `*_sping_*` pair
    is deliberately not used: it describes what the RTT ranking selected, while
    this label describes what the geometry makes possible.
    """
    if bool(row["has_discriminative_vp"]):
        return "discriminative"
    if bool(row["has_proximate_vp"]):
        return "proximate"
    return "no proximity"


def _top_seeds(dist_row: np.ndarray, k: int = TOP_K) -> list[int]:
    """Indices of the `k` seeds nearest the prediction, nearest first.

    `classify` stores the full `dist_km__seed_*` vector precisely so top-N for
    any N is derivable; `pred_seed_id` is this list's head by construction.
    """
    finite = np.isfinite(dist_row)
    if not finite.any():
        return []
    order = np.argsort(np.where(finite, dist_row, np.inf), kind="stable")
    return [int(i) for i in order[: min(k, int(finite.sum()))]]


def _inflation(rtt_ms: float, km: float) -> float | None:
    """Observed RTT over the speed-of-internet RTT for the same great circle.

    `THEORETICAL_SLOPE` is the round-trip 2/3-c slope (0.01 ms/km) that v2 used
    to produce `min_inflation`, so a per-VP value here and the per-target
    minimum in the proximity table are on one scale. Undefined at zero distance.
    """
    from scripts.libs.cbg.rtt_model import THEORETICAL_SLOPE

    if km is None or km <= 0:
        return None
    return _safe_float(rtt_ms / (THEORETICAL_SLOPE * km))


# ---- geometry ---------------------------------------------------------------


def conus_boundary():
    """Natural Earth's US, minus Alaska, Hawaii and the outlying islands.

    `resolve_landmass("US")` returns the whole country, Aleutians included, and
    the repo has no CONUS helper — the one import that sounds like it
    (`scripts/processing/source/filter_mainland_and_min_pair_observations.py`)
    names a module that does not exist. Filtering parts by representative point
    against `US_MAINLAND_EXTENT` is exact here because the lower 48 are a single
    contiguous polygon at 110 m.
    """
    from shapely.geometry import MultiPolygon
    from shapely.ops import unary_union

    from scripts.analysis.v3.modules.mapping import US_MAINLAND_EXTENT
    from scripts.visualization.cluster.voronoi import resolve_landmass

    geom, _label = resolve_landmass("US")
    lon_min, lon_max, lat_min, lat_max = US_MAINLAND_EXTENT
    parts = list(getattr(geom, "geoms", [geom]))
    keep = []
    for part in parts:
        rp = part.representative_point()
        if lon_min <= rp.x <= lon_max and lat_min <= rp.y <= lat_max:
            keep.append(part)
    if not keep:
        return geom
    return unary_union(keep) if len(keep) > 1 else MultiPolygon([keep[0]]).geoms[0]


def conus_voronoi_rings(seeds: pd.DataFrame) -> list[list[list[float]]]:
    """Nearest-seed partition clipped to the continental US, as `[lat, lon]` rings.

    Computed in an azimuthal-equidistant frame centred on the seed mean, then
    unprojected. This is not optional: `mapping.seed_voronoi` records that a
    diagram built in raw lon/lat misassigns ~10.5% of frame area at these
    latitudes, because a degree of longitude is 87.6 km at 38N against 111.2 km
    of latitude.

    Seeds outside CONUS still seed the diagram — dropping them would hand their
    territory to a neighbour and move boundaries that are drawn — but cells are
    clipped to the boundary, so only the mainland is drawn.
    """
    from pyproj import CRS, Transformer
    from shapely.ops import transform as shapely_transform

    from scripts.visualization.cluster.voronoi import clipped_voronoi_cells

    lats = seeds["seed_lat"].to_numpy(dtype=float)
    lons = seeds["seed_lon"].to_numpy(dtype=float)
    if len(lats) < 2:
        return []

    boundary = conus_boundary()
    aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={float(lats.mean())} +lon_0={float(lons.mean())} "
        "+datum=WGS84 +units=m +no_defs"
    )
    fwd = Transformer.from_crs("EPSG:4326", aeqd, always_xy=True)
    inv = Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True)

    bx = shapely_transform(lambda x, y, z=None: fwd.transform(x, y), boundary)
    sx, sy = fwd.transform(lons, lats)
    cells = clipped_voronoi_cells(np.asarray(sx), np.asarray(sy), bx, crs=None)

    rings: list[list[list[float]]] = []
    for geom in cells.geometry:
        for poly in getattr(geom, "geoms", [geom]):
            ext = getattr(poly, "exterior", None)
            if ext is None:
                continue
            xs, ys = zip(*list(ext.coords))
            lo, la = inv.transform(np.asarray(xs), np.asarray(ys))
            rings.append(
                _cw_ring([[round(float(a), 4), round(float(o), 4)] for a, o in zip(la, lo)])
            )
    return rings


def _cw_ring(points: list[list[float]]) -> list[list[float]]:
    """Force a closed `[lat, lon]` ring clockwise in (lon, lat).

    Plotly's scattergeo `fill: "toself"` treats a closed lat/lon path as a
    *spherical* polygon and fills the side on the right of the walk. A
    counter-clockwise ring therefore fills the antipodal complement — the whole
    globe minus the shape — and a handful of those stacked turns the entire map
    into one flat wash of colour. Nothing about the outline changes, so the bug
    is invisible until something is actually filled.

    Same normalization, and the same reason, as
    `mtl_world_map._polygon_to_ring`; the shoelace is computed here because
    these rings come from H3 and pyproj rather than from Shapely, so there is no
    `is_ccw` to ask.
    """
    area = 0.0
    for (lat_a, lon_a), (lat_b, lon_b) in zip(points, points[1:]):
        area += lon_a * lat_b - lon_b * lat_a
    return points[::-1] if area > 0 else points


def seed_rings(seeds: pd.DataFrame) -> list[list[list[float]]]:
    """Each seed's own grid cell as a closed `[lat, lon]` ring.

    `Grid.cell_boundaries` returns ragged `(lon, lat)` arrays — 6 vertices for a
    hexagon, 5 for a pentagon — already made contiguous in longitude, so they are
    flipped but never re-wrapped. H3 emits them counter-clockwise, so every ring
    is closed and passed through `_cw_ring` before it is filled.
    """
    grid = get_grid(str(seeds["grid_scheme"].iloc[0]))
    resolution = int(seeds["grid_resolution"].iloc[0])
    out: list[list[list[float]]] = []
    for ring in grid.cell_boundaries(seeds["cell_id"].to_numpy(), resolution):
        pts = [[round(float(la), 4), round(float(lo), 4)] for lo, la in ring]
        if pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        out.append(_cw_ring(pts))
    return out


def _region_task(task: tuple[str, str, list[dict]]) -> dict | None:
    """Replay one target's MTL and serialize the region. Runs in a worker process.

    Module-level and taking only plain data so it pickles; the MTL is
    instantiated per call because the registry object is not picklable and
    construction is negligible against the intersection itself.
    """
    import scripts.framework.v2  # noqa: F401  (populates the registries)
    from scripts.framework.v2.registry import MTL_REGISTRY
    from scripts.framework.v2.ltd.base import LTDResult
    from scripts.framework.v2.types import Coord, Distance, Latency, VpId
    from scripts.visualization.benchmark.v2.mtl_world_map import _serialize_intersection

    mtl_name, mtl_kwargs_json, participants = task
    mtl = MTL_REGISTRY[mtl_name](**json.loads(mtl_kwargs_json))
    results = [
        LTDResult(
            success=True,
            error=None,
            vp_id=VpId(p["vp_id"]),
            vp_coord=Coord(lat=p["vp_lat"], lon=p["vp_lon"]),
            latency=Latency(p["rtt_ms"]) if p["rtt_ms"] is not None else None,
            tg_distance=Distance(upper_km=p["upper_km"], lower_km=p["lower_km"]),
        )
        for p in participants
    ]
    if not results:
        return None
    result = mtl.multilaterate(results)
    if not result.success:
        return None
    return _serialize_intersection(result.intersection)


def replay_mtl(
    run: RunPaths,
    combo_id: str,
    folds: pd.DataFrame,
    *,
    cache_dir: Path | None = None,
    workers: int = 1,
    progress: Any = None,
) -> dict[str, dict]:
    """`{target_id: serialized feasible region}` recomputed from the run's own MTL.

    The benchmark stores `mtl_intersection_kind` but never the geometry, so the
    region has to be rebuilt. `mtl_participants[]` is the post-filter constraint
    set with `vp_lat`/`vp_lon`/`rtt_ms` inline — everything the MTL reads — so no
    join against the VP roster or the RTT table is needed. Verified to reproduce
    `mtl_intersection_kind` and `n_mtl_participants` exactly.

    **This is the expensive part of the map, by three orders of magnitude.** The
    planar face decompositions are the same computation the benchmark paid for:
    `mtl_ms` on as01/octant_cbg_hull averages 7.0 s per target (median 0.9 s,
    max 78 s), so a serial replay of 399 targets is ~47 minutes. Hence both the
    process pool and the on-disk cache — the cache also makes an interrupted
    render resume, and makes re-rendering after a template edit instant.
    """
    specs = {
        (int(r.fold), str(r.combo_id)): (r.mtl, r.mtl_kwargs)
        for r in io.load_run_configs(run, [combo_id]).itertuples(index=False)
    }

    _invalidate_stale_cache(cache_dir, combo_id, specs, progress=progress)

    regions: dict[str, dict] = {}
    tasks: list[tuple[str, tuple[str, str, list[dict]]]] = []
    for row in folds.itertuples(index=False):
        tid = str(row.target_id)
        spec = specs.get((int(row.fold), combo_id))
        if spec is None or not spec[0]:
            continue
        mtl_name, mtl_kwargs = spec

        cached = _read_region_cache(cache_dir, combo_id, tid)
        if cached is not None:
            if cached:  # `{}` is the memo for "no region", and stays skipped
                regions[tid] = cached
            continue

        participants = [
            {
                "vp_id": str(p["vp_id"]),
                "vp_lat": float(p["vp_lat"]),
                "vp_lon": float(p["vp_lon"]),
                "rtt_ms": None if p.get("rtt_ms") is None else float(p["rtt_ms"]),
                "upper_km": float(p["echoed_upper_km"]),
                "lower_km": float(p.get("echoed_lower_km") or 0.0),
            }
            for p in _nested(row.mtl_participants)
            if p.get("echoed_upper_km")
        ]
        if not participants:
            continue
        kwargs_json = mtl_kwargs if isinstance(mtl_kwargs, str) else json.dumps(mtl_kwargs or {})
        tasks.append((tid, (mtl_name, kwargs_json, participants)))

    if not tasks:
        return regions
    if progress is not None:
        progress(
            f"    {combo_id}: replaying MTL for {len(tasks)} target(s) on "
            f"{workers} worker(s), {len(regions)} already cached…"
        )

    if workers > 1:
        # `spawn` would re-import the world per task; the default fork start
        # method keeps the already-imported registries.
        from multiprocessing import Pool

        with Pool(processes=workers) as pool:
            outs = pool.map(_region_task, [t[1] for t in tasks], chunksize=1)
    else:
        outs = [_region_task(t[1]) for t in tasks]

    for (tid, _), region in zip(tasks, outs):
        _write_region_cache(cache_dir, combo_id, tid, region)
        if region is not None:
            regions[tid] = region
    return regions


#: Sidecar recording which MTL spec produced a method's cached regions.
REGION_SPEC_JSON = "_spec.json"


def _spec_digest(specs: dict) -> str:
    """Stable digest of a combo's `(mtl, mtl_kwargs)` across its folds."""
    payload = sorted(
        (int(fold), str(name), str(kwargs)) for (fold, _combo), (name, kwargs) in specs.items()
    )
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


def _invalidate_stale_cache(
    cache_dir: Path | None, combo_id: str, specs: dict, *, progress: Any = None
) -> None:
    """Drop cached regions that a *different* MTL spec produced.

    The cache is keyed on (method, target), which is not enough on its own:
    re-running the benchmark for the same run_id with different `mtl_kwargs`
    would otherwise leave the map showing the previous configuration's regions
    beside the new configuration's predictions. A missing sidecar is adopted
    rather than treated as a mismatch, so caches written before this guard
    existed survive.
    """
    if cache_dir is None or not specs:
        return
    digest = _spec_digest(specs)
    sidecar = cache_dir / combo_id / REGION_SPEC_JSON
    if sidecar.exists():
        try:
            previous = json.loads(sidecar.read_text()).get("digest")
        except json.JSONDecodeError:
            previous = None
        if previous is not None and previous != digest:
            for stale in sidecar.parent.glob("*.json"):
                stale.unlink()
            if progress is not None:
                progress(f"    {combo_id}: MTL spec changed, dropped the region cache")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({"digest": digest}, indent=2) + "\n")


def _region_cache_path(cache_dir: Path | None, combo_id: str, target_id: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / combo_id / f"{target_id}.json"


def _read_region_cache(cache_dir: Path | None, combo_id: str, target_id: str) -> dict | None:
    """Cached region, `{}` for a memoized "no region", or None when uncached."""
    path = _region_cache_path(cache_dir, combo_id, target_id)
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _write_region_cache(
    cache_dir: Path | None, combo_id: str, target_id: str, region: dict | None
) -> None:
    """Persist a region, or `{}` so an empty result is not recomputed forever."""
    path = _region_cache_path(cache_dir, combo_id, target_id)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(region if region is not None else {}, separators=(",", ":")))


# ---- payload ----------------------------------------------------------------


def build_payload(
    run: RunPaths,
    space: AnswerSpace,
    method: str,
    *,
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    edges: pd.DataFrame,
    crossing: np.ndarray | None,
    rings: list[list[list[float]]],
    voronoi: list[list[list[float]]],
    vps: pd.DataFrame,
    regions: dict[str, dict] | None = None,
    folds: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Assemble the whole viewer payload for one method.

    Everything the page ever shows is inlined; there is no lazy fetch, which is
    what lets the file open over `file://`. Per-VP rows are positional arrays
    rather than objects, and coordinates are rounded to 4 dp (~11 m) — together
    those two choices are most of the difference between a 4 MB page and a 20 MB
    one.
    """
    seeds = space.seeds.reset_index(drop=True)
    seed_ids = seeds["seed_id"].to_numpy()
    #: `dist_km__seed_<id>` columns are ordered by the seeds frame, so a
    #: positional argsort indexes straight back into it.
    dist_cols = [f"dist_km__seed_{int(s)}" for s in seed_ids]
    dist = scores[dist_cols].to_numpy(dtype=float)
    solved = io.solved_mask(scores)

    assign = space.assignments.set_index("target_id")
    lab = labels.set_index("target_id")

    # The Shortest-Ping baseline has no `targets.parquet`, so it contributes
    # neither constraints nor a region; the caller passes empty frames for it.
    #
    # The viewer needs to know that as a *fact about the method* rather than
    # infer it from the data: an ordinary CBG run whose every target happened to
    # produce no region would otherwise be rendered as a baseline. Deciding it
    # here keeps the method name in the one module that owns the constant.
    is_baseline = method == SHORTEST_PING
    if folds is None:
        folds = pd.DataFrame(columns=["target_id", "ltd_predictions", "mtl_participants"])
    regions = regions or {}

    # Which constraints actually formed the region. Every MTL runs
    # `filter_redundant_outer_disks` when `enable_circle_filter` is on — a disk
    # that fully contains another is not the binding constraint, so it is
    # dropped — and records the survivors as `MTLResult.participating_vp_ids`,
    # which the benchmark persists as `mtl_participants[]`. Reading it back is
    # exact; `mtl_world_map` instead re-implements the heuristic client-side,
    # which is a mirror that can drift from the geometry it describes.
    #
    # The filter is not cosmetic at this scale: on as01 `million_scale_cbg`
    # keeps 4.5 of 133 disks, so drawing the unfiltered set buries the four that
    # decide the answer.
    kept_by_target: dict[str, set] = {}
    ltd_by_target: dict[str, list] = {}
    for row in folds.itertuples(index=False):
        tid = str(row.target_id)
        kept = {str(p["vp_id"]) for p in _nested(row.mtl_participants)}
        kept_by_target[tid] = kept
        ltd_by_target[tid] = [
            [
                str(p["vp_id"]),
                round(float(p["upper_km"]), 2),
                round(float(p.get("lower_km") or 0.0), 2),
                1 if str(p["vp_id"]) in kept else 0,
            ]
            for p in _nested(row.ltd_predictions)
            if p.get("success") and p.get("upper_km")
        ]

    # Observations, deduped to the minimum RTT per (target, VP) — the same rule
    # `proximity` applies, so the inflation shown per VP and `min_inflation`
    # cannot disagree about which edge they describe.
    obs = (
        edges.groupby(["target_id", "vp_id"], as_index=False)["rtt_ms"].min()
        .merge(vps[["vp_id", "vp_lat", "vp_lon"]], on="vp_id", how="inner")
        .join(assign[["target_lat", "target_lon"]], on="target_id", how="inner")
    )
    obs_km = elementwise_km(
        obs["target_lat"].to_numpy(dtype=float), obs["target_lon"].to_numpy(dtype=float),
        obs["vp_lat"].to_numpy(dtype=float), obs["vp_lon"].to_numpy(dtype=float),
    )
    obs_by_target: dict[str, list] = {}
    for (tid_, vp_, rtt_), km_ in zip(
        zip(obs["target_id"], obs["vp_id"], obs["rtt_ms"]), obs_km
    ):
        infl = _inflation(float(rtt_), float(km_))
        obs_by_target.setdefault(str(tid_), []).append(
            [str(vp_), round(float(rtt_), 3), None if infl is None else round(infl, 3)]
        )

    n_total_vps = int(len(vps))
    mtl_kind = "disk"
    for rows in ltd_by_target.values():
        if any(r[2] > 0 for r in rows):
            mtl_kind = "annulus"
            break

    targets: list[dict[str, Any]] = []
    for i, row in enumerate(scores.itertuples(index=False)):
        tid = str(row.target_id)
        top = _top_seeds(dist[i], TOP_K)
        pred_lat, pred_lon = _safe_float(row.pred_lat), _safe_float(row.pred_lon)
        pred = [round(pred_lat, 4), round(pred_lon, 4)] if None not in (pred_lat, pred_lon) else None

        # The margin: half the geodesic between the prediction's top-1 and
        # top-2 seed. Inside it, any coordinate snaps to top-1, so it is the
        # radius at which coordinate error turns into a wrong label.
        margin_km = None
        if len(top) >= 2:
            a, b = top[0], top[1]
            margin_km = _safe_float(
                0.5 * float(space.seed_mesh_km.iloc[a, b])
            )

        crossed = None
        tg_seed = _safe_float(row.tg_seed_id)
        pred_seed = _safe_float(row.pred_seed_id)
        if crossing is not None and tg_seed is not None and pred_seed is not None and pred_seed >= 0:
            crossed = int(crossing[int(tg_seed), int(pred_seed)])

        prox = lab.loc[tid] if tid in lab.index else None
        a_row = assign.loc[tid]
        t_obs = obs_by_target.get(tid, [])

        targets.append(
            {
                "target_id": tid,
                "fold": int(row.fold),
                "true": [round(float(a_row["target_lat"]), 4), round(float(a_row["target_lon"]), 4)],
                "tg_seed_id": int(tg_seed) if tg_seed is not None else None,
                "cell_offset_km": _safe_float(a_row["cell_offset_km"]),
                "pred": pred,
                "status": _status_of(bool(solved[i]), row.tg_seed_rank),
                "error_km": _safe_float(row.error_to_target_km),
                "seeds_crossed": crossed,
                "top_seeds": top,
                "margin_km": margin_km,
                "n_measured": len(t_obs),
                "n_total": n_total_vps,
                "proximity": _proximity_label(prox) if prox is not None else None,
                "sping_vp_id": str(prox["sping_vp_id"]) if prox is not None else None,
                "min_inflation": _safe_float(prox["min_inflation"]) if prox is not None else None,
                "obs": t_obs,
                "rings": ltd_by_target.get(tid, []),
                "n_kept": len(kept_by_target.get(tid, ())),
                "region": regions.get(tid),
            }
        )

    # Error ascending over the answered targets — `correct` and `wrong`
    # interleaved, since the ordering is the distance and not the verdict — then
    # every `failed` target after them. So the list reads best to worst and a
    # position in it means the same thing as a percentile of error does.
    #
    # A fallback does carry a coordinate and an `error_km`, but they are the
    # Shortest-Ping VP's rather than the method's, so that number is not on the
    # same scale as the rest of the column and cannot be ranked against it.
    # Sorting the failed block by it is still the useful order *within* the
    # block; `None` (no prediction at all) sorts to the very end.
    def _order(t: dict[str, Any]) -> tuple[int, int, float]:
        err = t["error_km"]
        return (1 if t["status"] == "failed" else 0, 1 if err is None else 0, err or 0.0)

    targets.sort(key=_order)

    return {
        "run_id": run.run_id,
        "method": method,
        "source": run.source,
        "setup": run.setup,
        "grid": str(seeds["grid_scheme"].iloc[0]),
        "resolution": int(seeds["grid_resolution"].iloc[0]),
        "earth_radius_km": EARTH_RADIUS_KM,
        "mtl_kind": mtl_kind,
        "is_baseline": is_baseline,
        "n_seeds": int(len(seeds)),
        "vps": {
            str(r.vp_id): [
                round(float(r.vp_lat), 4),
                round(float(r.vp_lon), 4),
                None if pd.isna(r.vp_asn) else str(r.vp_asn),
                None if pd.isna(r.vp_country) else str(r.vp_country),
            ]
            for r in vps.itertuples(index=False)
        },
        "seeds": [
            {
                "id": int(s),
                "lat": round(float(la), 4),
                "lon": round(float(lo), 4),
                "cell_id": str(c),
                "n_targets": int(n),
                "ring": rings[k],
            }
            for k, (s, la, lo, c, n) in enumerate(
                zip(
                    seeds["seed_id"], seeds["seed_lat"], seeds["seed_lon"],
                    seeds["cell_id"], seeds["n_targets"],
                )
            )
        ],
        "voronoi": voronoi,
        "targets": targets,
    }


def render_html(payload: dict[str, Any]) -> str:
    """Assemble the standalone page from the HTML shell + viewer JS.

    Substitution order is load-bearing: the JS goes in first (it contains
    neither of the other tokens), then the title, then the payload last. The
    blob escapes `</` so an embedded `</script>` cannot close the data block
    early, and `allow_nan=False` makes a stray NaN fail here rather than produce
    JSON the browser silently rejects.
    """
    # "MTL map" is a misnomer for a method with no multilateration stage, so the
    # heading is built here in full rather than half-prefixed in the shell.
    kind = "Classification map" if payload["is_baseline"] else "MTL map"
    title = f"{kind} — {payload['run_id']} · {payload['method']}"
    html = _HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    js = _JS_TEMPLATE_PATH.read_text(encoding="utf-8")
    html = html.replace("__SCRIPT__", js).replace("__TITLE__", title)
    blob = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    return html.replace("__PAYLOAD__", blob)


def build_for_run(
    run: RunPaths,
    *,
    methods: list[str],
    grid: str = DEFAULT_GRID,
    resolution: int | None = None,
    analysis_root: Path | None = None,
    regions: bool = True,
    workers: int = 1,
    progress: Any = None,
) -> list[tuple[str, Path, dict[str, Any]]]:
    """Render every requested method for one run. Returns `(method, path, payload)`.

    The answer space, crossing matrix, proximity labels, VP roster, edge table,
    seed rings and CONUS partition are all run-level, so they are built once and
    shared; only the seed scoring and the MTL replay are per method.
    """
    from scripts.analysis.v3.modules import bipartite, classify, proximity
    from scripts.benchmark.v2.eval_source import load_canonical_csv

    space = build_answer_space_for_run(run, grid=grid, resolution=resolution)
    space_grid = str(space.seeds["grid_scheme"].iloc[0])
    space_res = int(space.seeds["grid_resolution"].iloc[0])

    vps = io.load_vps(run)
    edges = load_canonical_csv(Path(bipartite.resolve_source_csv(run)))
    labels = proximity.build_proximity(
        space,
        vps,
        edges,
        io.load_sping_vp(run),
        context=io.load_eval_per_target(run),
        source_label=f"{run.run_id}/{run.source}/{run.setup}",
        source_csv=str(bipartite.resolve_source_csv(run)),
    ).labels
    crossing = (
        seed_crossing_matrix(space.seeds)
        if space.n_seeds <= MAX_SEEDS_FOR_CROSSING
        else None
    )
    rings = seed_rings(space.seeds)
    voronoi = conus_voronoi_rings(space.seeds)

    out_dir = run.mtl_map_dir(root=analysis_root, grid=space_grid, resolution=space_res)
    cache_dir = out_dir / REGION_CACHE_DIR
    rendered: list[tuple[str, Path, dict[str, Any]]] = []
    for method in methods:
        is_baseline = method == SHORTEST_PING
        scores = (
            classify.score_shortest_ping(run, space)
            if is_baseline
            else classify.score_combo(run, space, method)
        )
        folds = (
            None if is_baseline else io.load_folds(run, method, include_nested=True)
        )
        method_regions: dict[str, dict] = {}
        if regions and folds is not None:
            method_regions = replay_mtl(
                run, method, folds,
                cache_dir=cache_dir, workers=workers, progress=progress,
            )
        payload = build_payload(
            run, space, method,
            scores=scores, labels=labels, edges=edges, crossing=crossing,
            rings=rings, voronoi=voronoi, vps=vps,
            regions=method_regions, folds=folds,
        )
        out = out_dir / MAP_HTML.format(method=method)
        out.write_text(render_html(payload), encoding="utf-8")
        rendered.append((method, out, payload))
    return rendered


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("plot-mtl-map")
    def plot_mtl_map_cmd(
        run_id: str = typer.Option(..., help="Run to render. One run per invocation."),
        method: list[str] = typer.Option(
            [],
            "--method",
            "-m",
            help="Method to render; repeatable. Defaults to every combo in the "
                 f"run plus the {SHORTEST_PING!r} baseline.",
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option(
            [], "--resolution", "-r", help=RESOLUTION_HELP
        ),
        no_regions: bool = typer.Option(
            False,
            "--no-regions",
            help="Skip the MTL feasible-region layer. It is the only expensive "
                 "part of this command — the regions are not stored by the "
                 "benchmark, so each one is a full re-run of the planar "
                 "intersection (7 s per target on average for the Octant "
                 "family). Results are cached under mtl-map/<grid>/regions/, so "
                 "the cost is paid once per (method, target).",
        ),
        workers: int = typer.Option(
            DEFAULT_WORKERS,
            "--workers",
            "-j",
            help="Processes used for the region replay.",
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """Interactive per-target map of one method's MTL result and its verdict.

        Writes one self-contained mtl_map.<method>.html into the run's
        mtl-map/<grid>-<resolution>/ dir.
        """
        g, resolutions = resolve_cli_grid(grid, resolution, sweep=False)
        if g.name != "h3":
            raise typer.BadParameter(
                f"plot-mtl-map draws grid cells and supports --grid h3 only, got {g.name!r}"
            )
        if len(resolutions) > 1:
            raise typer.BadParameter(
                "plot-mtl-map takes one --resolution; it renders a case, not a sweep"
            )

        run = resolve_run(run_id, outputs_root)
        methods = list(method) or [*run.combo_ids, SHORTEST_PING]
        rendered = build_for_run(
            run,
            methods=methods,
            grid=g.name,
            resolution=resolutions[0],
            analysis_root=analysis_root,
            regions=not no_regions,
            workers=max(1, workers),
            progress=typer.echo,
        )
        for name, out, payload in rendered:
            counts = {"correct": 0, "wrong": 0, "failed": 0}
            for t in payload["targets"]:
                counts[t["status"]] += 1
            label = f"{payload['grid']} {g.resolution_arg}={payload['resolution']}"
            typer.echo(
                f"{run.run_id}: {label} · {name} · {len(payload['targets'])} targets "
                f"({counts['correct']} correct / {counts['wrong']} wrong / "
                f"{counts['failed']} failed) · K={payload['n_seeds']} -> {out}"
            )
