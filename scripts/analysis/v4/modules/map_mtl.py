"""Interactive per-target map of one method's MTL output against the answer space.

The v4 figures are static: they show a distribution, not a case. This is the
case viewer — pick a target and read, on one map, every quantity the ring-graded
verdict is made of:

  - the **truth's cell** and its ring-1 and ring-2 neighbours, drawn as the
    HEALPix cells they actually are, so "correct" is a region you can see rather
    than a rank you have to trust;
  - the **prediction's own cell**, which is what `ring` compares against;
  - every VP's **LTD constraint** (disk, or annulus when the LTD emits a lower
    bound) and the **MTL feasible region** those constraints intersect to;
  - the **prediction** and the **truth**, joined by the error.

## What this is NOT, and why that matters

v3's viewer drew a nearest-seed verdict: a top-1/2/3 seed ramp, a margin circle,
and a Voronoi partition that *was* the decision boundary. v4 retired that rule —
a Voronoi partition over K seeds labels every point on Earth, which is how
`tg-e1a1545` was scored CORRECT for a prediction 2,360 km away in the Canadian
Arctic (see `classify`). None of those layers are ported.

The **Voronoi overlay survives as context**, seeded by the centres of every
occupied target cell (`answer_space` seeds each class at `H.pix2ang(cell)`, so
`space.seeds` is exactly that centroid set). It is drawn because it is useful
for reading how the class centres are distributed — but it is no longer the
boundary any verdict is read off, and the viewer's copy says so. At nside 128
the class cells are 51 km while the Voronoi cells spanning them are hundreds of
km across; that mismatch is the argument v4 makes, drawn.

Likewise not ported: v3's three-level `proximity` label. Both flags underneath
it are defined over unbounded nearest-seed rank and half-gap margin, so printing
it beside a `ring2` verdict would put the retired rule back in the popup looking
authoritative. The two fields worth keeping from that table — `sping_vp_id` and
`min_inflation` — are plain columns of `eval_per_target.csv` and are read
directly (`eval_context`).

## Nothing here reads a v4 artifact

The answer space and the scoring are rebuilt in-process by calling the same
functions that write them (`answer_space.build_answer_space`,
`classify.score_method`), which costs milliseconds on a 400-target run. So the
map runs on a bare benchmark run, depends on no other v4 command, cannot
disagree with `accuracy.csv`, and cannot go stale against it.

The MTL feasible region is likewise recomputed rather than read: it is never
serialized by the benchmark. `mtl_participants[]` carries `vp_id`, `rtt_ms`,
`echoed_upper_km`, `echoed_lower_km`, `vp_lat`, `vp_lon` inline — exactly an
`LTDResult` — so replaying `MTL_REGISTRY[run.json["mtl"]](**mtl_kwargs)` over
the participants reproduces the bench-time region.

Output is a **single self-contained HTML** per method: Plotly from CDN, payload
inlined, no sibling JSON, so it opens over `file://` with no web server.

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

from scripts.analysis.v4.modules import bipartite as B
from scripts.analysis.v4.modules import classify as C
from scripts.analysis.v4.modules import edges as E
from scripts.analysis.v4.modules import healpix as H
from scripts.analysis.v4.modules.answer_space import (
    AnswerSpace,
    build_answer_space,
    elementwise_km,
    load_targets,
)
from scripts.analysis.v4.modules.figure_outcome_bars import SEGMENT_INK
from scripts.analysis.v4.modules.paths import MissingArtifactError, RunPaths
from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_HTML_TEMPLATE_PATH = _TEMPLATE_DIR / "mtl_map.html"
_JS_TEMPLATE_PATH = _TEMPLATE_DIR / "mtl_map.js"

#: `<method>` is a filename infix, not a directory: one run's maps all share the
#: same rung tree, and a method that overwrote its neighbour would make the
#: directory unreadable.
MAP_HTML = "mtl_map.{method}.html"

#: Sidecar recording which MTL spec produced a method's cached regions.
REGION_SPEC_JSON = "_spec.json"

#: Re-exported so the shell driver asks one module for the whole default method
#: list, rather than knowing that the control lives in `classify`.
SHORTEST_PING = C.SHORTEST_PING

#: The five statuses, in ladder order. Exactly `classify.summarize`'s
#: `OUTCOME_COUNTS` partition — see `status_of`.
STATUSES = ("ring0", "ring1", "ring2", "beyond", "failed")

#: `region_mode` values. A geometric MTL answers with a feasible set that can be
#: drawn as a polygon; a density MTL answers with a probability field over a
#: grid, which is a different kind of object.
REGION_GEOMETRIC = "geometric"
REGION_DENSITY = "density"

#: The nested constraint columns, requested alongside `classify.TARGET_COLUMNS`.
#: `score_method` copies the frame and writes only named columns, so these ride
#: through scoring and the payload builder needs no second read.
NESTED_COLUMNS = ("ltd_predictions", "mtl_participants")

#: Region replay is CPU-bound and embarrassingly parallel. Capped below the core
#: count because each worker holds its own shapely arrangement.
DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))

#: nside for a density combo's cell, when its `mtl_kwargs` predate the key.
_DEFAULT_DENSITY_NSIDE = 128


# ---- small helpers ----------------------------------------------------------


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


def _safe_str(x: Any) -> str | None:
    """A real string, or None for anything absent.

    Not `str(x)`: a missing CSV field arrives as None or NaN, and `str` turns
    those into the literal `"None"` / `"nan"`, which the viewer would then
    print as though it were a VP id.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return str(x)


def _inflation(rtt_ms: float, km: float) -> float | None:
    """Observed RTT over the speed-of-internet RTT for the same great circle.

    `THEORETICAL_SLOPE` is the round-trip 2/3-c slope (0.01 ms/km) the benchmark
    used to produce `min_inflation`, so a per-VP value here and the per-target
    minimum from `eval_per_target.csv` are on one scale. Undefined at zero
    distance.
    """
    from scripts.libs.cbg.rtt_model import THEORETICAL_SLOPE

    if km is None or km <= 0:
        return None
    return _safe_float(rtt_ms / (THEORETICAL_SLOPE * km))


def _cw_ring(lons, lats) -> list[list[float]]:
    """A closed, rounded, clockwise `[lat, lon]` ring from parallel lon/lat.

    Clockwise is not cosmetic. Plotly's scattergeo `fill: "toself"` reads a
    closed lat/lon path as a **spherical** polygon under the right-hand
    convention, so a counter-clockwise ring fills the antipodal complement — the
    whole globe minus the shape — and a handful of those stacked turns the
    entire map into one flat wash of colour. Nothing about the outline changes,
    so the bug is invisible until something is actually filled.

    One function where v3 had two (`_cw_ring` over `[lat, lon]` lists and
    `_clockwise` over `(lon, lat)` arrays), because both of v4's ring sources —
    `healpix.cell_rings` and the pyproj inverse transform — hand back parallel
    lon/lat. The winding rule is therefore stated once.
    """
    lon = np.asarray(lons, dtype=float)
    lat = np.asarray(lats, dtype=float)
    # Shoelace in an (x=lon, y=lat) frame. Positive is counter-clockwise.
    twice_area = float(np.sum(lon * np.roll(lat, -1) - np.roll(lon, -1) * lat))
    if twice_area > 0:
        lon, lat = lon[::-1], lat[::-1]
    pts = [[round(float(a), 4), round(float(o), 4)] for a, o in zip(lat, lon)]
    if pts and pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def status_of(solved: bool, ring: Any) -> str:
    """The per-target verdict: `ring0` | `ring1` | `ring2` | `beyond` | `failed`.

    Deliberately the same five-way partition `classify.summarize` counts as
    `n_ring0 .. n_failed`, so the map and the outcome bars cannot disagree about
    one target. `TestStatusIsTheSamePartitionAsTheBars` pins that.

    `solved` must come from `classify.solved_mask`, not from an inline
    `status == "SUCCESS"`: the Shortest-Ping frame is all-`BASELINE`, and the
    inline test would score the control as universally failed.
    """
    if not solved:
        return "failed"
    r = _safe_float(ring)
    if r is not None and 0 <= int(r) <= H.MAX_RING:
        return f"ring{int(r)}"
    return "beyond"


# ---- geometry ---------------------------------------------------------------


def conus_boundary():
    """Natural Earth's US, minus Alaska, Hawaii and the outlying islands.

    `resolve_landmass("US")` returns the whole country, Aleutians included, and
    the repo has no CONUS helper. Filtering parts by representative point
    against `US_MAINLAND_EXTENT` is exact here because the lower 48 are a single
    contiguous polygon at 110 m.
    """
    from shapely.geometry import MultiPolygon
    from shapely.ops import unary_union

    from scripts.analysis.v4.modules.mapping import US_MAINLAND_EXTENT
    from scripts.visualization.cluster.voronoi import resolve_landmass

    geom, _label = resolve_landmass("US")
    lon_min, lon_max, lat_min, lat_max = US_MAINLAND_EXTENT
    keep = []
    for part in list(getattr(geom, "geoms", [geom])):
        rp = part.representative_point()
        if lon_min <= rp.x <= lon_max and lat_min <= rp.y <= lat_max:
            keep.append(part)
    if not keep:
        return geom
    return unary_union(keep) if len(keep) > 1 else MultiPolygon([keep[0]]).geoms[0]


def conus_voronoi_rings(seeds: pd.DataFrame) -> list[list[list[float]]]:
    """Nearest-class-centre partition clipped to CONUS, as `[lat, lon]` rings.

    **Context, not the verdict.** In v3 this was the classifier's own decision
    boundary; in v4 correctness is containment in a cell, and this is drawn
    because the distribution of class centres is worth seeing — not because
    anything is scored against it. The viewer's copy states that, because a
    reader who knows the v3 map will otherwise read the boundary as the answer.

    Computed in an azimuthal-equidistant frame centred on the seed mean, then
    unprojected. This is not optional: a diagram built in raw lon/lat misassigns
    ~10.5% of frame area at these latitudes, because a degree of longitude is
    87.6 km at 38N against 111.2 km of latitude.

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
            rings.append(_cw_ring(lo, la))
    return rings


def cell_polygons(cell_ids, nside: int) -> dict[int, list[list[float]]]:
    """`{cell_id: closed clockwise [lat, lon] ring}` for the given cells.

    The payload's one source of cell geometry. Every layer that draws a cell —
    the ring neighbourhood, the class cells, the prediction's cell — indexes
    this table, so they cannot draw the same cell two different ways, and a
    cell shared by several targets is serialized once.

    That sharing is the reason the table exists at all. Inlining a
    neighbourhood per target is ~25 cells x 33 vertices x 400 targets, a 12-20 MB
    page; but 399 targets occupy ~18 cells at nside 128, so the deduplicated
    table is a few hundred entries and the page stays openable over `file://`.
    """
    ids = np.unique(np.asarray(list(cell_ids), dtype=np.int64))
    ids = ids[ids >= 0]
    if ids.size == 0:
        return {}
    return {
        int(c): _cw_ring(ring[:, 0], ring[:, 1])
        for c, ring in zip(ids, H.cell_rings(ids, nside))
    }


def ring_neighbourhood(cell_ids, nside: int) -> dict[int, list[list[int]]]:
    """`{cell_id: [[cell], ring-1 cells, ring-2 cells]}`, memoised per cell.

    Keyed on the truth's cell rather than on the target, because targets in the
    same class share a neighbourhood exactly. On a 399-target run that is ~18
    `ring_cells` calls instead of 399.
    """
    return {
        int(c): H.ring_cells(int(c), nside)
        for c in np.unique(np.asarray(list(cell_ids), dtype=np.int64))
        if int(c) >= 0
    }


# ---- the run's own MTL ------------------------------------------------------


def load_mtl_specs(run: RunPaths, combo_id: str) -> dict[int, tuple[str, str]]:
    """`{fold: (mtl_name, mtl_kwargs_json)}` from each fold's `run.json`.

    Trimmed from v3's `io.load_run_configs`, which returns 24 columns for its
    cost and memory commands — v4 has neither, and porting the frame would put
    20 dead columns in the package for someone to build on. Keyed on `fold`
    alone because the `(fold, combo)` composite key was vestigial: every call
    site passes one combo.

    Missing or unreadable `run.json` files are skipped rather than raised on.
    The callers all degrade safely — an unknown MTL means "no region", which
    costs a layer, not correctness.
    """
    specs: dict[int, tuple[str, str]] = {}
    for fold in run.fold_ids:
        path = run.combo_dir(combo_id, fold) / "run.json"
        if not path.exists():
            continue
        try:
            spec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        name = spec.get("mtl")
        if not name:
            continue
        kwargs = spec.get("mtl_kwargs") or {}
        specs[int(fold.split("_")[1])] = (
            str(name),
            kwargs if isinstance(kwargs, str) else json.dumps(kwargs),
        )
    return specs


def is_density_mtl(mtl_name: str) -> bool:
    """True when this MTL's answer is a probability field, not a feasible set.

    Asked of the registry rather than matched against a name list, so a future
    density family is covered the day it is registered. `DensityMTLMethod` is
    the marker base; today `gaussian_density` is its only subclass and the four
    planar/spherical families are not.
    """
    if not mtl_name:
        return False
    import scripts.framework.v2  # noqa: F401  (populates the registries)
    from scripts.framework.v2.mtl.base import DensityMTLMethod
    from scripts.framework.v2.registry import MTL_REGISTRY

    cls = MTL_REGISTRY.get(mtl_name)
    return cls is not None and issubclass(cls, DensityMTLMethod)


def combo_region_mode(run: RunPaths, combo_id: str) -> str:
    """`REGION_DENSITY` if this combo's MTL is a density family, else geometric.

    Read from the combo's own `run.json` rather than from a config, so it
    describes what actually ran. An unreadable spec degrades to geometric — the
    status quo, and the safe default because it only costs a replay.
    """
    names = {name for name, _ in load_mtl_specs(run, combo_id).values()}
    return REGION_DENSITY if any(is_density_mtl(n) for n in names) else REGION_GEOMETRIC


def density_nside(run: RunPaths, combo_id: str) -> int | None:
    """The nside a density combo reported at, from its stored `mtl_kwargs`.

    Read off `run.json` rather than by constructing the MTL: the grid is the
    only thing needed, and constructing it would build a global grid and (for a
    combo predating the required `grid` kwarg) raise.

    Returns None when the combo's folds disagree — which cannot happen from one
    `run-combo`, but would make the drawn cell a lie if it did — or when the
    MTL reported on a grid that is not HEALPix.
    """
    seen: set[int] = set()
    for _name, kwargs_json in load_mtl_specs(run, combo_id).values():
        try:
            kwargs = json.loads(kwargs_json)
        except json.JSONDecodeError:
            continue
        if not isinstance(kwargs, dict):
            continue
        if kwargs.get("grid") not in (None, "healpix"):
            return None
        seen.add(int(kwargs.get("resolution", _DEFAULT_DENSITY_NSIDE)))
    if len(seen) != 1:
        return None
    return seen.pop()


def argmax_cell_regions(
    run: RunPaths, combo_id: str, scored: pd.DataFrame
) -> tuple[dict[str, dict], int | None]:
    """`({target_id: the cell the prediction fell in}, nside)` for a density combo.

    A density MTL's answer is a probability field, which `replay_mtl` cannot
    rebuild: `mtl_participants` stores the echoed band and not `mu_km`/
    `sigma_km`, and the band is explicitly not invertible back to the
    distribution. So this map would otherwise draw nothing at all for Spotter.

    None of that needs solving, because `density_argmax` returns a **cell
    centre**. Re-binning the persisted prediction therefore recovers the exact
    cell it came from — `ang2pix(pix2ang(p)) == p` is a pinned property of the
    grid — and the cell boundary is closed-form. So the region comes back for
    free: no MTL construction, no global grid, no replay, and nothing read that
    the benchmark did not already write.

    What is drawn is narrower than a geometric method's region, and the viewer
    says so: it is the argmax cell, i.e. the quantisation of the point estimate,
    not a feasible set and not a credible region. Its area is exactly
    `pixel_area_km2(nside)` because HEALPix cells are equal-area.

    Returns the nside alongside, because it is the **MTL's own** grid and need
    not equal the rung the map is scored on.
    """
    nside = density_nside(run, combo_id)
    if nside is None:
        return {}, None
    lat = pd.to_numeric(scored["pred_lat"], errors="coerce")
    lon = pd.to_numeric(scored["pred_lon"], errors="coerce")
    ok = lat.notna() & lon.notna()
    if not ok.any():
        return {}, nside
    pix = H.ang2pix(lat[ok].to_numpy(), lon[ok].to_numpy(), nside)
    rings = H.cell_rings(pix, nside)
    out: dict[str, dict] = {}
    for tid, ring in zip(scored.loc[ok, "target_id"].astype(str), rings):
        out[str(tid)] = {
            "kind": "healpix_cell",
            "rings": [{"outer": _cw_ring(ring[:, 0], ring[:, 1]), "holes": []}],
        }
    return out, nside


# ---- region replay ----------------------------------------------------------


def _region_task(task: tuple[str, str, list[dict]]) -> dict | None:
    """Replay one target's MTL and serialize the region. Runs in a worker process.

    Module-level and taking only plain data so it pickles; the MTL is
    instantiated per call because the registry object is not picklable and
    construction is negligible against the intersection itself.
    """
    import scripts.framework.v2  # noqa: F401  (populates the registries)
    from scripts.framework.v2.ltd.base import LTDResult
    from scripts.framework.v2.registry import MTL_REGISTRY
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


def _spec_digest(specs: dict[int, tuple[str, str]]) -> str:
    """Stable digest of a combo's `(mtl, mtl_kwargs)` across its folds."""
    payload = sorted(
        (int(fold), str(name), str(kwargs)) for fold, (name, kwargs) in specs.items()
    )
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


def _invalidate_stale_cache(
    cache_dir: Path | None,
    combo_id: str,
    specs: dict[int, tuple[str, str]],
    *,
    progress: Any = None,
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


def _region_cache_path(
    cache_dir: Path | None, combo_id: str, target_id: str
) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / combo_id / f"{target_id}.json"


def _read_region_cache(
    cache_dir: Path | None, combo_id: str, target_id: str
) -> dict | None:
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
    path.write_text(
        json.dumps(region if region is not None else {}, separators=(",", ":"))
    )


def replay_mtl(
    run: RunPaths,
    combo_id: str,
    frame: pd.DataFrame,
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

    The cache is rung-free (`RunPaths.mtl_region_cache_dir`): no nside enters
    this function, so a second rung must not re-pay the replay.
    """
    specs = load_mtl_specs(run, combo_id)

    # A density MTL cannot be replayed from what the benchmark persisted, and
    # this is a schema limit rather than something to work around here.
    # `mtl_participants` stores only `echoed_upper_km` / `echoed_lower_km`, so
    # the `Distance` rebuilt below carries no `mu_km` / `sigma_km`;
    # `GaussianDensityMTL` requires `has_distribution` and the bounds are
    # explicitly not invertible back to the distribution. Attempting it returned
    # INSUFFICIENT_DATA for every target -- while still paying a global grid
    # construction each time, and while memoising `{}` so no later render would
    # retry.
    #
    # Returning before the cache is touched is the point: an empty result here
    # means "not applicable", which must stay distinguishable from an MTL that
    # genuinely found nothing, and must not be frozen into the cache.
    #
    # The region is not lost, it just comes from somewhere cheaper: see
    # `argmax_cell_regions`, which re-bins the stored prediction to recover the
    # cell `density_argmax` chose. `build_for_run` routes density combos there.
    density = sorted({n for n, _ in specs.values() if n and is_density_mtl(n)})
    if density:
        if progress is not None:
            progress(
                f"    {combo_id}: {'/'.join(density)} answers with a probability "
                f"field, not a feasible set — no region to replay"
            )
        return {}

    _invalidate_stale_cache(cache_dir, combo_id, specs, progress=progress)

    regions: dict[str, dict] = {}
    tasks: list[tuple[str, tuple[str, str, list[dict]]]] = []
    for row in frame.itertuples(index=False):
        tid = str(row.target_id)
        spec = specs.get(int(row.fold))
        if spec is None or not spec[0]:
            continue
        mtl_name, kwargs_json = spec

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


# ---- eval-source context ----------------------------------------------------

#: The two fields worth keeping from v3's proximity table. Both are plain
#: columns of `eval_per_target.csv`; `proximity.py` reached them through pure
#: passthroughs, so 732 lines of nearest-seed machinery buy nothing here.
_CONTEXT_COLUMNS = {"shortest_ping_vp_id": "sping_vp_id", "min_inflation": "min_inflation"}


def eval_context(run: RunPaths) -> pd.DataFrame:
    """`target_id, sping_vp_id, min_inflation`, or an empty frame.

    Absent `eval_source/` is not an error: the map's own layers are all built
    from the benchmark tree, and these two are context. A run without them
    renders with both fields null rather than failing.
    """
    empty = pd.DataFrame(columns=["target_id", *_CONTEXT_COLUMNS.values()])
    try:
        raw = pd.read_csv(run.eval_file("eval_per_target.csv"))
    except (MissingArtifactError, FileNotFoundError):
        return empty
    have = [c for c in _CONTEXT_COLUMNS if c in raw.columns]
    if "target_id" not in raw.columns or not have:
        return empty
    out = raw[["target_id", *have]].drop_duplicates("target_id")
    out = out.rename(columns=_CONTEXT_COLUMNS)
    # Both columns, always. `build_payload` indexes each by name on any frame
    # this returns, so a CSV carrying one of the two would raise a KeyError
    # rather than degrade to the null field this is supposed to give it.
    for col in _CONTEXT_COLUMNS.values():
        if col not in out.columns:
            out[col] = None
    return out[["target_id", *_CONTEXT_COLUMNS.values()]]


# ---- payload ----------------------------------------------------------------


def build_payload(
    run: RunPaths,
    space: AnswerSpace,
    method: str,
    *,
    scored: pd.DataFrame,
    edges: pd.DataFrame,
    vps: pd.DataFrame,
    context: pd.DataFrame,
    voronoi: list[list[list[float]]],
    regions: dict[str, dict] | None = None,
    region_mode: str = REGION_GEOMETRIC,
    density_nside: int | None = None,
) -> dict[str, Any]:
    """Assemble the whole viewer payload for one method.

    Everything the page ever shows is inlined; there is no lazy fetch, which is
    what lets the file open over `file://`. Per-VP rows are positional arrays
    rather than objects, coordinates are rounded to 4 dp (~11 m), and cell
    geometry is shared through `cells` rather than repeated per target —
    together those three choices are most of the difference between a page under
    a megabyte and one over fifteen.
    """
    nside = space.nside
    seeds = space.seeds.reset_index(drop=True)
    solved = C.solved_mask(scored).to_numpy()
    assign = space.assignments.set_index("target_id")
    ctx = context.set_index("target_id") if len(context) else None

    if region_mode not in (REGION_GEOMETRIC, REGION_DENSITY):
        raise ValueError(f"unknown region_mode: {region_mode!r}")
    regions = regions or {}

    # The Shortest-Ping control has no `targets.parquet`, so it contributes
    # neither constraints nor a region.
    #
    # The viewer needs to know that as a *fact about the method* rather than
    # infer it from the data: an ordinary CBG run whose every target happened to
    # produce no region would otherwise be rendered as a baseline. Deciding it
    # here keeps the method name in the one module that owns the constant.
    is_baseline = method == SHORTEST_PING
    has_nested = NESTED_COLUMNS[1] in scored.columns

    # Which constraints actually formed the region. Every MTL runs
    # `filter_redundant_outer_disks` when `enable_circle_filter` is on — a disk
    # that fully contains another is not the binding constraint, so it is
    # dropped — and records the survivors as `MTLResult.participating_vp_ids`,
    # which the benchmark persists as `mtl_participants[]`. Reading it back is
    # exact, rather than re-implementing the heuristic client-side where it
    # could drift from the geometry it describes.
    #
    # The filter is not cosmetic at this scale: on as01 `million_scale_cbg`
    # keeps 4.5 of 133 disks, so drawing the unfiltered set buries the four that
    # decide the answer.
    kept_by_target: dict[str, set] = {}
    ltd_by_target: dict[str, list] = {}
    if has_nested:
        for row in scored.itertuples(index=False):
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

    # Observations, deduped to the minimum RTT per (target, VP) upstream in
    # `edges.load_min_rtt`, so the inflation shown per VP and the eval source's
    # `min_inflation` cannot disagree about which edge they describe.
    obs = (
        edges.merge(vps[["vp_id", "vp_lat", "vp_lon"]], on="vp_id", how="inner")
        .join(assign[["target_lat", "target_lon"]], on="target_id", how="inner")
    )
    obs_km = elementwise_km(
        obs["target_lat"].to_numpy(dtype=float),
        obs["target_lon"].to_numpy(dtype=float),
        obs["vp_lat"].to_numpy(dtype=float),
        obs["vp_lon"].to_numpy(dtype=float),
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

    nbhd = ring_neighbourhood(scored["tg_cell"], nside)

    targets: list[dict[str, Any]] = []
    for i, row in enumerate(scored.itertuples(index=False)):
        tid = str(row.target_id)
        pred_lat, pred_lon = _safe_float(row.pred_lat), _safe_float(row.pred_lon)
        pred = (
            [round(pred_lat, 4), round(pred_lon, 4)]
            if None not in (pred_lat, pred_lon)
            else None
        )
        tg_cell = int(row.tg_cell)
        pred_cell = int(row.pred_cell)
        a_row = assign.loc[tid]
        t_obs = obs_by_target.get(tid, [])
        c_row = ctx.loc[tid] if ctx is not None and tid in ctx.index else None

        targets.append(
            {
                "target_id": tid,
                "fold": int(row.fold),
                "true": [
                    round(float(a_row["target_lat"]), 4),
                    round(float(a_row["target_lon"]), 4),
                ],
                "tg_seed_id": int(row.tg_seed_id),
                "tg_cell": tg_cell,
                "pred_cell": pred_cell,
                "ring": int(row.ring),
                "ring_cells": nbhd.get(tg_cell, [[tg_cell], [], []]),
                "cell_offset_km": _safe_float(a_row["cell_offset_km"]),
                "pred": pred,
                "status": status_of(bool(solved[i]), row.ring),
                "error_km": _safe_float(row.error_km),
                "n_measured": len(t_obs),
                "n_total": n_total_vps,
                "sping_vp_id": (
                    None if c_row is None else _safe_str(c_row["sping_vp_id"])
                ),
                "min_inflation": (
                    None if c_row is None else _safe_float(c_row["min_inflation"])
                ),
                "obs": t_obs,
                "rings": ltd_by_target.get(tid, []),
                "n_kept": len(kept_by_target.get(tid, ())),
                "region": regions.get(tid),
            }
        )

    # Error ascending over the answered targets — the ring statuses interleave,
    # since the ordering is the distance and not the verdict — then every
    # `failed` target after them. So the list reads best to worst and a position
    # in it means the same thing as a percentile of error does.
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

    # One table for every cell any layer draws: the neighbourhoods, the class
    # cells, and each prediction's own cell.
    needed: set[int] = set(int(c) for c in seeds["cell_id"])
    for t in targets:
        for ring in t["ring_cells"]:
            needed.update(int(c) for c in ring)
        if t["pred_cell"] >= 0:
            needed.add(t["pred_cell"])
    cells = cell_polygons(needed, nside)

    return {
        "run_id": run.run_id,
        "method": method,
        "source": run.source,
        "setup": run.setup,
        "nside": nside,
        "cell_km": round(H.nominal_cell_km(nside), 1),
        "max_ring": H.MAX_RING,
        # The MTL's OWN grid, which is not necessarily the answer space's. The
        # seeds come from the rung the map was scored on; a density combo's
        # drawn cell comes from its `mtl_kwargs`. They are 128 on both sides
        # today and nothing requires that, so they stay separate fields.
        "density_nside": density_nside,
        # True when the two coincide, so the viewer draws ONE polygon with a
        # combined label instead of stacking two identical fills under two
        # different names.
        "region_is_pred_cell": bool(
            region_mode == REGION_DENSITY and density_nside == nside
        ),
        "earth_radius_km": EARTH_RADIUS_KM,
        "mtl_kind": mtl_kind,
        "region_mode": region_mode,
        "is_baseline": is_baseline,
        "n_seeds": int(len(seeds)),
        "palette": {
            "ring0": SEGMENT_INK["n_ring0"],
            "ring1": SEGMENT_INK["n_ring1"],
            "ring2": SEGMENT_INK["n_ring2"],
            "beyond": SEGMENT_INK["n_beyond"],
            "failed": SEGMENT_INK["n_failed"],
        },
        "cells": {str(c): ring for c, ring in cells.items()},
        "vps": {
            str(r.vp_id): [
                round(float(r.vp_lat), 4),
                round(float(r.vp_lon), 4),
                None if pd.isna(getattr(r, "vp_asn", None)) else str(r.vp_asn),
                None if pd.isna(getattr(r, "vp_country", None)) else str(r.vp_country),
            ]
            for r in vps.itertuples(index=False)
        },
        "seeds": [
            {
                "id": int(s),
                "lat": round(float(la), 4),
                "lon": round(float(lo), 4),
                "cell_id": int(c),
                "n_targets": int(n),
            }
            for s, la, lo, c, n in zip(
                seeds["seed_id"],
                seeds["seed_lat"],
                seeds["seed_lon"],
                seeds["cell_id"],
                seeds["n_targets"],
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
    title = (
        f"{kind} — {payload['run_id']} · {payload['method']} · "
        f"healpix nside {payload['nside']}"
    )
    html = _HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    js = _JS_TEMPLATE_PATH.read_text(encoding="utf-8")
    html = html.replace("__SCRIPT__", js).replace("__TITLE__", title)
    blob = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    return html.replace("__PAYLOAD__", blob)


def _load_edges(run: RunPaths, *, progress: Any = None) -> pd.DataFrame:
    """The per-VP observation table, or an empty one when the CSV is absent.

    Absence is degradable and the refusal is not, which is why
    `MeshSupersetError` is re-raised rather than swallowed. The canonical CSV
    lives under `datasets/`, outside the benchmark output tree, so a run whose
    dataset has been moved or archived would otherwise lose its whole map --
    and "renders on a bare benchmark run" is the property this module is built
    around. What is lost without it is one layer: the per-VP RTTs and their
    inflation. The verdict, the rings, the constraints and the region all come
    from the run's own parquets and are unaffected.

    A mesh superset is the opposite case. That file parses and every RTT in it
    is real, but they are edges this arm never measured, so drawing them would
    be a quiet lie about the population rather than a missing layer.
    """
    try:
        return E.load_min_rtt(run)
    except E.MeshSupersetError:
        raise
    except MissingArtifactError as exc:
        if progress is not None:
            progress(f"    no canonical edge CSV ({exc}); per-VP RTTs omitted")
        return pd.DataFrame(columns=list(E.MIN_RTT_COLUMNS))


def build_for_run(
    run: RunPaths,
    *,
    methods: list[str],
    nside: int = H.DEFAULT_NSIDE,
    analysis_root: Path | None = None,
    regions: bool = True,
    workers: int = 1,
    progress: Any = None,
) -> list[tuple[str, Path, dict[str, Any]]]:
    """Render one HTML per method. Returns `[(method, path, payload)]`.

    The answer space and the scoring are built in-process rather than read from
    `target-answer-space/` and `target-cls-accuracy/`. That is the module's
    headline property: the map depends on no v4 command, so it renders on a bare
    benchmark run, and it cannot show a verdict `accuracy.csv` disagrees with.
    """
    nside = H.validate_nside(nside)
    targets = load_targets(run)
    space = build_answer_space(
        targets,
        nside=nside,
        source_label=f"{run.run_id}/{run.source}/{run.setup}",
    )
    vps = B.load_vps(run)
    edge_table = _load_edges(run, progress=progress)
    context = eval_context(run)
    voronoi = conus_voronoi_rings(space.seeds)

    out_dir = run.mtl_map_dir(nside, root=analysis_root)
    cache_dir = run.mtl_region_cache_dir(root=analysis_root)

    rendered: list[tuple[str, Path, dict[str, Any]]] = []
    for method in methods:
        is_baseline = method == SHORTEST_PING
        if is_baseline:
            # The control is scored over the EVALUATED roster rather than over
            # the eval source, so it needs some combo's targets.parquet to say
            # which targets the run actually evaluated. Any combo will do --
            # K-fold splits targets, not methods. No guard for an empty
            # `combo_ids`: `load_targets` above raises on exactly that, so this
            # line is unreachable with one.
            frame = C.load_shortest_ping_frame(
                run, C.load_method_frame(run, run.combo_ids[0])
            )
        else:
            frame = C.load_method_frame(
                run, method, columns=C.TARGET_COLUMNS + NESTED_COLUMNS
            )
        scored = C.score_method(frame, space)

        region_mode = (
            REGION_GEOMETRIC if is_baseline else combo_region_mode(run, method)
        )
        method_regions: dict[str, dict] = {}
        method_density_nside: int | None = None
        if regions and not is_baseline:
            if region_mode == REGION_DENSITY:
                # Derived from the stored prediction, not replayed -- see
                # `argmax_cell_regions`. No cache, because it is closed-form.
                method_regions, method_density_nside = argmax_cell_regions(
                    run, method, scored
                )
                if progress is not None:
                    progress(
                        f"    {method}: {len(method_regions)} argmax cells from "
                        f"the stored predictions (no replay needed)"
                    )
            else:
                method_regions = replay_mtl(
                    run,
                    method,
                    scored,
                    cache_dir=cache_dir,
                    workers=workers,
                    progress=progress,
                )
        elif region_mode == REGION_DENSITY:
            # `--no-regions` still reports which grid the method answered on.
            method_density_nside = density_nside(run, method)

        payload = build_payload(
            run,
            space,
            method,
            scored=scored,
            edges=edge_table,
            vps=vps,
            context=context,
            voronoi=voronoi,
            regions=method_regions,
            region_mode=region_mode,
            density_nside=method_density_nside,
        )
        out = out_dir / MAP_HTML.format(method=method)
        out.write_text(render_html(payload), encoding="utf-8")
        rendered.append((method, out, payload))
    return rendered
