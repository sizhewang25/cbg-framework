"""Interactive per-TG map of one method's MTL output against both partitions.

The v5 figures are static: they show a distribution, not a case. This is the
case viewer -- pick a TG and read, on one map, every quantity the two verdicts
are made of:

  - the **grid axis**: the truth's grid and its ring-1 and ring-2 neighbours,
    drawn as the HEALPix grids they actually are, plus the grid the prediction
    landed in, so "correct" is a region you can see rather than a rank you have
    to trust;
  - the **cell axis**: the Voronoi cell of every seed, and in particular the
    TG's own cell against the cell the prediction fell in -- which is exactly
    what `cell_label` compares;
  - every VP's **LTD constraint** (disk, or annulus when the LTD emits a lower
    bound) and the **MTL feasible region** those constraints intersect to;
  - the **prediction** and the **truth**, joined by the error.

## Both partitions, because neither is a verdict alone

v4 drew one partition and called the Voronoi overlay context. v5 grades on
both, so this viewer draws both and says so. The cells are **unbounded**: every
point on Earth is nearest to some seed, so `cell_label` on its own will call a
prediction correct at any distance -- the Canadian Arctic case `classify`
documents. The grid axis is what bounds it. Read them together.

## The grid axis is an exact offset, banded here

`classify` writes `pred_dist_to_tg_grid`: exact grid steps, uncapped, with `-1`
meaning "no prediction" and nothing else. This module bands it into the same
five statuses `summarize` counts -- `ring0`, `ring1`, `ring2`, `beyond`,
`failed` -- via `status_of`, so the map and the outcome bars cannot disagree
about one TG.

What the popup prints, though, is the **offset itself**. `beyond` spans 3 to 68
grids out on these runs, and a case viewer whose whole job is one TG should say
"14 grids out" rather than repeat the bucket name.

**The drawn neighbourhood stays capped at `classify.MAX_RING`.** `ring_grids`
grows a disk, and the disk at offset 68 is some 15,000 grids -- undrawable, and
meaningless as a shaded region. Past the second ring the offset is a number and
a status, never a polygon.

## What is read, and what is rebuilt

The **answer space is loaded**, not rebuilt: `build-answer-space` must have run.
v4's viewer rebuilt it in-process and advertised that it ran on a bare
benchmark run. That property is not worth keeping here, because a v5 seed is a
complete-linkage cluster centroid rather than a closed-form pixel centre --
rebuilding risks drawing cells that are not the cells `classify` scored, which
is the one thing a case viewer must never do.

The **scoring is still in-process** (`classify.score_method`), because the
nested constraint columns have to ride through it on the same frame. So the map
still cannot disagree with `accuracy.csv`.

The **MTL feasible region is recomputed**: it is never serialized by the
benchmark. `mtl_participants[]` carries `vp_id`, `rtt_ms`, `echoed_upper_km`,
`echoed_lower_km`, `vp_lat`, `vp_lon` inline -- exactly an `LTDResult` -- so
replaying `MTL_REGISTRY[run.json["mtl"]](**mtl_kwargs)` over the participants
reproduces the bench-time region.

The **LTD cutoff is read from the fold's `fit_checkpoint.pkl`**, the only place
that fitted value is kept. Constraints whose RTT is past it are drawn as their
own layer: there the band is Octant/Spotter's sentinel extension (outer bound
parallel to 2/3*c, inner held at its cutoff value), not the fit.

Output is a **single self-contained HTML** per method: Plotly from CDN, payload
inlined, no sibling JSON, so it opens over `file://` with no web server.

Command: `plot-mtl-map`. Needs `build-answer-space`.
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

from scripts.analysis.v5.modules import cells as CL
from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import edges as E
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.answer_space import (
    META_JSON,
    AnswerSpace,
    load_answer_space,
    require_mesh_universe,
    site_n_scored,
)
from scripts.analysis.v5.modules.figure_outcome_bars import MODE_INK, TIER_INK
from scripts.analysis.v5.modules.geodesy import elementwise_km
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths
from scripts.analysis.v5.modules.status import SHORTEST_PING, solved_mask
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

#: The five statuses, in ladder order. Exactly `classify.summarize`'s
#: `OUTCOME_COUNTS` partition -- see `status_of`.
STATUSES = ("ring0", "ring1", "ring2", "beyond", "failed")

#: `region_mode` values. A geometric MTL answers with a feasible set that can be
#: drawn as a polygon; a density MTL answers with a probability field over a
#: grid, which is a different kind of object.
REGION_GEOMETRIC = "geometric"
REGION_DENSITY = "density"

#: The nested constraint columns, requested alongside the scoring columns.
#: `score_method` copies the frame and writes only named columns, so these ride
#: through scoring and the payload builder needs no second read.
NESTED_COLUMNS = ("ltd_predictions", "mtl_participants")

#: Region replay is CPU-bound and embarrassingly parallel. Capped below the core
#: count because each worker holds its own shapely arrangement.
DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))

#: nside for a density combo's grid, when its `mtl_kwargs` predate the key.
_DEFAULT_DENSITY_NSIDE = 128

# ---- the cell frame ---------------------------------------------------------

#: The extent the Voronoi cells are built against, and the one thing on this
#: map that is a **rendering** bound rather than a measured quantity.
#:
#: `cells.cell_polygons` needs a finite frame and raises when a corner leaves
#: EPSG:5070's usable domain -- it comes back from the inverse transform
#: wrapped past the antimeridian and draws as a line across the map. Measured
#: on as01's seeds, `US_MAINLAND_EXTENT` scaled 1.662x is the last frame that
#: builds and 1.75x raises; per axis, +-25 deg of longitude pad is fine and
#: +-30 deg raises, +-22 deg of latitude is fine and +-25 deg raises.
#:
#: That boundary is **not** a half-plane -- a fitted predicate gave 13
#: false-safes on 400 random extents -- so this is a validated constant rather
#: than a rule, with `fit_extent` as the backstop.
#:
#: Why this box and not `US_MAINLAND_EXTENT`: it contains 100% of every
#: prediction in every run and method on disk, where the CONUS box clips
#: Spotter's 64.57N one; and it is roughly 2x CONUS on both axes, so the frame
#: edge is visibly not a cell edge. Why not `mapping.auto_extent`: on these
#: sites it reproduces `US_MAINLAND_EXTENT` to within 2 deg, putting the frame
#: edge right against the data -- the one reading `cells.py` warns against.
CELL_FRAME: tuple[float, float, float, float] = (-136.0, -56.0, 14.0, 66.0)

#: Douglas-Peucker tolerance applied to the cells, in degrees, **here and not
#: in `cells.py`** -- the static map shares that module and has no size budget.
#:
#: `cells.py` densifies every edge to 10 km, which is what keeps a straight
#: EPSG:5070 edge from bowing away from the boundary it represents once it is
#: back in lon/lat. It also makes the seed-keyed table 274 KiB at 18 seeds and
#: 784 KiB at 250. At 0.02 deg that is an 18.5x reduction (to 9.9 and 42.3 KiB)
#: for a measured max Hausdorff deviation of 2.2 km against `grid_km` = 50.9,
#: and ~2.6 km once d3-geo's great-circle resampling of the longer segments is
#: included -- so the bow the densification exists to prevent stays ~1/20th of
#: a grid. `agreement_with_nearest_seed` moves by <= 0.0002 on all three runs.
#:
#: 0.01 deg costs ~1.6 km for 5x the bytes; 0.05 deg saves a further 12% for
#: 2.5x the deviation. This is the knee.
SIMPLIFY_DEG = 0.02

#: How far `fit_extent` shrinks per attempt, and how many attempts it makes.
_SHRINK = 0.93
_SHRINK_TRIES = 20


def shrink_extent(
    extent: tuple[float, float, float, float], centre: tuple[float, float], factor: float
) -> tuple[float, float, float, float]:
    """`extent` pulled `factor` of the way toward `centre` (lon, lat)."""
    lon_min, lon_max, lat_min, lat_max = extent
    clon, clat = centre
    return (
        clon + (lon_min - clon) * factor,
        clon + (lon_max - clon) * factor,
        clat + (lat_min - clat) * factor,
        clat + (lat_max - clat) * factor,
    )


def fit_extent(
    seeds: pd.DataFrame,
    extent: tuple[float, float, float, float],
    *,
    progress: Any = None,
) -> tuple[dict[int, Any], tuple[float, float, float, float]]:
    """`(cell polygons, the extent they were actually built at)`.

    Shrinks toward the seed centroid and retries when `cell_polygons` refuses
    the frame. `CELL_FRAME` has only about a degree of headroom, so a run whose
    seeds sit further out -- or a user passing `--cell-extent` by hand -- would
    otherwise get a traceback where a smaller honest frame is the right answer.

    Measured convergence: `CELL_FRAME` takes 0 steps, `(-160,-30,0,80)` 6 and
    `(-180,-20,-10,85)` 9.

    The frame it settles on is returned rather than assumed, because the page
    draws that rectangle and has to label it truthfully.
    """
    centre = (float(seeds["seed_lon"].mean()), float(seeds["seed_lat"].mean()))
    current = tuple(float(v) for v in extent)
    for attempt in range(_SHRINK_TRIES):
        try:
            return CL.cell_polygons(seeds, current), current
        except AssertionError as exc:
            # ONLY the out-of-domain frame is retryable. `cell_polygons` also
            # asserts on a degenerate seed set -- a Voronoi polygon holding two
            # generators, or a cell count that does not match -- and shrinking
            # the frame cannot fix either. Retrying those 19 times would pay a
            # full Voronoi build each time and then blame the projection.
            if "frame too large" not in str(exc) or attempt == _SHRINK_TRIES - 1:
                raise
            current = shrink_extent(current, centre, _SHRINK)
            if progress is not None and attempt == 0:
                progress(
                    f"    the requested cell frame left EPSG:5070's domain; "
                    f"shrinking toward the seeds"
                )
    raise AssertionError("unreachable")


def simplify_cells(polygons: dict[int, Any]) -> dict[int, Any]:
    """`cell_polygons`' output decimated at `SIMPLIFY_DEG`.

    Separate from `cell_rings` so the **simplified** geometry is what both the
    page and `agreement_with_nearest_seed` see. Measuring agreement on the
    dense polygons and drawing the decimated ones would publish a number about
    a picture nobody is looking at.

    Every cell is a single `Polygon` with no interior rings, and this asserts
    it rather than walking a `MultiPolygon` it can never see. That is
    structural, not luck: a planar Voronoi cell is convex, `cells._frame` is a
    box, convex-intersect-convex is convex, and the inverse transform is a
    homeomorphism inside the projection's domain -- which `cell_polygons`
    already checks. 6,753 cells over 40 seed sets were all single polygons
    with zero holes. A generic walker here would imply otherwise.
    """
    import shapely

    out: dict[int, Any] = {}
    for seed_id, geom in polygons.items():
        simple = shapely.simplify(geom, SIMPLIFY_DEG, preserve_topology=True)
        if simple.geom_type != "Polygon":
            raise AssertionError(
                f"cell {seed_id} is a {simple.geom_type}, not a Polygon; a Voronoi "
                f"cell cut to a box cannot be, so the frame or the projection is wrong"
            )
        if len(simple.interiors):
            raise AssertionError(f"cell {seed_id} has {len(simple.interiors)} holes")
        out[int(seed_id)] = simple
    return out


def cell_rings(polygons: dict[int, Any]) -> dict[int, list[list[float]]]:
    """`{seed_id: closed clockwise [lat, lon] ring}` from simplified cells."""
    out: dict[int, list[list[float]]] = {}
    for seed_id, geom in polygons.items():
        lon, lat = zip(*list(geom.exterior.coords))
        out[int(seed_id)] = _cw_ring(lon, lat)
    return out


def frame_ring(extent: tuple[float, float, float, float], *, step: int = 25) -> list[list[float]]:
    """The cell frame itself as a closed `[lat, lon]` ring, densified.

    Drawn on the page, and not decoration. The static answer-space map never
    needed it because its extent *was* the frame; this viewer pans and offers
    an orthographic projection, so without the rectangle a reader sees the
    cells stop in mid-Atlantic and reads the stop as a cell boundary. That is
    the exact misreading `cells.py` says must not happen.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    lons = np.linspace(lon_min, lon_max, step)
    lats = np.linspace(lat_min, lat_max, step)
    lon = np.concatenate([lons, np.full(step, lon_max), lons[::-1], np.full(step, lon_min)])
    lat = np.concatenate([np.full(step, lat_min), lats, np.full(step, lat_max), lats[::-1]])
    return _cw_ring(lon, lat)

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
    `value or []` raises "truth value of an array is ambiguous" -- quietly,
    only on rows that happen to be non-empty.
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
    used to produce `min_inflation`, so a per-VP value here and the per-TG
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
    convention, so a counter-clockwise ring fills the antipodal complement --
    the whole globe minus the shape -- and a handful of those stacked turns the
    entire map into one flat wash of colour. Nothing about the outline changes,
    so the bug is invisible until something is actually filled.

    It matters more for a cell than for a grid: a grid is 51 km across, a cell
    spans up to 59 degrees. Both of this module's ring sources -- `grid_rings`
    and a shapely exterior -- hand back parallel lon/lat, so the winding rule
    is stated once, here.
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


def status_of(solved: bool, offset: Any) -> str:
    """The grid verdict: `ring0` | `ring1` | `ring2` | `beyond` | `failed`.

    Bands `classify`'s `pred_dist_to_tg_grid` -- exact grid steps, uncapped --
    into the same five-way partition `summarize` counts as
    `n_ring0 .. n_failed`, so the map and the outcome bars cannot disagree
    about one TG. `TestStatusIsTheSamePartitionAsTheBars` pins that.

    Everything past `classify.MAX_RING` is `beyond`. The **offset itself** is
    carried into the payload alongside, because `beyond` spans 3 to 68 grids
    out on these runs and the popup should say which.

    `solved` must come from `status.solved_mask`, not from an inline
    `status == "SUCCESS"`: the Shortest-Ping frame is all-`BASELINE`, and the
    inline test would score the control as universally failed.
    """
    if not solved:
        return "failed"
    o = _safe_float(offset)
    if o is not None and 0 <= int(o) <= C.MAX_RING:
        return f"ring{int(o)}"
    return "beyond"


def cell_label_of(solved: bool, label: Any) -> str:
    """The cell verdict, masked by `solved` the way `summarize` masks it.

    `score_method` writes `correct`/`wrong` for any row that carries a
    coordinate, and a FALLBACK row carries one -- the Shortest-Ping VP's. But
    `summarize` ANDs every cell count with `answered`, so those rows land in
    `n_cell_unanswered`, and `guard_cross_tab` asserts that count equals the
    grid axis's `n_failed`.

    Taking `score_method`'s column at face value would therefore credit a
    method's cell accuracy with the baseline's answers exactly where it gave
    up: on as01 `vanilla_cbg` that is 270 `correct` against `accuracy.csv`'s
    163, a 107-row overstatement in the method's favour. This is the cell
    axis's `status_of`, and the two must be applied together or the map shows
    one axis masked and the other not.
    """
    if not solved:
        return C.UNANSWERED
    return str(label)

# ---- grid geometry ----------------------------------------------------------


def grid_polygons(grid_ids, nside: int) -> dict[int, list[list[float]]]:
    """`{grid_id: closed clockwise [lat, lon] ring}` for the given grids.

    The payload's one source of grid geometry. Every layer that draws a grid --
    the ring neighbourhood, the TG grids, the prediction's grid -- indexes this
    table, so they cannot draw the same grid two different ways, and a grid
    shared by several TGs is serialized once.

    That sharing is the reason the table exists at all. Inlining a
    neighbourhood per TG is ~25 grids x 33 vertices x 400 TGs, a 12-20 MB page;
    but 399 TGs occupy ~18 grids at nside 128, so the deduplicated table is a
    few hundred entries and the page stays openable over `file://`.
    """
    ids = np.unique(np.asarray(list(grid_ids), dtype=np.int64))
    ids = ids[ids >= 0]
    if ids.size == 0:
        return {}
    return {
        int(c): _cw_ring(ring[:, 0], ring[:, 1])
        for c, ring in zip(ids, G.grid_rings(ids, nside))
    }


def ring_neighbourhood(grid_ids, nside: int) -> dict[int, list[list[int]]]:
    """`{grid_id: [[grid], ring-1 grids, ring-2 grids]}`, memoised per grid.

    Keyed on the truth's grid rather than on the TG, because TGs in the same
    grid share a neighbourhood exactly. On a 399-TG run that is ~18
    `ring_grids` calls instead of 399.

    Capped at `classify.MAX_RING` and no further: this is the drawn
    neighbourhood, and `ring_grids` grows a disk. See the module docstring.
    """
    return {
        int(c): G.ring_grids(int(c), nside, C.MAX_RING)
        for c in np.unique(np.asarray(list(grid_ids), dtype=np.int64))
        if int(c) >= 0
    }


# ---- the run's own MTL ------------------------------------------------------


def load_mtl_specs(run: RunPaths, combo_id: str) -> dict[int, tuple[str, str]]:
    """`{fold: (mtl_name, mtl_kwargs_json)}` from each fold's `run.json`.

    Keyed on `fold` alone because the `(fold, combo)` composite key was
    vestigial: every call site passes one combo.

    Missing or unreadable `run.json` files are skipped rather than raised on.
    The callers all degrade safely -- an unknown MTL means "no region", which
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


# ---- the LTD's cutoff -------------------------------------------------------
#
# Octant and Spotter both trust their fit only up to `cutoff_rtt`, the right
# edge of the last dense RTT bin. Past it the band is no longer the fit: the
# outer bound runs on the line from `outer(cutoff)` to the sentinel on 2/3*c --
# parallel to the speed-of-internet line, below it by a constant -- and the
# inner bound is held at its cutoff value. A ring drawn past the cutoff is
# therefore a different kind of claim from one drawn inside it, and the map has
# to say which is which.
#
# The cutoff is fitted state, not configuration: it lives only in the fold's
# `fit_checkpoint.pkl`, never in `targets.parquet` or `run.json`.


def ltd_cutoffs(model: Any) -> dict[str, float] | float | None:
    """The `cutoff_rtt` a fitted LTD applies: `{vp_id: ms}`, one pooled ms, or None.

    Octant (`bounded_spline`) fits one submodel per VP, so the cutoff is per VP.
    Spotter (`normal_dist`) fits one pooled model, so one cutoff covers every VP.
    A stateless LTD (`speed_of_internet`) has no cutoff at all.

    `cutoff_rtt <= 0` is the models' own "unset" value -- both gate the regime
    on `cutoff_rtt > 0` -- so it is reported as no cutoff rather than as 0 ms,
    which would flag every constraint as past it.
    """
    submodels = getattr(model, "_submodels", None)
    if isinstance(submodels, dict):
        per_vp = {
            str(vp): float(sub.cutoff_rtt)
            for vp, sub in submodels.items()
            if getattr(sub, "fitted", True) and float(getattr(sub, "cutoff_rtt", 0.0) or 0.0) > 0
        }
        return per_vp or None
    pooled = float(getattr(getattr(model, "_model", None), "cutoff_rtt", 0.0) or 0.0)
    return pooled if pooled > 0 else None


def load_cutoffs(run: RunPaths, combo_id: str) -> dict[int, dict[str, float] | float]:
    """`{fold: ltd_cutoffs(model)}` from each fold's checkpoint.

    Folds with no checkpoint, a stateless marker, or no cutoff are left out, so
    an absent fold means "no cutoff to draw" -- the same degrade-to-no-layer
    rule `load_mtl_specs` follows.
    """
    from scripts.benchmark.v2.checkpoint import load_ltd_checkpoint

    out: dict[int, dict[str, float] | float] = {}
    for fold in run.fold_ids:
        combo_dir = run.combo_dir(combo_id, fold)
        if not (combo_dir / "targets.parquet").exists():
            continue
        try:
            model = load_ltd_checkpoint(combo_dir)
        except FileNotFoundError:
            continue
        cut = ltd_cutoffs(model) if model is not None else None
        if cut is not None:
            out[int(fold.split("_")[1])] = cut
    return out


def _cutoff_of(cutoffs: dict[int, dict[str, float] | float], fold: int, vp_id: str) -> float | None:
    cut = cutoffs.get(fold)
    if isinstance(cut, dict):
        return cut.get(vp_id)
    return cut


def _cutoff_summary(
    cutoffs: dict[int, dict[str, float] | float], ltd_by_tg: dict[str, list]
) -> dict[str, Any] | None:
    """Run-level cutoff facts for the page header: scope, pooled values, counts."""
    if not cutoffs:
        return None
    pooled = {str(f): round(c, 2) for f, c in sorted(cutoffs.items()) if not isinstance(c, dict)}
    rows = [r for rs in ltd_by_tg.values() for r in rs]
    return {
        "scope": "pooled" if pooled else "per_vp",
        "pooled_ms_by_fold": pooled,
        "n_constraints": len(rows),
        "n_past": sum(r[5] for r in rows),
        "n_past_kept": sum(r[5] for r in rows if r[3] == 1),
    }


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
    describes what actually ran. An unreadable spec degrades to geometric --
    the status quo, and the safe default because it only costs a replay.
    """
    names = {name for name, _ in load_mtl_specs(run, combo_id).values()}
    return REGION_DENSITY if any(is_density_mtl(n) for n in names) else REGION_GEOMETRIC


def density_nside(run: RunPaths, combo_id: str) -> int | None:
    """The nside a density combo reported at, from its stored `mtl_kwargs`.

    Read off `run.json` rather than by constructing the MTL: the grid is the
    only thing needed, and constructing it would build a global grid and (for a
    combo predating the required `grid` kwarg) raise.

    Returns None when the combo's folds disagree -- which cannot happen from
    one `run-combo`, but would make the drawn grid a lie if it did -- or when
    the MTL reported on a grid that is not HEALPix.
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


def argmax_grid_regions(
    run: RunPaths, combo_id: str, scored: pd.DataFrame
) -> tuple[dict[str, dict], int | None]:
    """`({tg_id: the grid the prediction fell in}, nside)` for a density combo.

    A density MTL's answer is a probability field, which `replay_mtl` cannot
    rebuild: `mtl_participants` stores the echoed band and not `mu_km`/
    `sigma_km`, and the band is explicitly not invertible back to the
    distribution. So this map would otherwise draw nothing at all for Spotter.

    None of that needs solving, because `density_argmax` returns a **grid
    centre**. Re-binning the persisted prediction therefore recovers the exact
    grid it came from -- `ang2pix(pix2ang(p)) == p` is a pinned property of the
    grid -- and the grid boundary is closed-form. So the region comes back for
    free: no MTL construction, no global grid, no replay, and nothing read that
    the benchmark did not already write.

    What is drawn is narrower than a geometric method's region, and the viewer
    says so: it is the argmax grid, i.e. the quantisation of the point
    estimate, not a feasible set and not a credible region.

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
    pix = G.ang2pix(lat[ok].to_numpy(), lon[ok].to_numpy(), nside)
    rings = G.grid_rings(pix, nside)
    out: dict[str, dict] = {}
    for tid, ring in zip(scored.loc[ok, "tg_id"].astype(str), rings):
        out[str(tid)] = {
            "kind": "healpix_grid",
            "rings": [{"outer": _cw_ring(ring[:, 0], ring[:, 1]), "holes": []}],
        }
    return out, nside

# ---- region replay ----------------------------------------------------------


def _region_task(task: tuple[str, str, list[dict]]) -> dict | None:
    """Replay one TG's MTL and serialize the region. Runs in a worker process.

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

    The cache is keyed on (method, tg), which is not enough on its own:
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


def _region_cache_path(cache_dir: Path | None, combo_id: str, tg_id: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / combo_id / f"{tg_id}.json"


def _read_region_cache(cache_dir: Path | None, combo_id: str, tg_id: str) -> dict | None:
    """Cached region, `{}` for a memoized "no region", or None when uncached."""
    path = _region_cache_path(cache_dir, combo_id, tg_id)
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _write_region_cache(
    cache_dir: Path | None, combo_id: str, tg_id: str, region: dict | None
) -> None:
    """Persist a region, or `{}` so an empty result is not recomputed forever."""
    path = _region_cache_path(cache_dir, combo_id, tg_id)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(region if region is not None else {}, separators=(",", ":")))


def replay_mtl(
    run: RunPaths,
    combo_id: str,
    frame: pd.DataFrame,
    *,
    cache_dir: Path | None = None,
    workers: int = 1,
    progress: Any = None,
) -> dict[str, dict]:
    """`{tg_id: serialized feasible region}` recomputed from the run's own MTL.

    The benchmark stores `mtl_intersection_kind` but never the geometry, so the
    region has to be rebuilt. `mtl_participants[]` is the post-filter constraint
    set with `vp_lat`/`vp_lon`/`rtt_ms` inline -- everything the MTL reads -- so
    no join against the VP roster or the RTT table is needed.

    **This is the expensive part of the map, by three orders of magnitude.** The
    planar face decompositions are the same computation the benchmark paid for:
    `mtl_ms` on as01/octant_cbg_hull averages 7.0 s per TG (median 0.9 s, max
    78 s), so a serial replay of 399 TGs is ~47 minutes. Hence both the process
    pool and the on-disk cache -- the cache also makes an interrupted render
    resume, and makes re-rendering after a template edit instant.

    The cache is rung-free (`RunPaths.mtl_region_cache_dir`): no nside enters
    this function, so a second rung must not re-pay the replay.
    """
    specs = load_mtl_specs(run, combo_id)

    # A density MTL cannot be replayed from what the benchmark persisted, and
    # this is a schema limit rather than something to work around here.
    # `mtl_participants` stores only `echoed_upper_km` / `echoed_lower_km`, so
    # the `Distance` rebuilt above carries no `mu_km` / `sigma_km`;
    # `GaussianDensityMTL` requires `has_distribution` and the bounds are
    # explicitly not invertible back to the distribution.
    #
    # Returning before the cache is touched is the point: an empty result here
    # means "not applicable", which must stay distinguishable from an MTL that
    # genuinely found nothing, and must not be frozen into the cache.
    #
    # The region is not lost, it just comes from somewhere cheaper: see
    # `argmax_grid_regions`. `build_for_run` routes density combos there.
    density = sorted({n for n, _ in specs.values() if n and is_density_mtl(n)})
    if density:
        if progress is not None:
            progress(
                f"    {combo_id}: {'/'.join(density)} answers with a probability "
                f"field, not a feasible set -- no region to replay"
            )
        return {}

    _invalidate_stale_cache(cache_dir, combo_id, specs, progress=progress)

    regions: dict[str, dict] = {}
    tasks: list[tuple[str, tuple[str, str, list[dict]]]] = []
    for row in frame.itertuples(index=False):
        tid = str(row.tg_id)
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
            f"    {combo_id}: replaying MTL for {len(tasks)} TG(s) on "
            f"{workers} worker(s), {len(regions)} already cached..."
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

# ---- run inputs -------------------------------------------------------------


def load_vps(run: RunPaths) -> pd.DataFrame:
    """`vps.csv` -- the run's vantage points, shared across every fold.

    Lives here rather than in a module of its own: v5 dropped v4's `bipartite`,
    and this viewer is the only thing left that needs the VP roster.
    """
    path = run.setup_dir / "vps.csv"
    if not path.exists():
        raise MissingArtifactError(f"{path} missing; cannot draw the VP side")
    vps = pd.read_csv(path)
    missing = {"vp_id", "vp_lat", "vp_lon"} - set(vps.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    return vps


def load_rung(run: RunPaths, nside: int, *, analysis_root: Path | None = None) -> AnswerSpace:
    """The answer space at one rung, read from disk. Never rebuilt.

    Same refusal as `map_answer_space.load_rung` and `classify.score_rung`. A
    v5 seed is a complete-linkage cluster centroid, not a closed-form grid
    centre, so rebuilding it here could quietly draw cells that are not the
    cells `classify` scored against.
    """
    d = run.answer_space_dir(nside, root=analysis_root)
    if not (d / META_JSON).exists():
        raise MissingArtifactError(
            f"{d / META_JSON} missing; run `build-answer-space --run-id {run.run_id}` first"
        )
    space = load_answer_space(d)
    require_mesh_universe(run, space)
    return space


#: Two fields of context, both plain columns of `eval_per_target.csv`.
_CONTEXT_COLUMNS = {"shortest_ping_vp_id": "sping_vp_id", "min_inflation": "min_inflation"}


def eval_context(run: RunPaths) -> pd.DataFrame:
    """`tg_id, sping_vp_id, min_inflation`, or an empty frame.

    Absent `eval_source/` is not an error: the map's own layers are all built
    from the benchmark tree and the answer space, and these two are context. A
    run without them renders with both fields null rather than failing.
    """
    empty = pd.DataFrame(columns=["tg_id", *_CONTEXT_COLUMNS.values()])
    try:
        raw = pd.read_csv(run.eval_file("eval_per_target.csv"))
    except (MissingArtifactError, FileNotFoundError):
        return empty
    have = [c for c in _CONTEXT_COLUMNS if c in raw.columns]
    if "target_id" not in raw.columns or not have:
        return empty
    out = raw[["target_id", *have]].drop_duplicates("target_id")
    out = out.rename(columns={"target_id": "tg_id", **_CONTEXT_COLUMNS})
    # Both columns, always. `build_payload` indexes each by name on any frame
    # this returns, so a CSV carrying one of the two would raise a KeyError
    # rather than degrade to the null field this is supposed to give it.
    for col in _CONTEXT_COLUMNS.values():
        if col not in out.columns:
            out[col] = None
    return out[["tg_id", *_CONTEXT_COLUMNS.values()]]


def _load_edges(run: RunPaths, *, progress: Any = None) -> pd.DataFrame:
    """The per-VP observation table, or an empty one when the CSV is absent.

    Absence is degradable and the refusal is not, which is why
    `MeshSupersetError` is re-raised rather than swallowed. The canonical CSV
    lives under `datasets/`, outside the benchmark output tree, so a run whose
    dataset has been moved or archived would otherwise lose a whole layer.
    What is lost without it is the per-VP RTTs and their inflation; the
    verdicts, the rings, the constraints and the region all come from the run's
    own parquets and the answer space, and are unaffected.

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
    cells: dict[int, list[list[float]]],
    cell_frame: tuple[float, float, float, float],
    cell_agreement: float,
    regions: dict[str, dict] | None = None,
    region_mode: str = REGION_GEOMETRIC,
    density_nside: int | None = None,
    cutoffs: dict[int, dict[str, float] | float] | None = None,
) -> dict[str, Any]:
    """Assemble the whole viewer payload for one method.

    Everything the page ever shows is inlined; there is no lazy fetch, which is
    what lets the file open over `file://`. Per-VP rows are positional arrays
    rather than objects, coordinates are rounded to 4 dp (~11 m), and grid
    geometry is shared through `grids` rather than repeated per TG -- together
    those three choices are most of the difference between a page under a
    megabyte and one over fifteen.

    `cutoffs` is `load_cutoffs`' output. Each constraint row then carries its
    VP's cutoff and whether the RTT it was predicted at is past it.
    """
    nside = space.nside
    seeds = space.seeds.reset_index(drop=True)
    solved = solved_mask(scored).to_numpy()
    placed = space.tgs.set_index("tg_id")
    ctx = context.set_index("tg_id") if len(context) else None

    if region_mode not in (REGION_GEOMETRIC, REGION_DENSITY):
        raise ValueError(f"unknown region_mode: {region_mode!r}")
    regions = regions or {}

    # The Shortest-Ping control has no `targets.parquet`, so it contributes
    # neither constraints nor a region.
    #
    # The viewer needs to know that as a *fact about the method* rather than
    # infer it from the data: an ordinary CBG run whose every TG happened to
    # produce no region would otherwise be rendered as a baseline.
    is_baseline = method == SHORTEST_PING
    has_nested = NESTED_COLUMNS[1] in scored.columns

    # Which constraints actually formed the region. Every MTL runs
    # `filter_redundant_outer_disks` when `enable_circle_filter` is on -- a disk
    # that fully contains another is not the binding constraint, so it is
    # dropped -- and records the survivors as `MTLResult.participating_vp_ids`,
    # which the benchmark persists as `mtl_participants[]`. Reading it back is
    # exact, rather than re-implementing the heuristic client-side where it
    # could drift from the geometry it describes.
    #
    # The filter is not cosmetic at this scale: on as01 `million_scale_cbg`
    # keeps 4.5 of 133 disks, so drawing the unfiltered set buries the four that
    # decide the answer.
    #
    # Row layout: `[vp_id, upper_km, lower_km, kept, cutoff_ms, past_cutoff]`.
    # `ltd_predictions[]` persists no RTT, so the RTT the constraint was
    # predicted at is the edge table's min RTT for the pair -- the same value
    # the benchmark fed the LTD (`mtl_participants[].rtt_ms` matches it exactly
    # on every kept pair of as01's folds). `past_cutoff` is strict, matching
    # the models' own `rtt > cutoff_rtt` gate.
    cutoffs = cutoffs or {}
    rtt_of: dict[tuple[str, str], float] = {}
    if cutoffs:
        rtt_of = {
            (str(t), str(v)): float(r)
            for t, v, r in zip(edges["tg_id"], edges["vp_id"], edges["rtt_ms"])
        }
    kept_by_tg: dict[str, set] = {}
    ltd_by_tg: dict[str, list] = {}
    if has_nested:
        for row in scored.itertuples(index=False):
            tid = str(row.tg_id)
            fold = int(row.fold)
            kept = {str(p["vp_id"]) for p in _nested(row.mtl_participants)}
            kept_by_tg[tid] = kept
            rows = []
            for p in _nested(row.ltd_predictions):
                if not (p.get("success") and p.get("upper_km")):
                    continue
                vid = str(p["vp_id"])
                cut = _cutoff_of(cutoffs, fold, vid)
                rtt = rtt_of.get((tid, vid))
                rows.append(
                    [
                        vid,
                        round(float(p["upper_km"]), 2),
                        round(float(p.get("lower_km") or 0.0), 2),
                        1 if vid in kept else 0,
                        None if cut is None else round(cut, 2),
                        1 if cut is not None and rtt is not None and rtt > cut else 0,
                    ]
                )
            ltd_by_tg[tid] = rows

    # Observations, deduped to the minimum RTT per (TG, VP) upstream in
    # `edges.load_min_rtt`, so the inflation shown per VP and the eval source's
    # `min_inflation` cannot disagree about which edge they describe.
    obs = edges.merge(vps[["vp_id", "vp_lat", "vp_lon"]], on="vp_id", how="inner").join(
        placed[["tg_lat", "tg_lon"]], on="tg_id", how="inner"
    )
    obs_km = elementwise_km(
        obs["tg_lat"].to_numpy(dtype=float),
        obs["tg_lon"].to_numpy(dtype=float),
        obs["vp_lat"].to_numpy(dtype=float),
        obs["vp_lon"].to_numpy(dtype=float),
    )
    obs_by_tg: dict[str, list] = {}
    for (tid_, vp_, rtt_), km_ in zip(zip(obs["tg_id"], obs["vp_id"], obs["rtt_ms"]), obs_km):
        infl = _inflation(float(rtt_), float(km_))
        obs_by_tg.setdefault(str(tid_), []).append(
            [str(vp_), round(float(rtt_), 3), None if infl is None else round(infl, 3)]
        )

    n_total_vps = int(len(vps))
    mtl_kind = "disk"
    for rows in ltd_by_tg.values():
        if any(r[2] > 0 for r in rows):
            mtl_kind = "annulus"
            break

    nbhd = ring_neighbourhood(scored["tg_grid_id"], nside)

    tgs: list[dict[str, Any]] = []
    for i, row in enumerate(scored.itertuples(index=False)):
        tid = str(row.tg_id)
        pred_lat, pred_lon = _safe_float(row.pred_lat), _safe_float(row.pred_lon)
        pred = (
            [round(pred_lat, 4), round(pred_lon, 4)]
            if None not in (pred_lat, pred_lon)
            else None
        )
        tg_grid = int(row.tg_grid_id)
        pred_grid = int(row.pred_grid_id)
        offset = int(getattr(row, C.GRID_OFFSET))
        p_row = placed.loc[tid]
        t_obs = obs_by_tg.get(tid, [])
        c_row = ctx.loc[tid] if ctx is not None and tid in ctx.index else None

        tgs.append(
            {
                "tg_id": tid,
                "fold": int(row.fold),
                "true": [
                    round(float(p_row["tg_lat"]), 4),
                    round(float(p_row["tg_lon"]), 4),
                ],
                "site_id": int(row.site_id),
                # -- the grid axis --
                "tg_grid": tg_grid,
                "pred_grid": pred_grid,
                "grid_offset": offset,
                "ring_grids": nbhd.get(tg_grid, [[tg_grid], [], []]),
                "status": status_of(bool(solved[i]), offset),
                "tg_dist_to_grid_centre_km": _safe_float(p_row["tg_dist_to_grid_centre_km"]),
                # -- the cell axis --
                "tg_seed_id": int(row.tg_seed_id),
                "pred_seed_id": int(row.pred_seed_id),
                "cell_label": cell_label_of(bool(solved[i]), row.cell_label),
                # What `score_method` wrote before the mask, kept so the page
                # can say "the fallback coordinate did land in the right cell"
                # without that ever entering a count.
                "cell_label_unmasked": str(row.cell_label),
                "pred_dist_to_seed_km": _safe_float(row.pred_dist_to_seed_km),
                "tg_dist_to_seed_km": _safe_float(p_row["tg_dist_to_seed_km"]),
                # -- the rest --
                "pred": pred,
                "pred_dist_to_tg_km": _safe_float(row.pred_dist_to_tg_km),
                "n_measured": len(t_obs),
                "n_total": n_total_vps,
                "sping_vp_id": None if c_row is None else _safe_str(c_row["sping_vp_id"]),
                "min_inflation": None if c_row is None else _safe_float(c_row["min_inflation"]),
                "obs": t_obs,
                "rings": ltd_by_tg.get(tid, []),
                "n_kept": len(kept_by_tg.get(tid, ())),
                "region": regions.get(tid),
            }
        )

    # Error ascending over the answered TGs -- the statuses interleave, since
    # the ordering is the distance and not the verdict -- then every `failed`
    # TG after them. So the list reads best to worst and a position in it means
    # the same thing as a percentile of error does.
    #
    # A fallback does carry a coordinate and a distance, but they are the
    # Shortest-Ping VP's rather than the method's, so that number is not on the
    # same scale as the rest of the column and cannot be ranked against it.
    # Sorting the failed block by it is still the useful order *within* the
    # block; `None` (no prediction at all) sorts to the very end.
    def _order(t: dict[str, Any]) -> tuple[int, int, float]:
        err = t["pred_dist_to_tg_km"]
        return (1 if t["status"] == "failed" else 0, 1 if err is None else 0, err or 0.0)

    tgs.sort(key=_order)

    # One table for every grid any layer draws: the neighbourhoods, the
    # occupied TG grids, and each prediction's own grid.
    needed: set[int] = {int(g) for g in space.grids["grid_id"]}
    for t in tgs:
        for ring in t["ring_grids"]:
            needed.update(int(c) for c in ring)
        if t["pred_grid"] >= 0:
            needed.add(t["pred_grid"])
    grids = grid_polygons(needed, nside)

    site_scored = site_n_scored(space).to_numpy()
    seed_scored = (
        pd.Series(site_scored, index=space.sites["seed_id"].to_numpy()).groupby(level=0).sum()
    )

    return {
        "run_id": run.run_id,
        "method": method,
        "source": run.source,
        "setup": run.setup,
        "nside": nside,
        "grid_km": round(space.grid_km, 1),
        "max_ring": C.MAX_RING,
        # The MTL's OWN grid, which is not necessarily the answer space's. A
        # density combo's drawn grid comes from its `mtl_kwargs`. They are 128
        # on both sides today and nothing requires that, so they stay separate.
        "density_nside": density_nside,
        # True when the two coincide, so the viewer draws ONE polygon with a
        # combined label instead of stacking two identical fills.
        "region_is_pred_grid": bool(region_mode == REGION_DENSITY and density_nside == nside),
        "earth_radius_km": EARTH_RADIUS_KM,
        "mtl_kind": mtl_kind,
        "region_mode": region_mode,
        "is_baseline": is_baseline,
        # `per_vp` (Octant), `pooled` (Spotter), or None when the LTD has no
        # cutoff -- in which case every row's `cutoff_ms` is None too.
        "ltd_cutoff": _cutoff_summary(cutoffs, ltd_by_tg),
        "n_seeds": int(len(seeds)),
        "n_sites": int(len(space.sites)),
        # The space's sites above, the run's scored ones here. They differ only
        # on a traffic-weighted run, whose space is its pre-filter mesh.
        "n_sites_scored": int((site_scored > 0).sum()),
        "targets_source": space.meta.get("targets_provenance", {}).get(
            "targets_source", "the run's own evaluated TGs"
        ),
        # The ring tiers come from the outcome bars rather than being copied,
        # so the map and the bars cannot drift into two greens for one outcome.
        # `failed` is the bars' `unanswered`: one row counted on two axes.
        "palette": {
            "ring0": TIER_INK["ring0"],
            "ring1": TIER_INK["ring1"],
            "ring2": TIER_INK["ring2"],
            "beyond": TIER_INK["beyond"],
            "failed": TIER_INK[C.UNANSWERED],
        },
        "cell_palette": {
            "correct": MODE_INK["correct"],
            "wrong": MODE_INK["wrong"],
            C.UNANSWERED: MODE_INK[C.UNANSWERED],
        },
        "grids": {str(c): ring for c, ring in grids.items()},
        # The OCCUPIED grids of the answer space, as a plain list. v4's viewer
        # enumerated these off `seeds[].cell_id`, which only worked while a
        # seed was a grid centre; a v5 seed is a cluster centroid and has no
        # grid, so the list is carried explicitly.
        "tg_grids": [int(g) for g in space.grids["grid_id"]],
        "cells": {str(s): ring for s, ring in cells.items()},
        "cell_frame": dict(zip(("lon_min", "lon_max", "lat_min", "lat_max"), cell_frame)),
        "cell_frame_ring": frame_ring(cell_frame),
        "cell_meta": {
            "projected_crs": "EPSG:5070",
            "segment_km": CL.SEGMENT_M / 1000.0,
            "simplify_deg": SIMPLIFY_DEG,
            "agreement": round(float(cell_agreement), 4),
            "note": (
                "planar Voronoi in EPSG:5070, unclipped, densified then simplified "
                "for the page; the frame is a RENDERING bound and every cell would "
                "continue past it. `agreement` is the share of points sampled in the "
                "frame whose drawn cell is their great-circle nearest seed -- the "
                "rule classify actually scores."
            ),
        },
        "vps": {
            str(r.vp_id): [
                round(float(r.vp_lat), 4),
                round(float(r.vp_lon), 4),
                None if pd.isna(getattr(r, "vp_asn", None)) else str(r.vp_asn),
                None if pd.isna(getattr(r, "vp_country", None)) else str(r.vp_country),
            ]
            for r in vps.itertuples(index=False)
        },
        # No `grid_id` on a seed: a v5 seed is the spherical centroid of the
        # sites complete linkage grouped, NOT a grid centre. v4's `cell_id`
        # here is exactly the assumption that no longer holds.
        "seeds": [
            {
                "id": int(s),
                "lat": round(float(la), 4),
                "lon": round(float(lo), 4),
                "n_sites": int(ns),
                "n_tgs": int(nt),
                "n_tgs_scored": int(seed_scored.get(int(s), 0)),
            }
            for s, la, lo, ns, nt in zip(
                seeds["seed_id"],
                seeds["seed_lat"],
                seeds["seed_lon"],
                seeds["n_sites"],
                seeds["n_tgs"],
            )
        ],
        # `[lat, lon, seed_id, n_tgs_scored]`. The last is 0 at a mesh site a
        # traffic-weighted run scores no TG at; the viewer draws those hollow.
        "sites": [
            [round(float(la), 4), round(float(lo), 4), int(sd), int(ns)]
            for la, lo, sd, ns in zip(
                space.sites["site_lat"], space.sites["site_lon"], space.sites["seed_id"],
                site_scored,
            )
        ],
        "tgs": tgs,
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
        f"{kind} -- {payload['run_id']} - {payload['method']} - "
        f"healpix nside {payload['nside']}"
    )
    html = _HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    js = _JS_TEMPLATE_PATH.read_text(encoding="utf-8")
    html = html.replace("__SCRIPT__", js).replace("__TITLE__", title)
    blob = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    return html.replace("__PAYLOAD__", blob)


def build_for_run(
    run: RunPaths,
    *,
    methods: list[str],
    nside: int = G.DEFAULT_NSIDE,
    extent: tuple[float, float, float, float] = CELL_FRAME,
    analysis_root: Path | None = None,
    regions: bool = True,
    workers: int = 1,
    progress: Any = None,
) -> list[tuple[str, Path, dict[str, Any]]]:
    """Render one HTML per method. Returns `[(method, path, payload)]`.

    The cells are built **once, before the method loop**, and every page of the
    run gets the identical frame. They are a property of the answer space, not
    of the method; deriving them per method would let two pages of one run
    disagree about where a boundary lies.
    """
    nside = G.validate_nside(nside)
    space = load_rung(run, nside, analysis_root=analysis_root)
    vps = load_vps(run)
    edge_table = _load_edges(run, progress=progress)
    context = eval_context(run)

    polygons, frame = fit_extent(space.seeds, extent, progress=progress)
    # Simplify FIRST, then measure: the number the page prints has to describe
    # the polygons the page draws.
    polygons = simplify_cells(polygons)
    cells = cell_rings(polygons)
    agreement = CL.agreement_with_nearest_seed(space.seeds, polygons, frame)
    if progress is not None:
        progress(
            f"    {len(cells)} cells at {tuple(round(v, 1) for v in frame)}, "
            f"{agreement:.4f} agreement with the nearest-seed rule"
        )

    out_dir = run.mtl_map_dir(nside, root=analysis_root)
    cache_dir = run.mtl_region_cache_dir(root=analysis_root)

    rendered: list[tuple[str, Path, dict[str, Any]]] = []
    for method in methods:
        is_baseline = method == SHORTEST_PING
        if is_baseline:
            # The control is scored over the EVALUATED roster rather than over
            # the eval source, so it needs some combo's targets.parquet to say
            # which TGs the run actually evaluated. Any combo will do -- K-fold
            # splits TGs, not methods.
            combos = run.combo_ids
            if not combos:
                raise MissingArtifactError(
                    f"{run.setup_dir} holds no combo with a targets.parquet; "
                    f"the Shortest-Ping control has no roster to be scored over"
                )
            frame_df = C.load_shortest_ping_frame(run, C.load_method_frame(run, combos[0]))
        else:
            frame_df = C.load_method_frame(run, method, columns=NESTED_COLUMNS)
        scored = C.score_method(frame_df, space)

        region_mode = REGION_GEOMETRIC if is_baseline else combo_region_mode(run, method)
        method_regions: dict[str, dict] = {}
        method_density_nside: int | None = None
        if regions and not is_baseline:
            if region_mode == REGION_DENSITY:
                # Derived from the stored prediction, not replayed -- see
                # `argmax_grid_regions`. No cache, because it is closed-form.
                method_regions, method_density_nside = argmax_grid_regions(run, method, scored)
                if progress is not None:
                    progress(
                        f"    {method}: {len(method_regions)} argmax grids from "
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

        # Loaded even under `--no-regions`: the cutoff marks the constraint
        # rings, which are drawn either way.
        cutoffs = {} if is_baseline else load_cutoffs(run, method)

        payload = build_payload(
            run,
            space,
            method,
            scored=scored,
            edges=edge_table,
            vps=vps,
            context=context,
            cells=cells,
            cell_frame=frame,
            cell_agreement=agreement,
            regions=method_regions,
            region_mode=region_mode,
            density_nside=method_density_nside,
            cutoffs=cutoffs,
        )
        if progress is not None and payload["ltd_cutoff"]:
            lc = payload["ltd_cutoff"]
            progress(
                f"    {method}: {lc['n_past']}/{lc['n_constraints']} constraints past the "
                f"{lc['scope']} LTD cutoff ({lc['n_past_kept']} of them kept by the MTL)"
            )
        out = out_dir / MAP_HTML.format(method=method)
        out.write_text(render_html(payload), encoding="utf-8")
        rendered.append((method, out, payload))
    return rendered
