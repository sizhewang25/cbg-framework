"""Score a completed benchmark run's CBG *outputs* per (combo, target).

`eval_source.py` is the precheck: dataset topology/RTT quality before any CBG
runs. This is the postcheck twin — given `targets.parquet` from a finished
run, it explains *why* each target succeeded or failed by reconstructing the
per-VP `LTDResult`s that fed the pipeline and re-deriving metrics the runner
itself never persists (region area, truth-inclusion, per-VP brittleness).

Per-target metrics, grouped by axis:

  participants   n_part, part_min/mean/med_dist_km, part_min/mean/med_rtt_ms,
                 part_min/mean_infl — geometry + RTT quality of the VPs that
                 actually decided the region (`mtl_participants`, post the
                 MTL's own redundant-disk filter).
  region         mtl_area_km2, mtl_n_components — recomputed from a fresh
                 `mtl.multilaterate()` call over the reconstructed LTDResults
                 (`targets.parquet` keeps `mtl_intersection_kind` but not the
                 geometry/area). Area is a planar approximation consistent
                 with how the region was actually built (same local
                 km-per-degree scale as `_circle_to_planar_polygon`) — fine
                 near the equator / short distances, degrades poleward or at
                 large radii, same caveat the codebase already documents for
                 that helper.
  inclusion      truth_in_region — does the recomputed region actually
                 contain the ground-truth point (independent of whether the
                 combo's CTR picked a nearby point); exclusion_reason —
                 when it doesn't, whether that's because some participant's
                 *outer* bound was too tight (truth is farther than the
                 disk reaches) or its *inner* bound wrongly excludes it (an
                 annulus hole swallowed a genuinely close point) — the two
                 mechanically different failure directions behind the
                 "erroneous containment" bucket in
                 scripts/analysis/partvp/characterize_failures.py.
  includer/      n_includers/n_excluders + per-group mean dist/RTT — splits
  excluder       the participant set by whether *that VP's own* band
                 contains the true distance, extending the aggregate
                 "blocker fraction" idea from characterize_failures.py into
                 per-group stats.
  bridge         closest_vp_id/dist_km, sping_vp_id/dist_km,
                 closest_is_sping, sping_vp_is_participant — the same
                 closest-VP / shortest-ping-VP identities `per_target_table.py`
                 computes, plus whether the shortest-ping VP actually made it
                 into *this combo's* deciding set. Computed over the fold's
                 full `eval_observations.parquet` (the materialized-available
                 set), not narrowed by any combo-level `pair_weight_min` —
                 this describes input availability, not this combo's own
                 runtime filter.
  answer-space   cell_gap_km — distance from the target's truth-cluster
                 centroid to the nearest *other* centroid, joined from the
                 run's existing `cluster-eval` output (no reclustering).
  brittleness    loo_computed/loo_n_tested/loo_trivial/loo_n_flips/
                 loo_any_flip/loo_max_error_delta_km/loo_critical_vp_id —
                 leave-one-participant-out sensitivity. For each
                 participating VP, drop it and re-run MTL+CTR (the *actual*
                 registered classes for this combo, from run.json's mtl/ctr
                 + kwargs — LTD is never refit, since per-VP predictions are
                 cached and independent of any other target's participant
                 set). A "flip" is the reduced set failing MTL or CTR
                 outright; `loo_max_error_delta_km` is the worst error
                 increase among non-flip removals. Formalizes the "brittle
                 to a single bad disk" finding into a per-target score. Only
                 computed for SUCCESS targets with >=1 participant;
                 `loo_trivial=True` flags targets with exactly one
                 participant, where any removal trivially empties the
                 region. This is the expensive metric group — cost scales
                 with sum(n_part) over eligible targets, and each rerun's
                 own cost depends on the combo's CTR (Monte Carlo medoid's
                 rejection sampling is far pricier than a Shapely centroid)
                 — so it's optional: pass `compute_loo=False` /
                 `--skip-loo` to skip the per-participant reruns entirely.
                 `loo_computed` records whether it actually ran for that
                 row (False everywhere when skipped, vs. `loo_trivial`
                 which is always computed cheaply from the participant
                 count alone).

`recompute_matches` is a validation column: the fresh `mtl.multilaterate()`
call's success flag + participating set should match what the original run
persisted. A False here means the LTDResult reconstruction (join of
`ltd_predictions` against `eval_observations.parquet`) has drifted from what
the run actually saw — treat any nonzero share as a bug, not noise.

Artifacts (under `<run_dir>/bench_eval/`, one call per run):
  <combo_id>_bench_per_target.csv   per-(combo,target) metrics above
  summary.parquet                   one row per combo: percentile blocks +
                                     inclusion/brittleness/validation shares
"""

from __future__ import annotations

import logging
from math import atan2
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from shapely.geometry import Point
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.multipolygon import MultiPolygon
from sklearn.neighbors import BallTree

from scripts.analysis._v2_io import (
    discover_combos,
    group_combos_by_id,
    load_run_json,
    load_targets,
)
from scripts.framework.v2.ctr.base import CTRMethod
from scripts.framework.v2.ctr.geometric_centroid import _dedupe_vertices, _local_project
from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.mtl.base import MTLMethod
from scripts.framework.v2.registry import CTR_REGISTRY, MTL_REGISTRY
from scripts.framework.v2.types import Coord, Distance, Error
from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM, THEORETICAL_SLOPE, haversine_distance

logger = logging.getLogger(__name__)

_PCTS = (5, 25, 50, 75, 95)
_KM_PER_DEG_LAT = 111.0
_DEDUPE_TOLERANCE_DEG = 1e-9

PER_TARGET_STAT_METRICS = (
    "n_part",
    "part_min_dist_km",
    "part_mean_infl",
    "mtl_area_km2",
    "cell_gap_km",
    "loo_max_error_delta_km",
)


# ---- geometry: unify Shapely regions and spherical vertex lists --------------

def _km_per_deg_lon(lat: float) -> float:
    return max(_KM_PER_DEG_LAT * np.cos(np.radians(lat)), 1.0)


def _deg2_to_km2(area_deg2: float, mean_lat: float) -> float:
    return float(area_deg2 * _KM_PER_DEG_LAT * _km_per_deg_lon(mean_lat))


def _vertices_to_polygon(vertices: list[Coord]) -> Optional[ShapelyPolygon]:
    """Order an unordered spherical-crossing vertex list into a degree-space
    polygon, reusing `GeometricCentroidCTR`'s own dedupe + local-projection
    ordering (`_dedupe_vertices` / `_local_project`) so the polygon this
    module reasons over (area, truth-inclusion) is the same region the
    combo's own CTR would have reasoned over — not a second, possibly
    divergent, reconstruction. Ordered indices are then applied to the
    original (lat, lon) pairs and built as a `(lon, lat)` Shapely polygon,
    matching `_circle_to_planar_polygon`'s convention so `_area_km2` applies
    uniformly across MTL families."""
    verts = [(c.lat, c.lon) for c in vertices]
    unique = _dedupe_vertices(verts, _DEDUPE_TOLERANCE_DEG)
    if len(unique) < 3:
        return None
    local_points, _, _, _ = _local_project(unique)
    order = sorted(range(len(local_points)), key=lambda i: atan2(local_points[i][1], local_points[i][0]))
    ordered_lonlat = [(unique[i][1], unique[i][0]) for i in order]
    poly = ShapelyPolygon(ordered_lonlat)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly is None or poly.is_empty:
        return None
    return poly


def _intersection_to_geometry(intersection: Any) -> Optional[BaseGeometry]:
    """Normalize an `MTLResult.intersection` (Polygon/MultiPolygon/list[Coord]/
    None) to a single Shapely geometry, or None when there's nothing to
    measure (empty/degenerate/absent region)."""
    if intersection is None:
        return None
    if isinstance(intersection, BaseGeometry):
        return None if intersection.is_empty else intersection
    if isinstance(intersection, list):
        return _vertices_to_polygon(intersection)
    return None


def _area_km2(geom: Optional[BaseGeometry]) -> Optional[float]:
    if geom is None or geom.is_empty:
        return None
    mean_lat = geom.centroid.y
    return round(_deg2_to_km2(geom.area, mean_lat), 3)


def _n_components(geom: Optional[BaseGeometry]) -> int:
    if geom is None:
        return 0
    if isinstance(geom, MultiPolygon):
        return len(geom.geoms)
    return 1


def _point_in_region(geom: Optional[BaseGeometry], lat: float, lon: float) -> Optional[bool]:
    if geom is None or geom.is_empty:
        return None
    return bool(geom.covers(Point(lon, lat)))


# ---- LTDResult reconstruction -------------------------------------------------

def _reconstruct_ltd_results(
    ltd_predictions: list[dict],
    obs_by_vp: dict[str, tuple[float, float, float]],
) -> list[LTDResult]:
    """Rebuild the full per-VP `LTDResult` list for one target.

    `ltd_predictions` (from targets.parquet) has success/error/upper_km/
    lower_km but no VP coords or latency; `obs_by_vp` (vp_id -> (lat, lon,
    latency_ms), from the matching fold's eval_observations.parquet) supplies
    the rest. The join must be exact — every vp_id in `ltd_predictions` came
    from that fold's eval_observations in the first place."""
    out: list[LTDResult] = []
    for p in ltd_predictions:
        vp_id = p["vp_id"]
        ob = obs_by_vp.get(vp_id)
        if ob is None:
            raise KeyError(
                f"vp_id {vp_id!r} in ltd_predictions has no matching row in "
                "eval_observations.parquet — inputs/outputs are out of sync"
            )
        vp_lat, vp_lon, latency_ms = ob
        success = bool(p["success"])
        tg_distance = (
            Distance(upper_km=float(p["upper_km"]), lower_km=float(p["lower_km"] or 0.0))
            if success else None
        )
        out.append(LTDResult(
            success=success,
            error=Error[p["error"]] if p["error"] else None,
            vp_id=vp_id,
            vp_coord=Coord(lat=vp_lat, lon=vp_lon),
            latency=latency_ms,
            tg_distance=tg_distance,
        ))
    return out


# ---- per-target metric groups -------------------------------------------------

def _participant_geo_stats(
    participants: list[dict], target_lat: float, target_lon: float,
) -> dict[str, Any]:
    keys = (
        "part_min_dist_km", "part_mean_dist_km", "part_med_dist_km",
        "part_min_rtt_ms", "part_mean_rtt_ms", "part_med_rtt_ms",
        "part_min_infl", "part_mean_infl",
    )
    out: dict[str, Any] = {k: np.nan for k in keys}
    out["n_part"] = len(participants)
    if not participants:
        return out
    dist = haversine_distance(
        target_lat, target_lon,
        np.array([p["vp_lat"] for p in participants], dtype=float),
        np.array([p["vp_lon"] for p in participants], dtype=float),
    )
    rtt = np.array([p["rtt_ms"] for p in participants], dtype=float)
    ideal = THEORETICAL_SLOPE * dist
    infl = np.where(ideal > 1e-9, rtt / ideal, np.nan)
    out.update(
        part_min_dist_km=float(np.nanmin(dist)),
        part_mean_dist_km=float(np.nanmean(dist)),
        part_med_dist_km=float(np.nanmedian(dist)),
        part_min_rtt_ms=float(np.nanmin(rtt)),
        part_mean_rtt_ms=float(np.nanmean(rtt)),
        part_med_rtt_ms=float(np.nanmedian(rtt)),
        part_min_infl=float(np.nanmin(infl)) if np.isfinite(infl).any() else np.nan,
        part_mean_infl=float(np.nanmean(infl)) if np.isfinite(infl).any() else np.nan,
    )
    return out


def _includer_excluder_stats(
    participants: list[dict], target_lat: float, target_lon: float,
) -> dict[str, Any]:
    """Split participants by whether *their own* echoed band contains the
    true VP->target distance; per-group count + mean dist/RTT, plus the
    outer- vs inner-bound exclusion counts that drive `exclusion_reason`."""
    keys = (
        "includer_mean_dist_km", "includer_mean_rtt_ms",
        "excluder_mean_dist_km", "excluder_mean_rtt_ms",
    )
    out: dict[str, Any] = {k: np.nan for k in keys}
    out.update(n_includers=0, n_excluders=0, n_outer_exclusions=0, n_inner_exclusions=0)
    if not participants:
        return out
    dist = haversine_distance(
        target_lat, target_lon,
        np.array([p["vp_lat"] for p in participants], dtype=float),
        np.array([p["vp_lon"] for p in participants], dtype=float),
    )
    rtt = np.array([p["rtt_ms"] for p in participants], dtype=float)
    lo = np.array([p["echoed_lower_km"] if p["echoed_lower_km"] is not None else 0.0
                   for p in participants], dtype=float)
    up = np.array([p["echoed_upper_km"] if p["echoed_upper_km"] is not None else np.inf
                   for p in participants], dtype=float)
    outer_excl = dist > up
    inner_excl = dist < lo
    excl = outer_excl | inner_excl
    incl = ~excl
    out["n_includers"] = int(incl.sum())
    out["n_excluders"] = int(excl.sum())
    out["n_outer_exclusions"] = int(outer_excl.sum())
    out["n_inner_exclusions"] = int(inner_excl.sum())
    if incl.any():
        out["includer_mean_dist_km"] = float(dist[incl].mean())
        out["includer_mean_rtt_ms"] = float(rtt[incl].mean())
    if excl.any():
        out["excluder_mean_dist_km"] = float(dist[excl].mean())
        out["excluder_mean_rtt_ms"] = float(rtt[excl].mean())
    return out


def _exclusion_reason(
    geom: Optional[BaseGeometry], truth_in_region: Optional[bool], incl_excl: dict[str, Any],
) -> Optional[str]:
    if geom is None:
        return "NO_REGION"
    if truth_in_region:
        return None
    if incl_excl["n_inner_exclusions"] > 0 and incl_excl["n_outer_exclusions"] > 0:
        return "BOTH"
    if incl_excl["n_inner_exclusions"] > 0:
        return "INNER"
    if incl_excl["n_outer_exclusions"] > 0:
        return "OUTER"
    return "OTHER"  # planar-approximation edge case / centroid-rule geometry


def _bridge_stats(
    obs_for_target: pd.DataFrame, participant_ids: set[str],
) -> dict[str, Any]:
    """closest-VP / shortest-ping-VP identities over the fold's full available
    set for this target (mirrors per_target_table.py), plus whether the
    shortest-ping VP made it into this combo's participant set."""
    dist = haversine_distance(
        obs_for_target["target_lat"].to_numpy(dtype=float),
        obs_for_target["target_lon"].to_numpy(dtype=float),
        obs_for_target["vp_lat"].to_numpy(dtype=float),
        obs_for_target["vp_lon"].to_numpy(dtype=float),
    )
    i_closest = int(np.argmin(dist))
    i_sping = int(obs_for_target["latency_ms"].to_numpy(dtype=float).argmin())
    closest_vp_id = str(obs_for_target["vp_id"].iloc[i_closest])
    sping_vp_id = str(obs_for_target["vp_id"].iloc[i_sping])
    return {
        "closest_vp_id": closest_vp_id,
        "closest_vp_dist_km": float(dist[i_closest]),
        "sping_vp_id": sping_vp_id,
        "sping_vp_dist_km": float(dist[i_sping]),
        "closest_is_sping": closest_vp_id == sping_vp_id,
        "sping_vp_is_participant": sping_vp_id in participant_ids,
    }


def _leave_one_out(
    ok_results: list[LTDResult],
    mtl_method: MTLMethod,
    ctr_method: CTRMethod,
    participant_ids: list[str],
    target_lat: float,
    target_lon: float,
    true_error_km: float,
    seed: Optional[int],
    compute: bool = True,
) -> dict[str, Any]:
    """Drop each participant in turn, re-run MTL+CTR, and record whether the
    pipeline flips to failure or how much the error degrades. Only
    participants are tested: a VP the MTL's own redundant-disk filter already
    dropped can't change the result by construction (removing an already-
    redundant disk doesn't alter which other disks are redundant), so testing
    only participants is both correct and cheaper.

    `compute=False` skips the per-participant reruns entirely (this is the
    expensive metric group — see the module docstring) while still reporting
    `loo_trivial` from the participant count alone, so callers can tell
    "not tested by request" (`loo_computed=False`) apart from "tested, found
    nothing to flip" (`loo_computed=True, loo_any_flip=False`)."""
    out: dict[str, Any] = {
        "loo_computed": False,
        "loo_n_tested": 0,
        "loo_trivial": len(participant_ids) <= 1,
        "loo_n_flips": 0,
        "loo_any_flip": False,
        "loo_max_error_delta_km": np.nan,
        "loo_critical_vp_id": None,
    }
    if not compute or not participant_ids or not np.isfinite(true_error_km):
        return out
    out["loo_computed"] = True

    flips: list[str] = []
    deltas: dict[str, float] = {}
    for vp in participant_ids:
        reduced = [r for r in ok_results if r.vp_id != vp]
        if seed is not None and hasattr(ctr_method, "rng"):
            ctr_method.rng = np.random.default_rng(int(seed))
        mtl_r = mtl_method.multilaterate(reduced)
        ctr_r = ctr_method.select_centroid(mtl_r) if mtl_r.success else None
        if (not mtl_r.success) or ctr_r is None or (not ctr_r.success) or ctr_r.tg_coord is None:
            flips.append(vp)
            continue
        new_err = float(haversine_distance(
            target_lat, target_lon, ctr_r.tg_coord.lat, ctr_r.tg_coord.lon,
        ))
        deltas[vp] = new_err - true_error_km

    out["loo_n_tested"] = len(participant_ids)
    out["loo_n_flips"] = len(flips)
    out["loo_any_flip"] = len(flips) > 0
    critical = flips[0] if flips else None
    if deltas:
        worst_vp = max(deltas, key=deltas.get)
        out["loo_max_error_delta_km"] = deltas[worst_vp]
        if critical is None:
            critical = worst_vp
    out["loo_critical_vp_id"] = critical
    return out


# ---- answer-space (cell gap) --------------------------------------------------

def _load_cell_gap(clusters_dir: Path) -> Optional[pd.DataFrame]:
    """target_id -> cell_gap_km from an existing `cluster-eval` output dir.

    `clusters.csv` doesn't store cell_gap_km itself, so it's rederived the
    same way `eval_source.py`'s `cluster_targets` does (BallTree k=2 over
    centroids) rather than reclustering targets from scratch. None if the
    dir doesn't exist (cluster-eval not yet run for this source/setup)."""
    assignments_path = clusters_dir / "assignments.csv"
    clusters_path = clusters_dir / "clusters.csv"
    if not (assignments_path.exists() and clusters_path.exists()):
        return None
    assignments = pd.read_csv(assignments_path)
    clusters = pd.read_csv(clusters_path)
    n = len(clusters)
    if n > 1:
        centroids_rad = np.radians(clusters[["centroid_lat", "centroid_lon"]].to_numpy())
        tree = BallTree(centroids_rad, metric="haversine")
        dist, _ = tree.query(centroids_rad, k=2)
        cell_gap = dist[:, 1] * EARTH_RADIUS_KM
    else:
        cell_gap = np.full(n, np.inf)
    clusters = clusters.assign(cell_gap_km=cell_gap)
    merged = assignments.merge(clusters[["cluster_id", "cell_gap_km"]], on="cluster_id", how="left")
    return merged[["target_id", "cell_gap_km"]]


# ---- orchestration -------------------------------------------------------------

def _fold_paths(combo_dir: Path, run_dir: Path, inputs_root: Path) -> tuple[Path, Path]:
    """(eval_observations.parquet path, clusters dir) for one combo's fold dir.

    Layout: `<run_dir>/<source>/<setup>/<fold>/<combo_id>/` (outputs) and
    `<inputs_root>/<source>/<run_id>/<setup>/<fold>/` (inputs) — same
    convention `per_target_table.py` / `extract_features.py` rely on."""
    fold_dir = combo_dir.parent
    setup_dir = fold_dir.parent
    source_dir = setup_dir.parent
    source, setup, fold = source_dir.name, setup_dir.name, fold_dir.name
    obs_path = inputs_root / source / run_dir.name / setup / fold / "eval_observations.parquet"
    clusters_dir = setup_dir / "clusters"
    return obs_path, clusters_dir


def compute_bench_metrics(
    run_dir: Path,
    inputs_root: Path,
    source: Optional[str] = None,
    combos: Optional[list[str]] = None,
    compute_loo: bool = True,
) -> dict[str, pd.DataFrame]:
    """Return `{combo_id: per_target_metrics_df}` for every discovered combo,
    pooled across folds (K-fold sets are disjoint, so pooling is a plain
    concat — same convention as `per_target_table.py`).

    `compute_loo=False` skips leave-one-out brittleness (see the module
    docstring's brittleness section) — every other metric group still runs,
    including the single full MTL recompute each target needs for
    area/inclusion. Use this on large runs or combos with an expensive CTR
    (Monte Carlo medoid) where the O(sum of n_part) LOO reruns dominate."""
    combo_dirs = discover_combos(run_dir, source, slice_=None, combos=combos)
    if not combo_dirs:
        raise FileNotFoundError(f"No combos found under {run_dir} (source={source})")
    grouped = group_combos_by_id(combo_dirs)

    cell_gap_cache: dict[Path, Optional[pd.DataFrame]] = {}
    results: dict[str, pd.DataFrame] = {}

    for combo_id, fold_dirs in sorted(grouped.items()):
        rows: list[dict[str, Any]] = []
        for fold_dir in fold_dirs:
            run_meta = load_run_json(fold_dir)
            mtl_method = MTL_REGISTRY[run_meta["mtl"]](**(run_meta.get("mtl_kwargs") or {}))
            ctr_method = CTR_REGISTRY[run_meta["ctr"]](**(run_meta.get("ctr_kwargs") or {}))

            obs_path, clusters_dir = _fold_paths(fold_dir, run_dir, inputs_root)
            if not obs_path.exists():
                raise FileNotFoundError(
                    f"{combo_id}/{fold_dir.name}: no eval_observations.parquet at "
                    f"{obs_path} — pass --inputs-root pointing at the matching inputs tree"
                )
            obs = pq.read_table(obs_path).to_pandas()
            obs_groups = {tid: g for tid, g in obs.groupby("target_id")}

            if clusters_dir not in cell_gap_cache:
                cell_gap_cache[clusters_dir] = _load_cell_gap(clusters_dir)
            cell_gap_df = cell_gap_cache[clusters_dir]
            cell_gap_by_target = (
                cell_gap_df.set_index("target_id")["cell_gap_km"].to_dict()
                if cell_gap_df is not None else {}
            )
            if cell_gap_df is None:
                logger.warning(
                    "%s/%s: no cluster-eval output at %s — cell_gap_km will be NaN "
                    "(run `cli.py cluster-eval --run-id %s` first)",
                    combo_id, fold_dir.name, clusters_dir, run_dir.name,
                )

            tbl = load_targets(fold_dir).to_pandas()
            for row in tbl.itertuples(index=False):
                target_id = row.target_id
                obs_for_target = obs_groups.get(target_id)
                if obs_for_target is None or obs_for_target.empty:
                    raise KeyError(
                        f"{combo_id}/{fold_dir.name}: target {target_id!r} has no "
                        f"eval_observations rows at {obs_path}"
                    )
                obs_by_vp = {
                    str(r.vp_id): (float(r.vp_lat), float(r.vp_lon), float(r.latency_ms))
                    for r in obs_for_target.itertuples(index=False)
                }

                ltd_results = _reconstruct_ltd_results(list(row.ltd_predictions), obs_by_vp)
                ok = [r for r in ltd_results if r.success]
                mtl_result = mtl_method.multilaterate(ok)

                participants = list(row.mtl_participants)
                participant_ids = [p["vp_id"] for p in participants]
                recompute_matches = (
                    mtl_result.success == bool(row.mtl_success)
                    and set(mtl_result.participating_vp_ids or ()) == set(participant_ids)
                )
                if not recompute_matches:
                    logger.warning(
                        "%s/%s target=%s: recomputed MTL disagrees with the stored "
                        "run (success %s vs %s) — check LTDResult reconstruction",
                        combo_id, fold_dir.name, target_id,
                        mtl_result.success, row.mtl_success,
                    )

                geom = _intersection_to_geometry(mtl_result.intersection) if mtl_result.success else None
                truth_in_region = _point_in_region(geom, row.target_lat, row.target_lon)
                part_stats = _participant_geo_stats(participants, row.target_lat, row.target_lon)
                incl_excl = _includer_excluder_stats(participants, row.target_lat, row.target_lon)
                bridge = _bridge_stats(obs_for_target, set(participant_ids))

                loo: dict[str, Any]
                if row.status == "SUCCESS" and mtl_result.success and len(participant_ids) >= 1:
                    loo = _leave_one_out(
                        ok, mtl_method, ctr_method, participant_ids,
                        row.target_lat, row.target_lon, float(row.error_km),
                        None if pd.isna(row.seed) else int(row.seed),
                        compute=compute_loo,
                    )
                else:
                    loo = _leave_one_out(
                        ok, mtl_method, ctr_method, [], 0.0, 0.0, np.nan, None,
                        compute=compute_loo,
                    )

                rows.append({
                    "combo_id": combo_id,
                    "target_id": target_id,
                    "target_lat": row.target_lat,
                    "target_lon": row.target_lon,
                    "status": row.status,
                    "error_km": row.error_km,
                    "n_obs": row.n_obs,
                    "n_ltd_success": row.n_ltd_success,
                    "n_mtl_participants": row.n_mtl_participants,
                    "mtl_success": bool(mtl_result.success),
                    "mtl_intersection_kind": row.mtl_intersection_kind,
                    "recompute_matches": recompute_matches,
                    "mtl_area_km2": _area_km2(geom),
                    "mtl_n_components": _n_components(geom),
                    "truth_in_region": truth_in_region,
                    "exclusion_reason": _exclusion_reason(geom, truth_in_region, incl_excl),
                    **part_stats,
                    **incl_excl,
                    **bridge,
                    "cell_gap_km": cell_gap_by_target.get(target_id, np.nan),
                    **loo,
                })
        results[combo_id] = pd.DataFrame(rows)
    return results


def _stat_block(series: pd.Series) -> dict[str, Any]:
    v = series.dropna().to_numpy(dtype=float)
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, _PCTS)
    return {
        "n": int(v.size),
        "mean": round(float(v.mean()), 3),
        "percentiles": {f"p{p}": round(float(x), 3) for p, x in zip(_PCTS, q)},
    }


def summarize_bench(df: pd.DataFrame) -> dict[str, Any]:
    """Dataset-level rollup for one combo's per-target metrics frame."""
    mtl_ok = df[df["mtl_success"]]
    eligible_loo = df[df["loo_n_tested"] > 0]
    return {
        "n_targets": int(len(df)),
        "n_success": int((df["status"] == "SUCCESS").sum()),
        "metrics": {m: _stat_block(df[m]) for m in PER_TARGET_STAT_METRICS},
        "truth_in_region_share": (
            round(float(mtl_ok["truth_in_region"].mean()), 4) if len(mtl_ok) else None
        ),
        "exclusion_reason_shares": (
            df.loc[df["exclusion_reason"].notna(), "exclusion_reason"]
            .value_counts(normalize=True).round(4).to_dict()
        ),
        "loo_any_flip_share": (
            round(float(eligible_loo["loo_any_flip"].mean()), 4) if len(eligible_loo) else None
        ),
        "loo_n_eligible": int(len(eligible_loo)),
        "loo_computed": bool(df["loo_computed"].any()),
        "recompute_matches_share": round(float(df["recompute_matches"].mean()), 4),
    }


def eval_bench_results(
    run_dir: Path,
    inputs_root: Path,
    out_dir: Optional[Path] = None,
    source: Optional[str] = None,
    combos: Optional[list[str]] = None,
    compute_loo: bool = True,
) -> dict[str, Any]:
    """Compute + write bench-output metrics for every combo under `run_dir`.

    Writes `<out_dir>/<combo_id>_bench_per_target.csv` per combo and
    `<out_dir>/summary.parquet` (one row per combo). `out_dir` defaults to
    `<run_dir>/bench_eval/`. Returns `{combo_id: summary_dict}` plus the
    output paths under `"per_target_csvs"` / `"summary_parquet"`.
    `compute_loo=False` skips leave-one-out brittleness — see
    `compute_bench_metrics`."""
    out_dir = out_dir or (run_dir / "bench_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_combo = compute_bench_metrics(
        run_dir, inputs_root, source=source, combos=combos, compute_loo=compute_loo,
    )

    stats: dict[str, Any] = {"run_dir": str(run_dir), "per_target_csvs": {}}
    summary_rows: list[dict[str, Any]] = []
    for combo_id, df in sorted(per_combo.items()):
        csv_path = out_dir / f"{combo_id}_bench_per_target.csv"
        df.to_csv(csv_path, index=False)
        stats["per_target_csvs"][combo_id] = str(csv_path)
        combo_summary = summarize_bench(df)
        stats[combo_id] = combo_summary
        summary_rows.append({"combo_id": combo_id, **_flatten_summary(combo_summary)})

    summary_path = out_dir / "summary.parquet"
    pd.DataFrame(summary_rows).to_parquet(summary_path, index=False)
    stats["summary_parquet"] = str(summary_path)
    return stats


def _flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten `summarize_bench`'s nested dict into scalar columns for the
    one-row-per-combo summary parquet (percentile blocks -> `<metric>_p50`
    etc., matching the flattening convention `cli.py`'s SUMMARY_SCHEMA uses)."""
    flat: dict[str, Any] = {
        "n_targets": summary["n_targets"],
        "n_success": summary["n_success"],
        "truth_in_region_share": summary["truth_in_region_share"],
        "loo_any_flip_share": summary["loo_any_flip_share"],
        "loo_n_eligible": summary["loo_n_eligible"],
        "loo_computed": summary["loo_computed"],
        "recompute_matches_share": summary["recompute_matches_share"],
    }
    for reason, share in summary["exclusion_reason_shares"].items():
        flat[f"exclusion_reason_{reason.lower()}_share"] = share
    for metric, block in summary["metrics"].items():
        flat[f"{metric}_n"] = block.get("n", 0)
        flat[f"{metric}_mean"] = block.get("mean")
        for p, val in block.get("percentiles", {}).items():
            flat[f"{metric}_{p}"] = val
    return flat
