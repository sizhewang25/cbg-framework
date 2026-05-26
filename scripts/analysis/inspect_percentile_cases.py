"""Geographic inspection of MTL circle intersection per percentile case.

Driven by the percentile CSV from
`scripts/analysis/inspect_cbg_vs_shortest_ping.py`: for each of the 5 rows
(p5, p25, p50, p75, p95) belonging to one combo (default vanilla_cbg),
reconstruct the runner's per-target geometry and render one world-map PNG
showing:

  - every VP's predicted disk boundary (transparent gray polyline)
  - the VP centers (gray dots)
  - the nearest-RTT VP (blue triangle) — what shortest-ping picked
  - the MTL feasible-region vertices (black dots) — output of multilateration
  - the CBG centroid (red star) — from the original run's targets.parquet
  - the true target (gold star)

Re-runs the LTD predict + MTL multilaterate using the saved checkpoint and
the eval-observations parquet, so what's plotted is exactly what the runner
computed (we cross-check the recomputed centroid against the recorded
pred_lat/pred_lon).

Sibling to `plot_circle_intersections.py` (the pedagogical 3D math
illustrator); both can live side by side.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Iterable, Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from scripts.analysis._v2_io import load_targets
from scripts.benchmark.v2.checkpoint import load_ltd_checkpoint
from scripts.framework.geometry import EARTH_RADIUS_KM, geo_to_cartesian
from scripts.framework.v2.mtl.spherical_circle import SphericalCircleMTL
from scripts.framework.v2.types import Coord, Latency, VpId
from scripts.libs.cbg.rtt_model import haversine_distance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Great-circle disk boundary on the real Earth
# ---------------------------------------------------------------------------

def great_circle_polygon(
    lat_c: float, lon_c: float, radius_km: float, n: int = 128,
) -> np.ndarray:
    """Sample `n` points on the great-circle boundary of a spherical disk.

    Returns an (n, 2) array of (lat, lon) in degrees. Same closed-form math
    as `cap_boundary` in `plot_circle_intersections.py`, scaled by the real
    Earth radius — the angular radius is `radius_km / EARTH_RADIUS_KM`.
    """
    center = np.array(geo_to_cartesian(lat_c, lon_c))
    # Orthonormal basis perpendicular to `center`.
    tmp = np.array([0.0, 0.0, 1.0]) if abs(center[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(center, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(center, e1)

    r = radius_km / EARTH_RADIUS_KM
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = (
        np.cos(r) * center[None, :]
        + np.sin(r) * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)
    )
    # ECEF → lat/lon (degrees).
    lat = np.degrees(np.arcsin(pts[:, 2]))
    lon = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    return np.column_stack([lat, lon])


# ---------------------------------------------------------------------------
# CSV / parquet loaders
# ---------------------------------------------------------------------------

def _read_percentile_csv(path: Path, combo: str) -> list[dict]:
    rows = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r["combo_id"] != combo:
                continue
            rows.append({
                "combo_id": r["combo_id"],
                "percentile": int(r["percentile"]),
                "fold": r["fold"],
                "target_id": r["target_id"],
                "target_lat": float(r["target_lat"]),
                "target_lon": float(r["target_lon"]),
                "nearest_vp_id": r["nearest_vp_id"],
                "error_cbg_km": float(r["error_cbg_km"]),
                "error_baseline_km": float(r["error_baseline_km"]),
                "delta_km": float(r["delta_km"]),
                "n_obs": int(r["n_obs"]),
                "n_ltd_success": int(r["n_ltd_success"]),
                "mtl_intersection_kind": r["mtl_intersection_kind"],
            })
    rows.sort(key=lambda r: r["percentile"])
    return rows


def _eval_obs_for_target(
    inputs_dir: Path, fold: str, target_id: str,
) -> list[tuple[VpId, Coord, Latency, str]]:
    """Build the runner's `obs` list for one target. Returns 4-tuples that
    include the vp_id string explicitly (for marker lookup later)."""
    path = inputs_dir / fold / "eval_observations.parquet"
    df = pq.read_table(path).to_pandas()
    df = df[df["target_id"] == target_id]
    if df.empty:
        raise ValueError(f"target {target_id} absent from {path}")
    obs = []
    for r in df.itertuples(index=False):
        obs.append((
            VpId(str(r.vp_id)),
            Coord(lat=float(r.vp_lat), lon=float(r.vp_lon)),
            Latency(float(r.latency_ms)),
            str(r.vp_id),
        ))
    return obs


def _saved_pred_for_target(
    run_dir: Path, fold: str, combo: str, target_id: str,
) -> tuple[Optional[float], Optional[float]]:
    combo_dir = run_dir / fold / combo
    tbl = load_targets(combo_dir)
    tids = tbl.column("target_id").to_pylist()
    try:
        i = tids.index(target_id)
    except ValueError as e:
        raise ValueError(f"target {target_id} absent from {combo_dir}/targets.parquet") from e
    pred_lat = tbl.column("pred_lat")[i].as_py()
    pred_lon = tbl.column("pred_lon")[i].as_py()
    return pred_lat, pred_lon


# ---------------------------------------------------------------------------
# Recompute LTD + MTL for one case
# ---------------------------------------------------------------------------

def _recompute_geometry(
    ltd, obs_with_vp_id: list[tuple[VpId, Coord, Latency, str]],
):
    """Run `ltd.predict_all` + `SphericalCircleMTL.multilaterate`, same kwargs
    as the canonical vanilla_cbg combo. Returns (ltd_results, mtl_result)."""
    obs = [(vp_id, vp_coord, lat) for vp_id, vp_coord, lat, _ in obs_with_vp_id]
    ltd_results = ltd.predict_all(obs)
    ok = [r for r in ltd_results if r.success]
    mtl = SphericalCircleMTL(speed_ratio=2.0 / 3.0, enable_circle_filter=True)
    return ltd_results, mtl.multilaterate(ok)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_case(
    row: dict,
    ltd_results,
    mtl_result,
    obs_with_vp_id: list[tuple[VpId, Coord, Latency, str]],
    saved_pred_lat: Optional[float],
    saved_pred_lon: Optional[float],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(14, 7.5))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f3f3f3", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#888888", zorder=1)
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.4, color="#bbbbbb", alpha=0.7, linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    # ---- disks (great-circle polygons) -----------------------------------
    n_ok = 0
    for r in ltd_results:
        if not r.success or r.vp_coord is None or r.tg_distance is None:
            continue
        n_ok += 1
        ring = great_circle_polygon(
            r.vp_coord.lat, r.vp_coord.lon, r.tg_distance.upper_km,
        )
        # Close the loop for the polyline.
        ring = np.vstack([ring, ring[:1]])
        ax.plot(
            ring[:, 1], ring[:, 0],
            color="#666666", alpha=0.08, linewidth=0.5,
            transform=ccrs.Geodetic(), zorder=2,
        )

    # ---- VP centers ------------------------------------------------------
    vp_lats = [vc.lat for _, vc, _, _ in obs_with_vp_id]
    vp_lons = [vc.lon for _, vc, _, _ in obs_with_vp_id]
    ax.scatter(
        vp_lons, vp_lats, s=6, c="#555555", alpha=0.6,
        transform=ccrs.PlateCarree(), zorder=3, label="VPs",
    )

    # ---- nearest VP (blue triangle) --------------------------------------
    nearest_vp_id = row["nearest_vp_id"]
    nearest = next(
        ((vc.lat, vc.lon) for _, vc, _, vid in obs_with_vp_id if vid == nearest_vp_id),
        None,
    )
    if nearest is not None:
        ax.scatter(
            [nearest[1]], [nearest[0]], s=110, marker="^",
            c="#0072B2", edgecolors="black", linewidths=0.6,
            transform=ccrs.PlateCarree(), zorder=6,
            label=f"shortest-ping VP ({nearest_vp_id})",
        )

    # ---- feasible-region vertices (black dots) ---------------------------
    n_vertices = 0
    if mtl_result.success and isinstance(mtl_result.intersection, list):
        verts = mtl_result.intersection
        n_vertices = len(verts)
        if verts:
            ax.scatter(
                [c.lon for c in verts], [c.lat for c in verts],
                s=15, c="black", alpha=0.85,
                transform=ccrs.PlateCarree(), zorder=5,
                label=f"feasible vertices (n={n_vertices})",
            )

    # ---- CBG centroid (red star) -----------------------------------------
    if saved_pred_lat is not None and saved_pred_lon is not None:
        ax.scatter(
            [saved_pred_lon], [saved_pred_lat], s=220, marker="*",
            c="#D55E00", edgecolors="black", linewidths=0.7,
            transform=ccrs.PlateCarree(), zorder=7,
            label="CBG centroid (from run)",
        )

    # ---- true target (gold star) -----------------------------------------
    ax.scatter(
        [row["target_lon"]], [row["target_lat"]], s=260, marker="*",
        c="#F0E442", edgecolors="black", linewidths=0.9,
        transform=ccrs.PlateCarree(), zorder=8,
        label=f"true target ({row['target_id']})",
    )

    title = (
        f"p{row['percentile']:02d} — {row['combo_id']}, fold={row['fold']}, "
        f"target={row['target_id']}\n"
        f"error_CBG={row['error_cbg_km']:.0f} km · "
        f"error_baseline={row['error_baseline_km']:.0f} km · "
        f"Δ={row['delta_km']:+.0f} km · "
        f"n_obs={row['n_obs']} · n_ltd_success={n_ok} · "
        f"n_vertices={n_vertices} · mtl={row['mtl_intersection_kind']}"
    )
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render world-map PNGs of the MTL circle intersection for each "
            "percentile case in the inspect_cbg_vs_shortest_ping CSV."
        ),
    )
    parser.add_argument("--percentile-csv", type=Path, required=True)
    parser.add_argument("--combo", default="vanilla_cbg")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="benchmark outputs <run_id>/<source>/<setup>/ (parent of fold_*/<combo>/)")
    parser.add_argument("--inputs-dir", type=Path, required=True,
                        help="benchmark inputs <source>/<run_id>/<setup>/ (parent of fold_*/eval_observations.parquet)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows = _read_percentile_csv(args.percentile_csv, args.combo)
    if not rows:
        raise SystemExit(
            f"No rows in {args.percentile_csv} for combo={args.combo!r}"
        )

    # LTD checkpoints are per-fold; cache to avoid reloading the same one
    # across percentile cases that share a fold.
    ltd_cache: dict[str, object] = {}

    for row in rows:
        fold = row["fold"]
        combo_dir = args.run_dir / fold / args.combo

        if fold not in ltd_cache:
            ltd = load_ltd_checkpoint(combo_dir)
            if ltd is None:
                raise SystemExit(
                    f"{args.combo} has no fitted state at {combo_dir} "
                    "(stateless LTD marker present). Cannot recompute geometry."
                )
            ltd_cache[fold] = ltd
        ltd = ltd_cache[fold]

        obs = _eval_obs_for_target(args.inputs_dir, fold, row["target_id"])
        ltd_results, mtl_result = _recompute_geometry(ltd, obs)

        saved_pred_lat, saved_pred_lon = _saved_pred_for_target(
            args.run_dir, fold, args.combo, row["target_id"],
        )

        # Sanity check: the recomputed centroid should match the run's
        # saved prediction. Compute the centroid the same way the runner
        # would (boundary-vertex mean) only if MTL succeeded.
        if mtl_result.success and isinstance(mtl_result.intersection, list) \
                and mtl_result.intersection and saved_pred_lat is not None:
            verts = mtl_result.intersection
            recomp_lat = float(np.mean([c.lat for c in verts]))
            recomp_lon = float(np.mean([c.lon for c in verts]))
            d_km = haversine_distance(
                recomp_lat, recomp_lon, saved_pred_lat, saved_pred_lon,
            )
            if d_km > 0.1:
                logger.warning(
                    "centroid mismatch on %s: recomputed=(%.4f, %.4f) "
                    "saved=(%.4f, %.4f) Δ=%.3f km",
                    row["target_id"], recomp_lat, recomp_lon,
                    saved_pred_lat, saved_pred_lon, d_km,
                )

        fname = (
            f"case_p{row['percentile']:02d}_{args.combo}_"
            f"{fold}_{row['target_id']}.png"
        )
        _plot_case(
            row, ltd_results, mtl_result, obs,
            saved_pred_lat, saved_pred_lon,
            args.out_dir / fname,
        )

    logger.info("done — %d cases written to %s", len(rows), args.out_dir)


if __name__ == "__main__":
    main()
