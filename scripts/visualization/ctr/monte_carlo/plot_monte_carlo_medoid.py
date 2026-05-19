"""Visualizations of MonteCarloMedoidCTR.

Renders four PNGs into `outputs/`:

1. concept_polygon.png            — Shapely-region input path. Shows a 3-disk
   planar intersection, the Sobol-QMC rejection samples inside it (colored by
   sum-of-distances to every other sample), the chosen medoid (the sample with
   minimum total distance), and the Shapely area centroid for contrast.
2. concept_vertex_list.png        — list[Coord] input path. The vertices ARE
   the point set — no sampling. The medoid is the vertex with smallest total
   distance to the others.
3. sample_convergence.png         — Same feasible region sampled at increasing
   n_samples (20, 100, 500, 2000) to show how the medoid stabilizes as the
   Sobol-QMC samples densify.
4. discrete_polygon_feasibility.png — A hand-defined non-convex L-shape
   polygon. Sobol-QMC samples inside the L → medoid stays inside by
   construction (it IS one of the samples). The Shapely area centroid falls
   into the L's notch — outside the polygon — showing why the medoid is the
   safer choice on irregular feasible regions.

Run as a script:
    python -m scripts.visualization.ctr.monte_carlo.plot_monte_carlo_medoid
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon

from scripts.framework.geometry import sample_points_in_region, sampled_medoid
from scripts.framework.v2.ctr.monte_carlo_medoid import MonteCarloMedoidCTR
from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.mtl.planar_circle import (
    PlanarCircleMTL,
    _circle_to_planar_polygon,
)
from scripts.framework.v2.mtl.spherical_circle import SphericalCircleMTL
from scripts.framework.v2.types import Coord, Distance, Latency, VpId

OUT_DIR = Path(__file__).parent / "outputs"

# Same VPs as the boundary_vertex_mean / geometric_centroid visualizations
# for direct comparability across CTR methods.
_VPS: list[tuple[float, float, float]] = [
    (0.0, 0.0, 400.0),
    (0.0, 4.0, 400.0),
    (3.5, 2.0, 400.0),
]
_VP_COLORS = ["tab:red", "tab:blue", "tab:green"]


def _make_ltd_results() -> list[LTDResult]:
    return [
        LTDResult(
            success=True,
            vp_id=VpId(f"vp{i}"),
            vp_coord=Coord(lat=lat, lon=lon),
            latency=Latency(10.0),
            tg_distance=Distance(upper_km=r),
        )
        for i, (lat, lon, r) in enumerate(_VPS)
    ]


def _draw_disk(
    ax,
    lat_c: float,
    lon_c: float,
    radius_km: float,
    color: str,
    fill_alpha: float = 0.08,
    edge_alpha: float = 0.45,
) -> None:
    poly = _circle_to_planar_polygon(lat_c, lon_c, radius_km, n_pts=256)
    xs, ys = poly.exterior.xy
    ax.fill(xs, ys, color=color, alpha=fill_alpha)
    ax.plot(xs, ys, color=color, lw=1.4, alpha=edge_alpha)
    ax.plot(lon_c, lat_c, "o", color=color, markersize=7, zorder=5)


def _plot_region(ax, region) -> None:
    xs, ys = region.exterior.xy
    ax.fill(xs, ys, color="gold", alpha=0.30, zorder=2)
    ax.plot(xs, ys, color="darkgoldenrod", lw=1.6, zorder=3)


def _total_distances(points: np.ndarray) -> np.ndarray:
    """Sum of pairwise distances from each sample to all others.

    Degrees-as-flat-plane is fine here (figure is at the equator over a few
    hundred km); the colormap only needs to show *relative* total-distance
    ordering, which matches the haversine-based ranking inside ``sampled_medoid``.
    """
    lat = points[:, 0]
    lon = points[:, 1]
    d_lat = lat[:, None] - lat[None, :]
    d_lon = lon[:, None] - lon[None, :]
    dist = np.sqrt(d_lat * d_lat + d_lon * d_lon)
    return dist.sum(axis=1)


# ---------------------------------------------------------------------------
# Figure 1 — Shapely region input path
# ---------------------------------------------------------------------------
def plot_concept_polygon(out_path: Path) -> None:
    _, ax = plt.subplots(figsize=(9, 8))

    ltd_results = _make_ltd_results()
    mtl_result = PlanarCircleMTL(n_pts=64).multilaterate(ltd_results)
    region = mtl_result.intersection

    for (lat, lon, r), color in zip(_VPS, _VP_COLORS):
        _draw_disk(ax, lat, lon, r, color)

    _plot_region(ax, region)

    rng = np.random.default_rng(0)
    n_samples = 600
    samples = sample_points_in_region(region, n_samples=n_samples, rng=rng)
    totals = _total_distances(samples)

    sc = ax.scatter(
        samples[:, 1],
        samples[:, 0],
        c=totals,
        cmap="viridis_r",
        s=12,
        zorder=4,
        label=f"Sobol-QMC samples (n={len(samples)})",
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Σ distance to every other sample (lower = more central)")

    medoid_lat, medoid_lon = sampled_medoid(samples)
    ax.plot(
        medoid_lon,
        medoid_lat,
        marker="*",
        color="crimson",
        markersize=26,
        markeredgecolor="black",
        zorder=6,
        label="MonteCarloMedoidCTR\n(argmin Σ distances over samples)",
    )

    area_centroid = region.centroid
    ax.plot(
        area_centroid.x,
        area_centroid.y,
        marker="P",
        color="purple",
        markersize=14,
        markeredgecolor="white",
        zorder=6,
        label="Shapely area centroid (for contrast)",
    )

    ax.set_aspect("equal")
    ax.set_xlabel("lon (degrees)")
    ax.set_ylabel("lat (degrees)")
    ax.set_title(
        "MonteCarloMedoidCTR — Shapely region input path\n"
        "Sobol-QMC rejection-sample inside the region, then take the sample\n"
        "with the smallest total distance to every other sample."
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2 — vertex-list input path
# ---------------------------------------------------------------------------
def plot_concept_vertex_list(out_path: Path) -> None:
    _, ax = plt.subplots(figsize=(8.5, 8))

    ltd_results = _make_ltd_results()
    mtl_result = SphericalCircleMTL().multilaterate(ltd_results)
    vertices: list[Coord] = mtl_result.intersection

    for (lat, lon, r), color in zip(_VPS, _VP_COLORS):
        _draw_disk(ax, lat, lon, r, color)

    verts_arr = np.array([(v.lat, v.lon) for v in vertices], dtype=float)
    totals = _total_distances(verts_arr)

    sc = ax.scatter(
        verts_arr[:, 1],
        verts_arr[:, 0],
        c=totals,
        cmap="viridis_r",
        s=200,
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
        label=(
            f"spherical crossings (n={len(vertices)})\n"
            "vertices ARE the point set — no sampling"
        ),
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Σ distance to every other vertex")

    ctr_result = MonteCarloMedoidCTR().select_centroid(mtl_result)
    ax.plot(
        ctr_result.tg_coord.lon,
        ctr_result.tg_coord.lat,
        marker="*",
        color="crimson",
        markersize=28,
        markeredgecolor="black",
        zorder=6,
        label="MonteCarloMedoidCTR\n(argmin Σ distances over vertices)",
    )

    ax.set_aspect("equal")
    ax.set_xlabel("lon (degrees)")
    ax.set_ylabel("lat (degrees)")
    ax.set_title(
        "MonteCarloMedoidCTR — vertex-list input path\n"
        "For SphericalCircleMTL output, the medoid is one of the crossings."
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3 — sample-count convergence
# ---------------------------------------------------------------------------
def plot_sample_convergence(out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), sharex=True, sharey=True)

    ltd_results = _make_ltd_results()
    mtl_result = PlanarCircleMTL(n_pts=64).multilaterate(ltd_results)
    region = mtl_result.intersection

    sample_counts = [20, 100, 500, 2000]
    medoids: list[tuple[float, float]] = []

    for ax, n in zip(axes.flat, sample_counts):
        for (lat, lon, r), color in zip(_VPS, _VP_COLORS):
            _draw_disk(ax, lat, lon, r, color, fill_alpha=0.05, edge_alpha=0.3)
        _plot_region(ax, region)

        rng = np.random.default_rng(0)
        samples = sample_points_in_region(region, n_samples=n, rng=rng)
        ax.scatter(
            samples[:, 1],
            samples[:, 0],
            color="black",
            s=max(4, 80 / np.sqrt(n)),
            alpha=0.6,
            zorder=4,
            label=f"Sobol samples (n={len(samples)})",
        )

        medoid_lat, medoid_lon = sampled_medoid(samples)
        medoids.append((medoid_lat, medoid_lon))
        ax.plot(
            medoid_lon,
            medoid_lat,
            marker="*",
            color="crimson",
            markersize=22,
            markeredgecolor="black",
            zorder=6,
            label=f"medoid  ({medoid_lat:.4f}, {medoid_lon:.4f})",
        )

        ax.set_aspect("equal")
        ax.set_title(f"n_samples = {n}")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.25)

    for ax in axes[-1, :]:
        ax.set_xlabel("lon (degrees)")
    for ax in axes[:, 0]:
        ax.set_ylabel("lat (degrees)")

    drift_km = []
    final_lat, final_lon = medoids[-1]
    for lat, lon in medoids[:-1]:
        d_lat = (lat - final_lat) * 111.0
        d_lon = (lon - final_lon) * 111.0
        drift_km.append(np.hypot(d_lat, d_lon))

    fig.suptitle(
        "MonteCarloMedoidCTR — convergence with n_samples.\n"
        "Sobol-QMC densifies the region uniformly; the medoid stabilizes as n grows.\n"
        f"Drift vs. n=2000: "
        f"n=20 → {drift_km[0]:.1f} km, "
        f"n=100 → {drift_km[1]:.1f} km, "
        f"n=500 → {drift_km[2]:.1f} km.",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4 — discrete polygon, sample-and-medoid, feasibility highlight
# ---------------------------------------------------------------------------
def _l_shape_polygon() -> ShapelyPolygon:
    """Non-convex L-shape; its area centroid lies *outside* the polygon."""
    return ShapelyPolygon([
        (0.0, 0.0),
        (5.0, 0.0),
        (5.0, 1.5),
        (1.5, 1.5),
        (1.5, 5.0),
        (0.0, 5.0),
    ])


def plot_discrete_polygon_feasibility(out_path: Path) -> None:
    _, ax = plt.subplots(figsize=(9, 8))

    region = _l_shape_polygon()
    xs, ys = region.exterior.xy
    ax.fill(xs, ys, color="gold", alpha=0.30, zorder=2)
    ax.plot(xs, ys, color="darkgoldenrod", lw=1.8, zorder=3)
    for vx, vy in zip(xs[:-1], ys[:-1]):
        ax.plot(vx, vy, "o", color="darkgoldenrod", markersize=6, zorder=4)

    rng = np.random.default_rng(0)
    n_samples = 800
    samples = sample_points_in_region(region, n_samples=n_samples, rng=rng)
    totals = _total_distances(samples)

    sc = ax.scatter(
        samples[:, 1],
        samples[:, 0],
        c=totals,
        cmap="viridis_r",
        s=12,
        zorder=4,
        label=f"Sobol-QMC samples (n={len(samples)})",
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Σ distance to every other sample (lower = more central)")

    medoid_lat, medoid_lon = sampled_medoid(samples)
    medoid_inside = region.contains(Point(medoid_lon, medoid_lat))
    ax.plot(
        medoid_lon,
        medoid_lat,
        marker="*",
        color="crimson",
        markersize=26,
        markeredgecolor="black",
        zorder=6,
        label=(
            f"MonteCarloMedoidCTR  ({medoid_lat:.2f}, {medoid_lon:.2f})\n"
            f"inside polygon? {medoid_inside}"
        ),
    )

    area_centroid = region.centroid
    centroid_inside = region.contains(area_centroid)
    ax.plot(
        area_centroid.x,
        area_centroid.y,
        marker="P",
        color="purple",
        markersize=14,
        markeredgecolor="white",
        zorder=6,
        label=(
            f"Shapely area centroid  ({area_centroid.y:.2f}, {area_centroid.x:.2f})\n"
            f"inside polygon? {centroid_inside}"
        ),
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        "MonteCarloMedoidCTR on a non-convex L-shape polygon.\n"
        "Sobol-QMC samples uniformly cover the L; the medoid is one of them,\n"
        "so it's guaranteed inside. The area centroid lands in the notch — outside."
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_concept_polygon(OUT_DIR / "concept_polygon.png")
    plot_concept_vertex_list(OUT_DIR / "concept_vertex_list.png")
    plot_sample_convergence(OUT_DIR / "sample_convergence.png")
    plot_discrete_polygon_feasibility(
        OUT_DIR / "discrete_polygon_feasibility.png"
    )
    print("Wrote:")
    for f in sorted(OUT_DIR.iterdir()):
        print(" ", f)


if __name__ == "__main__":
    main()
