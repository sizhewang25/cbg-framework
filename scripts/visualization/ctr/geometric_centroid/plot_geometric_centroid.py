"""Visualizations of GeometricCentroidCTR.

Renders three PNGs into `outputs/`:

1. concept_polygon.png             — Shapely-region input path. Shows a 3-disk
   planar intersection and the area-weighted centroid Shapely returns directly,
   contrasted with the arithmetic mean of the boundary vertices.
2. concept_vertex_list.png         — list[Coord] input path. Shows the dedupe
   → local-project → polar-angle sort → polygon → area-centroid pipeline that
   reconstructs a polygon from unordered spherical crossings.
3. vertex_density_invariance.png   — Companion to BoundaryVertexMeanCTR's
   density-bias figure. The SAME logical unit square sampled two ways (4
   corners vs. 4 corners + 40 co-linear top-edge points) is fed through the
   vertex-list pipeline; the area centroid stays at (0.5, 0.5) either way,
   while the arithmetic mean drifts toward the dense edge.

Run as a script:
    python -m scripts.visualization.ctr.geometric_centroid.plot_geometric_centroid
"""

from __future__ import annotations

from math import atan2
from pathlib import Path

import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon

from scripts.framework.geometry import arithmetic_mean_centroid
from scripts.framework.v2.ctr.boundary_vertex_mean import _extract_vertex_coords
from scripts.framework.v2.ctr.geometric_centroid import (
    GeometricCentroidCTR,
    _circular_mean_longitude,
    _local_project,
)
from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.mtl.base import MTLResult
from scripts.framework.v2.mtl.planar_circle import (
    PlanarCircleMTL,
    _circle_to_planar_polygon,
)
from scripts.framework.v2.mtl.spherical_circle import SphericalCircleMTL
from scripts.framework.v2.types import Coord, Distance, Latency, VpId

OUT_DIR = Path(__file__).parent / "outputs"

# Same VPs as the boundary_vertex_mean visualization for direct comparability.
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
    ax.fill(xs, ys, color="gold", alpha=0.45, zorder=2)
    ax.plot(xs, ys, color="darkgoldenrod", lw=1.8, zorder=3)


# ---------------------------------------------------------------------------
# Figure 1 — Shapely region input path
# ---------------------------------------------------------------------------
def plot_concept_polygon(out_path: Path) -> None:
    _, ax = plt.subplots(figsize=(8.5, 8))

    ltd_results = _make_ltd_results()
    mtl_result = PlanarCircleMTL(n_pts=64).multilaterate(ltd_results)
    region = mtl_result.intersection

    for (lat, lon, r), color in zip(_VPS, _VP_COLORS):
        _draw_disk(ax, lat, lon, r, color)

    _plot_region(ax, region)

    coords = _extract_vertex_coords(region)
    vert_lons = [lon for _, lon in coords]
    vert_lats = [lat for lat, _ in coords]
    ax.scatter(
        vert_lons,
        vert_lats,
        color="black",
        s=14,
        alpha=0.55,
        zorder=4,
        label=f"boundary vertices (n={len(coords)})",
    )

    ctr_result = GeometricCentroidCTR().select_centroid(mtl_result)
    ax.plot(
        ctr_result.tg_coord.lon,
        ctr_result.tg_coord.lat,
        marker="*",
        color="crimson",
        markersize=26,
        markeredgecolor="black",
        zorder=6,
        label="GeometricCentroidCTR\n(Shapely area-weighted centroid)",
    )

    mean_lat, mean_lon = arithmetic_mean_centroid(coords)
    ax.plot(
        mean_lon,
        mean_lat,
        marker="P",
        color="purple",
        markersize=14,
        markeredgecolor="white",
        zorder=6,
        label="arithmetic mean of vertices (for contrast)",
    )

    ax.set_aspect("equal")
    ax.set_xlabel("lon (degrees)")
    ax.set_ylabel("lat (degrees)")
    ax.set_title(
        "GeometricCentroidCTR — Shapely region input path\n"
        "Returns Shapely's area-weighted centroid of the feasible polygon."
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2 — vertex-list pipeline
# ---------------------------------------------------------------------------
def plot_concept_vertex_list(out_path: Path) -> None:
    _, ax = plt.subplots(figsize=(8.5, 8))

    ltd_results = _make_ltd_results()
    mtl_result = SphericalCircleMTL().multilaterate(ltd_results)
    vertices: list[Coord] = mtl_result.intersection

    for (lat, lon, r), color in zip(_VPS, _VP_COLORS):
        _draw_disk(ax, lat, lon, r, color)

    raw_coords = [(v.lat, v.lon) for v in vertices]
    lat0 = sum(lat for lat, _ in raw_coords) / len(raw_coords)
    lon0 = _circular_mean_longitude([lon for _, lon in raw_coords])
    local_points, _, _, _ = _local_project(raw_coords)
    ordered_idx = sorted(
        range(len(local_points)),
        key=lambda i: atan2(local_points[i][1], local_points[i][0]),
    )
    ordered_coords = [raw_coords[i] for i in ordered_idx]

    poly_lons = [c[1] for c in ordered_coords] + [ordered_coords[0][1]]
    poly_lats = [c[0] for c in ordered_coords] + [ordered_coords[0][0]]
    ax.fill(poly_lons, poly_lats, color="gold", alpha=0.45, zorder=2)
    ax.plot(
        poly_lons,
        poly_lats,
        color="darkgoldenrod",
        lw=1.8,
        linestyle="--",
        zorder=3,
        label="reconstructed planar polygon\n(after polar-angle sort)",
    )

    for raw_i, (lat, lon) in enumerate(raw_coords):
        ax.annotate(
            f"raw#{raw_i}",
            (lon, lat),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color="black",
        )
    for sort_pos, raw_i in enumerate(ordered_idx):
        lat, lon = raw_coords[raw_i]
        ax.annotate(
            f"sorted#{sort_pos}",
            (lon, lat),
            xytext=(8, -14),
            textcoords="offset points",
            fontsize=9,
            color="darkgoldenrod",
        )

    ax.scatter(
        [v.lon for v in vertices],
        [v.lat for v in vertices],
        color="black",
        s=110,
        zorder=5,
        label=f"feasible-region vertices (n={len(vertices)})",
    )

    ax.plot(
        lon0,
        lat0,
        marker="x",
        color="dimgray",
        markersize=12,
        markeredgewidth=2.5,
        zorder=5,
        label="local-projection origin (lat0, lon0)",
    )

    ctr_result = GeometricCentroidCTR().select_centroid(mtl_result)
    ax.plot(
        ctr_result.tg_coord.lon,
        ctr_result.tg_coord.lat,
        marker="*",
        color="crimson",
        markersize=28,
        markeredgecolor="black",
        zorder=6,
        label="GeometricCentroidCTR\n(area centroid of reconstructed polygon)",
    )

    ax.set_aspect("equal")
    ax.set_xlabel("lon (degrees)")
    ax.set_ylabel("lat (degrees)")
    ax.set_title(
        "GeometricCentroidCTR — vertex-list input path\n"
        "Pipeline: dedupe → local project → polar-angle sort → polygon → "
        "area centroid."
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3 — vertex-density invariance (companion to BoundaryVertexMeanCTR bias)
# ---------------------------------------------------------------------------
def _square_with_densified_top(n_top_extra: int) -> list[Coord]:
    """Unit-square corner set with extra co-linear vertices on the top edge."""
    bottom_right = Coord(lat=0.0, lon=1.0)
    top_right = Coord(lat=1.0, lon=1.0)
    top_left = Coord(lat=1.0, lon=0.0)
    bottom_left = Coord(lat=0.0, lon=0.0)
    top_extras = [
        Coord(lat=1.0, lon=1.0 - (i + 1) / (n_top_extra + 1))
        for i in range(n_top_extra)
    ]
    return [bottom_right, top_right, *top_extras, top_left, bottom_left]


def _mtl_from_coords(coords: list[Coord]) -> MTLResult:
    return MTLResult(success=True, intersection=coords)


def plot_vertex_density_invariance(out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)

    configs = [
        (0, "4 corners (uniform sampling)"),
        (40, "4 corners + 40 co-linear points on top edge"),
    ]

    ctr = GeometricCentroidCTR()

    for ax, (n_extra, label) in zip(axes, configs):
        coords = _square_with_densified_top(n_extra)
        poly = ShapelyPolygon([(c.lon, c.lat) for c in coords])
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, color="gold", alpha=0.35, zorder=1)
        ax.plot(xs, ys, color="darkgoldenrod", lw=2, zorder=2)

        ax.scatter(
            [c.lon for c in coords],
            [c.lat for c in coords],
            color="black",
            s=36,
            zorder=4,
            label=f"vertices (n={len(coords)})",
        )

        ctr_result = ctr.select_centroid(_mtl_from_coords(coords))
        ax.plot(
            ctr_result.tg_coord.lon,
            ctr_result.tg_coord.lat,
            marker="*",
            color="crimson",
            markersize=24,
            markeredgecolor="black",
            zorder=6,
            label=(
                "GeometricCentroidCTR\n"
                f"({ctr_result.tg_coord.lon:.3f}, {ctr_result.tg_coord.lat:.3f})"
            ),
        )

        mean_lat, mean_lon = arithmetic_mean_centroid(
            [(c.lat, c.lon) for c in coords]
        )
        ax.plot(
            mean_lon,
            mean_lat,
            marker="P",
            color="purple",
            markersize=14,
            markeredgecolor="white",
            zorder=6,
            label=f"arithmetic mean  ({mean_lon:.3f}, {mean_lat:.3f})",
        )

        ax.set_aspect("equal")
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel("lon")
        ax.set_title(label)
        ax.legend(loc="lower center", fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("lat")
    fig.suptitle(
        "Vertex-density invariance: identical polygon, different boundary "
        "sampling.\n"
        "After polar-angle sort + polygon reconstruction, the area centroid "
        "depends only on the polygon's shape — not on how the boundary is "
        "sampled.\n"
        "The arithmetic mean (purple +) drifts toward the dense edge; the "
        "area centroid (red ★) does not.",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_concept_polygon(OUT_DIR / "concept_polygon.png")
    plot_concept_vertex_list(OUT_DIR / "concept_vertex_list.png")
    plot_vertex_density_invariance(OUT_DIR / "vertex_density_invariance.png")
    print("Wrote:")
    for f in sorted(OUT_DIR.iterdir()):
        print(" ", f)


if __name__ == "__main__":
    main()
