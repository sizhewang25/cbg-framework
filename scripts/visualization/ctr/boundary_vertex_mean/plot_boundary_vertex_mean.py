"""Visualizations of BoundaryVertexMeanCTR.

Renders three PNGs into `outputs/`:

1. concept_polygon.png      — Shapely-region input path. Shows a 3-disk planar
   intersection, its boundary vertices, the arithmetic-mean centroid, and the
   Shapely area centroid for contrast.
2. concept_vertex_list.png  — list[Coord] input path. Shows the pairwise
   spherical great-circle crossings (filtered to those inside every disk) that
   SphericalCircleMTL hands the CTR, with the arithmetic mean marked.
3. vertex_density_bias.png  — Controlled experiment: the SAME square polygon
   sampled two ways (uniform corners vs. densified top edge). The arithmetic
   mean is pulled toward the dense side; the area centroid is invariant.

Run as a script:
    python -m scripts.visualization.ctr.boundary_vertex_mean.plot_boundary_vertex_mean
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon

from scripts.framework.v2.ctr.boundary_vertex_mean import (
    BoundaryVertexMeanCTR,
    _extract_vertex_coords,
)
from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.mtl.planar_circle import (
    PlanarCircleMTL,
    _circle_to_planar_polygon,
)
from scripts.framework.v2.mtl.spherical_circle import SphericalCircleMTL
from scripts.framework.v2.types import Coord, Distance, Latency, VpId

OUT_DIR = Path(__file__).parent / "outputs"

# Three disk constraints near the equator (planar approx is accurate at this
# latitude and scale). Each entry: (lat, lon, radius_km).
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
    """Render a disk as a high-resolution planar polygon for the figure."""
    poly = _circle_to_planar_polygon(lat_c, lon_c, radius_km, n_pts=256)
    xs, ys = poly.exterior.xy
    ax.fill(xs, ys, color=color, alpha=fill_alpha)
    ax.plot(xs, ys, color=color, lw=1.4, alpha=edge_alpha)
    ax.plot(lon_c, lat_c, "o", color=color, markersize=7, zorder=5)


def _plot_region(ax, region) -> None:
    """Fill and outline the feasible region (assumed Polygon for these inputs)."""
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
        s=22,
        zorder=4,
        label=f"boundary vertices (n={len(coords)})",
    )

    ctr_result = BoundaryVertexMeanCTR().select_centroid(mtl_result)
    ax.plot(
        ctr_result.tg_coord.lon,
        ctr_result.tg_coord.lat,
        marker="*",
        color="crimson",
        markersize=26,
        markeredgecolor="black",
        zorder=6,
        label="BoundaryVertexMeanCTR\n(arithmetic mean of vertices)",
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
        "BoundaryVertexMeanCTR — Shapely region input path\n"
        "Vertices = dedup'd exterior + interior-ring points of the "
        "feasible polygon."
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

    ax.scatter(
        [v.lon for v in vertices],
        [v.lat for v in vertices],
        color="black",
        s=110,
        zorder=5,
        label=(
            f"feasible-region vertices (n={len(vertices)})\n"
            "= great-circle crossings inside every disk"
        ),
    )

    ctr_result = BoundaryVertexMeanCTR().select_centroid(mtl_result)
    ax.plot(
        ctr_result.tg_coord.lon,
        ctr_result.tg_coord.lat,
        marker="*",
        color="crimson",
        markersize=28,
        markeredgecolor="black",
        zorder=6,
        label="BoundaryVertexMeanCTR\n(arithmetic mean of vertices)",
    )

    ax.set_aspect("equal")
    ax.set_xlabel("lon (degrees)")
    ax.set_ylabel("lat (degrees)")
    ax.set_title(
        "BoundaryVertexMeanCTR — vertex-list input path\n"
        "Input from SphericalCircleMTL: pairwise spherical disk crossings."
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3 — vertex-density bias caveat
# ---------------------------------------------------------------------------
def _square_with_densified_top(n_top_extra: int) -> ShapelyPolygon:
    """A unit square with extra co-linear vertices inserted along the top edge.

    The polygon's *shape* (and area centroid) is unchanged by adding co-linear
    vertices; only the boundary-vertex sampling distribution shifts.
    """
    bottom_right = (1.0, 0.0)
    top_right = (1.0, 1.0)
    top_left = (0.0, 1.0)
    bottom_left = (0.0, 0.0)
    top_extras = [
        (1.0 - (i + 1) / (n_top_extra + 1), 1.0) for i in range(n_top_extra)
    ]
    coords = [bottom_right, top_right, *top_extras, top_left, bottom_left]
    return ShapelyPolygon(coords)


def _mean_of_shapely_polygon(poly: ShapelyPolygon) -> tuple[float, float]:
    """Arithmetic mean over a Shapely polygon's deduplicated ring vertices."""
    coords = list(poly.exterior.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    mean_x = sum(x for x, _ in coords) / len(coords)
    mean_y = sum(y for _, y in coords) / len(coords)
    return mean_x, mean_y


def plot_vertex_density_bias(out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)

    configs = [
        (0, "4 corners (uniform sampling)"),
        (40, "4 corners + 40 co-linear points on top edge"),
    ]

    for ax, (n_extra, label) in zip(axes, configs):
        poly = _square_with_densified_top(n_extra)
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, color="gold", alpha=0.35, zorder=1)
        ax.plot(xs, ys, color="darkgoldenrod", lw=2, zorder=2)

        coords = list(poly.exterior.coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        ax.scatter(
            [c[0] for c in coords],
            [c[1] for c in coords],
            color="black",
            s=36,
            zorder=4,
            label=f"vertices (n={len(coords)})",
        )

        mx, my = _mean_of_shapely_polygon(poly)
        ax.plot(
            mx,
            my,
            marker="*",
            color="crimson",
            markersize=24,
            markeredgecolor="black",
            zorder=6,
            label=f"arithmetic mean  ({mx:.3f}, {my:.3f})",
        )

        area_centroid = poly.centroid
        ax.plot(
            area_centroid.x,
            area_centroid.y,
            marker="P",
            color="purple",
            markersize=14,
            markeredgecolor="white",
            zorder=6,
            label=f"area centroid  ({area_centroid.x:.3f}, {area_centroid.y:.3f})",
        )

        ax.set_aspect("equal")
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel("x")
        ax.set_title(label)
        ax.legend(loc="lower center", fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("y")
    fig.suptitle(
        "Vertex-density bias: identical polygon, different boundary sampling.\n"
        "Co-linear extra vertices don't change the shape — but they pull the "
        "arithmetic mean toward the dense edge.\n"
        "Shapely's area centroid is invariant to the vertex distribution.",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_concept_polygon(OUT_DIR / "concept_polygon.png")
    plot_concept_vertex_list(OUT_DIR / "concept_vertex_list.png")
    plot_vertex_density_bias(OUT_DIR / "vertex_density_bias.png")
    print("Wrote:")
    for f in sorted(OUT_DIR.iterdir()):
        print(" ", f)


if __name__ == "__main__":
    main()
