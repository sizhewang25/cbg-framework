"""Map + stats for the ground-truth answer-space clustering.

Visualizes the **results** of `scripts.benchmark.v2.sources.cluster_ground_truth`
(deterministic complete-linkage agglomerative clustering, centroid radius capped
at ``R``). Clustering and visualization are decoupled: this script reads the
written results directory (``clusters.csv`` + ``assignments.csv`` + ``meta.json``)
and never recomputes the clustering, so the figure always reflects exactly what
was produced. Generate the inputs first::

    python -m scripts.benchmark.v2.sources.cluster_ground_truth \\
        --targets datasets/ripe_atlas/asn_corpora/targets.csv --radius-km 50

Left panel — a PlateCarree map: member coordinates colored by region (singletons
greyed), with an ``R``-radius geodesic circle drawn around each multi-member
region so the "coherent region" footprint is literal. Right column — the
distributions that characterize the answer space: region-size histogram (with
the singleton share called out) and the per-region centroid-radius CDF (the
scoring-relevant tightness, against the ``R`` line).

CLI::

    python -m scripts.visualization.cluster.plot_ground_truth_clusters \\
        --clusters-dir datasets/ripe_atlas/asn_corpora/clusters
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter

from scripts.benchmark.v2.sources.cluster_ground_truth import ClusterResult
from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM
from scripts.visualization.cluster.voronoi import (
    LandmassVoronoi,
    build_landmass_voronoi,
)

logger = logging.getLogger(__name__)

_RADIUS_PCTILES = (50, 75, 90, 95, 99)


def load_clustering(
    clusters_dir: Path, radius_override: float | None = None
) -> tuple[pd.DataFrame, ClusterResult]:
    """Load a `cluster_ground_truth` results dir into (members, ClusterResult).

    Reads ``clusters.csv`` (the answer space) and ``assignments.csv`` (per-coord
    membership). The radius cap ``R`` comes from ``meta.json``; pass
    `radius_override` to force it, or fall back to the max observed region radius
    (with a warning) when neither is available."""
    clusters = pd.read_csv(clusters_dir / "clusters.csv").sort_values(
        "cluster_id").reset_index(drop=True)
    members = pd.read_csv(clusters_dir / "assignments.csv")

    meta_path = clusters_dir / "meta.json"
    if radius_override is not None:
        radius_km = float(radius_override)
    elif meta_path.exists():
        radius_km = float(json.loads(meta_path.read_text())["radius_km"])
    else:
        radius_km = float(clusters["radius_km"].max())
        logger.warning("no meta.json and no --radius-km; using max region radius "
                       "%.0f km for footprints/cap line", radius_km)

    diameter = (clusters["diameter_km"].to_numpy(dtype=float)
                if "diameter_km" in clusters.columns
                else np.zeros(len(clusters)))
    res = ClusterResult(
        labels=members["cluster_id"].to_numpy(dtype=int),
        centroid_lat=clusters["centroid_lat"].to_numpy(dtype=float),
        centroid_lon=clusters["centroid_lon"].to_numpy(dtype=float),
        member_counts=clusters["n_members"].to_numpy(dtype=int),
        radius_km=clusters["radius_km"].to_numpy(dtype=float),
        diameter_km=diameter,
        dist_km=members["dist_to_centroid_km"].to_numpy(dtype=float),
        radius_target_km=radius_km,
        n_clusters=len(clusters),
    )
    return members, res


def _geodesic_circle(lat: float, lon: float, radius_km: float, n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Lon/lat polyline of a great-circle circle of `radius_km` about a center
    (destination-point formula), for drawing a region footprint on the map."""
    ang = radius_km / EARTH_RADIUS_KM
    brg = np.linspace(0, 2 * np.pi, n)
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(ang) + np.cos(lat1) * np.sin(ang) * np.cos(brg))
    lon2 = lon1 + np.arctan2(
        np.sin(brg) * np.sin(ang) * np.cos(lat1),
        np.cos(ang) - np.sin(lat1) * np.sin(lat2),
    )
    return np.degrees(lon2), np.degrees(lat2)


def _plot_voronoi_underlay(ax, voronoi: LandmassVoronoi) -> None:
    """Draw the answer-space Voronoi partition as a background layer (zorder 1).

    Multi-member cells are tinted by ``cluster_id`` (tab20) to echo the member
    dots above; singleton cells are greyed. Fills are translucent so the
    clustered points and footprints stay legible on top.
    """
    cmap = plt.get_cmap("tab20")
    cells = voronoi.cells
    singletons = cells[cells["is_singleton"]]
    if not singletons.empty:
        ax.add_geometries(
            list(singletons.geometry), crs=ccrs.PlateCarree(),
            facecolor="#d9d9d9", edgecolor="#9a9a9a", linewidth=0.3,
            alpha=0.30, zorder=1,
        )
    for _, row in cells[~cells["is_singleton"]].iterrows():
        ax.add_geometries(
            [row.geometry], crs=ccrs.PlateCarree(),
            facecolor=cmap(int(row["cluster_id"]) % 20),
            edgecolor="#555555", linewidth=0.4, alpha=0.30, zorder=1,
        )


def _plot_map(ax, df, res: ClusterResult, *, extent, voronoi: LandmassVoronoi | None = None) -> None:
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="#eaf2f8")
    ax.add_feature(cfeature.LAND, facecolor="#f6f4ef")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#999999")
    ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="#cccccc")

    if voronoi is not None:
        _plot_voronoi_underlay(ax, voronoi)

    lat = df["target_lat"].to_numpy()
    lon = df["target_lon"].to_numpy()
    multi = res.member_counts[res.labels] > 1  # per-input: is in a multi-member region

    # Singletons (grey) vs clustered members (colored by region id, cycled).
    ax.scatter(lon[~multi], lat[~multi], s=6, c="#b0b0b0", marker="o",
               transform=ccrs.PlateCarree(), zorder=2, label="singleton")
    if multi.any():
        ax.scatter(lon[multi], lat[multi], s=10,
                   c=res.labels[multi], cmap="tab20", marker="o",
                   transform=ccrs.PlateCarree(), zorder=3, label="clustered member")

    # R-radius footprints for multi-member regions (no centroid markers).
    for c in range(res.n_clusters):
        if res.member_counts[c] <= 1:
            continue
        cx, cy = _geodesic_circle(res.centroid_lat[c], res.centroid_lon[c], res.radius_target_km)
        ax.plot(cx, cy, color="#444444", linewidth=0.4, alpha=0.5,
                transform=ccrs.PlateCarree(), zorder=4)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    title = (
        f"Answer space: {len(df):,} coords → {res.n_clusters:,} regions "
        f"(≤{res.radius_target_km:.0f} km)"
    )
    if voronoi is not None:
        title += f"\nnearest-centroid partition over {voronoi.label} ({len(voronoi.cells):,} cells)"
    ax.set_title(title, fontsize=12)


def _plot_size_hist(ax, res: ClusterResult, n_coords: int) -> None:
    counts = res.member_counts
    singletons = int((counts == 1).sum())
    bins = np.arange(1, counts.max() + 2) - 0.5
    ax.hist(counts, bins=bins, color="#1f77b4", edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("members per region")
    ax.set_ylabel("region count (log)")
    ax.set_title(
        f"region sizes — {singletons:,} singletons "
        f"({100 * singletons / res.n_clusters:.0f}% of regions)",
        fontsize=11,
    )
    ax.grid(True, axis="y", alpha=0.3)


def _plot_radius_cdf(ax, res: ClusterResult) -> None:
    r = np.sort(res.radius_km)
    ax.plot(r, np.arange(1, len(r) + 1) / len(r), color="#2ca02c", linewidth=2.0)
    ax.axvline(res.radius_target_km, color="red", linestyle=":", alpha=0.7)
    ax.text(res.radius_target_km, 0.05, f" cap {res.radius_target_km:.0f} km",
            color="red", fontsize=8, rotation=90, va="bottom", ha="right")
    ax.set_xlim(0, max(res.radius_target_km * 1.05, float(r[-1])))
    ax.set_ylim(0, 1)
    ax.set_xlabel("region radius (max member→centroid, km)")
    ax.set_ylabel("CDF")
    ax.set_title("region radius (cap utilization)", fontsize=11)
    ax.grid(True, alpha=0.3)
    box = "\n".join(
        f"p{p:<2d} {np.percentile(res.radius_km, p):>5.1f} km" for p in _RADIUS_PCTILES)
    ax.text(0.04, 0.96, box, transform=ax.transAxes, fontsize=7.5, va="top",
            ha="left", family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))


def plot_clusters(
    df, res: ClusterResult, out_path: Path, *, extent=None,
    voronoi: LandmassVoronoi | None = None,
) -> Path:
    fig = plt.figure(figsize=(17, 7.5))
    gs = GridSpec(2, 2, width_ratios=[2.4, 1.0], figure=fig, wspace=0.18, hspace=0.32)
    ax_map = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    _plot_map(ax_map, df, res, extent=extent, voronoi=voronoi)
    _plot_size_hist(fig.add_subplot(gs[0, 1]), res, len(df))
    _plot_radius_cdf(fig.add_subplot(gs[1, 1]), res)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--clusters-dir", type=Path,
        default=Path("datasets/ripe_atlas/asn_corpora/clusters"),
        help="cluster_ground_truth results dir (clusters.csv + assignments.csv "
             "+ meta.json).",
    )
    parser.add_argument("--radius-km", type=float, default=None,
                        help="Override the region radius R for footprints/cap line "
                             "(default: read from meta.json).")
    parser.add_argument(
        "--extent", type=float, nargs=4, default=None,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Optional map extent; default is global, or the landmass bounds "
             "when --landmass is given.",
    )
    parser.add_argument(
        "--landmass", type=str, default=None,
        help="Draw the nearest-centroid answer-space partition (Voronoi of the "
             "cluster centroids) clipped to this landmass as a background layer. "
             "Accepts a continent ('Europe', 'North America') or a country "
             "code/name ('US', 'USA', 'France').",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "ground_truth_clusters.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("clusters.csv", "assignments.csv"):
        if not (args.clusters_dir / name).exists():
            raise SystemExit(
                f"{args.clusters_dir / name} not found — run "
                "`python -m scripts.benchmark.v2.sources.cluster_ground_truth` first."
            )

    df, res = load_clustering(args.clusters_dir, radius_override=args.radius_km)

    voronoi = None
    extent = args.extent
    if args.landmass:
        clusters = pd.read_csv(args.clusters_dir / "clusters.csv")
        voronoi = build_landmass_voronoi(clusters, args.landmass)
        if extent is None:
            extent = voronoi.focus_extent()

    out = plot_clusters(df, res, args.out, extent=extent, voronoi=voronoi)

    n = len(df)
    print(f"{n:,} coords → {res.n_clusters:,} regions (cap {res.radius_target_km:.0f} km)")
    if voronoi is not None:
        print(f"  partition: {len(voronoi.cells):,} Voronoi cells over {voronoi.label}")
    print(f"  singletons {int((res.member_counts == 1).sum()):,} "
          f"({100 * (res.member_counts == 1).mean():.1f}% of regions)")
    print(f"  radius_km: max {res.radius_km.max():.1f}, "
          f"p95 {np.percentile(res.radius_km, 95):.1f}, median {np.median(res.radius_km):.1f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
