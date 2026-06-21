"""Map + stats for the ground-truth answer-space clustering.

Visualizes the regions produced by
`scripts.benchmark.v2.sources.cluster_ground_truth` (deterministic
complete-linkage agglomerative clustering, diameter capped at ``2R``): every
ground-truth coordinate is grouped into a coherent region, and each region's
spherical centroid is one point of the CBG classification answer space.

Left panel — a PlateCarree map: member coordinates colored by region (singletons
greyed), with an ``R``-radius geodesic circle drawn around each multi-member
region so the "coherent region" footprint is literal. Right column — the
distributions that characterize the answer space: region-size histogram (with
the singleton share called out) and the per-region centroid-radius CDF (the
scoring-relevant tightness, against the ``R`` line).

Clustering is recomputed from coordinates here (importing
`cluster_ground_truth`) so the figure always matches the requested radius — no
dependency on a pre-written clusters.csv.

CLI::

    python -m scripts.visualization.cluster.plot_ground_truth_clusters \\
        --targets datasets/ripe_atlas/asn_corpora/targets.csv --radius-km 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter

from scripts.benchmark.v2.sources.cluster_ground_truth import (
    ClusterResult,
    _load_targets,
    cluster_ground_truth,
)
from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM

_DEFAULT_RADIUS_KM = 50.0
_RADIUS_PCTILES = (50, 75, 90, 95, 99)


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


def _plot_map(ax, df, res: ClusterResult, *, extent) -> None:
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="#eaf2f8")
    ax.add_feature(cfeature.LAND, facecolor="#f6f4ef")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#999999")
    ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="#cccccc")

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
    ax.set_title(
        f"Answer space: {len(df):,} coords → {res.n_clusters:,} regions "
        f"(≤{res.radius_target_km:.0f} km)",
        fontsize=12,
    )


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


def plot_clusters(df, res: ClusterResult, out_path: Path, *, extent=None) -> Path:
    fig = plt.figure(figsize=(17, 7.5))
    gs = GridSpec(2, 2, width_ratios=[2.4, 1.0], figure=fig, wspace=0.18, hspace=0.32)
    ax_map = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    _plot_map(ax_map, df, res, extent=extent)
    _plot_size_hist(fig.add_subplot(gs[0, 1]), res, len(df))
    _plot_radius_cdf(fig.add_subplot(gs[1, 1]), res)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--targets", type=Path,
        default=Path("datasets/ripe_atlas/asn_corpora/targets.csv"),
        help="Ground-truth file (csv/json) with target_id/target_lat/target_lon.",
    )
    parser.add_argument("--radius-km", type=float, default=_DEFAULT_RADIUS_KM,
                        help="Region radius R; complete-linkage caps diameter at 2R. Default 50.")
    parser.add_argument(
        "--extent", type=float, nargs=4, default=None,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Optional map extent; default is global.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "ground_truth_clusters.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    if not args.targets.exists():
        raise SystemExit(f"targets file not found at {args.targets}")

    df = _load_targets(args.targets)
    res = cluster_ground_truth(
        df["target_lat"].to_numpy(), df["target_lon"].to_numpy(),
        radius_km=args.radius_km,
    )
    out = plot_clusters(df, res, args.out, extent=args.extent)

    n = len(df)
    print(f"{n:,} coords → {res.n_clusters:,} regions (cap {args.radius_km:.0f} km)")
    print(f"  singletons {int((res.member_counts == 1).sum()):,} "
          f"({100 * (res.member_counts == 1).mean():.1f}% of regions)")
    print(f"  radius_km: max {res.radius_km.max():.1f}, "
          f"p95 {np.percentile(res.radius_km, 95):.1f}, median {np.median(res.radius_km):.1f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
