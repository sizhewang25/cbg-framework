"""
Visualize the full Octant geolocation pipeline on real measurement data.

For each randomly-selected target probe, produces a map showing:
  - Annular constraints (one colored ring per landmark)
  - Feasible region overlay
  - Monte Carlo sampled points (colored by sum-of-distances weight)
  - Geometric median estimate (blue diamond)
  - True location (red star)
  - Error annotation

Usage:
    python -m scripts.libs.octant.visualize_octant_geolocation
    python -m scripts.libs.octant.visualize_octant_geolocation --asn 7922 --n-targets 10 --seed 42
    python -m scripts.libs.octant.visualize_octant_geolocation --output-dir /tmp/octant_viz
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon, MultiPolygon

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.libs.cbg_feasibility.rtt_model import haversine_distance
from scripts.libs.octant.octant_evaluation import fit_octant_models
from scripts.libs.octant.octant_geolocation import (
    AnnularConstraint,
    form_constraints,
    compute_feasible_region_unweighted,
    compute_feasible_region_weighted,
    sample_points_in_region,
    geometric_median_approx,
    _weighted_centroid_fallback,
)
from scripts.libs.octant.visualize_monte_carlo import COLORS_BLIND
from scripts.utils.helpers import haversine

PC = ccrs.PlateCarree()  # shorthand for the geographic CRS transform

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "octant_geolocation"


# =============================================================================
# Data types
# =============================================================================

@dataclass
class TargetVizData:
    """All intermediate artifacts for one target, retained for plotting."""
    probe_ip: str
    true_lat: float
    true_lon: float
    constraints: List[AnnularConstraint]
    region: Any  # Shapely geometry or None (weighted feasible region, used for MC sampling)
    sampled_points: np.ndarray  # shape (n, 2), columns [lat, lon]
    est_lat: float
    est_lon: float
    error_km: float
    method_used: str
    region_area_km2: float


# =============================================================================
# Data loading
# =============================================================================

def load_and_prepare_data(
    asn: int = 7922,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """Load Vultr dataset, filter by ASN, compute distances."""
    data_path = PROJECT_ROOT / "datasets" / "cbg_test" / "vultr_pings_us_only.csv"
    df = pd.read_csv(data_path)
    df_asn = df[df["probe_asn"] == float(asn)].copy()

    if df_asn.empty:
        raise ValueError(f"No data for ASN {asn} in {data_path}")

    df_asn["distance_km"] = df_asn.apply(
        lambda row: haversine_distance(
            row["probe_latitude"],
            row["probe_longitude"],
            row["anchor_latitude"],
            row["anchor_longitude"],
        ),
        axis=1,
    )

    anchors = df_asn[["dst_ip", "anchor_latitude", "anchor_longitude"]].drop_duplicates()
    anchor_coords: Dict[str, Tuple[float, float]] = {}
    for _, row in anchors.iterrows():
        anchor_coords[row["dst_ip"]] = (row["anchor_latitude"], row["anchor_longitude"])

    return df_asn, anchor_coords


# =============================================================================
# Per-target geolocation with retained intermediates
# =============================================================================

def geolocate_target_with_artifacts(
    probe_ip: str,
    probe_data: pd.DataFrame,
    anchor_coords: Dict[str, Tuple[float, float]],
    models: Dict[str, Any],
    delta: Optional[float],
    method: str = "weighted",
    n_samples: int = 3000,
    rng: Optional[np.random.Generator] = None,
) -> Optional[TargetVizData]:
    """Run full Octant pipeline for one target, retaining all intermediates."""
    true_lat = float(probe_data["probe_latitude"].iloc[0])
    true_lon = float(probe_data["probe_longitude"].iloc[0])

    # Collect RTT measurements
    rtt_measurements: Dict[str, float] = {}
    for _, row in probe_data.iterrows():
        anchor_ip = row["dst_ip"]
        if anchor_ip in models and models[anchor_ip].fitted:
            rtt_measurements[anchor_ip] = row["min_rtt"]

    if not rtt_measurements:
        return None

    # Form constraints
    constraints = form_constraints(
        probe_ip, rtt_measurements, anchor_coords, models, delta=delta,
    )
    if not constraints:
        return None

    # Compute feasible region — strict geometric intersection of annuli
    region = compute_feasible_region_unweighted(constraints, n_pts=128)
    method_used = "unweighted" if region is not None else method

    # Sample and estimate
    points = np.empty((0, 2))
    region_area_km2 = 0.0

    if region is not None and not region.is_empty:
        centroid = region.centroid
        center_lat = centroid.y
        km_per_deg_lat = 111.0
        km_per_deg_lon = max(111.0 * np.cos(np.radians(center_lat)), 1.0)
        region_area_km2 = region.area * km_per_deg_lat * km_per_deg_lon

        points = sample_points_in_region(region, n_samples=n_samples, rng=rng)

    if len(points) >= 2:
        est_lat, est_lon = geometric_median_approx(points)
    elif region is not None and not region.is_empty:
        est_lat, est_lon = region.centroid.y, region.centroid.x
        method_used = "region_centroid"
    else:
        est_lat, est_lon = _weighted_centroid_fallback(constraints)
        method_used = "centroid_fallback"

    error_km = haversine((est_lat, est_lon), (true_lat, true_lon))

    return TargetVizData(
        probe_ip=probe_ip,
        true_lat=true_lat,
        true_lon=true_lon,
        constraints=constraints,
        region=region,
        sampled_points=points,
        est_lat=est_lat,
        est_lon=est_lon,
        error_km=error_km,
        method_used=method_used,
        region_area_km2=region_area_km2,
    )


# =============================================================================
# Cartopy-aware plotting helpers
# =============================================================================

def _draw_annulus_on_map(ax, constraint, color, alpha=0.08, label=None):
    """Draw an annulus on a Cartopy GeoAxes (PlateCarree coordinates)."""
    km_per_deg_lat = 111.0
    km_per_deg_lon = max(111.0 * np.cos(np.radians(constraint.landmark_lat)), 1.0)
    angles = np.linspace(0, 2 * np.pi, 200)

    # Outer circle
    outer_lon = constraint.outer_radius_km / km_per_deg_lon
    outer_lat = constraint.outer_radius_km / km_per_deg_lat
    ox = constraint.landmark_lon + outer_lon * np.cos(angles)
    oy = constraint.landmark_lat + outer_lat * np.sin(angles)
    ax.plot(ox, oy, color=color, linewidth=1.2, alpha=0.6, linestyle="-",
            transform=PC)
    ax.fill(ox, oy, color=color, alpha=alpha, label=label, transform=PC)

    # Inner circle
    if constraint.inner_radius_km > 0:
        inner_lon = constraint.inner_radius_km / km_per_deg_lon
        inner_lat = constraint.inner_radius_km / km_per_deg_lat
        ix = constraint.landmark_lon + inner_lon * np.cos(angles)
        iy = constraint.landmark_lat + inner_lat * np.sin(angles)
        ax.plot(ix, iy, color=color, linewidth=1.0, alpha=0.5, linestyle="--",
                transform=PC)
        ax.fill(ix, iy, color="white", alpha=0.7, transform=PC)


def _plot_region_on_map(ax, region, facecolor=("yellow", 0.4),
                        edgecolor=("orange", 1.0), label=None):
    """Plot a Shapely geometry on a Cartopy GeoAxes.

    facecolor / edgecolor: (color, alpha) tuple or a plain color value.
    """
    if region is None or region.is_empty:
        return

    # Unpack (color, alpha) tuples
    if isinstance(facecolor, tuple) and len(facecolor) == 2:
        fc, fa = facecolor
    else:
        fc, fa = facecolor, 0.15
    if isinstance(edgecolor, tuple) and len(edgecolor) == 2:
        ec, ea = edgecolor
    else:
        ec, ea = edgecolor, 1.0

    geoms = []
    if isinstance(region, Polygon):
        geoms = [region]
    elif isinstance(region, MultiPolygon):
        geoms = list(region.geoms)
    else:
        for g in region.geoms:
            if isinstance(g, (Polygon, MultiPolygon)):
                geoms.append(g)

    for i, geom in enumerate(geoms):
        xs, ys = geom.exterior.xy
        ax.fill(
            list(xs), list(ys),
            color=fc, alpha=fa,
            edgecolor=ec, linewidth=2,
            label=label if i == 0 else None,
            transform=PC, zorder=3,
        )
        ax.plot(
            list(xs), list(ys),
            color=ec, alpha=ea, linewidth=2,
            transform=PC, zorder=4,
        )
        for interior in geom.interiors:
            xs, ys = interior.xy
            ax.fill(list(xs), list(ys), color="white", alpha=0.8,
                    transform=PC, zorder=3)


def _add_map_features(ax, compact=False):
    """Add geographic context to a Cartopy GeoAxes."""
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff")
    ax.add_feature(cfeature.STATES, linewidth=0.3 if compact else 0.5,
                   edgecolor="gray")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5 if compact else 0.8)
    ax.coastlines(resolution="50m", linewidth=0.5 if compact else 0.8)


# =============================================================================
# Main per-target plotting
# =============================================================================

def plot_target_map(
    ax,
    data: TargetVizData,
    show_mc_points: bool = True,
    compact: bool = False,
) -> None:
    """Plot a single target's geolocation on a Cartopy GeoAxes."""
    label_size = 7 if compact else 9
    marker_scale = 0.7 if compact else 1.0

    # Always show the full contiguous US for consistent context
    ax.set_extent([-130, -64, 24, 55], crs=PC)

    _add_map_features(ax, compact=compact)

    # Draw annuli
    for i, c in enumerate(data.constraints):
        color = COLORS_BLIND[i % len(COLORS_BLIND)]
        _draw_annulus_on_map(
            ax, c, color=color, alpha=0.06,
            label=f"{c.landmark_ip} (r={c.inner_radius_km:.0f}, "
                  f"R={c.outer_radius_km:.0f} km)",
        )
        ax.plot(
            c.landmark_lon, c.landmark_lat,
            marker="^", color=color, markersize=8 * marker_scale,
            markeredgecolor="black", markeredgewidth=0.5, zorder=5,
            transform=PC,
        )

    # Strict annulus intersection region
    if data.region is not None and not data.region.is_empty:
        _plot_region_on_map(
            ax, data.region,
            facecolor=("yellow", 0.4),
            edgecolor=("orange", 1.0),
            label="Annulus intersection",
        )

    # # MC scatter (temporarily disabled)
    # if show_mc_points and len(data.sampled_points) >= 2:
    #     lats, lons = data.sampled_points[:, 0], data.sampled_points[:, 1]
    #
    #     # Color by sum-of-distances
    #     lat_rad = np.radians(lats)
    #     lon_rad = np.radians(lons)
    #     dlat = lat_rad[:, None] - lat_rad[None, :]
    #     dlon = lon_rad[:, None] - lon_rad[None, :]
    #     a = (
    #         np.sin(dlat / 2) ** 2
    #         + np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :])
    #         * np.sin(dlon / 2) ** 2
    #     )
    #     dist_matrix = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    #     sum_dists = dist_matrix.sum(axis=1)
    #     norm_w = sum_dists - sum_dists.min()
    #     if norm_w.max() > 0:
    #         norm_w /= norm_w.max()
    #
    #     sc = ax.scatter(
    #         lons, lats, c=norm_w, cmap="RdYlGn_r",
    #         s=4 if compact else 8, alpha=0.6, edgecolors="none", zorder=4,
    #         transform=PC,
    #     )
    #     if not compact:
    #         cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    #         cbar.set_label("Normalized \u03a3 distances\n(lower = better)",
    #                        fontsize=8)

    # Geometric median estimate
    ax.plot(
        data.est_lon, data.est_lat,
        marker="D", color="blue", markersize=10 * marker_scale,
        markeredgecolor="white", markeredgewidth=1.2, zorder=10,
        label="Geometric median", transform=PC,
    )

    # True location
    ax.plot(
        data.true_lon, data.true_lat,
        marker="*", color="red", markersize=14 * marker_scale,
        markeredgecolor="black", markeredgewidth=0.8, zorder=10,
        label="True location", transform=PC,
    )

    # Error annotation
    ax.annotate(
        f"Error: {data.error_km:.0f} km\nMethod: {data.method_used}\n"
        f"Constraints: {len(data.constraints)}\n"
        f"Area: {data.region_area_km2:.0f} km\u00b2",
        xy=(0.02, 0.98), xycoords="axes fraction", fontsize=label_size,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        zorder=11,
    )

    ax.legend(fontsize=label_size - 1, loc="lower left")


# =============================================================================
# Main orchestration
# =============================================================================

def visualize_targets(
    asn: int = 7922,
    n_targets: int = 10,
    seed: int = 42,
    target_coverage: float = 0.80,
    cutoff_variant: str = "high_only",
    method: str = "weighted",
    n_samples: int = 3000,
    output_dir: Optional[Path] = None,
    grid_layout: bool = True,
) -> None:
    """Load data, fit models, select targets, and produce visualizations."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data for ASN {asn}...")
    df_asn, anchor_coords = load_and_prepare_data(asn)
    print(f"  {len(df_asn)} measurements, {df_asn['src_ip'].nunique()} probes, "
          f"{df_asn['dst_ip'].nunique()} anchors")

    # Fit models
    print(f"Fitting Octant models (variant={cutoff_variant}, coverage={target_coverage})...")
    models, delta = fit_octant_models(
        df_asn,
        target_coverage=target_coverage,
        cutoff_variant=cutoff_variant,
    )
    print(f"  Delta: {delta:.4f}" if delta else "  Delta: None (hull bounds only)")

    # Select targets
    probe_ips = df_asn["src_ip"].unique()
    rng = np.random.default_rng(seed)
    n_select = min(n_targets, len(probe_ips))
    selected = rng.choice(probe_ips, size=n_select, replace=False)
    print(f"Selected {n_select} targets (seed={seed})")

    # Geolocate each target
    viz_data: List[TargetVizData] = []
    for probe_ip in selected:
        probe_data = df_asn[df_asn["src_ip"] == probe_ip]
        result = geolocate_target_with_artifacts(
            probe_ip, probe_data, anchor_coords, models, delta,
            method=method, n_samples=n_samples, rng=rng,
        )
        if result is not None:
            viz_data.append(result)
            print(f"  {probe_ip}: error={result.error_km:.0f} km, "
                  f"method={result.method_used}, "
                  f"constraints={len(result.constraints)}")
        else:
            print(f"  {probe_ip}: skipped (no constraints)")

    if not viz_data:
        print("No targets produced results. Exiting.")
        return

    # Projection for all maps
    proj = ccrs.LambertConformal(central_longitude=-96, central_latitude=39)

    # Individual per-target figures
    for d in viz_data:
        fig, ax = plt.subplots(1, 1, figsize=(12, 9),
                               subplot_kw={"projection": proj})
        fig.suptitle(
            f"Octant Geolocation: {d.probe_ip}",
            fontsize=14, fontweight="bold",
        )
        plot_target_map(ax, d, show_mc_points=True, compact=False)
        safe_ip = d.probe_ip.replace(".", "_")
        out_path = output_dir / f"target_{safe_ip}.pdf"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # Grid summary figure
    if grid_layout and len(viz_data) > 1:
        n = len(viz_data)
        cols = min(5, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(
            rows, cols, figsize=(5 * cols, 5 * rows),
            subplot_kw={"projection": proj},
        )
        fig.suptitle(
            f"Octant Geolocation — ASN {asn}, {n} targets",
            fontsize=16, fontweight="bold", y=0.99,
        )
        axes_flat = np.asarray(axes).flatten()

        for i, d in enumerate(viz_data):
            ax = axes_flat[i]
            ax.set_title(
                f"{d.probe_ip}\nerr={d.error_km:.0f} km",
                fontsize=8,
            )
            plot_target_map(ax, d, show_mc_points=True, compact=True)

        # Hide unused axes
        for j in range(len(viz_data), len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        grid_path = output_dir / "octant_geolocation_grid.pdf"
        fig.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved grid: {grid_path}")

    # Summary table
    print("\n--- Summary ---")
    print(f"{'Probe IP':<20} {'Error (km)':>10} {'Method':<22} {'Constraints':>11} {'Area (km2)':>10}")
    for d in viz_data:
        print(f"{d.probe_ip:<20} {d.error_km:>10.0f} {d.method_used:<22} "
              f"{len(d.constraints):>11} {d.region_area_km2:>10.0f}")

    errors = [d.error_km for d in viz_data]
    print(f"\nMedian error: {np.median(errors):.0f} km")
    print(f"Mean error:   {np.mean(errors):.0f} km")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Octant geolocation pipeline on real measurement data",
    )
    parser.add_argument("--asn", type=int, default=7922,
                        help="Probe ASN to filter (default: 7922)")
    parser.add_argument("--n-targets", type=int, default=10,
                        help="Number of random targets to visualize (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for target selection (default: 42)")
    parser.add_argument("--target-coverage", type=float, default=0.80,
                        help="Spline delta coverage target (default: 0.80)")
    parser.add_argument("--cutoff-variant", default="high_only",
                        choices=["none", "high_only", "low_only", "both"],
                        help="Cutoff variant for model fitting")
    parser.add_argument("--method", default="weighted",
                        choices=["weighted", "unweighted"],
                        help="Feasible region method")
    parser.add_argument("--n-samples", type=int, default=3000,
                        help="Monte Carlo samples per target")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for figures")
    parser.add_argument("--no-grid", action="store_true",
                        help="Skip summary grid figure")
    args = parser.parse_args()

    visualize_targets(
        asn=args.asn,
        n_targets=args.n_targets,
        seed=args.seed,
        target_coverage=args.target_coverage,
        cutoff_variant=args.cutoff_variant,
        method=args.method,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        grid_layout=not args.no_grid,
    )


if __name__ == "__main__":
    main()
