"""
Visualize Monte Carlo Point Selection on Octant Feasible Regions

Produces a multi-panel figure showing:
  (a) Annular constraints from landmarks + their intersection region
  (b) Monte Carlo sampled points colored by sum-of-distances weight
  (c) Convergence: how the geometric median estimate stabilizes with n_samples

Usage:
    python -m scripts.libs.octant.visualize_monte_carlo          # built-in demo
    python -m scripts.libs.octant.visualize_monte_carlo --target pittsburgh
"""

import argparse
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Point

from scripts.libs.octant.octant_model import OctantRTTModel
from scripts.libs.octant.octant_geolocation import (
    AnnularConstraint,
    form_constraints,
    compute_feasible_region_unweighted,
    compute_feasible_region_weighted,
    sample_points_in_region,
    geometric_median_approx,
    _circle_to_shapely,
    _haversine_vectorized,
)
from scripts.utils.helpers import haversine

# Try descartes for polygon rendering; if unavailable use manual fallback
try:
    from descartes import PolygonPatch
    HAS_DESCARTES = True
except ImportError:
    HAS_DESCARTES = False


# =============================================================================
# Plotting helpers
# =============================================================================

COLORS_BLIND = [
    (0, 114/255, 178/255),       # blue
    (230/255, 159/255, 0),       # orange
    (204/255, 121/255, 167/255), # reddish purple
    (0, 158/255, 115/255),       # bluish green
    (86/255, 180/255, 233/255),  # sky blue
    (213/255, 94/255, 0),        # vermilion
]


def _plot_region(ax, region, facecolor=(0.2, 0.4, 0.8, 0.15),
                 edgecolor=(0.1, 0.2, 0.6, 0.6), label=None):
    """Plot a Shapely geometry on a matplotlib axes."""
    if region is None or region.is_empty:
        return

    from shapely.geometry import MultiPolygon, Polygon
    geoms = []
    if isinstance(region, Polygon):
        geoms = [region]
    elif isinstance(region, MultiPolygon):
        geoms = list(region.geoms)
    else:
        # GeometryCollection or similar
        for g in region.geoms:
            if isinstance(g, (Polygon, MultiPolygon)):
                geoms.append(g)

    for i, geom in enumerate(geoms):
        if HAS_DESCARTES:
            patch = PolygonPatch(
                geom, facecolor=facecolor, edgecolor=edgecolor,
                linewidth=1.5, label=label if i == 0 else None,
            )
            ax.add_patch(patch)
        else:
            # Manual fallback: extract exterior coords
            xs, ys = geom.exterior.xy
            ax.fill(xs, ys, alpha=0.15, color=facecolor[:3] if len(facecolor) > 3 else facecolor,
                    edgecolor=edgecolor[:3] if len(edgecolor) > 3 else edgecolor,
                    linewidth=1.5, label=label if i == 0 else None)
            for interior in geom.interiors:
                xs, ys = interior.xy
                ax.fill(xs, ys, color='white', alpha=0.8)


def _draw_annulus(ax, constraint, color, alpha=0.08, label=None):
    """Draw an annulus (outer - inner circle) for a constraint."""
    km_per_deg_lat = 111.0
    km_per_deg_lon = max(111.0 * np.cos(np.radians(constraint.landmark_lat)), 1.0)

    # Outer circle
    outer_lon = constraint.outer_radius_km / km_per_deg_lon
    outer_lat = constraint.outer_radius_km / km_per_deg_lat
    angles = np.linspace(0, 2 * np.pi, 200)
    outer_x = constraint.landmark_lon + outer_lon * np.cos(angles)
    outer_y = constraint.landmark_lat + outer_lat * np.sin(angles)
    ax.plot(outer_x, outer_y, color=color, linewidth=1.2, alpha=0.6, linestyle='-')
    ax.fill(outer_x, outer_y, color=color, alpha=alpha, label=label)

    # Inner circle
    if constraint.inner_radius_km > 0:
        inner_lon = constraint.inner_radius_km / km_per_deg_lon
        inner_lat = constraint.inner_radius_km / km_per_deg_lat
        inner_x = constraint.landmark_lon + inner_lon * np.cos(angles)
        inner_y = constraint.landmark_lat + inner_lat * np.sin(angles)
        ax.plot(inner_x, inner_y, color=color, linewidth=1.0, alpha=0.5, linestyle='--')
        ax.fill(inner_x, inner_y, color='white', alpha=0.7)


# =============================================================================
# Demo data
# =============================================================================

DEMO_TARGETS = {
    'pittsburgh': (40.4406, -79.9959),
    'atlanta':    (33.7490, -84.3880),
    'denver':     (39.7392, -104.9903),
    'nashville':  (36.1627, -86.7816),
}

LANDMARKS = {
    'lm_nyc': (40.7128, -74.0060),
    'lm_chi': (41.8781, -87.6298),
    'lm_la':  (34.0522, -118.2437),
    'lm_hou': (29.7604, -95.3698),
    'lm_sea': (47.6062, -122.3321),
}


def _make_demo_model(ip, lat, lon):
    rng = np.random.RandomState(hash(ip) % 2**31)
    rtts = np.linspace(5, 100, 60)
    distances = 100.0 * rtts + rng.uniform(-150, 150, len(rtts))
    distances = np.maximum(distances, 10.0)
    model = OctantRTTModel(anchor_ip=ip, anchor_lat=lat, anchor_lon=lon)
    model.fit(rtts, distances)
    return model


def _synthetic_rtt(true_lat, true_lon, lm_lat, lm_lon, noise_ms=2.0):
    dist_km = haversine((true_lat, true_lon), (lm_lat, lm_lon))
    rtt = 2 * dist_km / 200.0  # 200 km/ms ≈ 2/3 c
    rng = np.random.RandomState(int(abs(true_lat * 1000 + lm_lat * 100)) % 2**31)
    return max(1.0, rtt + rng.uniform(0, noise_ms))


# =============================================================================
# Main visualization
# =============================================================================

def visualize_monte_carlo(
    target_name: str = 'pittsburgh',
    n_samples: int = 3000,
    method: str = 'unweighted',
    seed: int = 42,
    output_path: str = None,
):
    """Generate 3-panel Monte Carlo visualization.

    Panel (a): Annular constraints + feasible region + true location
    Panel (b): Sampled points colored by sum-of-distances + geometric median
    Panel (c): Convergence curve — error vs n_samples
    """
    true_lat, true_lon = DEMO_TARGETS[target_name]
    models = {ip: _make_demo_model(ip, lat, lon) for ip, (lat, lon) in LANDMARKS.items()}
    rtts = {
        ip: _synthetic_rtt(true_lat, true_lon, lat, lon)
        for ip, (lat, lon) in LANDMARKS.items()
    }
    constraints = form_constraints(
        target_name, rtts, LANDMARKS, models, max_rtt_ms=200.0
    )

    if not constraints:
        print(f"No constraints formed for {target_name}. Check models.")
        return

    # Compute region
    if method == 'weighted':
        region = compute_feasible_region_weighted(
            constraints, weight_threshold=0.4, grid_resolution_deg=0.15
        )
    else:
        region = compute_feasible_region_unweighted(constraints, n_pts=128)

    rng = np.random.default_rng(seed)

    # Sample points
    if region is not None and not region.is_empty:
        points = sample_points_in_region(region, n_samples=n_samples, rng=rng)
    else:
        print("Feasible region is empty — showing fallback centroid only.")
        points = np.empty((0, 2))

    # =========================================================================
    # Figure
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5))
    fig.suptitle(
        f'Octant Monte Carlo Point Selection — target: {target_name.title()}',
        fontsize=18, fontweight='bold', y=0.98,
    )

    # --- Panel (a): Constraints + Region ---
    ax = axes[0]
    ax.set_title('(a) Annular Constraints & Feasible Region', fontsize=13)

    for i, c in enumerate(constraints):
        color = COLORS_BLIND[i % len(COLORS_BLIND)]
        dist_km = haversine((true_lat, true_lon), (c.landmark_lat, c.landmark_lon))
        _draw_annulus(ax, c, color=color,
                      label=f'{c.landmark_ip} (r={c.inner_radius_km:.0f}, R={c.outer_radius_km:.0f} km)')
        ax.plot(c.landmark_lon, c.landmark_lat, marker='^', color=color,
                markersize=9, markeredgecolor='black', markeredgewidth=0.5, zorder=5)

    if region is not None:
        _plot_region(ax, region, label='Feasible region')

    ax.plot(true_lon, true_lat, marker='*', color='red', markersize=16,
            markeredgecolor='black', markeredgewidth=0.8, zorder=10, label='True location')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_aspect('equal')

    # Zoom to region if available
    if region is not None and not region.is_empty:
        minx, miny, maxx, maxy = region.bounds
        pad = max(maxx - minx, maxy - miny) * 0.3
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)

    # --- Panel (b): Monte Carlo samples + geometric median ---
    ax = axes[1]
    ax.set_title('(b) Monte Carlo Samples & Geometric Median', fontsize=13)

    if len(points) >= 2:
        # Compute sum-of-distances for coloring
        lats, lons = points[:, 0], points[:, 1]
        lat_rad = np.radians(lats)
        lon_rad = np.radians(lons)
        dlat = lat_rad[:, None] - lat_rad[None, :]
        dlon = lon_rad[:, None] - lon_rad[None, :]
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) *
             np.sin(dlon / 2) ** 2)
        dist_matrix = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        sum_dists = dist_matrix.sum(axis=1)

        # Normalize for colormap
        norm_weights = (sum_dists - sum_dists.min())
        if norm_weights.max() > 0:
            norm_weights /= norm_weights.max()

        sc = ax.scatter(
            lons, lats, c=norm_weights, cmap='RdYlGn_r',
            s=8, alpha=0.6, edgecolors='none', zorder=3,
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('Normalized Σ distances\n(lower = better)', fontsize=9)

        # Geometric median
        med_lat, med_lon = geometric_median_approx(points)
        ax.plot(med_lon, med_lat, marker='D', color='blue', markersize=12,
                markeredgecolor='white', markeredgewidth=1.5, zorder=10,
                label=f'Geometric median')

        # Centroid for comparison
        centroid_lat, centroid_lon = lats.mean(), lons.mean()
        ax.plot(centroid_lon, centroid_lat, marker='o', color='orange', markersize=10,
                markeredgecolor='white', markeredgewidth=1.2, zorder=9,
                label='Arithmetic centroid')

        # True location
        ax.plot(true_lon, true_lat, marker='*', color='red', markersize=16,
                markeredgecolor='black', markeredgewidth=0.8, zorder=10,
                label='True location')

        # Errors
        err_median = haversine((med_lat, med_lon), (true_lat, true_lon))
        err_centroid = haversine((centroid_lat, centroid_lon), (true_lat, true_lon))
        ax.annotate(
            f'Median err: {err_median:.0f} km\nCentroid err: {err_centroid:.0f} km',
            xy=(0.02, 0.98), xycoords='axes fraction', fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
        )

        # Region outline
        if region is not None:
            _plot_region(ax, region,
                         facecolor=(0.5, 0.5, 0.5, 0.05),
                         edgecolor=(0.3, 0.3, 0.3, 0.4))

    else:
        ax.text(0.5, 0.5, 'No feasible region\n(fallback triggered)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(fontsize=9, loc='lower left')
    ax.set_aspect('equal')

    if region is not None and not region.is_empty:
        minx, miny, maxx, maxy = region.bounds
        pad = max(maxx - minx, maxy - miny) * 0.3
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)

    # --- Panel (c): Convergence curve ---
    ax = axes[2]
    ax.set_title('(c) Estimate Convergence vs. Sample Size', fontsize=13)

    if len(points) >= 10:
        checkpoints = np.unique(np.geomspace(10, len(points), num=30, dtype=int))
        errors_at_n = []
        median_lats = []
        median_lons = []

        for n in checkpoints:
            subset = points[:n]
            if len(subset) < 2:
                continue
            m_lat, m_lon = geometric_median_approx(subset)
            err = haversine((m_lat, m_lon), (true_lat, true_lon))
            errors_at_n.append(err)
            median_lats.append(m_lat)
            median_lons.append(m_lon)

        ax.plot(checkpoints[:len(errors_at_n)], errors_at_n,
                color=COLORS_BLIND[0], linewidth=2, marker='o', markersize=4,
                label='Geometric median error')

        # Centroid convergence for comparison
        centroid_errors = []
        for n in checkpoints:
            subset = points[:n]
            c_lat, c_lon = subset[:, 0].mean(), subset[:, 1].mean()
            err = haversine((c_lat, c_lon), (true_lat, true_lon))
            centroid_errors.append(err)

        ax.plot(checkpoints[:len(centroid_errors)], centroid_errors,
                color=COLORS_BLIND[1], linewidth=2, marker='s', markersize=4,
                linestyle='--', label='Centroid error')

        ax.set_xlabel('Number of samples')
        ax.set_ylabel('Error (km)')
        ax.set_xscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Annotate final values
        if errors_at_n:
            ax.axhline(y=errors_at_n[-1], color=COLORS_BLIND[0],
                        linestyle=':', alpha=0.4)
            ax.annotate(
                f'Final: {errors_at_n[-1]:.0f} km',
                xy=(checkpoints[len(errors_at_n) - 1], errors_at_n[-1]),
                fontsize=9, color=COLORS_BLIND[0],
                xytext=(10, 10), textcoords='offset points',
            )
    else:
        ax.text(0.5, 0.5, 'Insufficient samples\nfor convergence plot',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path is None:
        output_path = f'scripts/libs/octant/figures/monte_carlo_{target_name}.pdf'

    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)

    # Print summary
    if len(points) >= 2:
        med_lat, med_lon = geometric_median_approx(points)
        err = haversine((med_lat, med_lon), (true_lat, true_lon))
        print(f"Target: {target_name} at ({true_lat:.4f}, {true_lon:.4f})")
        print(f"Constraints: {len(constraints)}")
        print(f"Samples: {len(points)}")
        print(f"Geometric median: ({med_lat:.4f}, {med_lon:.4f})")
        print(f"Error: {err:.1f} km")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize Octant Monte Carlo point selection'
    )
    parser.add_argument(
        '--target', default='pittsburgh',
        choices=list(DEMO_TARGETS.keys()),
        help='Demo target city',
    )
    parser.add_argument('--n-samples', type=int, default=3000)
    parser.add_argument('--method', default='unweighted',
                        choices=['unweighted', 'weighted'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default=None, help='Output file path')
    parser.add_argument(
        '--all', action='store_true',
        help='Run for all demo targets',
    )

    args = parser.parse_args()

    if args.all:
        for target in DEMO_TARGETS:
            visualize_monte_carlo(
                target_name=target,
                n_samples=args.n_samples,
                method=args.method,
                seed=args.seed,
            )
    else:
        visualize_monte_carlo(
            target_name=args.target,
            n_samples=args.n_samples,
            method=args.method,
            seed=args.seed,
            output_path=args.output,
        )


if __name__ == '__main__':
    main()
