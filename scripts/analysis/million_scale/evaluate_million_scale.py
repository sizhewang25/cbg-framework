"""
Compare CBG multilateration methods on Vultr US dataset.

Two variants:
1. Million-Scale CBG: Theoretical 2/3c model + spherical circle intersection + closest-VP fallback
2. Vanilla CBG: LP bestline model + spherical circle intersection + closest-VP fallback
   (isolates the impact of the RTT-distance model from intersection geometry)

Produces:
  - Comparative Error CDF plot (2 lines)
  - Per-anchor RTT-distance scatter plots with both model lines
  - Statistics table
"""

import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from functools import reduce
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon

# Add project root and cbg_feasibility to path (filter_demonstration.py uses relative imports)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CBG_DIR = PROJECT_ROOT / 'scripts' / 'analysis' / 'cbg_feasibility'
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CBG_DIR))

from scripts.utils.helpers import (
    haversine,
    circle_intersections,
    polygon_centroid,
    get_middle_intersection,
)
from scripts.analysis.cbg_feasibility.rtt_model import (
    RTTDistanceModel,
    haversine_distance,
    THEORETICAL_SLOPE,
    fit_bestline_lp,
)

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

ASN = 7922
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'comparison'


def _circle_to_shapely_polygon(clat, clon, radius_km, n_pts=100):
    """Convert a circle (center lat/lon, radius km) to a Shapely polygon in lon/lat space."""
    angles = np.linspace(0, 2 * np.pi, n_pts)
    r_deg_lat = radius_km / 111.0
    r_deg_lon = radius_km / (111.0 * math.cos(math.radians(clat)))
    lons = clon + r_deg_lon * np.cos(angles)
    lats = clat + r_deg_lat * np.sin(angles)
    return ShapelyPolygon(zip(lons, lats))  # Shapely uses (x, y) = (lon, lat)


def compute_intersection_area(circles_data):
    """Compute area (km²) of the intersection of circles [(lat, lon, radius_km), ...]."""
    if len(circles_data) < 2:
        return 0.0
    try:
        shapely_circles = [
            _circle_to_shapely_polygon(clat, clon, radius_km)
            for clat, clon, radius_km in circles_data
        ]
        shapely_circles = [p for p in shapely_circles if p.is_valid and not p.is_empty]
        if len(shapely_circles) < 2:
            return 0.0
        intersection_poly = reduce(lambda a, b: a.intersection(b), shapely_circles)
        if intersection_poly.is_empty or intersection_poly.geom_type not in ('Polygon', 'MultiPolygon'):
            return 0.0
        area_deg2 = intersection_poly.area
        minx, miny, maxx, maxy = intersection_poly.bounds
        mid_lat = (miny + maxy) / 2  # bounds: (minlon, minlat, maxlon, maxlat)
        area_km2 = area_deg2 * 111.0 * (111.0 * math.cos(math.radians(mid_lat)))
        return area_km2
    except Exception:
        return 0.0


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load and prepare the Vultr US dataset."""
    data_path = PROJECT_ROOT / 'datasets' / 'cbg_test' / 'vultr_pings_us_only.csv'
    df = pd.read_csv(data_path)

    df_asn = df[df['probe_asn'] == float(ASN)].copy()

    df_asn['distance_km'] = df_asn.apply(
        lambda row: haversine_distance(
            row['probe_latitude'], row['probe_longitude'],
            row['anchor_latitude'], row['anchor_longitude']
        ),
        axis=1
    )

    print(f"Total measurements: {len(df)}")
    print(f"AS{ASN} measurements: {len(df_asn)}")
    print(f"Unique anchors: {df_asn['dst_ip'].nunique()}")
    print(f"Unique probes: {df_asn['src_ip'].nunique()}")

    return df, df_asn


# =============================================================================
# Million-Scale CBG Evaluation
# =============================================================================

def run_million_scale_cbg(df_asn):
    """
    Run Million-Scale CBG (inlined from select_best_guess_centroid).

    - RTT → distance via rtt_to_km(rtt, speed_threshold=2/3)
    - Spherical circle intersection
    - Polygon centroid or closest-VP fallback
    """
    # Build anchor coordinate lookup
    anchors = df_asn[['dst_ip', 'anchor_latitude', 'anchor_longitude']].drop_duplicates()
    anchor_coords = {}
    for _, row in anchors.iterrows():
        anchor_coords[row['dst_ip']] = (row['anchor_latitude'], row['anchor_longitude'])

    probe_ips = df_asn['src_ip'].unique()
    results = []
    all_radii = []
    all_areas = []

    for probe_ip in probe_ips:
        probe_data = df_asn[df_asn['src_ip'] == probe_ip]
        true_lat = probe_data['probe_latitude'].iloc[0]
        true_lon = probe_data['probe_longitude'].iloc[0]

        circles = []
        min_rtt_per_vp_ip = {}
        for _, row in probe_data.iterrows():
            anchor_ip = row['dst_ip']
            rtt = row['min_rtt']
            if rtt > 100:
                continue
            if anchor_ip not in anchor_coords:
                continue
            lat, lon = anchor_coords[anchor_ip]
            min_rtt_per_vp_ip[anchor_ip] = rtt
            circles.append((lat, lon, rtt, None, None))

        # Compute radii: d = (2/3) * rtt * 300 / 2 = 100 * rtt
        radii_km = [100.0 * rtt for _, _, rtt, _, _ in circles]
        all_radii.extend(radii_km)

        if not circles:
            results.append({
                'probe_ip': probe_ip, 'true_lat': true_lat, 'true_lon': true_lon,
                'estimated_lat': None, 'estimated_lon': None,
                'error_km': None, 'n_anchors': 0,
                'method': 'million_scale_cbg', 'intersection': False,
                'avg_radius_km': None, 'intersection_area_km2': 0.0
            })
            continue

        intersections, circles_out = circle_intersections(circles, speed_threshold=2/3)
        circles_data_for_area = [(lat, lon, 100.0 * rtt) for lat, lon, rtt, _, _ in circles_out]
        area_km2 = compute_intersection_area(circles_data_for_area)
        all_areas.append(area_km2)

        if len(intersections) > 2:
            centroid = polygon_centroid(intersections)
            did_intersect = True
        elif len(intersections) == 2:
            centroid = get_middle_intersection(intersections)
            did_intersect = True
        else:
            closest_vp, _ = min(min_rtt_per_vp_ip.items(), key=lambda x: x[1])
            centroid = anchor_coords[closest_vp]
            did_intersect = False

        est_lat, est_lon = centroid
        error_km = haversine((est_lat, est_lon), (true_lat, true_lon))

        results.append({
            'probe_ip': probe_ip,
            'true_lat': true_lat,
            'true_lon': true_lon,
            'estimated_lat': float(est_lat),
            'estimated_lon': float(est_lon),
            'error_km': float(error_km),
            'n_anchors': len(circles_out),
            'method': 'million_scale_cbg',
            'intersection': did_intersect,
            'avg_radius_km': float(np.mean(radii_km)),
            'intersection_area_km2': area_km2
        })

    return results, np.array(all_radii), np.array(all_areas)


# =============================================================================
# LP Model Fitting
# =============================================================================

def fit_lp_models(df_asn):
    """Fit LP bestline models per anchor."""
    anchors = df_asn[['dst_ip', 'anchor_latitude', 'anchor_longitude', 'anchor_city']].drop_duplicates()
    models = {}

    for _, anchor in anchors.iterrows():
        anchor_ip = anchor['dst_ip']
        anchor_data = df_asn[df_asn['dst_ip'] == anchor_ip]

        distances = anchor_data['distance_km'].values
        rtts = anchor_data['min_rtt'].values

        model = RTTDistanceModel(
            fit_method='lp',
            anchor_ip=anchor_ip,
            anchor_lat=anchor['anchor_latitude'],
            anchor_lon=anchor['anchor_longitude'],
        )
        model.fit(
            distances=distances,
            rtts=rtts,
            method='lp',
            baseline_slope=THEORETICAL_SLOPE,
            n_std=1.0,
            global_n_std=1.0,
            bin_percentile=0.05,
            enable_baseline_filter=True,
            enable_bin_filter=False,
            enable_percentile_filter=False,
            enable_global_filter=False,
        )

        models[anchor_ip] = model
        status = f"slope={model.slope:.5f}, intercept={model.intercept:.2f}" if model.fitted else "FAILED"
        print(f"  LP model {anchor_ip}: {status}")

    return models


# =============================================================================
# Vanilla CBG Evaluation (LP model + Million-Scale multilateration)
# =============================================================================

def run_vanilla_cbg(df_asn, lp_models):
    """
    Run Vanilla CBG: LP bestline RTT→distance + Million-Scale spherical multilateration.

    This isolates the impact of the RTT-distance model by using:
    - RTT → distance via LP bestline inversion (per-anchor calibrated)
    - Spherical circle intersection (from helpers.py)
    - Polygon centroid / closest-VP fallback (from helpers.py)

    Pre-fills (lat, lon, rtt, d, r) tuples so circle_preprocessing() skips
    its own rtt_to_km(speed_threshold=2/3) conversion.
    """
    probe_ips = df_asn['src_ip'].unique()
    results = []
    all_radii = []
    all_areas = []

    for probe_ip in probe_ips:
        probe_data = df_asn[df_asn['src_ip'] == probe_ip]
        true_lat = probe_data['probe_latitude'].iloc[0]
        true_lon = probe_data['probe_longitude'].iloc[0]

        circles = []
        min_rtt_per_vp_ip = {}

        for _, row in probe_data.iterrows():
            anchor_ip = row['dst_ip']
            rtt = row['min_rtt']

            if anchor_ip not in lp_models:
                continue
            model = lp_models[anchor_ip]
            if not model.fitted:
                continue

            d = model.predict_distance(rtt)
            if d is None or d <= 0:
                continue

            min_rtt_per_vp_ip[anchor_ip] = rtt
            r = d / 6371  # radians
            circles.append((model.anchor_lat, model.anchor_lon, rtt, d, r))

        # Radii from LP model (d is already in km)
        radii_km = [d for _, _, _, d, _ in circles]
        all_radii.extend(radii_km)

        if not circles:
            results.append({
                'probe_ip': probe_ip, 'true_lat': true_lat, 'true_lon': true_lon,
                'estimated_lat': None, 'estimated_lon': None,
                'error_km': None, 'n_anchors': 0, 'method': 'vanilla_cbg',
                'intersection': False, 'avg_radius_km': None,
                'intersection_area_km2': 0.0
            })
            continue

        # Use Million-Scale spherical intersection pipeline
        intersections, circles_out = circle_intersections(circles, speed_threshold=2/3)
        circles_data_for_area = [(lat, lon, d) for lat, lon, _, d, _ in circles_out]
        area_km2 = compute_intersection_area(circles_data_for_area)
        all_areas.append(area_km2)

        if len(intersections) > 2:
            centroid = polygon_centroid(intersections)
            did_intersect = True
        elif len(intersections) == 2:
            centroid = get_middle_intersection(intersections)
            did_intersect = True
        else:
            # Fallback: closest anchor by min RTT
            closest_vp, _ = min(min_rtt_per_vp_ip.items(), key=lambda x: x[1])
            model = lp_models[closest_vp]
            centroid = (model.anchor_lat, model.anchor_lon)
            did_intersect = False

        est_lat, est_lon = centroid
        error_km = haversine((est_lat, est_lon), (true_lat, true_lon))

        results.append({
            'probe_ip': probe_ip,
            'true_lat': true_lat,
            'true_lon': true_lon,
            'estimated_lat': float(est_lat),
            'estimated_lon': float(est_lon),
            'error_km': float(error_km),
            'n_anchors': len(circles_out),
            'method': 'vanilla_cbg',
            'intersection': did_intersect,
            'avg_radius_km': float(np.mean(radii_km)),
            'intersection_area_km2': area_km2
        })

    return results, np.array(all_radii), np.array(all_areas)


# =============================================================================
# Plotting
# =============================================================================

def plot_error_cdf_comparison(ms_errors, vanilla_errors, output_path=None):
    """Plot comparative Error CDF for both methods."""
    fig, ax = plt.subplots(figsize=(12, 8))

    all_series = [
        (vanilla_errors, 'Vanilla CBG', 'black', '-'),
        (ms_errors, 'Million-Scale CBG', 'blue', '-'),
    ]

    all_errors_list = [vanilla_errors, ms_errors]

    for errors, label, color, ls in all_series:
        sorted_e = np.sort(errors)
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        median = np.median(errors)
        ax.plot(sorted_e, cdf, color=color, linestyle=ls, linewidth=2,
                label=f'{label}\n  Median: {median:.0f} km, N={len(errors)}')

    # Threshold lines
    for thresh, color in [(100, 'green'), (500, 'orange'), (1000, 'red')]:
        parts = []
        for errors, name in zip(all_errors_list, ['Van', 'MS']):
            parts.append(f'{name}={np.mean(errors <= thresh) * 100:.1f}%')
        ax.axvline(x=thresh, color=color, linestyle='--', alpha=0.5,
                   label=f'{thresh} km: {", ".join(parts)}')

    ax.hlines(y=0.5, xmin=0, xmax=3000, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Error Distance (km)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title(f'CBG Geolocation Error CDF Comparison — AS{ASN}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.9), fontsize=9)
    ax.grid(True, alpha=0.3)
    max_err = max(e.max() for e in all_errors_list)
    ax.set_xlim(0, min(max_err * 1.05, 3000))
    ax.set_ylim(0, 1)

    # Statistics text box
    def _stats_block(name, errors):
        return (
            f"{name}:\n"
            f"  Median: {np.median(errors):.0f} km\n"
            f"  Mean: {np.mean(errors):.0f} km\n"
            f"  75th: {np.percentile(errors, 75):.0f} km\n"
            f"  90th: {np.percentile(errors, 90):.0f} km"
        )

    blocks = [
        _stats_block('Vanilla', vanilla_errors),
        _stats_block('Million-Scale', ms_errors),
    ]
    stats = '\n\n'.join(blocks)

    ax.text(0.98, 0.02, stats, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_radius_cdf(ms_radii, van_radii, output_path=None):
    """Plot CDF of all individual circle radii for both methods."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for radii, label, color in [
        (van_radii, 'Vanilla CBG (LP)', 'black'),
        (ms_radii, 'Million-Scale CBG (2/3c)', 'blue'),
    ]:
        sorted_r = np.sort(radii)
        cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        median = np.median(radii)
        ax.plot(sorted_r, cdf, color=color, linewidth=2,
                label=f'{label}\n  Median: {median:.0f} km, N={len(radii)}')

    ax.hlines(y=0.5, xmin=0, xmax=ax.get_xlim()[1], color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Circle Radius (km)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title(f'CBG Circle Radius CDF — AS{ASN}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_area_cdf(ms_areas, van_areas, output_path=None):
    """Plot CDF of intersection areas (km²) for both methods, with zoomed inset."""
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(6, 4))

    # Convert to million km² for readability (US ≈ 9.8 million km²)
    scale = 1e6
    unit = 'million km²'

    series_data = []
    # Only include probes with area > 0 (successful polygon intersection)
    for areas, label, color in [
        (van_areas, 'Vanilla CBG (LP)', 'black'),
        (ms_areas, 'Million-Scale CBG (2/3c)', 'blue'),
    ]:
        valid = areas[areas > 0]
        sorted_a = np.sort(valid) / scale
        cdf = np.arange(1, len(sorted_a) + 1) / len(sorted_a)
        median = np.median(valid) / scale
        ax.plot(sorted_a, cdf, color=color, linewidth=2,
                label=f'{label}\n  Median: {median:,.1f} {unit}, N={len(valid)}')
        series_data.append((sorted_a, cdf, color))

    ax.hlines(y=0.5, xmin=0, xmax=ax.get_xlim()[1], color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(f'Intersection Area ({unit})', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title(f'CBG Intersection Area CDF — AS{ASN}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0)
    ax.set_ylim(0, 1)

    # Inset: zoom into area <= 20 million km²
    inset_limit = 20.0  # 20 million km²

    # Inset position in axes fraction: [left, bottom, width, height]
    inset_pos = [0.42, 0.25, 0.35, 0.35]
    ax_inset = ax.inset_axes(inset_pos)
    for sorted_a, cdf, color in series_data:
        mask = sorted_a <= inset_limit
        if mask.any():
            ax_inset.plot(sorted_a[mask], cdf[mask], color=color, linewidth=2)
    ax_inset.set_xlim(0, inset_limit)
    ax_inset.set_ylim(0, 1)
    ax_inset.set_xlabel(f'Area ({unit})', fontsize=8)
    ax_inset.set_ylabel('CDF', fontsize=8)
    ax_inset.set_title(f'Area ≤ {inset_limit:.0f} {unit}', fontsize=9)
    ax_inset.tick_params(labelsize=8)
    ax_inset.grid(True, alpha=0.3)

    # Draw zoom indicator rectangle on main plot and dashed connector lines
    from matplotlib.patches import Rectangle, ConnectionPatch
    # Indicator box in data coords on main axes
    rect = Rectangle((0, 0), inset_limit, 1.0, linewidth=1.2,
                      edgecolor='gray', facecolor='lightgray', alpha=0.2,
                      linestyle='-', zorder=0)
    ax.add_patch(rect)
    # Connector lines: top-right of indicator box → top-left of inset,
    #                  bottom-right of indicator box → bottom-left of inset
    for (xy_main, xy_inset) in [
        ((inset_limit, 1.0), (0, 1)),   # top corners
        ((inset_limit, 0.0), (0, 0)),   # bottom corners
    ]:
        con = ConnectionPatch(
            xyA=xy_main, coordsA=ax.transData,
            xyB=xy_inset, coordsB=ax_inset.transAxes,
            color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
        fig.add_artist(con)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_circles_on_map(probe_result, circles_data, intersections, method_label,
                        output_path=None):
    """
    Plot CBG circles on a US map for one probe.

    Parameters:
        probe_result: dict with probe_ip, true_lat/lon, estimated_lat/lon, error_km, etc.
        circles_data: list of (lat, lon, radius_km) for each anchor circle
        intersections: list of (lat, lon) intersection points
        method_label: str for the title
        output_path: Path to save
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    proj = ccrs.LambertConformal(central_longitude=-96, central_latitude=39)
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={'projection': proj})

    # US extent
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff')
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
    ax.add_feature(cfeature.BORDERS, linewidth=0.8)
    ax.coastlines(resolution='50m', linewidth=0.8)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(circles_data), 10)))

    # Draw circles
    for i, (clat, clon, radius_km) in enumerate(circles_data):
        # Approximate circle as polygon: radius_km → degrees
        n_pts = 100
        angles = np.linspace(0, 2 * np.pi, n_pts)
        r_deg_lat = radius_km / 111.0
        r_deg_lon = radius_km / (111.0 * math.cos(math.radians(clat)))
        circle_lons = clon + r_deg_lon * np.cos(angles)
        circle_lats = clat + r_deg_lat * np.sin(angles)
        ax.plot(circle_lons, circle_lats, color=colors[i], linewidth=1.5,
                alpha=0.7, transform=ccrs.PlateCarree())
        ax.fill(circle_lons, circle_lats, color=colors[i], alpha=0.05,
                transform=ccrs.PlateCarree())
        # Anchor marker
        ax.plot(clon, clat, 's', color=colors[i], markersize=8,
                transform=ccrs.PlateCarree(), zorder=5,
                label=f'VP ({clat:.1f}, {clon:.1f}) r={radius_km:.0f}km')

    # Intersection polygon via Shapely
    if len(circles_data) >= 2:
        try:
            shapely_circles = [
                _circle_to_shapely_polygon(clat, clon, radius_km)
                for clat, clon, radius_km in circles_data
            ]
            shapely_circles = [p for p in shapely_circles if p.is_valid and not p.is_empty]
            if shapely_circles:
                intersection_poly = reduce(lambda a, b: a.intersection(b), shapely_circles)
                if not intersection_poly.is_empty:
                    polys = (list(intersection_poly.geoms)
                             if isinstance(intersection_poly, MultiPolygon)
                             else [intersection_poly])
                    for k, poly in enumerate(polys):
                        if poly.is_empty or poly.geom_type != 'Polygon':
                            continue
                        xs, ys = poly.exterior.xy  # xs=lons, ys=lats
                        label = 'Intersection region' if k == 0 else None
                        ax.fill(list(xs), list(ys), color='yellow', alpha=0.4,
                                transform=ccrs.PlateCarree(), zorder=3, label=label)
                        ax.plot(list(xs), list(ys), color='orange', linewidth=2,
                                transform=ccrs.PlateCarree(), zorder=4)
        except Exception:
            pass

    # Pairwise circle-boundary crossing points (planar), filtered to inside intersection
    if len(circles_data) >= 2:
        from itertools import combinations
        from shapely.geometry import MultiPoint, Point as ShapelyPoint
        planar_circles = [
            _circle_to_shapely_polygon(clat, clon, radius_km)
            for clat, clon, radius_km in circles_data
        ]
        # Compute intersection region for filtering
        valid_polys = [p for p in planar_circles if p.is_valid and not p.is_empty]
        region = reduce(lambda a, b: a.intersection(b), valid_polys) if valid_polys else None
        cross_lons, cross_lats = [], []
        for ca, cb in combinations(planar_circles, 2):
            try:
                pts = ca.exterior.intersection(cb.exterior)
                if pts.is_empty:
                    continue
                candidates = []
                if isinstance(pts, MultiPoint):
                    candidates = list(pts.geoms)
                elif pts.geom_type == 'Point':
                    candidates = [pts]
                for pt in candidates:
                    # Use buffer to include points on boundary
                    if region and region.buffer(0.1).contains(ShapelyPoint(pt.x, pt.y)):
                        cross_lons.append(pt.x)
                        cross_lats.append(pt.y)
            except Exception:
                continue
        if cross_lons:
            ax.scatter(cross_lons, cross_lats, c='gray', s=30, marker='o',
                       alpha=0.8, edgecolors='black', linewidths=0.5,
                       transform=ccrs.PlateCarree(), zorder=6,
                       label=f'Circle crossings (n={len(cross_lons)})')

    # True location
    true_lat = probe_result['true_lat']
    true_lon = probe_result['true_lon']
    ax.plot(true_lon, true_lat, '*', color='green', markersize=18,
            markeredgecolor='black', markeredgewidth=1,
            transform=ccrs.PlateCarree(), zorder=10, label='True location')

    # Estimated location
    est_lat = probe_result['estimated_lat']
    est_lon = probe_result['estimated_lon']
    if est_lat is not None:
        ax.plot(est_lon, est_lat, 'X', color='red', markersize=14,
                markeredgecolor='black', markeredgewidth=1,
                transform=ccrs.PlateCarree(), zorder=10, label='Estimated location')

    error_km = probe_result['error_km']
    area_km2 = probe_result.get('intersection_area_km2', 0)
    ax.set_title(f'{method_label} — Error: {error_km:.0f} km, '
                 f'Area: {area_km2 / 1e6:.1f} million km²\n'
                 f'Probe: {probe_result["probe_ip"]}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8, ncol=2)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_percentile_maps(ms_results, van_results, df_asn, lp_models,
                         percentiles=(5, 50, 75), anchor_method='vanilla'):
    """Generate circle-on-map plots at error percentiles of the anchor method.

    Selects probes based on the anchor method's error distribution, then plots
    both methods for the same probe to enable direct comparison.

    Args:
        anchor_method: 'vanilla' or 'million_scale' — which method's error
                       distribution to use for selecting percentile probes.
    """
    anchor_info = df_asn[['dst_ip', 'anchor_latitude', 'anchor_longitude']].drop_duplicates()
    anchor_coords = {}
    for _, row in anchor_info.iterrows():
        anchor_coords[row['dst_ip']] = (row['anchor_latitude'], row['anchor_longitude'])

    # Index both results by probe_ip
    ms_by_ip = {r['probe_ip']: r for r in ms_results}
    van_by_ip = {r['probe_ip']: r for r in van_results}

    # Select anchor results for percentile selection
    if anchor_method == 'vanilla':
        anchor_success = [r for r in van_results if r['error_km'] is not None]
        anchor_label = 'Vanilla'
        prefix = 'van'
    else:
        anchor_success = [r for r in ms_results if r['error_km'] is not None]
        anchor_label = 'MS'
        prefix = 'ms'

    anchor_errors = [r['error_km'] for r in anchor_success]

    for pct in percentiles:
        target_err = np.percentile(anchor_errors, pct)
        anchor_probe = min(anchor_success, key=lambda r: abs(r['error_km'] - target_err))
        probe_ip = anchor_probe['probe_ip']
        probe_data = df_asn[df_asn['src_ip'] == probe_ip]

        # Plot both methods for this probe
        for method_name, probe_result, build_fn in [
            ('Vanilla CBG (LP)', van_by_ip.get(probe_ip),
             lambda pd: _build_lp_circles(pd, lp_models)),
            ('Million-Scale (2/3c)', ms_by_ip.get(probe_ip),
             lambda pd: _build_ms_circles(pd, anchor_coords)),
        ]:
            if probe_result is None or probe_result['error_km'] is None:
                continue

            circles_data, circle_tuples = build_fn(probe_data)
            if circle_tuples:
                intersections, _ = circle_intersections(circle_tuples, speed_threshold=2/3)
            else:
                intersections = []

            tag = method_name.split('(')[0].strip().lower().replace(' ', '_').replace('-', '_')
            out_path = OUTPUT_DIR / f'map_{prefix}_p{pct}_{tag}.png'
            fig = plot_circles_on_map(probe_result, circles_data, intersections,
                                      f'{method_name} — {anchor_label} P{pct}',
                                      output_path=out_path)
            plt.close(fig)


def _build_ms_circles(probe_data, anchor_coords):
    """Build circle data for Million-Scale method."""
    circles_data = []  # (lat, lon, radius_km) for plotting
    circle_tuples = []  # (lat, lon, rtt, None, None) for intersection
    for _, row in probe_data.iterrows():
        anchor_ip = row['dst_ip']
        rtt = row['min_rtt']
        if rtt > 100 or anchor_ip not in anchor_coords:
            continue
        lat, lon = anchor_coords[anchor_ip]
        radius_km = 100.0 * rtt
        circles_data.append((lat, lon, radius_km))
        circle_tuples.append((lat, lon, rtt, None, None))
    return circles_data, circle_tuples


def _build_lp_circles(probe_data, lp_models):
    """Build circle data for Vanilla CBG (LP) method."""
    circles_data = []
    circle_tuples = []
    for _, row in probe_data.iterrows():
        anchor_ip = row['dst_ip']
        rtt = row['min_rtt']
        if anchor_ip not in lp_models:
            continue
        model = lp_models[anchor_ip]
        if not model.fitted:
            continue
        d = model.predict_distance(rtt)
        if d is None or d <= 0:
            continue
        r = d / 6371
        circles_data.append((model.anchor_lat, model.anchor_lon, d))
        circle_tuples.append((model.anchor_lat, model.anchor_lon, rtt, d, r))
    return circles_data, circle_tuples


def plot_rtt_distance_scatter(anchor_ip, df_anchor, lp_model, max_rtt_ms=150, output_path=None):
    """Plot RTT-distance scatter with both model lines for one anchor."""
    fig, ax = plt.subplots(figsize=(12, 8))

    distances = df_anchor['distance_km'].values
    rtts = df_anchor['min_rtt'].values
    city = df_anchor['anchor_city'].iloc[0] if 'anchor_city' in df_anchor.columns else ''

    # Filter for plotting
    plot_mask = rtts <= max_rtt_ms
    ax.scatter(distances[plot_mask], rtts[plot_mask], alpha=0.3, s=20, c='gray',
               label=f'Measurements (n={len(distances)})', edgecolors='none')

    dist_range = np.linspace(0, distances.max(), 100)

    # Theoretical 2/3c line (Million-Scale)
    theoretical_rtts = THEORETICAL_SLOPE * dist_range
    ax.plot(dist_range, theoretical_rtts, 'b--', linewidth=2, alpha=0.8,
            label=f'Million-Scale (2/3c): {THEORETICAL_SLOPE:.4f} ms/km')

    # Lower Envelope (LP bestline)
    if lp_model is not None and lp_model.fitted:
        lp_rtts = lp_model.slope * dist_range + lp_model.intercept
        ax.plot(dist_range, lp_rtts, 'r-', linewidth=2.5,
                label=f'Lower Envelope: {lp_model.slope:.4f} ms/km + {lp_model.intercept:.1f} ms')

    ax.set_xlabel('Distance (km)', fontsize=12)
    ax.set_ylabel('RTT (ms)', fontsize=12)
    ax.set_title(f'RTT-Distance Model Comparison — {city} ({anchor_ip})\nAS{ASN} Probes',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, distances.max() * 1.05)
    ax.set_ylim(0, max_rtt_ms)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def print_statistics(ms_errors, vanilla_errors):
    """Print comparison statistics table."""
    w = 70
    print("\n" + "=" * w)
    print("CBG MULTILATERATION COMPARISON — STATISTICS")
    print("=" * w)

    cols = [('Million-Scale', ms_errors), ('Vanilla', vanilla_errors)]

    header = f"{'Metric':<25}" + "".join(f" {name:>20}" for name, _ in cols)
    print(header)
    print("-" * w)

    metrics = [
        ('N (probes)', lambda e: f"{len(e)}"),
        ('Median (km)', lambda e: f"{np.median(e):.1f}"),
        ('Mean (km)', lambda e: f"{np.mean(e):.1f}"),
        ('Std (km)', lambda e: f"{np.std(e):.1f}"),
        ('Min (km)', lambda e: f"{np.min(e):.1f}"),
        ('Max (km)', lambda e: f"{np.max(e):.1f}"),
        ('25th pct (km)', lambda e: f"{np.percentile(e, 25):.1f}"),
        ('75th pct (km)', lambda e: f"{np.percentile(e, 75):.1f}"),
        ('90th pct (km)', lambda e: f"{np.percentile(e, 90):.1f}"),
        ('95th pct (km)', lambda e: f"{np.percentile(e, 95):.1f}"),
    ]
    for label, fn in metrics:
        row = f"{label:<25}" + "".join(f" {fn(e):>20}" for _, e in cols)
        print(row)

    print()
    header2 = f"{'Accuracy Thresholds':<25}" + "".join(f" {name:>20}" for name, _ in cols)
    print(header2)
    print("-" * w)
    for thresh in [50, 100, 250, 500, 1000]:
        row = f"  Within {thresh:4d} km        "
        for _, e in cols:
            pct = np.mean(e <= thresh) * 100
            row += f" {pct:>19.1f}%"
        print(row)

    print("=" * w)


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df, df_asn = load_data()

    # Fit LP models
    print("\n" + "=" * 60)
    print("FITTING LP MODELS")
    print("=" * 60)
    lp_models = fit_lp_models(df_asn)

    # Run Vanilla CBG (LP + Spherical)
    print("\n" + "=" * 60)
    print("RUNNING VANILLA CBG")
    print("=" * 60)
    van_results, van_all_radii, van_all_areas = run_vanilla_cbg(df_asn, lp_models)
    van_success = [r for r in van_results if r['error_km'] is not None]
    van_errors = np.array([r['error_km'] for r in van_success])
    van_intersected = sum(1 for r in van_results if r['intersection'])
    print(f"  Successful: {len(van_success)}/{len(van_results)} probes")
    print(f"  Intersection succeeded: {van_intersected}/{len(van_results)} probes")
    print(f"  Median error: {np.median(van_errors):.1f} km")
    print(f"  Circle radii: N={len(van_all_radii)}, mean={np.mean(van_all_radii):.1f} km, median={np.median(van_all_radii):.1f} km")
    van_valid_areas = van_all_areas[van_all_areas > 0]
    print(f"  Intersection areas: N={len(van_valid_areas)}, median={np.median(van_valid_areas):,.0f} km²")

    # Run Million-Scale CBG (2/3c + Spherical)
    print("\n" + "=" * 60)
    print("RUNNING MILLION-SCALE CBG")
    print("=" * 60)
    ms_results, ms_all_radii, ms_all_areas = run_million_scale_cbg(df_asn)
    ms_success = [r for r in ms_results if r['error_km'] is not None]
    ms_errors = np.array([r['error_km'] for r in ms_success])
    ms_intersected = sum(1 for r in ms_results if r['intersection'])
    print(f"  Successful: {len(ms_success)}/{len(ms_results)} probes")
    print(f"  Intersection succeeded: {ms_intersected}/{len(ms_results)} probes")
    print(f"  Median error: {np.median(ms_errors):.1f} km")
    print(f"  Circle radii: N={len(ms_all_radii)}, mean={np.mean(ms_all_radii):.1f} km, median={np.median(ms_all_radii):.1f} km")
    ms_valid_areas = ms_all_areas[ms_all_areas > 0]
    print(f"  Intersection areas: N={len(ms_valid_areas)}, median={np.median(ms_valid_areas):,.0f} km²")

    # Per-anchor RTT-distance scatter
    print("\n" + "=" * 60)
    print("GENERATING RTT-DISTANCE SCATTER PLOTS")
    print("=" * 60)
    anchor_ips = df_asn['dst_ip'].unique()
    for anchor_ip in anchor_ips:
        df_anchor = df_asn[df_asn['dst_ip'] == anchor_ip]
        lp_model = lp_models.get(anchor_ip)
        output_path = OUTPUT_DIR / f"scatter_{anchor_ip.replace('.', '_')}.png"
        fig = plot_rtt_distance_scatter(anchor_ip, df_anchor, lp_model, output_path=output_path)
        plt.close(fig)

    # Comparative Error CDF
    print("\n" + "=" * 60)
    print("GENERATING ERROR CDF COMPARISON")
    print("=" * 60)
    cdf_path = OUTPUT_DIR / 'error_cdf_comparison.png'
    fig = plot_error_cdf_comparison(ms_errors, van_errors, output_path=cdf_path)
    plt.close(fig)

    # Circle Radius CDF
    radius_cdf_path = OUTPUT_DIR / 'radius_cdf_comparison.png'
    fig = plot_radius_cdf(ms_all_radii, van_all_radii, output_path=radius_cdf_path)
    plt.close(fig)

    # Intersection Area CDF
    area_cdf_path = OUTPUT_DIR / 'intersection_area_cdf.png'
    fig = plot_area_cdf(ms_all_areas, van_all_areas, output_path=area_cdf_path)
    plt.close(fig)

    # Circle maps at P5, P50, P75 — anchored on Vanilla CBG percentiles
    print("\n" + "=" * 60)
    print("GENERATING CIRCLE MAP PLOTS (Vanilla P5, P50, P75)")
    print("=" * 60)
    plot_percentile_maps(ms_results, van_results, df_asn, lp_models,
                         percentiles=(5, 50, 75), anchor_method='vanilla')

    # Circle maps at P5, P50, P75 — anchored on Million-Scale percentiles
    print("\n" + "=" * 60)
    print("GENERATING CIRCLE MAP PLOTS (MS P5, P50, P75)")
    print("=" * 60)
    plot_percentile_maps(ms_results, van_results, df_asn, lp_models,
                         percentiles=(5, 50, 75), anchor_method='million_scale')

    # Statistics
    print_statistics(ms_errors, van_errors)

    # Save results JSON
    results_json = {
        'asn': ASN,
        'million_scale_cbg': {
            'total_probes': len(ms_results),
            'successful': len(ms_success),
            'median_km': float(np.median(ms_errors)),
            'mean_km': float(np.mean(ms_errors)),
            'p75_km': float(np.percentile(ms_errors, 75)),
            'p90_km': float(np.percentile(ms_errors, 90)),
        },
        'vanilla_cbg': {
            'total_probes': len(van_results),
            'successful': len(van_success),
            'median_km': float(np.median(van_errors)),
            'mean_km': float(np.mean(van_errors)),
            'p75_km': float(np.percentile(van_errors, 75)),
            'p90_km': float(np.percentile(van_errors, 90)),
        },
    }
    json_path = OUTPUT_DIR / 'comparison_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved: {json_path}")


if __name__ == '__main__':
    main()
