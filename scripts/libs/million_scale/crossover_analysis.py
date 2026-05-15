"""
Crossover Analysis: Why does Million-Scale CBG (2/3c) beat LP CBG at lower error percentiles?

Investigates the CDF crossover by:
1. Running both CBG pipelines and collecting per-probe diagnostics
2. Classifying crossover probes by mechanism (fallback vs intersection)
3. Quantifying the is_within_cirle bug impact on LP results
4. Generating targeted visualizations

Output: scripts/libs/million_scale/outputs/crossover_analysis/
"""

import sys
import json
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from functools import reduce
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CBG_DIR = PROJECT_ROOT / 'scripts' / 'analysis' / 'cbg_feasibility'
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CBG_DIR))

from scripts.utils.helpers import (
    haversine,
    circle_preprocessing,
    polygon_centroid,
    get_middle_intersection,
    geo_to_cartesian,
    rtt_to_km,
)
from scripts.libs.cbg_feasibility.rtt_model import (
    RTTDistanceModel,
    haversine_distance,
    THEORETICAL_SLOPE,
)

# Import reusable functions from evaluate_million_scale
from scripts.libs.million_scale.evaluate_million_scale import (
    load_data,
    fit_lp_models,
    run_million_scale_cbg,
    run_vanilla_cbg,
    _build_ms_circles,
    _build_lp_circles,
    _circle_to_shapely_polygon,
    compute_intersection_area,
    ASN,
)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'crossover_analysis'
MAPS_DIR = OUTPUT_DIR / 'maps'


# =============================================================================
# Corrected circle_intersections (uses pre-filled d instead of rtt_to_km)
# =============================================================================

def corrected_circle_intersections(circles, speed_threshold=None):
    """
    Same as helpers.circle_intersections but with corrected point filtering.

    Bug in original: is_within_cirle() always uses rtt_to_km(rtt, speed_threshold)
    for the radius check, ignoring the pre-filled d value. This means LP circles
    are filtered with MS-sized radii (100*rtt) instead of LP radii.

    This version uses d_c directly when available.
    """
    circles = circle_preprocessing(circles, speed_threshold=speed_threshold)

    if len(circles) == 0:
        return [], circles

    if len(circles) == 1:
        from scripts.utils.helpers import get_points_on_circle
        single_circle = list(circles)[0]
        lat, lon, rtt, d, r = single_circle
        filtered_points = get_points_on_circle(lat, lon, d)
        return filtered_points, circles

    intersect_points = []
    for c_1, c_2 in itertools.combinations(circles, 2):
        lat_1, lon_1, rtt_1, d_1, r_1 = c_1
        lat_2, lon_2, rtt_2, d_2, r_2 = c_2

        x1 = np.array(list(geo_to_cartesian(lat_1, lon_1)))
        x2 = np.array(list(geo_to_cartesian(lat_2, lon_2)))

        q = np.dot(x1, x2)
        denom = 1 - q**2
        if abs(denom) < 1e-12:
            continue

        a = (np.cos(r_1) - np.cos(r_2) * q) / denom
        b = (np.cos(r_2) - np.cos(r_1) * q) / denom

        x0 = a * x1 + b * x2
        n = np.cross(x1, x2)
        nn = np.dot(n, n)
        if nn < 1e-12:
            continue
        val = (1 - np.dot(x0, x0)) / nn
        if val <= 0:
            continue

        t = np.sqrt(val)

        for sign in [1, -1]:
            pt = x0 + sign * t * n
            i_lon = np.arctan2(pt[1], pt[0]) * (180 / np.pi)
            i_lat = np.arctan(pt[2] / np.sqrt(pt[0]**2 + pt[1]**2)) / (np.pi / 180)
            intersect_points.append((i_lat, i_lon))

    # Corrected filtering: use d_c directly instead of rtt_to_km(rtt_c)
    filtered_points = []
    for point_geo in intersect_points:
        inside_all = True
        for lat_c, lon_c, rtt_c, d_c, r_c in circles:
            dist_to_center = haversine((lat_c, lon_c), point_geo)
            if d_c < dist_to_center:
                inside_all = False
                break
        if inside_all:
            filtered_points.append(point_geo)

    return filtered_points, circles


def run_corrected_lp_cbg(df_asn, lp_models):
    """Run LP CBG with corrected is_within_circle filtering."""
    probe_ips = df_asn['src_ip'].unique()
    results = []

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
            r = d / 6371
            circles.append((model.anchor_lat, model.anchor_lon, rtt, d, r))

        if not circles:
            results.append({
                'probe_ip': probe_ip, 'true_lat': true_lat, 'true_lon': true_lon,
                'estimated_lat': None, 'estimated_lon': None,
                'error_km': None, 'n_anchors': 0, 'method': 'corrected_lp_cbg',
                'intersection': False,
            })
            continue

        intersections, circles_out = corrected_circle_intersections(circles, speed_threshold=2/3)

        if len(intersections) > 2:
            centroid = polygon_centroid(intersections)
            did_intersect = True
        elif len(intersections) == 2:
            centroid = get_middle_intersection(intersections)
            did_intersect = True
        else:
            closest_vp, _ = min(min_rtt_per_vp_ip.items(), key=lambda x: x[1])
            model = lp_models[closest_vp]
            centroid = (model.anchor_lat, model.anchor_lon)
            did_intersect = False

        est_lat, est_lon = centroid
        error_km = haversine((est_lat, est_lon), (true_lat, true_lon))

        results.append({
            'probe_ip': probe_ip,
            'true_lat': true_lat, 'true_lon': true_lon,
            'estimated_lat': float(est_lat), 'estimated_lon': float(est_lon),
            'error_km': float(error_km),
            'n_anchors': len(circles_out),
            'method': 'corrected_lp_cbg',
            'intersection': did_intersect,
        })

    return results


# =============================================================================
# Diagnostic merge and classification
# =============================================================================

def build_merged_diagnostics(ms_results, van_results, corrected_results, df_asn):
    """Merge per-probe results from all methods into a single DataFrame."""
    # Build anchor coordinate lookup for distance computation
    anchors = df_asn[['dst_ip', 'anchor_latitude', 'anchor_longitude']].drop_duplicates()
    anchor_locs = [(row['anchor_latitude'], row['anchor_longitude'])
                   for _, row in anchors.iterrows()]
    anchor_ips = anchors['dst_ip'].tolist()

    rows = []
    ms_by_ip = {r['probe_ip']: r for r in ms_results}
    van_by_ip = {r['probe_ip']: r for r in van_results}
    corr_by_ip = {r['probe_ip']: r for r in corrected_results}

    all_ips = set(ms_by_ip.keys()) | set(van_by_ip.keys())

    for ip in all_ips:
        ms = ms_by_ip.get(ip, {})
        van = van_by_ip.get(ip, {})
        corr = corr_by_ip.get(ip, {})

        ms_err = ms.get('error_km')
        lp_err = van.get('error_km')
        corr_err = corr.get('error_km')

        if ms_err is None or lp_err is None:
            continue

        true_lat = ms.get('true_lat', van.get('true_lat'))
        true_lon = ms.get('true_lon', van.get('true_lon'))

        # Distance to nearest anchor
        dists = [haversine((true_lat, true_lon), loc) for loc in anchor_locs]
        nearest_idx = int(np.argmin(dists))

        rows.append({
            'probe_ip': ip,
            'true_lat': true_lat,
            'true_lon': true_lon,
            'ms_error': ms_err,
            'lp_error': lp_err,
            'corr_lp_error': corr_err,
            'error_diff': lp_err - ms_err,  # positive = MS wins
            'ms_intersected': ms.get('intersection', False),
            'lp_intersected': van.get('intersection', False),
            'corr_lp_intersected': corr.get('intersection', False),
            'ms_n_anchors': ms.get('n_anchors', 0),
            'lp_n_anchors': van.get('n_anchors', 0),
            'ms_avg_radius': ms.get('avg_radius_km'),
            'lp_avg_radius': van.get('avg_radius_km'),
            'ms_est_lat': ms.get('estimated_lat'),
            'ms_est_lon': ms.get('estimated_lon'),
            'lp_est_lat': van.get('estimated_lat'),
            'lp_est_lon': van.get('estimated_lon'),
            'dist_to_nearest_anchor': dists[nearest_idx],
            'nearest_anchor_ip': anchor_ips[nearest_idx],
        })

    return pd.DataFrame(rows)


def classify_probes(df_merged):
    """Classify each probe by crossover mechanism."""
    categories = []
    for _, row in df_merged.iterrows():
        ms_wins = row['ms_error'] < row['lp_error']
        ms_int = row['ms_intersected']
        lp_int = row['lp_intersected']

        if not ms_wins:
            if ms_int and not lp_int:
                cat = 'lp_fallback_ms_intersected'
            elif not ms_int and lp_int:
                cat = 'lp_intersected_ms_fallback'
            elif ms_int and lp_int:
                cat = 'both_intersected_lp_better'
            else:
                cat = 'both_fallback_lp_better'
        else:
            if not ms_int and lp_int:
                cat = 'ms_fallback_near'
            elif not ms_int and not lp_int:
                cat = 'both_fallback_ms_better'
            elif ms_int and lp_int:
                cat = 'both_intersected_ms_better'
            else:
                cat = 'ms_intersected_lp_fallback'

        categories.append(cat)

    df_merged['category'] = categories
    return df_merged


# =============================================================================
# Visualizations
# =============================================================================

def plot_crossover_cdf(df_merged, output_path):
    """Error CDF with crossover region shaded, including corrected LP."""
    ms_errors = np.sort(df_merged['ms_error'].values)
    lp_errors = np.sort(df_merged['lp_error'].values)
    corr_lp_errors = np.sort(df_merged['corr_lp_error'].dropna().values)
    cdf = np.arange(1, len(ms_errors) + 1) / len(ms_errors)
    cdf_corr = np.arange(1, len(corr_lp_errors) + 1) / len(corr_lp_errors)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(lp_errors, cdf, color='black', linewidth=2,
            label=f'LP CBG (buggy filter) — Median: {np.median(lp_errors):.0f} km')
    ax.plot(corr_lp_errors, cdf_corr, color='green', linewidth=2, linestyle='--',
            label=f'LP CBG (corrected filter) — Median: {np.median(corr_lp_errors):.0f} km')
    ax.plot(ms_errors, cdf, color='blue', linewidth=2,
            label=f'MS CBG (2/3c) — Median: {np.median(ms_errors):.0f} km')

    # Shade crossover region: where MS CDF is to the left of LP CDF
    cdf_grid = np.linspace(0.01, 0.99, 1000)
    ms_interp = np.interp(cdf_grid, cdf, ms_errors)
    lp_interp = np.interp(cdf_grid, cdf, lp_errors)

    ms_wins_mask = ms_interp < lp_interp
    if ms_wins_mask.any():
        ax.fill_betweenx(cdf_grid, ms_interp, lp_interp,
                         where=ms_wins_mask,
                         alpha=0.15, color='blue', label='MS better than buggy LP')

    # Also shade MS vs corrected LP crossover
    corr_interp = np.interp(cdf_grid, cdf_corr, corr_lp_errors)
    ms_vs_corr_mask = ms_interp < corr_interp
    if ms_vs_corr_mask.any():
        ax.fill_betweenx(cdf_grid, ms_interp, corr_interp,
                         where=ms_vs_corr_mask,
                         alpha=0.15, color='red', label='MS better than corrected LP')

    # Find crossover percentiles
    crossover_indices = np.where(ms_wins_mask)[0]
    if len(crossover_indices) > 0:
        # Find first crossing point (where MS stops being better)
        transitions = np.diff(ms_wins_mask.astype(int))
        cross_points = np.where(transitions == -1)[0]
        if len(cross_points) > 0:
            crossover_pct = cdf_grid[cross_points[0]]
            ax.axhline(y=crossover_pct, color='blue', linestyle=':', alpha=0.5,
                       label=f'MS vs buggy LP crossover ~{crossover_pct:.0%}')

    corr_cross_indices = np.where(ms_vs_corr_mask)[0]
    if len(corr_cross_indices) > 0:
        transitions_corr = np.diff(ms_vs_corr_mask.astype(int))
        cross_points_corr = np.where(transitions_corr == -1)[0]
        if len(cross_points_corr) > 0:
            crossover_pct_corr = cdf_grid[cross_points_corr[0]]
            ax.axhline(y=crossover_pct_corr, color='red', linestyle=':', alpha=0.5,
                       label=f'MS vs corrected LP crossover ~{crossover_pct_corr:.0%}')

    # Threshold lines
    for thresh, color in [(100, 'green'), (500, 'orange'), (1000, 'red')]:
        ms_pct = np.mean(df_merged['ms_error'] <= thresh) * 100
        lp_pct = np.mean(df_merged['lp_error'] <= thresh) * 100
        corr_pct = np.mean(df_merged['corr_lp_error'].dropna() <= thresh) * 100
        ax.axvline(x=thresh, color=color, linestyle=':', alpha=0.4,
                   label=f'{thresh}km: MS={ms_pct:.1f}%, LP={lp_pct:.1f}%, corr={corr_pct:.1f}%')

    ax.set_xlabel('Error Distance (km)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title(f'Error CDF: MS vs LP (buggy) vs LP (corrected) — AS{ASN}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_error_scatter(df_merged, output_path):
    """Scatter: LP error vs MS error, colored by category."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Category colors
    cat_colors = {
        'ms_fallback_near': 'red',
        'both_fallback_ms_better': 'orange',
        'both_intersected_ms_better': 'blue',
        'ms_intersected_lp_fallback': 'purple',
        'both_intersected_lp_better': 'gray',
        'lp_intersected_ms_fallback': 'lightgray',
        'both_fallback_lp_better': 'silver',
        'lp_fallback_ms_intersected': 'lightblue',
    }

    max_err = max(df_merged['ms_error'].max(), df_merged['lp_error'].max())
    ax.plot([0, max_err], [0, max_err], 'k--', alpha=0.5, linewidth=1, label='y=x')

    for cat, color in cat_colors.items():
        mask = df_merged['category'] == cat
        if mask.sum() == 0:
            continue
        subset = df_merged[mask]
        ax.scatter(subset['lp_error'], subset['ms_error'],
                   c=color, s=40, alpha=0.7, edgecolors='black', linewidths=0.3,
                   label=f'{cat} (n={mask.sum()})', zorder=3)

    ax.set_xlabel('LP CBG Error (km)', fontsize=12)
    ax.set_ylabel('MS CBG Error (km)', fontsize=12)
    ax.set_title(f'Per-Probe Error: MS vs LP — AS{ASN}\n(Below diagonal = MS wins)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(max_err * 1.05, 3000))
    ax.set_ylim(0, min(max_err * 1.05, 3000))
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_dist_vs_advantage(df_merged, output_path):
    """Distance to nearest anchor vs error advantage (LP_err - MS_err)."""
    fig, ax = plt.subplots(figsize=(12, 8))

    ms_wins = df_merged['error_diff'] > 0
    lp_wins = df_merged['error_diff'] <= 0

    ax.scatter(df_merged.loc[ms_wins, 'dist_to_nearest_anchor'],
               df_merged.loc[ms_wins, 'error_diff'],
               c='blue', s=30, alpha=0.6, label=f'MS wins (n={ms_wins.sum()})',
               edgecolors='black', linewidths=0.3)
    ax.scatter(df_merged.loc[lp_wins, 'dist_to_nearest_anchor'],
               df_merged.loc[lp_wins, 'error_diff'],
               c='gray', s=30, alpha=0.4, label=f'LP wins (n={lp_wins.sum()})',
               edgecolors='black', linewidths=0.3)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Distance to Nearest Anchor (km)', fontsize=12)
    ax.set_ylabel('LP Error − MS Error (km)\n(Positive = MS wins)', fontsize=12)
    ax.set_title(f'Error Advantage vs Distance to Nearest Anchor — AS{ASN}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_mechanism_breakdown(df_merged, output_path):
    """Stacked bar chart: category breakdown for MS-wins vs LP-wins."""
    ms_wins = df_merged[df_merged['error_diff'] > 0]
    lp_wins = df_merged[df_merged['error_diff'] <= 0]

    # MS-wins categories
    ms_cats = ms_wins['category'].value_counts()
    lp_cats = lp_wins['category'].value_counts()

    all_cats = sorted(set(ms_cats.index) | set(lp_cats.index))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title, color_base in [
        (axes[0], ms_cats, f'MS Wins (n={len(ms_wins)})', 'Blues'),
        (axes[1], lp_cats, f'LP Wins (n={len(lp_wins)})', 'Grays'),
    ]:
        if len(data) == 0:
            ax.set_title(title)
            continue
        cats = data.index.tolist()
        counts = data.values
        colors = plt.cm.get_cmap(color_base)(np.linspace(0.3, 0.8, len(cats)))
        bars = ax.barh(cats, counts, color=colors, edgecolor='black', linewidth=0.5)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    str(count), va='center', fontsize=10)
        ax.set_xlabel('Number of Probes')
        ax.set_title(title, fontsize=13, fontweight='bold')

    plt.suptitle(f'Crossover Mechanism Breakdown — AS{ASN}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_side_by_side_map(probe_ip, ms_result, lp_result,
                          ms_circles_data, lp_circles_data,
                          ms_circle_tuples, lp_circle_tuples,
                          output_path):
    """Side-by-side map: MS (left) vs LP (right) for one probe."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    proj = ccrs.LambertConformal(central_longitude=-96, central_latitude=39)
    fig, axes = plt.subplots(1, 2, figsize=(24, 10),
                              subplot_kw={'projection': proj})

    true_lat = ms_result['true_lat']
    true_lon = ms_result['true_lon']

    for ax, result, circles_data, circle_tuples, method_name in [
        (axes[0], ms_result, ms_circles_data, ms_circle_tuples, 'Million-Scale (2/3c)'),
        (axes[1], lp_result, lp_circles_data, lp_circle_tuples, 'LP CBG'),
    ]:
        ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')
        ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff')
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
        ax.add_feature(cfeature.BORDERS, linewidth=0.8)
        ax.coastlines(resolution='50m', linewidth=0.8)

        colors = plt.cm.tab10(np.linspace(0, 1, max(len(circles_data), 10)))

        # Draw circles
        for i, (clat, clon, radius_km) in enumerate(circles_data):
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
            ax.plot(clon, clat, 's', color=colors[i], markersize=8,
                    transform=ccrs.PlateCarree(), zorder=5,
                    label=f'VP ({clat:.1f},{clon:.1f}) r={radius_km:.0f}km')

        # Shapely intersection polygon
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
                            xs, ys = poly.exterior.xy
                            label = 'Intersection region' if k == 0 else None
                            ax.fill(list(xs), list(ys), color='yellow', alpha=0.4,
                                    transform=ccrs.PlateCarree(), zorder=3, label=label)
                            ax.plot(list(xs), list(ys), color='orange', linewidth=2,
                                    transform=ccrs.PlateCarree(), zorder=4)
            except Exception:
                pass

        # True location
        ax.plot(true_lon, true_lat, '*', color='green', markersize=18,
                markeredgecolor='black', markeredgewidth=1,
                transform=ccrs.PlateCarree(), zorder=10, label='True location')

        # Estimated location
        est_lat = result.get('estimated_lat')
        est_lon = result.get('estimated_lon')
        if est_lat is not None:
            ax.plot(est_lon, est_lat, 'X', color='red', markersize=14,
                    markeredgecolor='black', markeredgewidth=1,
                    transform=ccrs.PlateCarree(), zorder=10, label='Estimate')

        err = result.get('error_km', 0)
        did_int = result.get('intersection', False)
        status = 'Intersection' if did_int else 'Fallback'
        ax.set_title(f'{method_name} — {status}\nError: {err:.0f} km',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='lower left', fontsize=7, ncol=2)

    plt.suptitle(f'Probe: {probe_ip}', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Bug impact analysis
# =============================================================================

def analyze_bug_impact(van_results, corrected_results):
    """Compare buggy LP vs corrected LP filtering."""
    van_by_ip = {r['probe_ip']: r for r in van_results}
    corr_by_ip = {r['probe_ip']: r for r in corrected_results}

    changed_intersection = 0
    changed_error = []
    total = 0

    for ip in van_by_ip:
        van = van_by_ip[ip]
        corr = corr_by_ip.get(ip)
        if van['error_km'] is None or corr is None or corr['error_km'] is None:
            continue
        total += 1

        if van['intersection'] != corr['intersection']:
            changed_intersection += 1

        changed_error.append(corr['error_km'] - van['error_km'])

    changed_error = np.array(changed_error)
    return {
        'total_probes': total,
        'changed_intersection_status': changed_intersection,
        'median_error_change': float(np.median(changed_error)) if len(changed_error) > 0 else 0,
        'mean_error_change': float(np.mean(changed_error)) if len(changed_error) > 0 else 0,
        'probes_with_error_change': int(np.sum(np.abs(changed_error) > 0.1)),
        'corrected_median_error': float(np.median([r['error_km'] for r in corrected_results if r['error_km'] is not None])),
        'buggy_median_error': float(np.median([r['error_km'] for r in van_results if r['error_km'] is not None])),
    }


# =============================================================================
# Summary
# =============================================================================

def print_summary(df_merged, bug_impact):
    """Print diagnostic summary."""
    w = 70
    print("\n" + "=" * w)
    print("CROSSOVER ANALYSIS SUMMARY")
    print("=" * w)

    total = len(df_merged)
    ms_wins = (df_merged['error_diff'] > 0).sum()
    lp_wins = (df_merged['error_diff'] < 0).sum()
    tied = (df_merged['error_diff'] == 0).sum()

    print(f"\nTotal probes: {total}")
    print(f"MS wins:  {ms_wins} ({ms_wins/total*100:.1f}%)")
    print(f"LP wins:  {lp_wins} ({lp_wins/total*100:.1f}%)")
    print(f"Tied:     {tied} ({tied/total*100:.1f}%)")

    print(f"\n{'Category':<40} {'Count':>6} {'Med Δ (km)':>12} {'Med MS err':>12} {'Med LP err':>12}")
    print("-" * w)
    for cat in sorted(df_merged['category'].unique()):
        subset = df_merged[df_merged['category'] == cat]
        n = len(subset)
        med_diff = np.median(subset['error_diff'])
        med_ms = np.median(subset['ms_error'])
        med_lp = np.median(subset['lp_error'])
        print(f"  {cat:<38} {n:>6} {med_diff:>12.1f} {med_ms:>12.1f} {med_lp:>12.1f}")

    # Intersection breakdown
    print(f"\n{'Intersection Status':<40} {'MS':>6} {'LP':>6}")
    print("-" * 52)
    ms_int = df_merged['ms_intersected'].sum()
    lp_int = df_merged['lp_intersected'].sum()
    print(f"  {'Intersected':<38} {ms_int:>6} {lp_int:>6}")
    print(f"  {'Fallback':<38} {total - ms_int:>6} {total - lp_int:>6}")

    # Bug impact
    print(f"\n{'Bug Impact (is_within_cirle)'}")
    print("-" * 52)
    print(f"  Probes with changed intersection status: {bug_impact['changed_intersection_status']}")
    print(f"  Probes with error change >0.1 km:        {bug_impact['probes_with_error_change']}")
    print(f"  LP median error (buggy):                 {bug_impact['buggy_median_error']:.1f} km")
    print(f"  LP median error (corrected):             {bug_impact['corrected_median_error']:.1f} km")
    print(f"  Median error change (corrected-buggy):   {bug_impact['median_error_change']:.1f} km")
    print("=" * w)


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data and run pipelines
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df, df_asn = load_data()

    print("\n" + "=" * 60)
    print("FITTING LP MODELS")
    print("=" * 60)
    lp_models = fit_lp_models(df_asn)

    print("\n" + "=" * 60)
    print("RUNNING MILLION-SCALE CBG")
    print("=" * 60)
    ms_results, _, _ = run_million_scale_cbg(df_asn)
    ms_success = [r for r in ms_results if r['error_km'] is not None]
    ms_intersected = sum(1 for r in ms_results if r['intersection'])
    print(f"  Successful: {len(ms_success)}/{len(ms_results)}")
    print(f"  Intersected: {ms_intersected}/{len(ms_results)}")
    print(f"  Median error: {np.median([r['error_km'] for r in ms_success]):.1f} km")

    print("\n" + "=" * 60)
    print("RUNNING LP CBG (buggy filter)")
    print("=" * 60)
    van_results, _, _ = run_vanilla_cbg(df_asn, lp_models)
    van_success = [r for r in van_results if r['error_km'] is not None]
    van_intersected = sum(1 for r in van_results if r['intersection'])
    print(f"  Successful: {len(van_success)}/{len(van_results)}")
    print(f"  Intersected: {van_intersected}/{len(van_results)}")
    print(f"  Median error: {np.median([r['error_km'] for r in van_success]):.1f} km")

    print("\n" + "=" * 60)
    print("RUNNING LP CBG (corrected filter)")
    print("=" * 60)
    corrected_results = run_corrected_lp_cbg(df_asn, lp_models)
    corr_success = [r for r in corrected_results if r['error_km'] is not None]
    corr_intersected = sum(1 for r in corrected_results if r['intersection'])
    print(f"  Successful: {len(corr_success)}/{len(corrected_results)}")
    print(f"  Intersected: {corr_intersected}/{len(corrected_results)}")
    print(f"  Median error: {np.median([r['error_km'] for r in corr_success]):.1f} km")

    # Step 2: Merge and classify
    print("\n" + "=" * 60)
    print("BUILDING DIAGNOSTICS")
    print("=" * 60)
    df_merged = build_merged_diagnostics(ms_results, van_results, corrected_results, df_asn)
    df_merged = classify_probes(df_merged)
    print(f"  Merged probes: {len(df_merged)}")
    print(f"  MS wins: {(df_merged['error_diff'] > 0).sum()}")
    print(f"  LP wins: {(df_merged['error_diff'] < 0).sum()}")

    # Bug impact
    bug_impact = analyze_bug_impact(van_results, corrected_results)

    # Step 3: Summary
    print_summary(df_merged, bug_impact)

    # Step 4: Plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    plot_crossover_cdf(df_merged, OUTPUT_DIR / 'crossover_cdf.png')
    plot_error_scatter(df_merged, OUTPUT_DIR / 'error_scatter.png')
    plot_dist_vs_advantage(df_merged, OUTPUT_DIR / 'dist_vs_advantage.png')
    plot_mechanism_breakdown(df_merged, OUTPUT_DIR / 'mechanism_breakdown.png')

    # Step 5: Per-probe maps for crossover probes (MS wins)
    print("\n" + "=" * 60)
    print("GENERATING PER-PROBE MAPS (MS wins)")
    print("=" * 60)

    anchor_info = df_asn[['dst_ip', 'anchor_latitude', 'anchor_longitude']].drop_duplicates()
    anchor_coords = {}
    for _, row in anchor_info.iterrows():
        anchor_coords[row['dst_ip']] = (row['anchor_latitude'], row['anchor_longitude'])

    ms_by_ip = {r['probe_ip']: r for r in ms_results}
    van_by_ip = {r['probe_ip']: r for r in van_results}

    crossover_probes = df_merged[df_merged['error_diff'] > 0].sort_values('error_diff', ascending=False)
    print(f"  Generating maps for {len(crossover_probes)} crossover probes...")

    for _, row in crossover_probes.iterrows():
        probe_ip = row['probe_ip']
        probe_data = df_asn[df_asn['src_ip'] == probe_ip]

        ms_circles_data, ms_circle_tuples = _build_ms_circles(probe_data, anchor_coords)
        lp_circles_data, lp_circle_tuples = _build_lp_circles(probe_data, lp_models)

        safe_ip = probe_ip.replace('.', '_')
        out_path = MAPS_DIR / f'probe_{safe_ip}.png'

        try:
            plot_side_by_side_map(
                probe_ip, ms_by_ip[probe_ip], van_by_ip[probe_ip],
                ms_circles_data, lp_circles_data,
                ms_circle_tuples, lp_circle_tuples,
                out_path,
            )
        except Exception as e:
            print(f"  WARNING: Failed to plot {probe_ip}: {e}")

    # Step 6: Save results JSON
    results_json = {
        'asn': ASN,
        'total_probes': len(df_merged),
        'ms_wins': int((df_merged['error_diff'] > 0).sum()),
        'lp_wins': int((df_merged['error_diff'] < 0).sum()),
        'category_breakdown': df_merged['category'].value_counts().to_dict(),
        'category_stats': {},
        'bug_impact': bug_impact,
    }
    for cat in df_merged['category'].unique():
        subset = df_merged[df_merged['category'] == cat]
        results_json['category_stats'][cat] = {
            'count': len(subset),
            'median_error_diff': float(np.median(subset['error_diff'])),
            'median_ms_error': float(np.median(subset['ms_error'])),
            'median_lp_error': float(np.median(subset['lp_error'])),
        }

    # Per-probe details
    results_json['probes'] = df_merged.to_dict(orient='records')

    json_path = OUTPUT_DIR / 'crossover_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
