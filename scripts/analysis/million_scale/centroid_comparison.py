"""
Compare arithmetic-mean vs Shapely geometric (area-weighted) centroid for CBG geolocation.

The arithmetic centroid averages intersection vertex coordinates, which can fall
outside the intersection polygon for skewed distributions.  The geometric centroid
is the area-weighted center of the actual intersection polygon via Shapely.

Produces:
  - 4-line Error CDF (MS-arith, MS-geom, Vanilla-arith, Vanilla-geom)
  - Per-probe delta scatter plots (one per CBG variant)
  - Statistics table and JSON results
"""

import json
import math
import sys
import time
import types
from functools import reduce
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# evaluate_million_scale imports Octant helpers at module load time.
# Provide a narrow fallback stub when geom_median is unavailable.
# try:
#     from geom_median.numpy import compute_geometric_median as _check  # noqa: F401
# except ModuleNotFoundError:
#     geom_median_module = types.ModuleType("geom_median")
#     geom_median_numpy_module = types.ModuleType("geom_median.numpy")

#     def _missing(*args, **kwargs):
#         raise ModuleNotFoundError("geom_median not needed for centroid comparison.")

#     geom_median_numpy_module.compute_geometric_median = _missing
#     geom_median_module.numpy = geom_median_numpy_module
#     sys.modules.setdefault("geom_median", geom_median_module)
#     sys.modules.setdefault("geom_median.numpy", geom_median_numpy_module)

from scripts.utils.helpers import haversine, circle_intersections  # noqa: E402
from scripts.analysis.million_scale.evaluate_million_scale import (  # noqa: E402
    ASN,
    load_data,
    fit_lp_models,
    run_million_scale_cbg,
    run_vanilla_cbg,
    _circle_to_shapely_polygon,
    compute_intersection_area,
)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'centroid_comparison'


# =============================================================================
# Core: Shapely geometric centroid
# =============================================================================

def compute_shapely_centroid(circles_data):
    """Compute the area-weighted centroid of the intersection of circles.

    Args:
        circles_data: list of (lat, lon, radius_km)

    Returns:
        (lat, lon) or None if intersection is empty/degenerate.
    """
    if len(circles_data) == 0:
        return None
    if len(circles_data) == 1:
        return (circles_data[0][0], circles_data[0][1])
    try:
        shapely_circles = [
            _circle_to_shapely_polygon(clat, clon, radius_km)
            for clat, clon, radius_km in circles_data
        ]
        shapely_circles = [p for p in shapely_circles if p.is_valid and not p.is_empty]
        if len(shapely_circles) < 2:
            return (circles_data[0][0], circles_data[0][1]) if shapely_circles else None
        intersection_poly = reduce(lambda a, b: a.intersection(b), shapely_circles)
        if intersection_poly.is_empty:
            return None
        if intersection_poly.geom_type not in ('Polygon', 'MultiPolygon'):
            return None
        c = intersection_poly.centroid
        return (c.y, c.x)  # Shapely: x=lon, y=lat
    except Exception:
        return None


# =============================================================================
# Geometric centroid CBG runner
# =============================================================================

def run_geom_centroid_cbg(df_asn, method, lp_models=None):
    """Run CBG with Shapely geometric centroid instead of arithmetic mean.

    Args:
        df_asn: DataFrame filtered to target ASN.
        method: 'million_scale' or 'vanilla'.
        lp_models: required when method == 'vanilla'.

    Returns:
        (results, all_radii, all_areas) — same structure as the arithmetic versions.
    """
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

        if method == 'million_scale':
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
        else:  # vanilla
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
                'error_km': None, 'n_anchors': 0,
                'method': f'{method}_geom', 'intersection': False,
                'avg_radius_km': None, 'intersection_area_km2': 0.0,
            })
            continue

        # Preprocess circles (removes fully-contained circles, etc.)
        _, circles_out = circle_intersections(circles, speed_threshold=2/3)

        # Build (lat, lon, radius_km) for Shapely
        if method == 'million_scale':
            circles_data = [(lat, lon, 100.0 * rtt) for lat, lon, rtt, _, _ in circles_out]
        else:
            circles_data = [(lat, lon, d) for lat, lon, _, d, _ in circles_out]

        radii_km = [r for _, _, r in circles_data]
        all_radii.extend(radii_km)

        # Geometric centroid
        centroid = compute_shapely_centroid(circles_data)
        area_km2 = compute_intersection_area(circles_data)
        all_areas.append(area_km2)

        if centroid is not None:
            did_intersect = True
        else:
            # Fallback: closest VP by min RTT
            closest_vp, _ = min(min_rtt_per_vp_ip.items(), key=lambda x: x[1])
            if method == 'million_scale':
                centroid = anchor_coords[closest_vp]
            else:
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
            'method': f'{method}_geom',
            'intersection': did_intersect,
            'avg_radius_km': float(np.mean(radii_km)),
            'intersection_area_km2': area_km2,
        })

    return results, np.array(all_radii), np.array(all_areas)


# =============================================================================
# Plotting
# =============================================================================

def plot_centroid_comparison_cdf(ms_arith_errors, ms_geom_errors,
                                 van_arith_errors, van_geom_errors,
                                 output_path=None, title_suffix=''):
    """Plot 4-line Error CDF comparing centroid methods."""
    fig, ax = plt.subplots(figsize=(12, 8))

    all_series = [
        (van_arith_errors, 'Vanilla — Arithmetic', 'black', '-'),
        (van_geom_errors,  'Vanilla — Geometric',  'black', '--'),
        (ms_arith_errors,  'Million-Scale — Arithmetic', 'blue', '-'),
        (ms_geom_errors,   'Million-Scale — Geometric',  'blue', '--'),
    ]
    short_names = ['V-Ar', 'V-Ge', 'MS-Ar', 'MS-Ge']

    for errors, label, color, ls in all_series:
        sorted_e = np.sort(errors)
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        median = np.median(errors)
        ax.plot(sorted_e, cdf, color=color, linestyle=ls, linewidth=2,
                label=f'{label}\n  Median: {median:.0f} km, N={len(errors)}')

    # Threshold lines
    all_errors_list = [van_arith_errors, van_geom_errors,
                       ms_arith_errors, ms_geom_errors]
    for thresh, color in [(100, 'green'), (500, 'orange'), (1000, 'red')]:
        parts = []
        for errors, name in zip(all_errors_list, short_names):
            parts.append(f'{name}={np.mean(errors <= thresh) * 100:.1f}%')
        ax.axvline(x=thresh, color=color, linestyle='--', alpha=0.5,
                   label=f'{thresh} km: {", ".join(parts)}')

    ax.hlines(y=0.5, xmin=0, xmax=3000, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Error Distance (km)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title(f'Centroid Comparison — Error CDF — AS{ASN}{title_suffix}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.88), fontsize=9)
    ax.grid(True, alpha=0.3)
    max_err = max(e.max() for e in all_errors_list)
    ax.set_xlim(0, min(max_err * 1.05, 3000))
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    return fig


def plot_centroid_delta_scatter(arith_errors, geom_errors, method_name,
                                output_path=None):
    """Scatter: arithmetic error (x) vs geometric error (y). Below y=x → geometric wins."""
    fig, ax = plt.subplots(figsize=(10, 10))

    max_val = max(arith_errors.max(), geom_errors.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5, label='y = x')

    geom_better = geom_errors < arith_errors
    arith_better = arith_errors < geom_errors
    tied = geom_errors == arith_errors

    ax.scatter(arith_errors[geom_better], geom_errors[geom_better],
               c='green', s=30, alpha=0.6, edgecolors='none',
               label=f'Geometric better ({geom_better.sum()})')
    ax.scatter(arith_errors[arith_better], geom_errors[arith_better],
               c='red', s=30, alpha=0.6, edgecolors='none',
               label=f'Arithmetic better ({arith_better.sum()})')
    ax.scatter(arith_errors[tied], geom_errors[tied],
               c='gray', s=30, alpha=0.6, edgecolors='none',
               label=f'Tied ({tied.sum()})')

    ax.set_xlabel('Arithmetic Centroid Error (km)', fontsize=12)
    ax.set_ylabel('Geometric Centroid Error (km)', fontsize=12)
    ax.set_title(f'{method_name} — Arithmetic vs Geometric Centroid Error\nAS{ASN}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(max_val, 3000))
    ax.set_ylim(0, min(max_val, 3000))
    ax.set_aspect('equal')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    return fig


# =============================================================================
# Statistics
# =============================================================================

def print_centroid_stats(ms_arith_errors, ms_geom_errors,
                         van_arith_errors, van_geom_errors):
    """Print side-by-side statistics for all 4 variants."""
    cols = [
        ('MS Arith', ms_arith_errors),
        ('MS Geom', ms_geom_errors),
        ('Van Arith', van_arith_errors),
        ('Van Geom', van_geom_errors),
    ]

    w = 25 + 16 * len(cols)
    print("\n" + "=" * w)
    print("CENTROID COMPARISON — STATISTICS")
    print("=" * w)

    header = f"{'Metric':<25}" + "".join(f" {name:>15}" for name, _ in cols)
    print(header)
    print("-" * w)

    metrics = [
        ('N (probes)', lambda e: f"{len(e)}"),
        ('Median (km)', lambda e: f"{np.median(e):.1f}"),
        ('Mean (km)', lambda e: f"{np.mean(e):.1f}"),
        ('Std (km)', lambda e: f"{np.std(e):.1f}"),
        ('25th pct (km)', lambda e: f"{np.percentile(e, 25):.1f}"),
        ('75th pct (km)', lambda e: f"{np.percentile(e, 75):.1f}"),
        ('90th pct (km)', lambda e: f"{np.percentile(e, 90):.1f}"),
        ('95th pct (km)', lambda e: f"{np.percentile(e, 95):.1f}"),
    ]
    for label, fn in metrics:
        row = f"{label:<25}" + "".join(f" {fn(e):>15}" for _, e in cols)
        print(row)

    print()
    header2 = f"{'Accuracy Thresholds':<25}" + "".join(f" {name:>15}" for name, _ in cols)
    print(header2)
    print("-" * w)
    for thresh in [50, 100, 250, 500, 1000]:
        row = f"  Within {thresh:4d} km        "
        for _, e in cols:
            pct = np.mean(e <= thresh) * 100
            row += f" {pct:>14.1f}%"
        print(row)

    print("=" * w)


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    # --- Load data & fit models ---
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df, df_asn = load_data()

    print("\n" + "=" * 60)
    print("FITTING LP MODELS")
    print("=" * 60)
    lp_models = fit_lp_models(df_asn)

    # --- Run arithmetic baselines ---
    print("\n" + "=" * 60)
    print("RUNNING ARITHMETIC CENTROID — VANILLA CBG")
    print("=" * 60)
    van_arith_results, _, _ = run_vanilla_cbg(df_asn, lp_models)
    van_arith_success = [r for r in van_arith_results if r['error_km'] is not None]
    van_arith_errors = np.array([r['error_km'] for r in van_arith_success])
    print(f"  Median error: {np.median(van_arith_errors):.1f} km")

    print("\n" + "=" * 60)
    print("RUNNING ARITHMETIC CENTROID — MILLION-SCALE CBG")
    print("=" * 60)
    ms_arith_results, _, _ = run_million_scale_cbg(df_asn)
    ms_arith_success = [r for r in ms_arith_results if r['error_km'] is not None]
    ms_arith_errors = np.array([r['error_km'] for r in ms_arith_success])
    print(f"  Median error: {np.median(ms_arith_errors):.1f} km")

    # --- Run geometric centroid variants ---
    print("\n" + "=" * 60)
    print("RUNNING GEOMETRIC CENTROID — VANILLA CBG")
    print("=" * 60)
    van_geom_results, _, _ = run_geom_centroid_cbg(df_asn, 'vanilla', lp_models)
    van_geom_success = [r for r in van_geom_results if r['error_km'] is not None]
    van_geom_errors = np.array([r['error_km'] for r in van_geom_success])
    print(f"  Median error: {np.median(van_geom_errors):.1f} km")

    print("\n" + "=" * 60)
    print("RUNNING GEOMETRIC CENTROID — MILLION-SCALE CBG")
    print("=" * 60)
    ms_geom_results, _, _ = run_geom_centroid_cbg(df_asn, 'million_scale')
    ms_geom_success = [r for r in ms_geom_results if r['error_km'] is not None]
    ms_geom_errors = np.array([r['error_km'] for r in ms_geom_success])
    print(f"  Median error: {np.median(ms_geom_errors):.1f} km")

    # --- Plots ---
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    fig = plot_centroid_comparison_cdf(
        ms_arith_errors, ms_geom_errors,
        van_arith_errors, van_geom_errors,
        output_path=OUTPUT_DIR / 'centroid_comparison_cdf.png',
    )
    plt.close(fig)

    # Per-probe delta scatter — pair by probe_ip
    for method_name, arith_results, geom_results in [
        ('Million-Scale', ms_arith_success, ms_geom_success),
        ('Vanilla', van_arith_success, van_geom_success),
    ]:
        arith_by_ip = {r['probe_ip']: r['error_km'] for r in arith_results}
        geom_by_ip = {r['probe_ip']: r['error_km'] for r in geom_results}
        common_ips = sorted(set(arith_by_ip) & set(geom_by_ip))
        if not common_ips:
            continue
        paired_arith = np.array([arith_by_ip[ip] for ip in common_ips])
        paired_geom = np.array([geom_by_ip[ip] for ip in common_ips])
        tag = method_name.lower().replace('-', '_').replace(' ', '_')
        fig = plot_centroid_delta_scatter(
            paired_arith, paired_geom, method_name,
            output_path=OUTPUT_DIR / f'centroid_delta_scatter_{tag}.png',
        )
        plt.close(fig)

    # --- Statistics ---
    print_centroid_stats(ms_arith_errors, ms_geom_errors,
                         van_arith_errors, van_geom_errors)

    elapsed = time.perf_counter() - total_start
    print(f"\nTotal runtime: {elapsed:.1f} s")

    # --- Save JSON ---
    def _stats(errors):
        return {
            'n_probes': len(errors),
            'median_km': float(np.median(errors)),
            'mean_km': float(np.mean(errors)),
            'p75_km': float(np.percentile(errors, 75)),
            'p90_km': float(np.percentile(errors, 90)),
        }

    results_json = {
        'asn': ASN,
        'runtime_sec': elapsed,
        'million_scale_arith': _stats(ms_arith_errors),
        'million_scale_geom': _stats(ms_geom_errors),
        'vanilla_arith': _stats(van_arith_errors),
        'vanilla_geom': _stats(van_geom_errors),
    }
    json_path = OUTPUT_DIR / 'centroid_comparison_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == '__main__':
    main()
