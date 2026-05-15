"""
Compare CBG Error CDFs for Different Filter Configurations

This script fits RTT-distance models with 4 different filter configurations
and compares their CBG geolocation accuracy via error CDFs.

Filter Configurations:
1. Minimal: Only invalid + baseline filter (Stages 1 & 4)
2. + Bin σ: Add per-bin mean±σ filter (Stages 1, 2, & 4)
3. + Global σ: Add global bin-min filter (Stages 1, 2, 3, & 4)
4. Full: Add per-bin 5th percentile (all 5 stages)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Polygon
import json
import argparse

from rtt_model import (
    RTTDistanceModel,
    haversine_distance,
    fit_bestline_lp,
    THEORETICAL_SLOPE
)


# Filter configurations
FILTER_CONFIGS = {
    '1. Minimal (Invalid + Baseline)': {
        'enable_bin_filter': False,
        'enable_percentile_filter': False,
        'enable_global_filter': False,
        'enable_baseline_filter': True,
        'color': 'red',
        'linestyle': '--'
    },
    '2. + Bin σ Filter': {
        'enable_bin_filter': True,
        'enable_percentile_filter': False,
        'enable_global_filter': False,
        'enable_baseline_filter': True,
        'color': 'orange',
        'linestyle': '-.'
    },
    '3. + Global σ Filter': {
        'enable_bin_filter': True,
        'enable_percentile_filter': False,
        'enable_global_filter': True,
        'enable_baseline_filter': True,
        'color': 'blue',
        'linestyle': ':'
    },
    '4. Full (+ Percentile)': {
        'enable_bin_filter': True,
        'enable_percentile_filter': True,
        'enable_global_filter': True,
        'enable_baseline_filter': True,
        'color': 'green',
        'linestyle': '-'
    }
}


def create_circle_polygon(center_lat, center_lon, radius_km, num_points=64):
    """Create polygon approximation of a circle on Earth's surface."""
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(center_lat))
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radius_deg_lat = radius_km / km_per_deg_lat
    radius_deg_lon = radius_km / km_per_deg_lon
    lons = center_lon + radius_deg_lon * np.cos(angles)
    lats = center_lat + radius_deg_lat * np.sin(angles)
    return Polygon(list(zip(lons, lats)))


def find_circles_intersection(circles):
    """Find intersection of multiple circles, return centroid."""
    if len(circles) == 0:
        return None, None
    lat, lon, radius = circles[0]
    intersection = create_circle_polygon(lat, lon, radius)
    for lat, lon, radius in circles[1:]:
        circle_poly = create_circle_polygon(lat, lon, radius)
        intersection = intersection.intersection(circle_poly)
        if intersection.is_empty:
            return None, None
    centroid = intersection.centroid
    return centroid.y, centroid.x


def estimate_location_fallback(circles):
    """Fallback: weighted average of anchor positions."""
    total_weight = 0
    weighted_lat = 0
    weighted_lon = 0
    for lat, lon, radius in circles:
        weight = 1.0 / max(radius, 1.0)
        weighted_lat += lat * weight
        weighted_lon += lon * weight
        total_weight += weight
    if total_weight > 0:
        return weighted_lat / total_weight, weighted_lon / total_weight
    return None, None


def fit_models_with_config(df_asn, anchors, config):
    """Fit RTT-distance models for all anchors with given filter config."""
    models = {}
    for _, anchor in anchors.iterrows():
        anchor_ip = anchor['ip']
        anchor_data = df_asn[df_asn['dst_ip'] == anchor_ip]
        distances = anchor_data['distance_km'].values
        rtts = anchor_data['min_rtt'].values

        result = fit_bestline_lp(
            distances=distances,
            rtts=rtts,
            baseline_slope=THEORETICAL_SLOPE,
            filter_outliers=True,
            bin_size_km=100.0,
            n_std=1.0,
            global_n_std=1.0,
            bin_percentile=0.05,
            enable_bin_filter=config['enable_bin_filter'],
            enable_percentile_filter=config['enable_percentile_filter'],
            enable_global_filter=config['enable_global_filter'],
            enable_baseline_filter=config['enable_baseline_filter']
        )

        if result['success']:
            model = RTTDistanceModel(
                anchor_ip=anchor_ip,
                anchor_lat=anchor['lat'],
                anchor_lon=anchor['lon']
            )
            model.slope = result['slope']
            model.intercept = result['intercept']
            model.fitted = True
            models[anchor_ip] = model

    return models


def evaluate_cbg(df_asn, models):
    """Evaluate CBG for all probes, return error distances."""
    probe_ips = df_asn['src_ip'].unique()
    errors = []

    for probe_ip in probe_ips:
        probe_data = df_asn[df_asn['src_ip'] == probe_ip]
        true_lat = probe_data['probe_latitude'].iloc[0]
        true_lon = probe_data['probe_longitude'].iloc[0]

        circles = []
        for _, row in probe_data.iterrows():
            anchor_ip = row['dst_ip']
            rtt = row['min_rtt']
            if anchor_ip not in models:
                continue
            model = models[anchor_ip]
            max_distance = model.predict_distance(rtt)
            if max_distance is None or max_distance <= 0:
                continue
            circles.append((model.anchor_lat, model.anchor_lon, max_distance))

        if len(circles) < 2:
            continue

        est_lat, est_lon = find_circles_intersection(circles)
        if est_lat is None:
            est_lat, est_lon = estimate_location_fallback(circles)
        if est_lat is None:
            continue

        error_km = haversine_distance(true_lat, true_lon, est_lat, est_lon)
        errors.append(error_km)

    return np.array(errors)


def plot_cdf_comparison(all_errors, asn, output_path=None):
    """Plot CDF comparison for all filter configurations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    for config_name, config in FILTER_CONFIGS.items():
        errors = all_errors[config_name]
        errors_sorted = np.sort(errors)
        cdf = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)

        median = np.median(errors)
        label = f'{config_name} (median: {median:.0f} km)'

        ax.plot(errors_sorted, cdf,
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=2.5,
                label=label)

    ax.set_xlabel('Error Distance (km)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title(f'CBG Error CDF by Filter Configuration\nAS{asn}, 7 Vultr anchors',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 1)

    # Add threshold lines
    for thresh in [100, 500, 1000]:
        ax.axvline(x=thresh, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(thresh + 20, 0.02, f'{thresh}km', fontsize=9, color='gray')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {output_path}')

    return fig


def main():
    parser = argparse.ArgumentParser(description='Compare CBG Error CDFs for Different Filters')
    parser.add_argument('--data', type=str, default='data/vultr_pings_us_only.csv',
                        help='Path to input data CSV')
    parser.add_argument('--asn', type=int, default=7922,
                        help='Target ASN (default: 7922 Comcast)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for CDF plot')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_path = script_dir / args.data

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = script_dir / f'outputs/vultr-{args.asn}-rtt-models/error_cdf_filter_comparison.png'

    print("=" * 60)
    print("CBG ERROR CDF FILTER COMPARISON")
    print("=" * 60)
    print(f"Data: {data_path}")
    print(f"ASN: {args.asn}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Load data
    df = pd.read_csv(data_path)
    df_asn = df[df['probe_asn'] == float(args.asn)].copy()

    # Calculate distances
    df_asn['distance_km'] = df_asn.apply(
        lambda row: haversine_distance(
            row['probe_latitude'], row['probe_longitude'],
            row['anchor_latitude'], row['anchor_longitude']
        ), axis=1
    )

    # Get anchor info
    anchors = df_asn[['dst_ip', 'anchor_latitude', 'anchor_longitude']].drop_duplicates()
    anchors = anchors.rename(columns={'dst_ip': 'ip', 'anchor_latitude': 'lat', 'anchor_longitude': 'lon'})

    print(f"\nDataset: {len(df_asn['src_ip'].unique())} probes, {len(anchors)} anchors")

    # Evaluate each filter configuration
    all_errors = {}
    results_summary = []

    for config_name, config in FILTER_CONFIGS.items():
        print(f"\nProcessing: {config_name}")

        # Fit models
        models = fit_models_with_config(df_asn, anchors, config)
        print(f"  Fitted {len(models)} models")

        # Evaluate CBG
        errors = evaluate_cbg(df_asn, models)
        all_errors[config_name] = errors

        # Statistics
        median = np.median(errors)
        mean = np.mean(errors)
        p25 = np.percentile(errors, 25)
        p75 = np.percentile(errors, 75)
        within_100 = np.mean(errors <= 100) * 100
        within_500 = np.mean(errors <= 500) * 100
        within_1000 = np.mean(errors <= 1000) * 100

        print(f"  Probes: {len(errors)}")
        print(f"  Median: {median:.0f} km, Mean: {mean:.0f} km")
        print(f"  25th-75th: {p25:.0f}-{p75:.0f} km")
        print(f"  Within 100km: {within_100:.1f}%, 500km: {within_500:.1f}%, 1000km: {within_1000:.1f}%")

        results_summary.append({
            'config': config_name,
            'n_probes': len(errors),
            'median_km': median,
            'mean_km': mean,
            'p25_km': p25,
            'p75_km': p75,
            'within_100km': within_100,
            'within_500km': within_500,
            'within_1000km': within_1000
        })

    # Plot comparison
    print("\n" + "=" * 60)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_cdf_comparison(all_errors, args.asn, output_path)

    # Save summary JSON
    summary_path = output_path.with_suffix('.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'asn': args.asn,
            'configs': results_summary
        }, f, indent=2)
    print(f'Saved: {summary_path}')

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
