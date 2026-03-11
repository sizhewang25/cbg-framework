"""
Compare Million-Scale CBG vs Vanilla CBG on Vultr US dataset.

Million-Scale CBG: Theoretical 2/3c model with spherical circle intersection
  (from IMC 2012 paper replication, scripts/utils/helpers.py)

Vanilla CBG: LP bestline model with Shapely polygon intersection
  (from filter_demonstration.py, scripts/analysis/cbg_feasibility/rtt_model.py)

Produces:
  - Comparative Error CDF plot
  - Per-anchor RTT-distance scatter plots with both model lines
  - Statistics table
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root and cbg_feasibility to path (filter_demonstration.py uses relative imports)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CBG_DIR = PROJECT_ROOT / 'scripts' / 'analysis' / 'cbg_feasibility'
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CBG_DIR))

from scripts.utils.helpers import (
    select_best_guess_centroid,
    haversine,
)
from scripts.analysis.cbg_feasibility.rtt_model import (
    RTTDistanceModel,
    haversine_distance,
    THEORETICAL_SLOPE,
    fit_bestline_lp,
)
from scripts.analysis.cbg_feasibility.filter_demonstration import (
    evaluate_cbg_probe,
    find_circles_intersection,
    estimate_location_fallback,
)

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

ASN = 7922
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'comparison'


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
    Run Million-Scale CBG using select_best_guess_centroid().

    This uses the original code path:
    - RTT → distance via rtt_to_km(rtt, speed_threshold=2/3)
    - Spherical circle intersection
    - Polygon centroid or closest-VP fallback
    """
    # Build anchor coordinate lookup
    anchors = df_asn[['dst_ip', 'anchor_latitude', 'anchor_longitude']].drop_duplicates()
    vp_coordinates_per_ip = {}
    for _, row in anchors.iterrows():
        vp_coordinates_per_ip[row['dst_ip']] = (row['anchor_latitude'], row['anchor_longitude'])

    # Also add probe coordinates (needed for error computation — the target must be
    # in vp_coordinates_per_ip for compute_error, but select_best_guess_centroid
    # skips self-measurement via target_ip == vp_ip check)
    probes = df_asn[['src_ip', 'probe_latitude', 'probe_longitude']].drop_duplicates('src_ip')
    for _, row in probes.iterrows():
        vp_coordinates_per_ip[row['src_ip']] = (row['probe_latitude'], row['probe_longitude'])

    probe_ips = df_asn['src_ip'].unique()
    results = []

    for probe_ip in probe_ips:
        probe_data = df_asn[df_asn['src_ip'] == probe_ip]
        true_lat = probe_data['probe_latitude'].iloc[0]
        true_lon = probe_data['probe_longitude'].iloc[0]

        # Build RTT dict: {anchor_ip: [min_rtt]}
        rtt_per_vp_to_target = {}
        for _, row in probe_data.iterrows():
            anchor_ip = row['dst_ip']
            rtt_per_vp_to_target[anchor_ip] = [row['min_rtt']]

        # Run million-scale CBG
        result = select_best_guess_centroid(
            probe_ip, vp_coordinates_per_ip, rtt_per_vp_to_target
        )

        if result is not None:
            (est_lat, est_lon), circles = result
            error_km = haversine((est_lat, est_lon), (true_lat, true_lon))
            results.append({
                'probe_ip': probe_ip,
                'true_lat': true_lat,
                'true_lon': true_lon,
                'estimated_lat': float(est_lat),
                'estimated_lon': float(est_lon),
                'error_km': float(error_km),
                'n_anchors': len(circles),
                'method': 'million_scale_cbg'
            })
        else:
            results.append({
                'probe_ip': probe_ip,
                'true_lat': true_lat,
                'true_lon': true_lon,
                'estimated_lat': None,
                'estimated_lon': None,
                'error_km': None,
                'n_anchors': 0,
                'method': 'million_scale_cbg'
            })

    return results


# =============================================================================
# Vanilla CBG Evaluation
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


def run_cbg_multilateration(df, df_asn, models):
    """
    Run Vanilla CBG Multilateration using evaluate_cbg_probe().

    This uses the existing code path from filter_demonstration.py:
    - RTT → distance via LP bestline inversion
    - Shapely polygon intersection
    - Weighted average fallback
    """
    probe_ips = df_asn['src_ip'].unique()
    results = []

    for probe_ip in probe_ips:
        probe_data = df_asn[df_asn['src_ip'] == probe_ip]
        result = evaluate_cbg_probe(probe_ip, probe_data, models)
        result['method'] = 'vanilla_cbg'
        results.append(result)

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_error_cdf_comparison(ms_errors, cal_errors, output_path=None):
    """Plot comparative Error CDF for both methods."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for errors, label, color, ls in [
        (ms_errors, 'Million-Scale CBG (2/3c + Spherical)', 'blue', '-'),
        (cal_errors, 'Vanilla CBG (LP + Shapely)', 'green', '-'),
    ]:
        sorted_e = np.sort(errors)
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        median = np.median(errors)
        ax.plot(sorted_e, cdf, color=color, linestyle=ls, linewidth=2,
                label=f'{label}\n  Median: {median:.0f} km, N={len(errors)}')

    # Threshold lines
    for thresh, color in [(100, 'green'), (500, 'orange'), (1000, 'red')]:
        ms_pct = np.mean(ms_errors <= thresh) * 100
        cal_pct = np.mean(cal_errors <= thresh) * 100
        ax.axvline(x=thresh, color=color, linestyle='--', alpha=0.5,
                   label=f'{thresh} km: MS={ms_pct:.1f}%, Cal={cal_pct:.1f}%')

    ax.hlines(y=0.5, xmin=0, xmax=3000, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Error Distance (km)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title(f'CBG Geolocation Error CDF Comparison — AS{ASN}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.9), fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(max(ms_errors.max(), cal_errors.max()) * 1.05, 3000))
    ax.set_ylim(0, 1)

    # Statistics text box
    stats = (
        f"Million-Scale CBG:\n"
        f"  Median: {np.median(ms_errors):.0f} km\n"
        f"  Mean: {np.mean(ms_errors):.0f} km\n"
        f"  75th: {np.percentile(ms_errors, 75):.0f} km\n"
        f"  90th: {np.percentile(ms_errors, 90):.0f} km\n\n"
        f"Vanilla CBG:\n"
        f"  Median: {np.median(cal_errors):.0f} km\n"
        f"  Mean: {np.mean(cal_errors):.0f} km\n"
        f"  75th: {np.percentile(cal_errors, 75):.0f} km\n"
        f"  90th: {np.percentile(cal_errors, 90):.0f} km"
    )
    ax.text(0.98, 0.02, stats, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


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


def print_statistics(ms_errors, cal_errors):
    """Print comparison statistics table."""
    print("\n" + "=" * 70)
    print("CBG MULTILATERATION COMPARISON — STATISTICS")
    print("=" * 70)

    header = f"{'Metric':<25} {'Million-Scale CBG':>20} {'Vanilla CBG':>20}"
    print(header)
    print("-" * 70)

    rows = [
        ('N (probes)', f"{len(ms_errors)}", f"{len(cal_errors)}"),
        ('Median (km)', f"{np.median(ms_errors):.1f}", f"{np.median(cal_errors):.1f}"),
        ('Mean (km)', f"{np.mean(ms_errors):.1f}", f"{np.mean(cal_errors):.1f}"),
        ('Std (km)', f"{np.std(ms_errors):.1f}", f"{np.std(cal_errors):.1f}"),
        ('Min (km)', f"{np.min(ms_errors):.1f}", f"{np.min(cal_errors):.1f}"),
        ('Max (km)', f"{np.max(ms_errors):.1f}", f"{np.max(cal_errors):.1f}"),
        ('25th pct (km)', f"{np.percentile(ms_errors, 25):.1f}", f"{np.percentile(cal_errors, 25):.1f}"),
        ('75th pct (km)', f"{np.percentile(ms_errors, 75):.1f}", f"{np.percentile(cal_errors, 75):.1f}"),
        ('90th pct (km)', f"{np.percentile(ms_errors, 90):.1f}", f"{np.percentile(cal_errors, 90):.1f}"),
        ('95th pct (km)', f"{np.percentile(ms_errors, 95):.1f}", f"{np.percentile(cal_errors, 95):.1f}"),
    ]
    for label, ms_val, cal_val in rows:
        print(f"{label:<25} {ms_val:>20} {cal_val:>20}")

    print()
    print(f"{'Accuracy Thresholds':<25} {'Million-Scale CBG':>20} {'Vanilla CBG':>20}")
    print("-" * 70)
    for thresh in [50, 100, 250, 500, 1000]:
        ms_pct = np.mean(ms_errors <= thresh) * 100
        cal_pct = np.mean(cal_errors <= thresh) * 100
        print(f"  Within {thresh:4d} km         {ms_pct:>19.1f}% {cal_pct:>19.1f}%")

    print("=" * 70)


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

    # Step 2: Run Million-Scale CBG
    print("\n" + "=" * 60)
    print("RUNNING MILLION-SCALE CBG")
    print("=" * 60)
    ms_results = run_million_scale_cbg(df_asn)
    ms_success = [r for r in ms_results if r['error_km'] is not None]
    ms_errors = np.array([r['error_km'] for r in ms_success])
    print(f"  Successful: {len(ms_success)}/{len(ms_results)} probes")
    print(f"  Median error: {np.median(ms_errors):.1f} km")

    # Step 3: Fit LP models and run Vanilla CBG
    print("\n" + "=" * 60)
    print("FITTING LP MODELS & RUNNING Vanilla CBG")
    print("=" * 60)
    lp_models = fit_lp_models(df_asn)
    cal_results = run_cbg_multilateration(df, df_asn, lp_models)
    cal_success = [r for r in cal_results if r['error_km'] is not None]
    cal_errors = np.array([r['error_km'] for r in cal_success])
    print(f"  Successful: {len(cal_success)}/{len(cal_results)} probes")
    print(f"  Median error: {np.median(cal_errors):.1f} km")

    # Step 4: Per-anchor RTT-distance scatter
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

    # Step 5: Comparative Error CDF
    print("\n" + "=" * 60)
    print("GENERATING ERROR CDF COMPARISON")
    print("=" * 60)
    cdf_path = OUTPUT_DIR / 'error_cdf_comparison.png'
    fig = plot_error_cdf_comparison(ms_errors, cal_errors, output_path=cdf_path)
    plt.close(fig)

    # Step 6: Statistics
    print_statistics(ms_errors, cal_errors)

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
            'total_probes': len(cal_results),
            'successful': len(cal_success),
            'median_km': float(np.median(cal_errors)),
            'mean_km': float(np.mean(cal_errors)),
            'p75_km': float(np.percentile(cal_errors, 75)),
            'p90_km': float(np.percentile(cal_errors, 90)),
        },
    }
    json_path = OUTPUT_DIR / 'comparison_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved: {json_path}")


if __name__ == '__main__':
    main()
