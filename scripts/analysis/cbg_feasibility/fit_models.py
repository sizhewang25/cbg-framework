"""
Fit RTT-Distance Models for CBG Feasibility Analysis

This script:
1. Loads the Vultr ping dataset
2. Filters by target ASN (default: 7922 Comcast)
3. Fits RTT-distance models for each Vultr anchor
4. Generates scatter plots with bestline overlay
5. Saves models and summary to output directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

from rtt_model import RTTDistanceModel, haversine_distance, THEORETICAL_SLOPE


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the Vultr ping dataset."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} measurements")
    print(f"  Unique anchors: {df['dst_ip'].nunique()}")
    print(f"  Unique probes: {df['src_ip'].nunique()}")
    print(f"  Unique ASNs: {df['probe_asn'].nunique()}")
    return df


def get_anchor_info(df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique anchor information."""
    anchors = df[['dst_ip', 'anchor_latitude', 'anchor_longitude', 'anchor_city']].drop_duplicates()
    anchors = anchors.rename(columns={
        'dst_ip': 'ip',
        'anchor_latitude': 'lat',
        'anchor_longitude': 'lon',
        'anchor_city': 'city'
    })
    return anchors.reset_index(drop=True)


def filter_by_asn(df: pd.DataFrame, asn: int) -> pd.DataFrame:
    """Filter dataset to specific ASN."""
    filtered = df[df['probe_asn'] == float(asn)].copy()
    print(f"\nFiltered to ASN {asn}:")
    print(f"  Measurements: {len(filtered)}")
    print(f"  Unique probes: {filtered['src_ip'].nunique()}")
    return filtered


def calculate_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance column to dataframe."""
    df = df.copy()
    df['distance_km'] = df.apply(
        lambda row: haversine_distance(
            row['probe_latitude'], row['probe_longitude'],
            row['anchor_latitude'], row['anchor_longitude']
        ),
        axis=1
    )
    return df


def fit_anchor_model(
    df: pd.DataFrame,
    anchor_ip: str,
    anchor_lat: float,
    anchor_lon: float,
    bin_size_km: float = 50.0,
    percentile: float = 0.05
) -> RTTDistanceModel:
    """Fit RTT-distance model for a single anchor."""
    # Filter to this anchor
    anchor_data = df[df['dst_ip'] == anchor_ip].copy()

    if len(anchor_data) == 0:
        model = RTTDistanceModel(
            anchor_ip=anchor_ip,
            anchor_lat=anchor_lat,
            anchor_lon=anchor_lon
        )
        model.fit_message = "No data for this anchor"
        return model

    # Get distances and RTTs
    distances = anchor_data['distance_km'].values
    rtts = anchor_data['min_rtt'].values

    # Create and fit model
    model = RTTDistanceModel(
        anchor_ip=anchor_ip,
        anchor_lat=anchor_lat,
        anchor_lon=anchor_lon
    )
    model.fit(distances, rtts, bin_size_km=bin_size_km, percentile=percentile)

    return model


def plot_scatter_with_bestline(
    df: pd.DataFrame,
    model: RTTDistanceModel,
    output_path: Path,
    asn: int,
    show_theoretical: bool = True
) -> None:
    """
    Create scatter plot of RTT vs distance with bestline overlay.

    Shows:
    - All data points as scatter
    - Binned percentile points used for fitting
    - Fitted bestline
    - Theoretical 2/3c baseline for comparison
    """
    # Filter to this anchor
    anchor_data = df[df['dst_ip'] == model.anchor_ip].copy()

    if len(anchor_data) == 0:
        print(f"  No data to plot for {model.anchor_ip}")
        return

    distances = anchor_data['distance_km'].values
    rtts = anchor_data['min_rtt'].values

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot all data points
    ax.scatter(distances, rtts, alpha=0.4, s=20, c='blue', label='Measurements', edgecolors='none')

    # Plot range for lines
    dist_min, dist_max = distances.min(), distances.max()
    dist_range = np.linspace(dist_min, dist_max, 100)

    # Plot theoretical baseline (2/3 c)
    if show_theoretical:
        theoretical_rtts = THEORETICAL_SLOPE * dist_range
        ax.plot(dist_range, theoretical_rtts, 'g--', linewidth=2, alpha=0.7,
                label=f'Theoretical (2/3c): {THEORETICAL_SLOPE:.4f} ms/km')

    if model.fitted:
        # Plot bin centers and percentile RTTs
        ax.scatter(model.bin_centers, model.bin_rtts, s=100, c='red', marker='s',
                   edgecolors='darkred', linewidth=1.5, zorder=5,
                   label=f'5th percentile per bin (n={model.n_bins})')

        # Plot fitted bestline
        bestline_rtts = model.slope * dist_range + model.intercept
        ax.plot(dist_range, bestline_rtts, 'r-', linewidth=2.5,
                label=f'Bestline: {model.slope:.4f} ms/km + {model.intercept:.1f} ms (R²={model.r_squared:.3f})')

    # Labels and title
    ax.set_xlabel('Distance (km)', fontsize=12)
    ax.set_ylabel('RTT (ms)', fontsize=12)
    ax.set_title(
        f'RTT vs Distance - Anchor {model.anchor_ip}\n'
        f'ASN {asn} probes (n={len(anchor_data)})',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set axis limits with some padding
    ax.set_xlim(0, dist_max * 1.05)
    ax.set_ylim(0, rtts.max() * 1.1)

    # Add statistics text box
    stats_text = (
        f"Measurements: {len(anchor_data)}\n"
        f"Distance range: {dist_min:.0f} - {dist_max:.0f} km\n"
        f"RTT range: {rtts.min():.1f} - {rtts.max():.1f} ms"
    )
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {output_path.name}")


def fit_all_anchors(
    df: pd.DataFrame,
    asn: int,
    output_dir: Path,
    bin_size_km: float = 50.0,
    percentile: float = 0.05
) -> Dict[str, RTTDistanceModel]:
    """
    Fit models for all anchors and save results.

    Returns dict of anchor_ip -> RTTDistanceModel
    """
    # Filter by ASN
    asn_df = filter_by_asn(df, asn)

    if len(asn_df) == 0:
        print(f"No data for ASN {asn}")
        return {}

    # Calculate distances
    asn_df = calculate_distances(asn_df)

    # Get anchor info
    anchors = get_anchor_info(asn_df)
    print(f"\nAnchors to process: {len(anchors)}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fit model for each anchor
    models = {}
    summary = []

    for _, anchor in anchors.iterrows():
        anchor_ip = anchor['ip']
        print(f"\nProcessing anchor {anchor_ip}...")

        # Fit model
        model = fit_anchor_model(
            asn_df,
            anchor_ip=anchor_ip,
            anchor_lat=anchor['lat'],
            anchor_lon=anchor['lon'],
            bin_size_km=bin_size_km,
            percentile=percentile
        )

        models[anchor_ip] = model

        # Print results
        if model.fitted:
            print(f"  SUCCESS: slope={model.slope:.6f} ms/km, "
                  f"intercept={model.intercept:.2f} ms, "
                  f"R²={model.r_squared:.4f}, "
                  f"bins={model.n_bins}")
        else:
            print(f"  FAILED: {model.fit_message}")

        # Save model pickle
        model_path = output_dir / f"{anchor_ip.replace('.', '_')}.pkl"
        model.save(model_path)
        print(f"  Saved model: {model_path.name}")

        # Generate scatter plot
        plot_path = output_dir / f"scatter_{anchor_ip.replace('.', '_')}.png"
        plot_scatter_with_bestline(asn_df, model, plot_path, asn)

        # Add to summary
        summary.append(model.to_dict())

    # Save summary JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'asn': asn,
            'total_measurements': len(asn_df),
            'unique_probes': asn_df['src_ip'].nunique(),
            'bin_size_km': bin_size_km,
            'percentile': percentile,
            'theoretical_slope': THEORETICAL_SLOPE,
            'models': summary
        }, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    return models


def print_summary(models: Dict[str, RTTDistanceModel]) -> None:
    """Print summary of all fitted models."""
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    successful = [m for m in models.values() if m.fitted]
    failed = [m for m in models.values() if not m.fitted]

    print(f"Total anchors: {len(models)}")
    print(f"Successfully fitted: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        slopes = [m.slope for m in successful]
        intercepts = [m.intercept for m in successful]
        r_squareds = [m.r_squared for m in successful]

        print(f"\nSlope statistics (ms/km):")
        print(f"  Mean: {np.mean(slopes):.6f}")
        print(f"  Std:  {np.std(slopes):.6f}")
        print(f"  Min:  {np.min(slopes):.6f}")
        print(f"  Max:  {np.max(slopes):.6f}")
        print(f"  Theoretical (2/3c): {THEORETICAL_SLOPE:.6f}")

        print(f"\nIntercept statistics (ms):")
        print(f"  Mean: {np.mean(intercepts):.2f}")
        print(f"  Std:  {np.std(intercepts):.2f}")

        print(f"\nR² statistics:")
        print(f"  Mean: {np.mean(r_squareds):.4f}")
        print(f"  Min:  {np.min(r_squareds):.4f}")

    if failed:
        print(f"\nFailed models:")
        for m in failed:
            print(f"  {m.anchor_ip}: {m.fit_message}")


def main():
    parser = argparse.ArgumentParser(description='Fit RTT-Distance models for CBG analysis')
    parser.add_argument('--data', type=str, default='data/vultr_pings_us_only.csv',
                        help='Path to input data CSV')
    parser.add_argument('--asn', type=int, default=7922,
                        help='Target ASN (default: 7922 Comcast)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: outputs/vultr-{ASN}-rtt-models)')
    parser.add_argument('--bin-size', type=float, default=50.0,
                        help='Distance bin size in km (default: 50)')
    parser.add_argument('--percentile', type=float, default=0.05,
                        help='Percentile for lower envelope (default: 0.05)')

    args = parser.parse_args()

    # Set paths
    script_dir = Path(__file__).parent
    data_path = script_dir / args.data

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = script_dir / f"outputs/vultr-{args.asn}-rtt-models"

    print("=" * 60)
    print("CBG FEASIBILITY - RTT-DISTANCE MODEL FITTING")
    print("=" * 60)
    print(f"Data: {data_path}")
    print(f"Target ASN: {args.asn}")
    print(f"Output: {output_dir}")
    print(f"Bin size: {args.bin_size} km")
    print(f"Percentile: {args.percentile}")
    print("=" * 60)

    # Load data
    df = load_data(data_path)

    # Fit models
    models = fit_all_anchors(
        df,
        asn=args.asn,
        output_dir=output_dir,
        bin_size_km=args.bin_size,
        percentile=args.percentile
    )

    # Print summary
    print_summary(models)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
