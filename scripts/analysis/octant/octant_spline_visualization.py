"""
Octant Spline Visualization

Visualizes the piecewise linear spline RTT-distance model against real Vultr ping data.
Axes: RTT (ms) on x-axis, distance (km) on y-axis — matching OctantRTTModel's interface.

Per anchor, plots:
- Raw (RTT, distance) scatter
- Upper hull R_L (red) and lower hull r_L (blue) with shaded annular region
- Piecewise linear spline (orange)
- Low / high cutoff RTT vertical lines (bilateral)
- Theoretical 2/3c line (black dashed)
- Spline extended with 2/3c slope outside [low_cutoff, high_cutoff]
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Path setup
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent

cbg_dir = str(script_dir.parent / 'cbg_feasibility')
if cbg_dir not in sys.path:
    sys.path.insert(0, cbg_dir)
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from rtt_model import haversine_distance
from octant_model import (
    OctantRTTModel,
    hull_rtt_to_distance,
    find_delta_for_coverage,
    THEORETICAL_SLOPE,
)

ASN = 7922
plt.style.use('seaborn-v0_8-whitegrid')


def plot_anchor(anchor_ip, anchor_city, rtts, distances, model, output_path):
    fig, ax = plt.subplots(figsize=(12, 8))

    # --- Scatter ---
    ax.scatter(rtts, distances, alpha=0.25, s=15, c='gray', edgecolors='none',
               label=f'Measurements (n={len(rtts)}, valid after 2/3c filter)')

    # --- RTT range for curves ---
    max_rtt = min(rtts.max(), model.cutoff_rtt * 1.5)
    rtt_range = np.linspace(0.1, max_rtt, 300)

    # --- Theoretical 2/3c: distance = rtt / THEORETICAL_SLOPE ---
    theoretical_dists = rtt_range / THEORETICAL_SLOPE
    ax.plot(rtt_range, theoretical_dists, 'k--', linewidth=1.5, alpha=0.5,
            label=f'Theoretical 2/3c')

    # --- Low / high cutoff vertical lines ---
    low_cut = getattr(model, 'low_cutoff_rtt', 0.0)
    if low_cut > 0:
        ax.axvline(x=low_cut, color='purple', linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f'Low cutoff RTT = {low_cut:.1f} ms')
    ax.axvline(x=model.cutoff_rtt, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
               label=f'High cutoff RTT = {model.cutoff_rtt:.1f} ms')

    # --- Upper and lower hull ---
    upper_dists = np.array([
        hull_rtt_to_distance(r, model.hull_upper_rtts, model.hull_upper_distances,
                             model.cutoff_rtt, is_upper=True)
        for r in rtt_range
    ])
    lower_dists = np.array([
        hull_rtt_to_distance(r, model.hull_lower_rtts, model.hull_lower_distances,
                             model.cutoff_rtt, is_upper=False)
        for r in rtt_range
    ])
    # Below low cutoff: linear ramp from (0,0) to value at low_cutoff (upper); lower stays 0
    if low_cut > 0:
        upper_at_low_cut = float(np.interp(low_cut, rtt_range, upper_dists))
        upper_dists = np.where(
            rtt_range < low_cut,
            (rtt_range / low_cut) * upper_at_low_cut,
            upper_dists
        )
        lower_dists = np.where(rtt_range < low_cut, 0.0, lower_dists)

    ax.plot(rtt_range, upper_dists, 'r-', linewidth=2,
            label=f'Upper hull R_L ({len(model.hull_upper_rtts)} vertices)')
    ax.plot(rtt_range, lower_dists, 'b-', linewidth=2,
            label=f'Lower hull r_L ({len(model.hull_lower_rtts)} vertices)')
    ax.fill_between(rtt_range, lower_dists, upper_dists, color='green', alpha=0.07,
                    label='Annular region')

    # --- Piecewise linear spline with 2/3c extension outside reliable region ---
    if model.spline_rtt_knots is not None:
        knot_rtts = np.array(model.spline_rtt_knots)
        knot_dists = np.array(model.spline_dist_knots)
        spline_at_cutoff = float(np.interp(model.cutoff_rtt, knot_rtts, knot_dists))

        # Base interpolation (flat extrapolation outside knot range)
        spline_base = np.interp(rtt_range, knot_rtts, knot_dists)
        # Below low cutoff: linear ramp from (0,0) to spline value at low_cutoff
        if low_cut > 0:
            spline_at_low_cut = float(np.interp(low_cut, knot_rtts, knot_dists))
            spline_base = np.where(
                rtt_range < low_cut,
                (rtt_range / low_cut) * spline_at_low_cut,
                spline_base
            )
        # Above high cutoff: extend with 2/3c slope from cutoff value
        spline_dists = np.where(
            rtt_range > model.cutoff_rtt,
            spline_at_cutoff + (rtt_range - model.cutoff_rtt) / THEORETICAL_SLOPE,
            spline_base
        )
        # Spline: dotted within reliable region, solid outside
        in_reliable = (rtt_range >= low_cut) & (rtt_range <= model.cutoff_rtt)
        # Outside reliable region (solid)
        outside = ~in_reliable
        ax.plot(rtt_range, np.where(outside, spline_dists, np.nan),
                color='darkorange', linewidth=2,
                label=f'Spline ({model.spline_n_knots} knots)')
        # Inside reliable region (dotted)
        ax.plot(rtt_range, np.where(in_reliable, spline_dists, np.nan),
                color='darkorange', linewidth=2, linestyle=':')

        # --- Delta spline band (80% coverage) ---
        try:
            reliable_mask = (rtts >= low_cut) & (rtts <= model.cutoff_rtt)
            rel_rtts = rtts[reliable_mask]
            rel_dists = distances[reliable_mask]
            delta, delta_meta = find_delta_for_coverage(
                rel_rtts, rel_dists, knot_rtts, knot_dists,
                target_coverage=0.80
            )
            coverage_pct = delta_meta['actual_coverage'] * 100

            # Band only within reliable region; outside → collapse to spline
            in_band = (rtt_range >= low_cut) & (rtt_range <= model.cutoff_rtt)

            delta_upper = np.where(in_band, spline_base * delta, spline_dists)
            delta_lower = np.where(in_band, np.maximum(spline_base / delta, 0.0), spline_dists)

            # Clip by hull bounds within reliable region
            delta_upper = np.where(in_band, np.minimum(delta_upper, upper_dists), delta_upper)
            delta_lower = np.where(in_band, np.maximum(delta_lower, lower_dists), delta_lower)

            ax.plot(rtt_range, np.where(in_band, delta_upper, np.nan),
                    color='darkorange', linewidth=2,
                    linestyle='-', alpha=0.8, label=f'Delta upper (δ={delta:.2f})')
            ax.plot(rtt_range, np.where(in_band, delta_lower, np.nan),
                    color='darkorange', linewidth=2,
                    linestyle='-', alpha=0.8, label=f'Delta lower ({coverage_pct:.0f}% coverage)')
            ax.fill_between(rtt_range, delta_lower, delta_upper,
                            where=in_band, color='darkorange', alpha=0.12,
                            label=f'Delta band ({coverage_pct:.0f}%)')
        except Exception:
            pass  # Skip delta band if delta search fails

    # --- Labels ---
    ax.set_xlabel('RTT (ms)', fontsize=12)
    ax.set_ylabel('Distance (km)', fontsize=12)
    ax.set_title(f'Octant Spline Model — {anchor_city} ({anchor_ip})\nAS{ASN}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(0, max_rtt * 1.05)
    ax.set_ylim(0, distances.max() * 1.05)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.yaxis.set_major_locator(plt.MultipleLocator(500))

    # --- Stats box ---
    stats_text = model.fit_message
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    data_path = project_root / 'datasets' / 'cbg_test' / 'vultr_pings_us_only.csv'
    df = pd.read_csv(data_path)
    df_asn = df[df['probe_asn'] == float(ASN)].copy()

    df_asn['distance_km'] = df_asn.apply(
        lambda row: haversine_distance(
            row['probe_latitude'], row['probe_longitude'],
            row['anchor_latitude'], row['anchor_longitude']
        ), axis=1
    )

    anchors = (df_asn[['dst_ip', 'anchor_latitude', 'anchor_longitude', 'anchor_city']]
               .drop_duplicates()
               .rename(columns={'dst_ip': 'ip', 'anchor_latitude': 'lat',
                                'anchor_longitude': 'lon', 'anchor_city': 'city'}))

    output_dir = script_dir / 'outputs' / 'vultr-7922-octant-spline'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"AS{ASN}: {len(anchors)} anchors, {len(df_asn)} measurements")
    print(f"Output: {output_dir}\n")
    print(f"{'Anchor IP':<18} {'City':<20} {'n_pts':<8} {'removed':<10} {'cutoff_rtt':<12} {'n_knots':<10} {'R²':<8}")
    print("-" * 86)

    for _, anchor in anchors.iterrows():
        anchor_ip = anchor['ip']
        anchor_data = df_asn[df_asn['dst_ip'] == anchor_ip]
        rtts = anchor_data['min_rtt'].values
        distances = anchor_data['distance_km'].values

        # Remove points violating speed-of-internet constraint (RTT < distance * THEORETICAL_SLOPE)
        valid_mask = rtts >= distances * THEORETICAL_SLOPE
        n_removed = (~valid_mask).sum()
        rtts = rtts[valid_mask]
        distances = distances[valid_mask]

        model = OctantRTTModel(anchor_ip=anchor_ip, anchor_lat=anchor['lat'], anchor_lon=anchor['lon'])
        model.fit(rtts, distances)

        if not model.fitted:
            print(f"{anchor_ip:<18} {'FAILED':<20} {len(rtts):<8}")
            continue

        # Extract R² from fit_message
        r2_str = ''
        if 'R²=' in model.fit_message:
            r2_str = model.fit_message.split('R²=')[-1].split()[0]

        n_knots = model.spline_n_knots if model.spline_rtt_knots is not None else 'N/A'
        print(f"{anchor_ip:<18} {anchor['city']:<20} {len(rtts):<8} "
              f"{n_removed:<10} {model.cutoff_rtt:<12.1f} {str(n_knots):<10} {r2_str:<8}")

        output_path = output_dir / f"scatter_{anchor_ip.replace('.', '_')}.png"
        plot_anchor(anchor_ip, anchor['city'], rtts, distances, model, output_path)

    print(f"\nDone. {len(anchors)} plots saved to {output_dir}")


if __name__ == '__main__':
    main()
