"""
Compare Octant cutoff variants on the Vultr US dataset.

Variants:
1. baseline_no_cutoff
2. high_cutoff_only
3. low_cutoff_only
4. low_high_cutoff
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.libs.cbg_feasibility.rtt_model import haversine_distance
from scripts.libs.octant.octant_evaluation import fit_octant_models, run_octant_cbg


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

DEFAULT_ASN = 7922
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'cutoff_variants'
THRESHOLDS_KM = (50, 100, 250, 500, 1000)
VARIANT_CONFIGS = {
    'baseline_no_cutoff': {
        'cutoff_variant': 'none',
        'label': 'Baseline (No Cutoff)',
        'color': 'black',
        'linestyle': '-',
    },
    'high_cutoff_only': {
        'cutoff_variant': 'high_only',
        'label': 'High Cutoff Only',
        'color': 'tab:blue',
        'linestyle': '--',
    },
    'low_cutoff_only': {
        'cutoff_variant': 'low_only',
        'label': 'Low Cutoff Only',
        'color': 'tab:orange',
        'linestyle': '-.',
    },
    'low_high_cutoff': {
        'cutoff_variant': 'both',
        'label': 'Low + High Cutoffs',
        'color': 'tab:green',
        'linestyle': ':',
    },
}


def load_data(asn: int):
    """Load the Vultr US dataset and compute probe-anchor distances."""
    data_path = PROJECT_ROOT / 'datasets' / 'cbg_test' / 'vultr_pings_us_only.csv'
    df = pd.read_csv(data_path)

    df_asn = df[df['probe_asn'] == float(asn)].copy()
    df_asn['distance_km'] = df_asn.apply(
        lambda row: haversine_distance(
            row['probe_latitude'],
            row['probe_longitude'],
            row['anchor_latitude'],
            row['anchor_longitude'],
        ),
        axis=1,
    )

    print(f"Total measurements: {len(df)}")
    print(f"AS{asn} measurements: {len(df_asn)}")
    print(f"Unique anchors: {df_asn['dst_ip'].nunique()}")
    print(f"Unique probes: {df_asn['src_ip'].nunique()}")
    return data_path, df_asn


def compute_statistics(errors: np.ndarray) -> dict:
    """Return summary statistics for an error array."""
    if len(errors) == 0:
        return {
            'n': 0,
            'median_km': None,
            'mean_km': None,
            'std_km': None,
            'min_km': None,
            'max_km': None,
            'p25_km': None,
            'p75_km': None,
            'p90_km': None,
            'p95_km': None,
        }

    return {
        'n': int(len(errors)),
        'median_km': float(np.median(errors)),
        'mean_km': float(np.mean(errors)),
        'std_km': float(np.std(errors)),
        'min_km': float(np.min(errors)),
        'max_km': float(np.max(errors)),
        'p25_km': float(np.percentile(errors, 25)),
        'p75_km': float(np.percentile(errors, 75)),
        'p90_km': float(np.percentile(errors, 90)),
        'p95_km': float(np.percentile(errors, 95)),
    }


def compute_accuracy_thresholds(errors: np.ndarray) -> dict:
    """Return threshold accuracy percentages for an error array."""
    return {
        f'within_{threshold}km': (
            float(np.mean(errors <= threshold) * 100.0) if len(errors) > 0 else None
        )
        for threshold in THRESHOLDS_KM
    }


def collect_anchor_metadata(models: dict) -> dict:
    """Collect per-anchor fit metadata for JSON reporting."""
    anchor_metadata = {}
    for anchor_ip, model in models.items():
        anchor_metadata[anchor_ip] = {
            'fit_success': bool(model.fitted),
            'fit_message': model.fit_message,
            'cutoff_variant': model.cutoff_variant,
            'detected_low_cutoff_rtt': float(model.low_cutoff_rtt),
            'detected_high_cutoff_rtt': float(model.cutoff_rtt),
            'reliable_min_rtt': float(model.reliable_min_rtt),
            'reliable_max_rtt': float(model.reliable_max_rtt),
            'n_measurements': int(model.n_measurements),
        }
    return anchor_metadata


def print_summary_table(variant_results: dict) -> None:
    """Print a compact console summary table for all variants."""
    print("\n" + "=" * 168)
    print("OCTANT CUTOFF VARIANT COMPARISON")
    print("=" * 168)
    header = (
        f"{'Variant':<24} {'N':>6} {'Delta':>10} {'Median':>10} {'Mean':>10} "
        f"{'Std':>10} {'P25':>10} {'P75':>10} {'P90':>10} {'P95':>10} "
        f"{'50km':>8} {'100km':>8} {'250km':>8} {'500km':>8} {'1000km':>8}"
    )
    print(header)
    print("-" * 168)

    for variant_name, result in variant_results.items():
        stats = result['statistics']
        thresholds = result['accuracy_thresholds']

        def _fmt(value, digits=1):
            if value is None:
                return "N/A"
            return f"{value:.{digits}f}"

        row = (
            f"{variant_name:<24} "
            f"{stats['n']:>6d} "
            f"{_fmt(result['shared_delta'], 4):>10} "
            f"{_fmt(stats['median_km']):>10} "
            f"{_fmt(stats['mean_km']):>10} "
            f"{_fmt(stats['std_km']):>10} "
            f"{_fmt(stats['p25_km']):>10} "
            f"{_fmt(stats['p75_km']):>10} "
            f"{_fmt(stats['p90_km']):>10} "
            f"{_fmt(stats['p95_km']):>10} "
            f"{_fmt(thresholds['within_50km']):>8} "
            f"{_fmt(thresholds['within_100km']):>8} "
            f"{_fmt(thresholds['within_250km']):>8} "
            f"{_fmt(thresholds['within_500km']):>8} "
            f"{_fmt(thresholds['within_1000km']):>8}"
        )
        print(row)

    print("=" * 168)


def plot_error_cdf(variant_results: dict, output_path: Path) -> None:
    """Plot a four-line error CDF comparison."""
    fig, ax = plt.subplots(figsize=(12, 8))
    max_err = 0.0

    for variant_name, config in VARIANT_CONFIGS.items():
        errors = variant_results[variant_name]['errors']
        if len(errors) == 0:
            continue

        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        max_err = max(max_err, float(sorted_errors[-1]))
        ax.plot(
            sorted_errors,
            cdf,
            color=config['color'],
            linestyle=config['linestyle'],
            linewidth=2.5,
            label=(
                f"{config['label']}\n"
                f"  Median: {np.median(errors):.0f} km, N={len(errors)}"
            ),
        )

    ax.hlines(y=0.5, xmin=0, xmax=3000, color='gray', linestyle='--', alpha=0.4)
    ax.set_xlabel('Error Distance (km)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title(
        'Octant Cutoff Variant Error CDF Comparison',
        fontsize=14,
        fontweight='bold',
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, min(max_err * 1.05 if max_err > 0 else 3000, 3000))
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compare Octant cutoff variants on the Vultr US dataset.'
    )
    parser.add_argument('--asn', type=int, default=DEFAULT_ASN)
    parser.add_argument('--target-coverage', type=float, default=0.80)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    dataset_path, df_asn = load_data(args.asn)
    variant_results = {}

    for variant_name, config in VARIANT_CONFIGS.items():
        print("\n" + "=" * 72)
        print(f"PROCESSING {variant_name.upper()}")
        print("=" * 72)

        fit_start = time.perf_counter()
        models, delta = fit_octant_models(
            df_asn,
            target_coverage=args.target_coverage,
            cutoff_variant=config['cutoff_variant'],
            verbose=True,
        )
        fit_runtime_sec = time.perf_counter() - fit_start

        eval_start = time.perf_counter()
        results, _, _, benchmarks = run_octant_cbg(
            df_asn,
            models,
            delta,
            method_name=variant_name,
        )
        evaluation_runtime_sec = time.perf_counter() - eval_start

        success = [result for result in results if result['error_km'] is not None]
        errors = np.array([result['error_km'] for result in success], dtype=float)
        stats = compute_statistics(errors)
        thresholds = compute_accuracy_thresholds(errors)

        print(f"  Successful: {len(success)}/{len(results)} probes")
        if len(errors) > 0:
            print(f"  Median error: {np.median(errors):.1f} km")
            print(f"  Mean error: {np.mean(errors):.1f} km")
            print(f"  Shared delta: {delta:.4f}" if delta is not None else "  Shared delta: N/A")

        variant_results[variant_name] = {
            'label': config['label'],
            'cutoff_variant': config['cutoff_variant'],
            'shared_delta': float(delta) if delta is not None else None,
            'total_probes': int(len(results)),
            'successful_probes': int(len(success)),
            'statistics': stats,
            'accuracy_thresholds': thresholds,
            'fit_runtime_sec': float(fit_runtime_sec),
            'evaluation_runtime_sec': float(evaluation_runtime_sec),
            'geolocation_benchmarks': benchmarks,
            'anchors': collect_anchor_metadata(models),
            'probes': results,
            'errors': errors,
        }

    print_summary_table(variant_results)

    cdf_path = output_dir / 'error_cdf.png'
    plot_error_cdf(variant_results, cdf_path)

    json_payload = {
        'asn': int(args.asn),
        'target_coverage': float(args.target_coverage),
        'generated_at': datetime.now().isoformat(),
        'dataset_path': str(dataset_path),
        'total_runtime_sec': float(time.perf_counter() - total_start),
        'variants': {},
    }

    for variant_name, result in variant_results.items():
        payload = dict(result)
        payload.pop('errors', None)
        json_payload['variants'][variant_name] = payload

    json_path = output_dir / 'comparison_results.json'
    with open(json_path, 'w') as handle:
        json.dump(json_payload, handle, indent=2)
    print(f"Saved: {json_path}")


if __name__ == '__main__':
    main()
