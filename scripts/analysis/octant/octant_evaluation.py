"""
Shared Octant model fitting and geolocation evaluation helpers.

These helpers centralize the Octant comparison flow so multiple scripts can
reuse the same location-estimation path and benchmark collection.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from scripts.analysis.octant.octant_model import (
    OctantRTTModel,
    find_delta_for_coverage,
)
from scripts.analysis.octant.octant_geolocation import (
    estimate_location,
    form_constraints,
)
from scripts.utils.helpers import haversine


DEFAULT_ESTIMATE_LOCATION_KWARGS: Dict[str, Any] = {
    'method': 'weighted',
    'n_samples': 5000,
    'weight_threshold': 0.5,
    'grid_resolution_deg': 0.25,
    'n_pts': 128,
    'collect_benchmark': True,
}


def fit_octant_models(
    df_asn,
    target_coverage: float = 0.80,
    cutoff_variant: str = 'high_only',
    cutoff_min_points: int = 5,
    bin_size_ms: float = 5.0,
    verbose: bool = True,
) -> Tuple[Dict[str, OctantRTTModel], Optional[float]]:
    """Fit Octant RTT-distance models per anchor and compute a shared delta."""
    anchors = df_asn[
        ['dst_ip', 'anchor_latitude', 'anchor_longitude', 'anchor_city']
    ].drop_duplicates()
    models: Dict[str, OctantRTTModel] = {}

    all_rtts = []
    all_distances = []

    for _, anchor in anchors.iterrows():
        anchor_ip = anchor['dst_ip']
        anchor_data = df_asn[df_asn['dst_ip'] == anchor_ip]

        rtts = anchor_data['min_rtt'].values
        distances = anchor_data['distance_km'].values

        model = OctantRTTModel(
            anchor_ip=anchor_ip,
            anchor_lat=anchor['anchor_latitude'],
            anchor_lon=anchor['anchor_longitude'],
            cutoff_variant=cutoff_variant,
        )
        success = model.fit(
            rtts,
            distances,
            cutoff_min_points=cutoff_min_points,
            bin_size_ms=bin_size_ms,
        )
        models[anchor_ip] = model

        if success and model.spline_rtt_knots is not None:
            all_rtts.extend(rtts.tolist())
            all_distances.extend(distances.tolist())

        if verbose:
            status = model.fit_message if model.fitted else "FAILED"
            print(f"  Octant model {anchor_ip} [{cutoff_variant}]: {status}")

    delta = None
    fitted_models = [
        model
        for model in models.values()
        if model.fitted and model.spline_rtt_knots is not None
    ]
    if all_rtts and fitted_models:
        ref_model = fitted_models[0]
        try:
            delta, delta_meta = find_delta_for_coverage(
                np.array(all_rtts),
                np.array(all_distances),
                np.array(ref_model.spline_rtt_knots),
                np.array(ref_model.spline_dist_knots),
                target_coverage=target_coverage,
            )
            if verbose:
                print(
                    f"  Shared delta [{cutoff_variant}]: "
                    f"{delta:.4f} (coverage={delta_meta['actual_coverage']:.3f})"
                )
        except Exception as exc:  # pragma: no cover - exercised in script flows
            if verbose:
                print(
                    f"  Delta search failed [{cutoff_variant}]: {exc}, "
                    "using hull bounds only"
                )

    return models, delta


def run_octant_cbg(
    df_asn,
    octant_models: Dict[str, OctantRTTModel],
    delta: Optional[float],
    method_name: str = 'octant_cbg',
    rng_seed: int = 42,
    estimate_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Run Octant CBG using the shared weighted-region geolocation flow.

    This preserves the current evaluation configuration used by the
    million-scale comparison script.
    """
    anchors = df_asn[['dst_ip', 'anchor_latitude', 'anchor_longitude']].drop_duplicates()
    anchor_coords = {}
    for _, row in anchors.iterrows():
        anchor_coords[row['dst_ip']] = (row['anchor_latitude'], row['anchor_longitude'])

    probe_ips = df_asn['src_ip'].unique()
    results = []
    all_outer_radii = []
    all_areas = []
    benchmark_totals = {
        'collect_rtts_sec': 0.0,
        'form_constraints_sec': 0.0,
        'estimate_location_sec': 0.0,
        'weighted_region_sec': 0.0,
        'weighted_low_threshold_sec': 0.0,
        'unweighted_region_sec': 0.0,
        'sample_points_sec': 0.0,
        'geometric_median_sec': 0.0,
        'region_centroid_sec': 0.0,
        'centroid_fallback_sec': 0.0,
        'probe_total_sec': 0.0,
    }
    method_counts: Dict[str, int] = {}
    fallback_count = 0
    rng = np.random.default_rng(rng_seed)
    location_kwargs = dict(DEFAULT_ESTIMATE_LOCATION_KWARGS)
    if estimate_kwargs:
        location_kwargs.update(estimate_kwargs)

    for probe_ip in probe_ips:
        probe_start = time.perf_counter()
        probe_data = df_asn[df_asn['src_ip'] == probe_ip]
        true_lat = probe_data['probe_latitude'].iloc[0]
        true_lon = probe_data['probe_longitude'].iloc[0]

        collect_start = time.perf_counter()
        rtt_measurements = {}
        for _, row in probe_data.iterrows():
            anchor_ip = row['dst_ip']
            rtt = row['min_rtt']
            if anchor_ip in octant_models and octant_models[anchor_ip].fitted:
                rtt_measurements[anchor_ip] = rtt
        benchmark_totals['collect_rtts_sec'] += time.perf_counter() - collect_start

        if not rtt_measurements:
            benchmark_totals['probe_total_sec'] += time.perf_counter() - probe_start
            results.append({
                'probe_ip': probe_ip,
                'true_lat': true_lat,
                'true_lon': true_lon,
                'estimated_lat': None,
                'estimated_lon': None,
                'error_km': None,
                'n_anchors': 0,
                'method': method_name,
                'intersection': False,
                'avg_radius_km': None,
                'intersection_area_km2': 0.0,
            })
            continue

        constraints_start = time.perf_counter()
        constraints = form_constraints(
            probe_ip,
            rtt_measurements,
            anchor_coords,
            octant_models,
            delta=delta,
        )
        benchmark_totals['form_constraints_sec'] += time.perf_counter() - constraints_start

        outer_radii = [c.outer_radius_km for c in constraints]
        all_outer_radii.extend(outer_radii)

        if not constraints:
            results.append({
                'probe_ip': probe_ip,
                'true_lat': true_lat,
                'true_lon': true_lon,
                'estimated_lat': None,
                'estimated_lon': None,
                'error_km': None,
                'n_anchors': 0,
                'method': method_name,
                'intersection': False,
                'avg_radius_km': None,
                'intersection_area_km2': 0.0,
            })
            continue

        estimate_start = time.perf_counter()
        estimate = estimate_location(
            constraints,
            rng=rng,
            **location_kwargs,
        )
        benchmark_totals['estimate_location_sec'] += time.perf_counter() - estimate_start

        if estimate is None:
            benchmark_totals['probe_total_sec'] += time.perf_counter() - probe_start
            results.append({
                'probe_ip': probe_ip,
                'true_lat': true_lat,
                'true_lon': true_lon,
                'estimated_lat': None,
                'estimated_lon': None,
                'error_km': None,
                'n_anchors': len(constraints),
                'method': method_name,
                'intersection': False,
                'avg_radius_km': float(np.mean(outer_radii)) if outer_radii else None,
                'intersection_area_km2': 0.0,
                'geolocation_method': None,
                'fallback': True,
            })
            continue

        benchmark = estimate.get('benchmark_sec', {})
        for key in (
            'weighted_region_sec',
            'weighted_low_threshold_sec',
            'unweighted_region_sec',
            'sample_points_sec',
            'geometric_median_sec',
            'region_centroid_sec',
            'centroid_fallback_sec',
        ):
            benchmark_totals[key] += benchmark.get(key, 0.0)

        est_lat = estimate['lat']
        est_lon = estimate['lon']
        area_km2 = float(estimate['region_area_km2'])
        all_areas.append(area_km2)
        did_intersect = estimate['method'] != 'centroid_fallback'
        method_counts[estimate['method']] = method_counts.get(estimate['method'], 0) + 1
        fallback_count += int(bool(estimate['fallback']))

        error_km = haversine((est_lat, est_lon), (true_lat, true_lon))
        benchmark_totals['probe_total_sec'] += time.perf_counter() - probe_start

        results.append({
            'probe_ip': probe_ip,
            'true_lat': true_lat,
            'true_lon': true_lon,
            'estimated_lat': float(est_lat),
            'estimated_lon': float(est_lon),
            'error_km': float(error_km),
            'n_anchors': len(constraints),
            'method': method_name,
            'intersection': did_intersect,
            'avg_radius_km': float(np.mean(outer_radii)) if outer_radii else None,
            'intersection_area_km2': area_km2,
            'geolocation_method': estimate['method'],
            'fallback': bool(estimate['fallback']),
        })

    benchmark_summary = {
        'totals_sec': benchmark_totals,
        'method_counts': method_counts,
        'fallback_count': fallback_count,
        'n_probes': len(probe_ips),
        'avg_constraints_per_probe': (
            float(np.mean([r['n_anchors'] for r in results])) if results else 0.0
        ),
    }

    return results, np.array(all_outer_radii), np.array(all_areas), benchmark_summary
