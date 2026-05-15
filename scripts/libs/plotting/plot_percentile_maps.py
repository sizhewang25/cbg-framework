"""Percentile map plots: Cartopy US maps at error percentiles.

Reuses plot_circles_on_map() from evaluate_million_scale.py.
For each combo × percentile, selects the nearest probe and plots circles.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

from scripts.libs.core.combinations import PipelineSpec
from scripts.libs.core.evaluate import ProbeResult, get_errors


def plot_percentile_maps(
    all_results: Dict[str, List[ProbeResult]],
    specs_by_id: Dict[str, PipelineSpec],
    artifacts_by_combo: Dict[str, object],
    output_dir: Path,
    combos_to_plot: Optional[List[str]] = None,
    percentiles: tuple = (5, 25, 50, 75, 95),
) -> None:
    """Generate circle-on-map plots at error percentiles.

    Args:
        combos_to_plot: combo IDs to plot (default: S1, L1, B1)
        percentiles: error percentiles to visualize
    """
    from scripts.libs.core.evaluate import build_pipeline
    from scripts.libs.million_scale.evaluate_million_scale import (
        plot_circles_on_map,
    )

    if combos_to_plot is None:
        combos_to_plot = ["S1", "L1", "B1"]

    output_dir.mkdir(parents=True, exist_ok=True)

    for combo_id in combos_to_plot:
        spec = specs_by_id[combo_id]
        artifact = artifacts_by_combo[combo_id]
        results = all_results[combo_id]
        success = [r for r in results if r.error_km is not None]
        if not success:
            continue

        errors = np.array([r.error_km for r in success])

        # Rebuild pipeline to re-run geolocate for selected probes
        pipe = build_pipeline(
            spec,
            artifact.lp_models,
            artifact.octant_models,
            artifact.octant_delta,
        )

        for pct in percentiles:
            target_err = float(np.percentile(errors, pct))
            probe_result = min(success, key=lambda r: abs(r.error_km - target_err))
            probe_ip = probe_result.probe_ip

            # Re-run to get circles_used
            target = artifact.probe_targets[probe_ip]
            location, circles_used = pipe.geolocate(
                target["measurements"], artifact.anchor_coords
            )

            # Build circles_data for plot_circles_on_map: (lat, lon, radius_km)
            circles_data = [
                (c.vp_lat, c.vp_lon, c.radius_km) for c in circles_used
            ]

            # Build result dict expected by plot_circles_on_map
            result_dict = {
                "probe_ip": probe_result.probe_ip,
                "true_lat": probe_result.true_lat,
                "true_lon": probe_result.true_lon,
                "estimated_lat": probe_result.estimated_lat,
                "estimated_lon": probe_result.estimated_lon,
                "error_km": probe_result.error_km,
                "intersection_area_km2": 0,
            }

            out_path = output_dir / f"map_{combo_id}_p{pct}.png"
            try:
                fig = plot_circles_on_map(
                    result_dict, circles_data, [],
                    f"{combo_id}: {spec.label} — P{pct}",
                    output_path=out_path,
                )
                plt.close(fig)
            except Exception as e:
                logger.warning("map %s p%d failed: %s", combo_id, pct, e)
