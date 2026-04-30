"""Error CDF plot: 3 subfigures grouped by distance model.

Each panel shows one RTT-to-distance method with its multilateration+centroid
variants as separate lines, enabling direct comparison of Phase 3+4 choices
within a fixed Phase 1.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

from scripts.analysis.cbg_evaluation.combinations import PipelineSpec
from scripts.analysis.cbg_evaluation.evaluate import ProbeResult, get_errors

# Group specs by distance model for the 3-panel layout.
# Within each panel, color distinguishes multilateration+centroid path.
_PATH_STYLE = {
    # (multilateration, centroid) → (color, short_label)
    ("spherical_circle", "arithmetic_mean"): ("#D55E00", "spherical_circle + Arith"),
    ("spherical_circle", "geometric_centroid"): ("#CC79A7", "spherical_circle + Geom"),
    ("planar_circle", "arithmetic_mean"): ("#009E73", "planar_circle + Arith"),
    ("planar_circle", "geometric_centroid"): ("#0072B2", "planar_circle + Geom"),
    ("planar_circle", "monte_carlo_median"): ("#882255", "planar_circle + MC Median"),
    ("planar_annulus", "arithmetic_mean"): ("#E69F00", "planar_annulus + Arith"),
    ("planar_annulus", "geometric_centroid"): ("#56B4E9", "planar_annulus + Geom"),
    ("planar_annulus", "monte_carlo_median"): ("#999933", "planar_annulus + MC Median"),
}

_DISTANCE_TITLES = {
    "speed_of_internet": "Speed-of-Internet (2/3 c)",
    "low_envelope":      "LP Lower Envelope",
    "bounded_spline":    "Octant Bounded Spline",
}

# Canonical distance model ordering for the 3 panels
_DISTANCE_ORDER = ["speed_of_internet", "low_envelope", "bounded_spline"]


def plot_error_cdf(
    all_results: Dict[str, List[ProbeResult]],
    specs: List[PipelineSpec],
    output_path: Path,
    thresholds: tuple = (100, 500, 1000),
    max_x_km: float = 3000.0,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot Error CDF as 1x3 subfigures, one per distance model.

    Args:
        all_results: {combo_id: [ProbeResult]}
        specs: list of PipelineSpec (determines order, color, linestyle)
        output_path: where to save PNG
        thresholds: vertical threshold lines (km)
        max_x_km: x-axis upper limit
        title: figure title
    """
    # Group specs by distance model
    groups: Dict[str, List[PipelineSpec]] = {d: [] for d in _DISTANCE_ORDER}
    for spec in specs:
        if spec.distance in groups:
            groups[spec.distance].append(spec)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    threshold_colors = {100: "green", 500: "orange", 1000: "red"}

    for ax, dist_name in zip(axes, _DISTANCE_ORDER):
        panel_specs = groups[dist_name]
        panel_data = []

        for spec in panel_specs:
            errors = get_errors(all_results[spec.combo_id])
            if len(errors) == 0:
                continue
            key = (spec.multilateration, spec.centroid)
            color, short_label = _PATH_STYLE.get(
                key, ("#999999", f"{spec.multilateration}+{spec.centroid}")
            )
            sorted_e = np.sort(errors)
            cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
            # median = np.median(errors)
            ax.plot(
                sorted_e, cdf,
                color=color, linestyle="-", linewidth=2, alpha=0.7,
                label=f"{spec.combo_id}: {short_label}\n"
                      f"  (Samples: {len(errors):,})",
            )
            panel_data.append((spec, errors, color))

        # Threshold vertical lines (no per-combo percentages to keep panels clean)
        for thresh in thresholds:
            color = threshold_colors.get(thresh, "gray")
            ax.axvline(x=thresh, color=color, linestyle=":", alpha=0.4)

        ax.hlines(y=0.5, xmin=0, xmax=max_x_km, color="gray", linestyle="--", alpha=0.3)
        ax.set_title(
            _DISTANCE_TITLES.get(dist_name, dist_name),
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("Error Distance (km)", fontsize=11)
        ax.legend(loc="center right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_x_km)
        ax.set_ylim(0, 1)

        # Stats text box
        lines = []
        header = "      p5   p25   p50   p75   p95"
        lines.append(header)
        for spec, errors, _ in panel_data:
            lines.append(
                f"{spec.combo_id}: {np.percentile(errors, 5):5.0f} "
                f"{np.percentile(errors, 25):5.0f} "
                f"{np.median(errors):5.0f} "
                f"{np.percentile(errors, 75):5.0f} "
                f"{np.percentile(errors, 95):5.0f}"
            )
        if lines:
            ax.text(
                0.98, 0.02, "\n".join(lines),
                transform=ax.transAxes, fontsize=7,
                verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                family="monospace",
            )

    axes[0].set_ylabel("CDF", fontsize=12)

    fig.suptitle(
        title or "Error CDF by Distance Model",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    return fig
