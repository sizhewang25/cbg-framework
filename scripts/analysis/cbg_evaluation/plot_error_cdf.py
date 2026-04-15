"""Error CDF plot: all pipeline combinations on one figure.

Generalizes evaluate_million_scale.py:plot_error_cdf_comparison() to N series.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from scripts.analysis.cbg_evaluation.combinations import PipelineSpec
from scripts.analysis.cbg_evaluation.evaluate import ProbeResult, get_errors


def plot_error_cdf(
    all_results: Dict[str, List[ProbeResult]],
    specs: List[PipelineSpec],
    output_path: Path,
    thresholds: tuple = (100, 500, 1000),
    max_x_km: float = 3000.0,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot Error CDF with one line per combination.

    Args:
        all_results: {combo_id: [ProbeResult]}
        specs: list of PipelineSpec (determines order, color, linestyle)
        output_path: where to save PNG
        thresholds: vertical threshold lines (km)
        max_x_km: x-axis upper limit
        title: figure title
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    series_data = []
    for spec in specs:
        errors = get_errors(all_results[spec.combo_id])
        if len(errors) == 0:
            continue
        sorted_e = np.sort(errors)
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        median = np.median(errors)
        ax.plot(
            sorted_e, cdf,
            color=spec.color, linestyle=spec.linestyle, linewidth=2,
            label=f"{spec.combo_id}: {spec.label}\n"
                  f"  Median={median:.0f} km, N={len(errors)}",
        )
        series_data.append((spec, errors))

    # Threshold vertical lines
    threshold_colors = {100: "green", 500: "orange", 1000: "red"}
    for thresh in thresholds:
        parts = []
        for spec, errors in series_data:
            pct = np.mean(errors <= thresh) * 100
            parts.append(f"{spec.combo_id}={pct:.0f}%")
        color = threshold_colors.get(thresh, "gray")
        ax.axvline(
            x=thresh, color=color, linestyle="--", alpha=0.5,
            label=f"{thresh} km: {', '.join(parts)}",
        )

    ax.hlines(y=0.5, xmin=0, xmax=max_x_km, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Error Distance (km)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title(
        title or "CBG Pipeline Error CDF Comparison",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.95), fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_x_km)
    ax.set_ylim(0, 1)

    # Stats text box
    lines = []
    for spec, errors in series_data:
        lines.append(
            f"{spec.combo_id}: med={np.median(errors):.0f}, "
            f"mean={np.mean(errors):.0f}, "
            f"p75={np.percentile(errors, 75):.0f}, "
            f"p90={np.percentile(errors, 90):.0f}"
        )
    stats_text = "\n".join(lines)
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes, fontsize=7,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        family="monospace",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    return fig
