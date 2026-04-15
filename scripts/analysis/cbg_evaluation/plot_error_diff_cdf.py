"""Error-Diff CDF: pairwise per-probe error difference.

For each pair (A, B): compute error_A - error_B per probe (inner join).
Plot CDF of deltas. Negative = A is better.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from scripts.analysis.cbg_evaluation.combinations import PipelineSpec
from scripts.analysis.cbg_evaluation.evaluate import ProbeResult


def compute_error_diff(
    results_a: List[ProbeResult],
    results_b: List[ProbeResult],
) -> np.ndarray:
    """Per-probe error difference: error_A - error_B.

    Only includes probes that succeeded in both pipelines.
    """
    by_ip_a = {r.probe_ip: r.error_km for r in results_a if r.error_km is not None}
    by_ip_b = {r.probe_ip: r.error_km for r in results_b if r.error_km is not None}
    common = sorted(set(by_ip_a) & set(by_ip_b))
    if not common:
        return np.array([])
    return np.array([by_ip_a[ip] - by_ip_b[ip] for ip in common])


# Colorblind-friendly palette for diff pairs
_DIFF_COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#000000", "#F0E442",
]


def plot_error_diff_cdf(
    all_results: Dict[str, List[ProbeResult]],
    specs_by_id: Dict[str, PipelineSpec],
    diff_pairs: List[Tuple[str, str]],
    output_path: Path,
) -> plt.Figure:
    """CDF of per-probe error differences for selected pairs.

    X-axis: error_A - error_B (km). Negative = A better, Positive = B better.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (id_a, id_b) in enumerate(diff_pairs):
        deltas = compute_error_diff(all_results[id_a], all_results[id_b])
        if len(deltas) == 0:
            continue
        sorted_d = np.sort(deltas)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)

        label_a = specs_by_id[id_a].label
        label_b = specs_by_id[id_b].label
        pct_a_better = np.mean(deltas < 0) * 100
        median_delta = np.median(deltas)

        color = _DIFF_COLORS[i % len(_DIFF_COLORS)]
        ax.plot(
            sorted_d, cdf, color=color, linewidth=2,
            label=(
                f"{id_a} vs {id_b}\n"
                f"  {label_a} − {label_b}\n"
                f"  {id_a} better: {pct_a_better:.0f}%, "
                f"med Δ={median_delta:+.0f} km, N={len(deltas)}"
            ),
        )

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Error Difference (km): A − B", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title(
        "Error-Diff CDF — Pairwise Pipeline Comparison",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Annotate left/right halves
    ax.text(0.02, 0.98, "← A better", transform=ax.transAxes,
            fontsize=10, color="green", va="top", ha="left", alpha=0.7)
    ax.text(0.98, 0.98, "B better →", transform=ax.transAxes,
            fontsize=10, color="red", va="top", ha="right", alpha=0.7)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    return fig
