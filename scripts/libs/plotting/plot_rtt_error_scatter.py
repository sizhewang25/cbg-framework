"""RTT-Error scatter plot: NxM grid, one panel per combination.

Generalizes rtt_error_scatter.py to N combinations with binned median trend.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from scripts.libs.core.combinations import PipelineSpec
from scripts.libs.core.evaluate import ProbeResult, get_errors


BIN_WIDTH_MS = 10
MIN_BIN_COUNT = 5


def compute_binned_trend(
    rtt_values: np.ndarray,
    error_values: np.ndarray,
    bin_width_ms: float = BIN_WIDTH_MS,
    min_bin_count: int = MIN_BIN_COUNT,
) -> pd.DataFrame:
    """Compute fixed-width RTT bins and median error per bin.

    Adapted from rtt_error_scatter.py:compute_binned_trend().
    """
    if len(rtt_values) == 0:
        return pd.DataFrame(columns=["bin_mid_ms", "median_error_km", "count"])

    max_rtt = float(rtt_values.max())
    max_bin_edge = max(bin_width_ms, int(math.ceil(max_rtt / bin_width_ms) * bin_width_ms))
    bins = np.arange(0, max_bin_edge + bin_width_ms + 1e-9, bin_width_ms)

    bin_indices = np.digitize(rtt_values, bins) - 1  # 0-based
    records = []
    for bi in range(len(bins) - 1):
        mask = bin_indices == bi
        count = int(mask.sum())
        if count < min_bin_count:
            continue
        records.append({
            "bin_mid_ms": bins[bi] + bin_width_ms / 2.0,
            "median_error_km": float(np.median(error_values[mask])),
            "count": count,
        })
    return pd.DataFrame(records)


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> Optional[float]:
    """Compute correlation, return None if invalid."""
    if len(x) < 3:
        return None
    s = pd.Series(x).corr(pd.Series(y), method=method)
    return None if pd.isna(s) else float(s)


def _fmt_corr(c: Optional[float]) -> str:
    return "nan" if c is None else f"{c:.2f}"


def plot_rtt_error_scatter(
    all_results: Dict[str, List[ProbeResult]],
    specs: List[PipelineSpec],
    output_path: Path,
    bin_width_ms: float = BIN_WIDTH_MS,
    min_bin_count: int = MIN_BIN_COUNT,
) -> plt.Figure:
    """Subplot grid: RTT-error scatter with binned median trend per combination."""
    n = len(specs)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # Compute global axis limits
    all_rtt = []
    all_err = []
    for spec in specs:
        for r in all_results[spec.combo_id]:
            if r.error_km is not None:
                all_rtt.append(r.min_rtt_ms)
                all_err.append(r.error_km)
    x_limit = math.ceil(max(all_rtt) / BIN_WIDTH_MS) * BIN_WIDTH_MS if all_rtt else 100
    y_limit = max(100.0, max(all_err) * 1.05) if all_err else 1000

    for idx, spec in enumerate(specs):
        ax = axes_flat[idx]
        results = all_results[spec.combo_id]
        valid = [r for r in results if r.error_km is not None]
        if not valid:
            ax.set_title(f"{spec.combo_id}: {spec.label}\n(no data)", fontsize=10)
            continue

        rtts = np.array([r.min_rtt_ms for r in valid])
        errors = np.array([r.error_km for r in valid])

        ax.scatter(rtts, errors, s=30, alpha=0.45, c=spec.color,
                   edgecolors="none", label=f"n={len(valid)}")

        # Binned median trend
        trend = compute_binned_trend(rtts, errors, bin_width_ms, min_bin_count)
        if not trend.empty:
            ax.plot(
                trend["bin_mid_ms"], trend["median_error_km"],
                color=spec.color, linestyle=spec.linestyle,
                linewidth=2.5, marker="o", markersize=4,
                markeredgecolor="white", zorder=4,
                label=f"{int(bin_width_ms)}ms-bin median",
            )

        pearson = _safe_corr(rtts, errors, "pearson")
        spearman = _safe_corr(rtts, errors, "spearman")
        ax.set_title(
            f"{spec.combo_id}: {spec.label}\n"
            f"r={_fmt_corr(pearson)}, ρ={_fmt_corr(spearman)}",
            fontsize=9, fontweight="bold",
        )
        ax.set_xlim(0, x_limit)
        ax.set_ylim(0, y_limit)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7)

    # Hide unused axes
    for idx in range(len(specs), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared axis labels
    for ax in axes_flat[: len(specs)]:
        if ax.get_subplotspec().is_last_row() or not ax.get_subplotspec().is_first_col():
            pass  # matplotlib handles shared axes
    fig.supxlabel("Probe Minimum RTT (ms)", fontsize=12)
    fig.supylabel("Geolocation Error (km)", fontsize=12)
    fig.suptitle("RTT vs Error — All Combinations", fontsize=14, fontweight="bold")

    plt.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    return fig
