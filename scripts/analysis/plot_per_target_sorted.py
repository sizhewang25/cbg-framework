"""Sorted per-target error plot from per_target_table.parquet.

For each combo, targets are sorted by that combo's error_km ascending
(rank 1 = easiest target for that combo). X axis = rank (1…N), Y axis =
error_km. Each combo contributes one line, so the plot shows the shape of
each combo's error distribution rather than aggregate statistics.

Horizontal reference lines at 100, 500, 1000, 2500, 5000 km mark the same
threshold distances used in the CDF plots.

Usage::

    python -m scripts.analysis.plot_per_target_sorted \\
        --parquet  scripts/analysis/outputs/<run_id>/<source>/<setup>/merged/per_target_table.parquet \\
        --out      scripts/analysis/outputs/<run_id>/<source>/<setup>/merged/plot_per_target_sorted.png

    # restrict to specific combos
    python -m scripts.analysis.plot_per_target_sorted \\
        --parquet  ... --out ... \\
        --combos octant_cbg_top octant_cbg vanilla_cbg spotter_cbg_c80
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from scripts.analysis._v2_io import palette

logger = logging.getLogger(__name__)

_THRESHOLDS = [100, 500, 1000, 2500, 5000]
_THRESHOLD_STYLE = dict(linewidth=0.7, linestyle="--", alpha=0.45, zorder=1)
_THRESHOLD_COLORS = {100: "green", 500: "goldenrod", 1000: "orange",
                     2500: "tomato", 5000: "red"}


def plot_per_target_sorted(
    df: pd.DataFrame,
    combo_ids: list[str],
    output_path: Path,
    *,
    log_y: bool = True,
    title: Optional[str] = None,
    colors: Optional[dict[str, str]] = None,
) -> plt.Figure:
    """Draw per-target error lines with targets sorted by target_id ASC.

    Targets are ordered once by target_id (ascending) and all combos share
    the same x positions — position k corresponds to the same physical target
    for every combo, enabling direct per-target comparison across combos.
    """
    if colors is None:
        colors = palette(combo_ids)

    df_sorted = df.sort_values("target_id").reset_index(drop=True)
    x = np.arange(1, len(df_sorted) + 1)

    fig, ax = plt.subplots(figsize=(12, 7))

    for combo_id in combo_ids:
        col = f"error_km_{combo_id}"
        if col not in df_sorted.columns:
            logger.warning("Column %s not found — skipping", col)
            continue
        ax.plot(x, df_sorted[col].to_numpy(), label=combo_id,
                color=colors.get(combo_id), linewidth=0.9, alpha=0.75, zorder=2)

    for km in _THRESHOLDS:
        ax.axhline(km, color=_THRESHOLD_COLORS[km], label=f"{km} km",
                   **_THRESHOLD_STYLE)

    ax.set_xlabel("Target (sorted by target_id ASC)", fontsize=11)
    ax.set_ylabel("Error (km)", fontsize=11)
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_yticks([10, 50, 100, 500, 1000, 2500, 5000, 10000, 20000])

    ax.set_xlim(1, len(df_sorted))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
    ax.grid(True, which="major", linewidth=0.4, alpha=0.5)
    ax.grid(True, which="minor", linewidth=0.2, alpha=0.25)

    # Legend: combos first, then thresholds
    handles, labels = ax.get_legend_handles_labels()
    n_combos = len([c for c in combo_ids if f"error_km_{c}" in df.columns])
    combo_hl = list(zip(handles[:n_combos], labels[:n_combos]))
    thresh_hl = list(zip(handles[n_combos:], labels[n_combos:]))
    combo_hl.sort(key=lambda x: x[1])
    leg_handles = [h for h, _ in combo_hl] + [h for h, _ in thresh_hl]
    leg_labels  = [l for _, l in combo_hl] + [l for _, l in thresh_hl]
    ax.legend(leg_handles, leg_labels, fontsize=7.5, ncol=2,
              loc="upper left", framealpha=0.85)

    if title:
        ax.set_title(title, fontsize=12)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved %s", output_path)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sorted per-target error lines from per_target_table.parquet.",
    )
    parser.add_argument("--parquet", type=Path, required=True,
                        help="Path to per_target_table.parquet.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output PNG path.")
    parser.add_argument("--combos", nargs="*", default=None,
                        help="Combo IDs to plot (default: all error_km_* columns).")
    parser.add_argument("--no-log-y", dest="log_y", action="store_false",
                        default=True, help="Use linear Y scale.")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = pd.read_parquet(args.parquet)
    logger.info("Loaded %d rows from %s", len(df), args.parquet)

    if args.combos:
        combo_ids = args.combos
    else:
        combo_ids = sorted(
            c.removeprefix("error_km_")
            for c in df.columns if c.startswith("error_km_")
        )
    logger.info("Plotting %d combos: %s", len(combo_ids), combo_ids)

    fig = plot_per_target_sorted(
        df, combo_ids, args.out,
        log_y=args.log_y,
        title=args.title,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
