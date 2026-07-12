"""Heatmap of flow WEIGHT for (VP, target-city) pairs.

Reads a weighted matched dataset (e.g.
``datasets/test05-mainland.edr-final-weighted.csv``) and renders a
heatmap where:

- columns  = VP (``VP_ID``)
- rows     = target city (``TARGET_NORM_CITY``)
- cell     = summed ``WEIGHT`` for that (VP, city) pair

Rows are ranked top-to-bottom by total city weight (heaviest city first).
Columns are ordered by total VP weight (heaviest VP left) so the dense corner
sits top-left.

CLI:
    python -m scripts.analysis.pair_weight.plot_pair_weight_heatmap \\
        --input datasets/test05-mainland.edr-final-weighted.csv \\
        --output datasets/test05-pair-weight-heatmap.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

logger = logging.getLogger(__name__)


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to city x VP weight matrix, rows/cols ranked by total weight."""
    pivot = (
        df.pivot_table(
            index="TARGET_NORM_CITY",
            columns="VP_ID",
            values="WEIGHT",
            aggfunc="sum",
            fill_value=0.0,
        )
    )
    # Rank rows (cities) by total weight, heaviest first.
    row_order = pivot.sum(axis=1).sort_values(ascending=False).index
    # Rank columns (VPs) by total weight, heaviest first.
    col_order = pivot.sum(axis=0).sort_values(ascending=False).index
    return pivot.loc[row_order, col_order]


def plot_heatmap(
    matrix: pd.DataFrame,
    output: Path,
    top_cities: int | None = None,
    top_vps: int | None = None,
    log_scale: bool = True,
) -> None:
    if top_cities is not None:
        matrix = matrix.iloc[:top_cities, :]
    if top_vps is not None:
        matrix = matrix.iloc[:, :top_vps]

    n_rows, n_cols = matrix.shape
    values = matrix.to_numpy()

    if log_scale:
        positive = values[values > 0]
        vmin = positive.min() if positive.size else 1e-12
        norm = LogNorm(vmin=vmin, vmax=values.max())
        # Mask zeros so they render as the "empty" background color.
        plot_values = np.ma.masked_where(values <= 0, values)
    else:
        norm = None
        plot_values = values

    fig_w = max(8.0, n_cols * 0.16 + 3)
    fig_h = max(4.0, n_rows * 0.32 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(plot_values, aspect="auto", cmap=cmap, norm=norm)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=5)

    ax.set_xlabel(f"VP  (n={n_cols}, ordered by total weight)")
    ax.set_ylabel(f"Target city  (n={n_rows}, ranked by total weight)")
    ax.set_title("Flow weight per (VP, target city) pair")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("WEIGHT" + (" (log)" if log_scale else ""))

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    logger.info("saved %s (%d cities x %d VPs)", output, n_rows, n_cols)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/test05-mainland.edr-final-weighted.csv"),
        help="Weighted matched CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/test05-pair-weight-heatmap.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="Drop rows with WEIGHT below this before aggregating.",
    )
    parser.add_argument(
        "--top-cities",
        type=int,
        default=None,
        help="Keep only the N heaviest cities (rows).",
    )
    parser.add_argument(
        "--top-vps",
        type=int,
        default=None,
        help="Keep only the N heaviest VPs (columns).",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use a linear color scale instead of log.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = pd.read_csv(args.input)
    df = df[df["WEIGHT"].notna()]
    if args.min_weight > 0:
        df = df[df["WEIGHT"] >= args.min_weight]

    matrix = build_matrix(df)
    plot_heatmap(
        matrix,
        args.output,
        top_cities=args.top_cities,
        top_vps=args.top_vps,
        log_scale=not args.linear,
    )


if __name__ == "__main__":
    main()
