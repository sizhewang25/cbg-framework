"""Plot stacked proximity shares across datasets from characterization CSV.

Creates one stacked bar chart where:
- x-axis: dataset
- y-axis: percentage of target samples
- bottom: no_proximity_share
- middle: has_not_used_proximity_share
- top: has_used_proximity_share

Legend labels are fixed to:
- NO PROX
- HAS PROX & NO SPING
- HAS PROX & HAS SPING

CLI:
    .venv/bin/python -m scripts.analysis.dataset_props.topology.plot_proximity_stacked_bars \
        --csv scripts/analysis/outputs/cross_dataset_comparison02/dataset_props/dataset_characterization_aggregated.csv \
        --out-dir scripts/analysis/outputs/cross_dataset_comparison02/dataset_props
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _to_percent(series: pd.Series) -> pd.Series:
    # The characterization CSV stores shares in [0, 1] for current datasets.
    # Keep this tolerant in case input is already in [0, 100].
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if numeric.max() <= 1.0 + 1e-9:
        return numeric * 100.0
    return numeric


def _ordered_datasets(index: list[str]) -> list[str]:
    desired = [
        "AS7018",
        "RIPE",
    ]
    ordered = [label for label in desired if label in index]
    ordered.extend([label for label in index if label not in ordered])
    return ordered


def plot_proximity_stacked_bars(csv_path: Path, out_dir: Path) -> tuple[Path, Path]:
    df = pd.read_csv(csv_path)
    required = {
        "dataset",
        "no_proximity_share",
        "has_not_used_proximity_share",
        "has_used_proximity_share",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {', '.join(missing)}")

    plot_df = df[[
        "dataset",
        "no_proximity_share",
        "has_not_used_proximity_share",
        "has_used_proximity_share",
    ]].copy()
    plot_df = plot_df.set_index("dataset")

    plot_df["no_proximity_share"] = _to_percent(plot_df["no_proximity_share"])
    plot_df["has_not_used_proximity_share"] = _to_percent(plot_df["has_not_used_proximity_share"])
    plot_df["has_used_proximity_share"] = _to_percent(plot_df["has_used_proximity_share"])

    order = _ordered_datasets(plot_df.index.tolist())
    plot_df = plot_df.loc[order]

    x = np.arange(len(plot_df.index))
    bottom = plot_df["no_proximity_share"].to_numpy()
    mid = plot_df["has_not_used_proximity_share"].to_numpy()
    top = plot_df["has_used_proximity_share"].to_numpy()

    fig, ax = plt.subplots(figsize=(14, 7))

    b1 = ax.bar(
        x,
        bottom,
        width=0.72,
        color="#c0392b",
        edgecolor="black",
        linewidth=0.6,
        label="NO PROX",
    )
    b2 = ax.bar(
        x,
        mid,
        width=0.72,
        bottom=bottom,
        color="#f39c12",
        edgecolor="black",
        linewidth=0.6,
        label="HAS PROX & NO SPING",
    )
    b3 = ax.bar(
        x,
        top,
        width=0.72,
        bottom=bottom + mid,
        color="#27ae60",
        edgecolor="black",
        linewidth=0.6,
        label="HAS PROX & HAS SPING",
    )

    # Label each stack segment if visible.
    for i in range(len(plot_df.index)):
        segments = [
            (b1[i], bottom[i], 0.0),
            (b2[i], mid[i], bottom[i]),
            (b3[i], top[i], bottom[i] + mid[i]),
        ]
        for bar, value, offset in segments:
            if value <= 0:
                continue
            y = offset + (value / 2.0)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value >= 10 else "black",
                fontweight="bold",
            )

    totals = bottom + mid + top
    for i, total in enumerate(totals):
        ax.text(x[i], total + 1.0, f"{total:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax.set_ylabel("Percentage of Target Samples (%)", fontsize=12, fontweight="bold")
    ax.set_title("Proximity Breakdown Across Datasets", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=20, ha="right")
    ax.set_ylim(0, max(105, float(np.max(totals)) + 8.0))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "proximity_stacked_bars.png"
    csv_out = out_dir / "proximity_stacked_bars_values.csv"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    out_values = plot_df.copy()
    out_values["sum_pct"] = totals
    out_values.to_csv(csv_out)

    logger.info("Saved %s", png_path)
    logger.info("Saved %s", csv_out)
    return png_path, csv_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("scripts/analysis/outputs/cross_dataset_comparison02/dataset_props/dataset_characterization_aggregated.csv"),
        help="Input aggregated characterization CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("scripts/analysis/outputs/cross_dataset_comparison02/dataset_props"),
        help="Output directory for figure and values CSV.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plot_proximity_stacked_bars(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
