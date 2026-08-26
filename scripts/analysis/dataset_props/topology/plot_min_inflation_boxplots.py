"""Plot min RTT inflation boxplots across datasets from characterization CSV.

Creates one boxplot chart where:
- x-axis: dataset
- y-axis: min RTT inflation rate
- whiskers: p5 and p95 from characterization summaries

Weighted dataset variants are excluded from the figure.
The y-axis is capped at 2.5 to keep the main span readable.

CLI:
    .venv/bin/python -m scripts.analysis.dataset_props.topology.plot_min_inflation_boxplots \
        --csv scripts/analysis/outputs/cross_dataset_comparison02/dataset_props/dataset_characterization_aggregated.csv \
        --out-dir scripts/analysis/outputs/cross_dataset_comparison02/dataset_props
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def _ordered_datasets(index: list[str]) -> list[str]:
    desired = [
        "AS7018",
        "RIPE",
    ]
    ordered = [label for label in desired if label in index]
    ordered.extend([label for label in index if label not in ordered])
    return ordered


def _is_weighted_dataset(name: str) -> bool:
    lowered = name.lower()
    return "weighted" in lowered or lowered.endswith("-w")


def plot_min_inflation_boxplots(csv_path: Path, out_dir: Path) -> tuple[Path, Path, Path]:
    df = pd.read_csv(csv_path)
    required = {
        "dataset",
        "min_inflation_p5",
        "min_inflation_p25",
        "min_inflation_p50",
        "min_inflation_p75",
        "min_inflation_p95",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {', '.join(missing)}")

    plot_df = df[[
        "dataset",
        "min_inflation_p5",
        "min_inflation_p25",
        "min_inflation_p50",
        "min_inflation_p75",
        "min_inflation_p95",
    ]].copy()
    plot_df = plot_df[~plot_df["dataset"].astype(str).map(_is_weighted_dataset)]
    plot_df = plot_df.set_index("dataset")
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce")
    plot_df = plot_df.dropna(subset=[
        "min_inflation_p5",
        "min_inflation_p25",
        "min_inflation_p50",
        "min_inflation_p75",
        "min_inflation_p95",
    ])

    order = _ordered_datasets(plot_df.index.tolist())
    plot_df = plot_df.loc[order]

    stats: list[dict[str, object]] = []
    for dataset, row in plot_df.iterrows():
        stats.append(
            {
                "label": dataset,
                "whislo": float(row["min_inflation_p5"]),
                "q1": float(row["min_inflation_p25"]),
                "med": float(row["min_inflation_p50"]),
                "q3": float(row["min_inflation_p75"]),
                "whishi": float(row["min_inflation_p95"]),
                "fliers": [],
            }
        )

    fig, ax = plt.subplots(figsize=(14, 7))
    bp = ax.bxp(stats, showfliers=False, patch_artist=True, widths=0.7)

    for box in bp["boxes"]:
        box.set_facecolor("#4e79a7")
        box.set_alpha(0.75)
        box.set_edgecolor("black")
        box.set_linewidth(0.8)

    for whisker in bp["whiskers"]:
        whisker.set_color("black")
        whisker.set_linewidth(0.9)

    for cap in bp["caps"]:
        cap.set_color("black")
        cap.set_linewidth(0.9)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.4)

    ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax.set_ylabel("Min RTT Inflation Rate", fontsize=12, fontweight="bold")
    ax.set_title("Min RTT Inflation by Dataset", fontsize=14, fontweight="bold")
    ax.set_xticklabels(plot_df.index, rotation=20, ha="right")
    ax.set_ylim(0, 2.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "min_inflation_boxplots_unweighted_ymax2p5.png"
    pdf_path = out_dir / "min_inflation_boxplots_unweighted_ymax2p5.pdf"
    csv_out = out_dir / "min_inflation_boxplots_values.csv"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    out_values = plot_df.copy()
    out_values.to_csv(csv_out)

    logger.info("Saved %s", png_path)
    logger.info("Saved %s", pdf_path)
    logger.info("Saved %s", csv_out)
    return png_path, pdf_path, csv_out


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
    plot_min_inflation_boxplots(args.csv, args.out_dir)


if __name__ == "__main__":
    main()