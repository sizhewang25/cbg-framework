"""Plot target classification categories across multiple datasets.

Generates classification tables for multiple configs, categorizes targets into
5 mutually exclusive groups based on their classification results, and plots
a grouped bar chart showing distribution across datasets.

Categories:
  1. all_true: All columns (shortest_ping + all CBGs) are True
  2. all_false: All columns are False
  3. shortest_ping_only: shortest_ping=True, all CBGs=False
  4. cbg_only: At least one CBG=True, shortest_ping=False
  5. Mixed: shortest_ping=True AND at least one CBG=True

CLI:
    python -m scripts.analysis.classification.plot_target_categories \\
        --configs \\
            scripts/analysis/config/clusters/as7018_us_test01.yaml \\
        --out-dir scripts/analysis/outputs/multi_dataset_comparison \\
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from scripts.analysis._v2_io import resolve_run_dir

logger = logging.getLogger(__name__)


CATEGORIES = ["all_true", "all_false", "shortest_ping_only", "cbg_only", "Mixed"]

# Color palette for datasets (7 distinct colors)
DATASET_COLORS = [
    "#e67e22",  # dark orange - AS7018
]


def categorize_targets(table: pd.DataFrame) -> pd.Series:
    """Categorize each target based on classification table.

    Assumes first column is shortest_ping baseline, remaining are CBG variants.
    Returns Series with index=target_id, values=category name.
    """
    categories = []
    
    # Extract shortest_ping (first column) and CBGs (remaining)
    shortest_ping = table.iloc[:, 0].values
    cbgs = table.iloc[:, 1:].values
    
    for i in range(len(table)):
        sp = shortest_ping[i]
        cbg_vals = cbgs[i]
        any_cbg_true = cbg_vals.any()
        all_cbg_false = (~cbg_vals).all()
        
        # Check categories in order (mutually exclusive)
        if sp and (cbg_vals).all():
            # all_true: shortest_ping=True AND all CBGs=True
            categories.append("all_true")
        elif not sp and (~cbg_vals).all():
            # all_false: shortest_ping=False AND all CBGs=False
            categories.append("all_false")
        elif sp and all_cbg_false:
            # shortest_ping_only: shortest_ping=True, all CBGs=False
            categories.append("shortest_ping_only")
        elif not sp and any_cbg_true:
            # cbg_only: shortest_ping=False, at least one CBG=True
            categories.append("cbg_only")
        elif sp and any_cbg_true:
            # Mixed: shortest_ping=True AND at least one CBG=True
            categories.append("Mixed")
        else:
            # Shouldn't reach here if logic is exhaustive
            raise ValueError(f"Target {table.index[i]}: cannot categorize (sp={sp}, cbgs={cbg_vals})")
    
    return pd.Series(categories, index=table.index, name="category")


def process_config(
    config: Path,
    outputs_root: Path | None = None,
    radius_km: float = 50.0,
    top_n: int = 1,
) -> tuple[str, dict[str, int]]:
    """Load pre-generated top-N classification table for a config and count categories.

    Returns: (dataset_label, {category_name: count})
    """

    try:
        run_dir = resolve_run_dir(config, None, outputs_root)
    except Exception as e:
        logger.error(f"Failed to resolve run_dir for {config}: {e}")
        raise

    cfg = yaml.safe_load(config.read_text()) or {}
    source = cfg.get("source")
    if not source:
        raise ValueError(f"Config {config} is missing required key: source")
    setup = cfg.get("setup", "anchors_to_probes")

    table_base = run_dir / source / setup
    parquet_path = table_base / f"{run_dir.name}_classification_table_top{top_n}.parquet"
    csv_path = table_base / f"{run_dir.name}_classification_table_top{top_n}.csv"
    
    logger.info(f"Loading classification table (top-{top_n}) for {config.stem}...")
    try:
        if parquet_path.exists():
            table = pd.read_parquet(parquet_path)
            logger.info(f"  Loaded parquet: {parquet_path}")
        elif csv_path.exists():
            table = pd.read_csv(csv_path, index_col=0)
            logger.info(f"  Loaded csv: {csv_path}")
        else:
            raise FileNotFoundError(
                f"Missing classification table for top-{top_n}: {parquet_path} (or CSV equivalent). "
                "Run scripts/analysis/evaluation.smk first."
            )
    except Exception as e:
        logger.error(f"Failed to load table for {run_dir}: {e}")
        raise
    
    logger.info(f"  Table shape: {table.shape}")
    
    categories = categorize_targets(table)
    counts = categories.value_counts().to_dict()
    
    # Ensure all categories are present (with 0 count if not seen)
    for cat in CATEGORIES:
        counts.setdefault(cat, 0)
    
    logger.info(f"  Categories: {counts}")
    return (config.stem, counts)


def plot_categories(
    results: dict[str, dict[str, int]],
    labels: list[str] | None = None,
    out_dir: Path = Path("scripts/analysis/outputs"),
    top_n: int = 1,
) -> None:
    """Plot grouped bar chart of category counts across datasets.

    Args:
        results: {dataset_name: {category: count}}
        labels: Optional friendly labels for datasets (replaces dataset names)
        out_dir: Output directory for PNG and CSV
        top_n: Top-N ranking level (for title)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Build DataFrame: rows=datasets, columns=categories
    df = pd.DataFrame(results).T  # Transpose to get datasets as rows
    
    # Reindex to ensure consistent category order
    for cat in CATEGORIES:
        if cat not in df.columns:
            df[cat] = 0
    df = df[CATEGORIES]
    
    if labels is None:
        labels = list(results.keys())
    else:
        if len(labels) != len(results):
            raise ValueError(f"Expected {len(results)} labels, got {len(labels)}")
    
    df.index = labels
    
    # Calculate percentages per dataset (row-wise)
    df_pct = df.div(df.sum(axis=1), axis=0) * 100
    
    logger.info(f"Summary table (counts):\n{df}")
    logger.info(f"Summary table (percentages):\n{df_pct}")
    
    # Map dataset labels to colors and hatches
    # Group weighted/unweighted pairs with same base color
    dataset_info = {}
    color_map = {
        "AS7018": (DATASET_COLORS[6], None),                 # dark orange, solid
        "RIPE": (DATASET_COLORS[6], None),                   # dark orange, solid
    }
    
    for label in df.index:
        if label in color_map:
            dataset_info[label] = color_map[label]
        else:
            # Fallback for unexpected labels
            idx = list(df.index).index(label)
            dataset_info[label] = (DATASET_COLORS[idx % len(DATASET_COLORS)], None)
    
    # Reorder datasets to group weighted/unweighted pairs
    desired_order = [
        "AS7018",
        "RIPE",
    ]
    ordered_labels = [l for l in desired_order if l in df.index]
    # Preserve any unexpected labels rather than dropping them.
    ordered_labels.extend([l for l in df.index if l not in ordered_labels])
    df = df.loc[ordered_labels]
    df_pct = df_pct.loc[ordered_labels]
    
    # Plot grouped bar chart with categories on x-axis, datasets as grouped bars
    fig, ax = plt.subplots(figsize=(16, 7))
    
    x = range(len(CATEGORIES))
    n_datasets = len(df.index)
    width = 0.12
    
    # Plot bars for each dataset (using percentages for heights)
    for i, dataset_label in enumerate(df.index):
        color, hatch = dataset_info[dataset_label]
        offset = (i - (n_datasets - 1) / 2) * width  # Center the bars
        bars = ax.bar(
            [xi + offset for xi in x],
            df_pct.loc[dataset_label],
            width=width,
            label=dataset_label,
            color=color,
            hatch=hatch,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        
        # Add percentage + count labels on top of bars (black text)
        for j, bar in enumerate(bars):
            count = df.loc[dataset_label, CATEGORIES[j]]
            pct = df_pct.loc[dataset_label, CATEGORIES[j]]
            height = bar.get_height()
            if height > 0:  # Only show label if bar is visible
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{pct:.1f}%\n({int(count)})",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="black",
                    fontweight="bold",
                )
    
    ax.set_xlabel("Classification Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Percentage of Targets (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"Top-{top_n} Classification: Target Categories Across Datasets", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([cat.replace("_", " ").title() for cat in CATEGORIES])
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", frameon=True, fontsize=10, ncol=1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    
    # Save outputs
    png_path = out_dir / f"target_categories_comparison_top{top_n}.png"
    csv_path = out_dir / f"target_categories_comparison_top{top_n}.csv"
    
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved {png_path}")
    
    # Save both counts and percentages to CSV
    df.to_csv(csv_path)
    logger.info(f"Saved {csv_path}")
    
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--configs",
        type=Path,
        nargs="+",
        required=True,
        help="Benchmark config YAML files (one per dataset).",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=None,
        help="Override the outputs root used with configs.",
    )
    parser.add_argument(
        "--radius-km",
        type=float,
        default=50.0,
        help="Cluster centroid-radius cap. Default 50.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="Top-N ranking to check. Default 1.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("scripts/analysis/outputs/multi_dataset_comparison"),
        help="Output directory for PNG and CSV.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Friendly labels for datasets (default: config stems).",
    )
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    results = {}
    for config in args.configs:
        dataset_name, counts = process_config(config, args.outputs_root, args.radius_km, args.top_n)
        results[dataset_name] = counts
    
    plot_categories(results, labels=args.labels, out_dir=args.out_dir, top_n=args.top_n)


if __name__ == "__main__":
    main()
