"""Plot min RTT inflation split by VP-proximity category across datasets.

For each dataset listed in a cross-datasets YAML the script loads the
eval_per_target CSV and splits targets into three groups via ``proximity_label``:

- ``NO_PROXIMITY``         → targets with no nearby VP
- ``HAS_NOT_USED_PROXIMITY`` → targets that have a close VP but the closest VP
                               is NOT the one with the shortest RTT (no SPING)
- ``HAS_USED_PROXIMITY``   → targets where the closest VP IS the shortest-ping VP

Three outputs are produced in ``<out_dir>/``:

1. ``proximity_min_inflation_grouped_boxplot.{png,pdf}``
   One figure with side-by-side boxes per dataset, coloured by proximity group.

2. ``min_inflation_<group>.csv``
   Per-group aggregated CSV with the same schema expected by
   ``plot_min_inflation_boxplots.py``, so that script can be re-used downstream.

CLI::

    .venv/bin/python -m scripts.analysis.dataset_props.topology.plot_proximity_filtered_min_inflation \\
        --cross-config scripts/analysis/config/cross_datasets/cross_dataset_comparison02.yaml \\
        --out-dir scripts/analysis/outputs/cross_dataset_comparison02/dataset_props/proximity_inflation
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROXIMITY_GROUPS: list[tuple[str, str, str]] = [
    ("HAS_USED_PROXIMITY",        "HAS PROX & HAS SPING", "#27ae60"),
    ("HAS_NOT_USED_PROXIMITY",    "HAS PROX & NO SPING",  "#f39c12"),
]

# Groups included in the grouped boxplot (NO_PROXIMITY is still aggregated to
# CSV but excluded from the figure).
_PLOT_GROUPS = PROXIMITY_GROUPS

DEFAULT_OUTPUTS_ROOT = Path(__file__).resolve().parents[3] / "benchmark" / "v2" / "outputs"


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _resolve_dataset_configs(cross_cfg_path: Path, cross_cfg: dict[str, Any]) -> list[Path]:
    datasets = cross_cfg.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"'datasets' must be a non-empty list in {cross_cfg_path}")
    out: list[Path] = []
    for i, item in enumerate(datasets):
        if not isinstance(item, str):
            raise ValueError(f"datasets[{i}] must be a string path")
        p = Path(item)
        if not p.is_absolute():
            p = (cross_cfg_path.parent / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Dataset config not found: {p}")
        out.append(p)
    return out


def _resolve_outputs_root(cross_cfg: dict[str, Any], cwd: Path) -> Path:
    raw = cross_cfg.get("v2_outputs_root")
    if raw is None:
        return DEFAULT_OUTPUTS_ROOT
    p = Path(str(raw))
    if not p.is_absolute():
        p = (cwd / p).resolve()
    return p


# ---------------------------------------------------------------------------
# Per-target CSV loader
# ---------------------------------------------------------------------------

def _find_eval_per_target(eval_source_dir: Path) -> Path:
    candidates = sorted(eval_source_dir.glob("*_eval_per_target.csv"))
    if not candidates:
        raise FileNotFoundError(f"No *_eval_per_target.csv under {eval_source_dir}")
    if len(candidates) == 1:
        return candidates[0]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    logger.warning(
        "Multiple eval_per_target files under %s; using newest: %s",
        eval_source_dir,
        candidates[0],
    )
    return candidates[0]


# ---------------------------------------------------------------------------
# Inflation summary
# ---------------------------------------------------------------------------

def _min_inflation_summary(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {k: float("nan") for k in ("n", "min", "mean", "p5", "p25", "p50", "p75", "p95", "max")}
    q = numeric.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "n": float(len(numeric)),
        "min": float(numeric.min()),
        "mean": float(numeric.mean()),
        "p5": float(q.loc[0.05]),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.50]),
        "p75": float(q.loc[0.75]),
        "p95": float(q.loc[0.95]),
        "max": float(numeric.max()),
    }


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def build_proximity_inflation(
    cross_config: Path,
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load per-target CSVs, split by proximity_label, return per-group DataFrames."""
    cross_config = cross_config.resolve()
    cross_cfg = _load_yaml(cross_config)
    dataset_cfg_paths = _resolve_dataset_configs(cross_config, cross_cfg)
    labels = cross_cfg.get("labels")
    if labels is not None and (not isinstance(labels, list) or len(labels) != len(dataset_cfg_paths)):
        raise ValueError("'labels' must be a list with the same length as 'datasets'")
    outputs_root = _resolve_outputs_root(cross_cfg, Path.cwd())

    # group_label → list of per-dataset rows (all three groups written to CSV)
    all_groups = [
        ("NO_PROXIMITY",           "NO PROX",             "#c0392b"),
        ("HAS_NOT_USED_PROXIMITY", "HAS PROX & NO SPING", "#f39c12"),
        ("HAS_USED_PROXIMITY",     "HAS PROX & HAS SPING","#27ae60"),
    ]
    by_group: dict[str, list[dict[str, Any]]] = {g: [] for g, _, _ in all_groups}

    for i, cfg_path in enumerate(dataset_cfg_paths):
        cfg = _load_yaml(cfg_path)
        run_id = cfg.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"Missing/invalid run_id in {cfg_path}")

        dataset_label = labels[i] if isinstance(labels, list) else cfg_path.stem
        eval_source_dir = outputs_root / run_id / "eval_source"
        per_target_path = _find_eval_per_target(eval_source_dir)

        df = pd.read_csv(per_target_path, usecols=["proximity_label", "min_inflation"])

        for group_key, _, _ in all_groups:
            subset = df[df["proximity_label"] == group_key]["min_inflation"]
            summary = _min_inflation_summary(subset)
            by_group[group_key].append(
                {
                    "dataset": dataset_label,
                    "n_targets": int(summary["n"]) if not np.isnan(summary["n"]) else 0,
                    "min_inflation_n": int(summary["n"]) if not np.isnan(summary["n"]) else 0,
                    "min_inflation_min": summary["min"],
                    "min_inflation_mean": summary["mean"],
                    "min_inflation_p5": summary["p5"],
                    "min_inflation_p25": summary["p25"],
                    "min_inflation_p50": summary["p50"],
                    "min_inflation_p75": summary["p75"],
                    "min_inflation_p95": summary["p95"],
                    "min_inflation_max": summary["max"],
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, pd.DataFrame] = {}
    for group_key, _, _ in all_groups:
        gdf = pd.DataFrame(by_group[group_key])
        csv_path = out_dir / f"min_inflation_{group_key.lower()}.csv"
        gdf.to_csv(csv_path, index=False)
        logger.info("Wrote %s", csv_path)
        result[group_key] = gdf

    return result


# ---------------------------------------------------------------------------
# Grouped boxplot
# ---------------------------------------------------------------------------

def _bxp_stats(label: str, row: pd.Series) -> dict[str, Any]:
    """Build a single bxp stat dict from a DataFrame row."""
    return {
        "label": label,
        "whislo": float(row["min_inflation_p5"]),
        "q1": float(row["min_inflation_p25"]),
        "med": float(row["min_inflation_p50"]),
        "q3": float(row["min_inflation_p75"]),
        "whishi": float(row["min_inflation_p95"]),
        "fliers": [],
    }


def _is_weighted(name: str) -> bool:
    lowered = name.lower()
    return "weighted" in lowered or lowered.endswith("-w")


def plot_grouped_boxplot(
    by_group: dict[str, pd.DataFrame],
    out_dir: Path,
    y_min: float = 1.0,
    y_max: float = 2.5,
) -> tuple[Path, Path]:
    """Produce a grouped boxplot: 2 proximity-group boxes per dataset."""
    # Use the dataset order from the first plotted group, excluding weighted variants.
    first_df = by_group[_PLOT_GROUPS[0][0]]
    datasets = [d for d in first_df["dataset"].tolist() if not _is_weighted(d)]
    n = len(datasets)
    n_groups = len(_PLOT_GROUPS)

    box_w = 0.28                        # width of each box
    pair_gap = 0.04                     # gap between the two boxes in a pair
    cluster_gap = 0.30                  # extra gap between dataset clusters
    pair_span = n_groups * box_w + (n_groups - 1) * pair_gap
    group_span = pair_span + cluster_gap
    offsets = np.array(
        [i * (box_w + pair_gap) - pair_span / 2 + box_w / 2 for i in range(n_groups)]
    )

    fig, ax = plt.subplots(figsize=(max(12, n * 2.2), 7))
    legend_handles: list[plt.Artist] = []

    for g_idx, (group_key, group_label, color) in enumerate(_PLOT_GROUPS):
        gdf = by_group[group_key].set_index("dataset")
        stats: list[dict[str, Any]] = []
        positions: list[float] = []

        for d_idx, dataset in enumerate(datasets):
            x_center = d_idx * group_span
            positions.append(x_center + offsets[g_idx])
            if dataset in gdf.index:
                row = gdf.loc[dataset]
                if pd.notna(row["min_inflation_p50"]):
                    stats.append(_bxp_stats(dataset, row))
                    continue
            # Placeholder for missing / all-NaN datasets.
            stats.append({
                "label": dataset,
                "whislo": float("nan"),
                "q1": float("nan"),
                "med": float("nan"),
                "q3": float("nan"),
                "whishi": float("nan"),
                "fliers": [],
            })

        bp = ax.bxp(
            stats,
            positions=positions,
            widths=box_w,
            showfliers=False,
            patch_artist=True,
        )
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.75)
            box.set_edgecolor("black")
            box.set_linewidth(0.8)
        for whisker in bp["whiskers"]:
            whisker.set_color("black")
            whisker.set_linewidth(0.8)
        for cap in bp["caps"]:
            cap.set_color("black")
            cap.set_linewidth(0.8)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.4)

        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.75, edgecolor="black", label=group_label)
        )

    # Dataset tick labels centred under each cluster.
    x_ticks = [d_idx * group_span + (pair_span / 2 - box_w / 2) for d_idx in range(n)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=10)

    ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax.set_ylabel("Min RTT Inflation Rate", fontsize=12, fontweight="bold")
    ax.set_title("Min RTT Inflation by VP-Proximity Category", fontsize=14, fontweight="bold")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "proximity_min_inflation_grouped_boxplot.png"
    pdf_path = out_dir / "proximity_min_inflation_grouped_boxplot.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    logger.info("Saved %s", png_path)
    logger.info("Saved %s", pdf_path)
    return png_path, pdf_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cross-config",
        type=Path,
        required=True,
        help="Cross-datasets config YAML with a 'datasets' list.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "scripts/analysis/outputs/cross_dataset_comparison02/dataset_props/proximity_inflation"
        ),
        help="Output directory for grouped boxplot and per-group CSVs.",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=1.0,
        help="Y-axis lower limit for the grouped boxplot (default: 1.0).",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=2.5,
        help="Y-axis upper limit for the grouped boxplot (default: 2.5).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    by_group = build_proximity_inflation(
        cross_config=args.cross_config,
        out_dir=args.out_dir,
    )
    png_path, pdf_path = plot_grouped_boxplot(by_group, args.out_dir, y_min=args.y_min, y_max=args.y_max)
    print(f"Wrote grouped boxplot: {png_path}")
    print(f"Wrote grouped boxplot: {pdf_path}")


if __name__ == "__main__":
    main()
