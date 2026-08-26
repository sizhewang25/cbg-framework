"""Build and plot category-filtered topology summaries across datasets.

For each dataset listed in a cross-datasets config, this script joins:
- top-N classification table (per target booleans)
- eval-per-target CSV (per target topology metrics)

Targets are grouped into 4 categories:
- all_true
- all_false
- cbg_only
- mixed

Category definitions:
- all_true: shortest_ping=True and all CBG variants=True
- all_false: shortest_ping=False and all CBG variants=False
- cbg_only: shortest_ping=False and at least one CBG variant=True
- mixed: everything else (including shortest_ping_only and partial overlaps)

For each category, it writes one aggregated CSV with the columns needed by:
- plot_proximity_stacked_bars.py
- plot_min_inflation_boxplots.py

Then it calls those two plotting functions and writes outputs to:
<out_dir>/<category>/

CLI:
    .venv/bin/python -m scripts.analysis.dataset_props.topology.plot_category_filtered_topology \
      --cross-config scripts/analysis/config/cross_datasets/cross_dataset_comparison02.yaml \
      --out-dir scripts/analysis/outputs/cross_dataset_comparison02/classification
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scripts.analysis.dataset_props.topology.plot_min_inflation_boxplots import (
    plot_min_inflation_boxplots,
)
from scripts.analysis.dataset_props.topology.plot_proximity_stacked_bars import (
    plot_proximity_stacked_bars,
)


logger = logging.getLogger(__name__)

CATEGORIES = ("all_true", "all_false", "cbg_only", "mixed")

DEFAULT_OUTPUTS_ROOT = Path(__file__).resolve().parents[3] / "benchmark" / "v2" / "outputs"


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


def _load_classification_table(path_base: Path, run_id: str, top_n: int) -> pd.DataFrame:
    parquet_path = path_base / f"{run_id}_classification_table_top{top_n}.parquet"
    csv_path = path_base / f"{run_id}_classification_table_top{top_n}.csv"
    if parquet_path.exists():
        table = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        table = pd.read_csv(csv_path, index_col=0)
    else:
        raise FileNotFoundError(
            f"Missing classification table for top-{top_n}: {parquet_path} (or CSV equivalent)"
        )

    if table.empty:
        raise ValueError(f"Classification table is empty: {path_base}")
    return table


def _find_eval_per_target(eval_source_dir: Path) -> Path:
    candidates = sorted(eval_source_dir.glob("*_eval_per_target.csv"))
    if not candidates:
        raise FileNotFoundError(f"No *_eval_per_target.csv under {eval_source_dir}")
    if len(candidates) == 1:
        return candidates[0]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    logger.warning("Multiple eval_per_target files under %s; using newest: %s", eval_source_dir, candidates[0])
    return candidates[0]


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "yes", "y", "t"})


def _categorize_targets_4(table: pd.DataFrame) -> pd.Series:
    if table.shape[1] < 2:
        raise ValueError("Classification table needs at least shortest_ping + one CBG column")

    shortest_ping = _as_bool_series(table.iloc[:, 0])
    cbgs = table.iloc[:, 1:].apply(_as_bool_series)

    all_cbg_true = cbgs.all(axis=1)
    all_cbg_false = ~cbgs.any(axis=1)
    any_cbg_true = cbgs.any(axis=1)

    out = pd.Series("mixed", index=table.index, dtype="object")
    out[(shortest_ping) & (all_cbg_true)] = "all_true"
    out[(~shortest_ping) & (all_cbg_false)] = "all_false"
    out[(~shortest_ping) & (any_cbg_true)] = "cbg_only"
    return out


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _min_inflation_summary(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            "n": 0.0,
            "min": float("nan"),
            "mean": float("nan"),
            "p5": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
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


def _summarize_one_dataset_for_category(
    dataset_label: str,
    run_id: str,
    source: str,
    setup: str,
    outputs_root: Path,
    category: str,
    top_n: int,
) -> dict[str, Any]:
    run_dir = outputs_root / run_id
    table_base = run_dir / source / setup
    eval_source_dir = run_dir / "eval_source"

    class_table = _load_classification_table(table_base, run_id=run_id, top_n=top_n)
    categories = _categorize_targets_4(class_table)

    eval_per_target_path = _find_eval_per_target(eval_source_dir)
    eval_df = pd.read_csv(eval_per_target_path)
    if "target_id" not in eval_df.columns:
        raise ValueError(f"Missing target_id in {eval_per_target_path}")

    class_df = pd.DataFrame({"target_id": categories.index.astype(str), "category": categories.values})
    eval_df["target_id"] = eval_df["target_id"].astype(str)
    merged = eval_df.merge(class_df, on="target_id", how="inner")
    filtered = merged[merged["category"] == category].copy()

    n_targets = int(len(filtered))
    prox_counts = filtered.get("proximity_label", pd.Series(dtype="object")).value_counts()
    no_prox = float(prox_counts.get("NO_PROXIMITY", 0))
    has_not_used = float(prox_counts.get("HAS_NOT_USED_PROXIMITY", 0))
    has_used = float(prox_counts.get("HAS_USED_PROXIMITY", 0))

    min_inf = _min_inflation_summary(filtered.get("min_inflation", pd.Series(dtype="float64")))

    return {
        "dataset": dataset_label,
        "run_id": run_id,
        "top_n": top_n,
        "category": category,
        "n_targets_in_category": n_targets,
        "no_proximity_share": _safe_div(no_prox, n_targets),
        "has_not_used_proximity_share": _safe_div(has_not_used, n_targets),
        "has_used_proximity_share": _safe_div(has_used, n_targets),
        "min_inflation_n": int(min_inf["n"]),
        "min_inflation_min": min_inf["min"],
        "min_inflation_mean": min_inf["mean"],
        "min_inflation_p5": min_inf["p5"],
        "min_inflation_p25": min_inf["p25"],
        "min_inflation_p50": min_inf["p50"],
        "min_inflation_p75": min_inf["p75"],
        "min_inflation_p95": min_inf["p95"],
        "min_inflation_max": min_inf["max"],
    }


def build_category_aggregated(
    cross_config: Path,
    out_dir: Path,
    top_n: int,
) -> dict[str, pd.DataFrame]:
    cross_cfg = _load_yaml(cross_config)
    dataset_cfg_paths = _resolve_dataset_configs(cross_config, cross_cfg)
    labels = cross_cfg.get("labels")
    if labels is not None and (not isinstance(labels, list) or len(labels) != len(dataset_cfg_paths)):
        raise ValueError("'labels' must be a list with the same length as 'datasets'")

    outputs_root = _resolve_outputs_root(cross_cfg, Path.cwd())

    by_category: dict[str, list[dict[str, Any]]] = {k: [] for k in CATEGORIES}
    for i, dataset_cfg_path in enumerate(dataset_cfg_paths):
        cfg = _load_yaml(dataset_cfg_path)
        run_id = cfg.get("run_id")
        source = cfg.get("source")
        setup = cfg.get("setup")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"Missing/invalid run_id in {dataset_cfg_path}")
        if not isinstance(source, str) or not source:
            raise ValueError(f"Missing/invalid source in {dataset_cfg_path}")
        if not isinstance(setup, str) or not setup:
            raise ValueError(f"Missing/invalid setup in {dataset_cfg_path}")

        dataset_label = labels[i] if isinstance(labels, list) else dataset_cfg_path.stem
        for category in CATEGORIES:
            row = _summarize_one_dataset_for_category(
                dataset_label=dataset_label,
                run_id=run_id,
                source=source,
                setup=setup,
                outputs_root=outputs_root,
                category=category,
                top_n=top_n,
            )
            by_category[category].append(row)

    out: dict[str, pd.DataFrame] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for category, rows in by_category.items():
        cat_dir = out_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        csv_path = cat_dir / f"dataset_characterization_aggregated_{category}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Wrote %s", csv_path)
        out[category] = df
    return out


def plot_from_category_aggregated(out_dir: Path) -> None:
    for category in CATEGORIES:
        cat_dir = out_dir / category
        csv_path = cat_dir / f"dataset_characterization_aggregated_{category}.csv"
        if not csv_path.exists():
            logger.warning("Skipping missing CSV: %s", csv_path)
            continue

        plot_proximity_stacked_bars(csv_path, cat_dir)

        df = pd.read_csv(csv_path)
        finite_mask = pd.to_numeric(df.get("min_inflation_p50"), errors="coerce").notna()
        if finite_mask.any():
            plot_min_inflation_boxplots(csv_path, cat_dir)
        else:
            logger.warning("Skipping min-inflation boxplot for %s: no finite values", category)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cross-config",
        type=Path,
        required=True,
        help="Cross-datasets config YAML (for example cross_dataset_comparison02.yaml).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("scripts/analysis/outputs/cross_dataset_comparison02/classification"),
        help="Output root where per-category CSVs and plots will be written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="Top-N classification table to load (default: 1).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cross_config = args.cross_config.resolve()
    build_category_aggregated(cross_config=cross_config, out_dir=args.out_dir, top_n=args.top_n)
    plot_from_category_aggregated(args.out_dir)


if __name__ == "__main__":
    main()
