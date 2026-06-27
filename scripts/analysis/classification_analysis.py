"""Classification result distribution analysis for the per-target truth table.

Loads the boolean classification table produced by `classification_table.py`,
computes per-target summary statistics, and joins with `targets.csv` (from the
benchmark run dir) so each row carries geographic metadata alongside its
classification outcome.

Per-target columns added:
  n_methods      — total number of method columns in the table
  n_correct      — how many methods returned True for this target
  category       — "all_false" | "all_true" | "partial"
  correct_methods — comma-separated list of methods that returned True (empty for all_false)
  wrong_methods   — comma-separated list of methods that returned False (empty for all_true)

Outputs (written to the same cluster/ dir as the classification table):
  <run_id>_classification_analysis.csv  — one row per target, sorted by n_correct

CLI:
    python -m scripts.analysis.classification_analysis \\
        --config scripts/benchmark/v2/config/north_america_as7018_final_us.yaml
    python -m scripts.analysis.classification_analysis \\
        --run-dir scripts/benchmark/v2/outputs/north_america_as7018_final_us
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from scripts.analysis._v2_io import (
    add_geo_filter_args,
    analysis_out_dir,
    resolve_run_dir,
    route_geo_path,
    set_geo_filter_from_args,
)

logger = logging.getLogger(__name__)


def _targets_csv(run_dir: Path) -> Path | None:
    """Locate targets.csv under the run dir (first match, source/setup level)."""
    hits = sorted(run_dir.rglob("targets.csv"))
    return hits[0] if hits else None


def analyze(table: pd.DataFrame) -> pd.DataFrame:
    """Compute per-target distribution stats from the boolean classification table.

    Returns a DataFrame indexed by target_id with columns:
      n_methods, n_correct, category, correct_methods, wrong_methods."""
    n_methods = table.shape[1]
    n_correct = table.sum(axis=1).astype(int)

    category = pd.Series("partial", index=table.index, dtype=object)
    category[n_correct == 0] = "all_false"
    category[n_correct == n_methods] = "all_true"

    correct_methods = table.apply(
        lambda row: ",".join(col for col in table.columns if row[col]), axis=1
    )
    wrong_methods = table.apply(
        lambda row: ",".join(col for col in table.columns if not row[col]), axis=1
    )

    return pd.DataFrame({
        "n_methods": n_methods,
        "n_correct": n_correct,
        "category": category,
        "correct_methods": correct_methods,
        "wrong_methods": wrong_methods,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None,
                        help="Benchmark config YAML; its run_id resolves the run dir.")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Explicit outputs/<run_id>/ (overrides --config).")
    parser.add_argument("--outputs-root", type=Path, default=None,
                        help="Override the outputs root used with --config.")
    parser.add_argument("--table", type=Path, default=None,
                        help="Path to classification table parquet (auto-resolved when omitted).")
    parser.add_argument("--targets-csv", type=Path, default=None,
                        help="Path to targets.csv (auto-resolved from run dir when omitted).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir (default: scripts/analysis/outputs/<run_id>/cluster).")
    add_geo_filter_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_geo_filter_from_args(args)

    run_dir = resolve_run_dir(args.config, args.run_dir, args.outputs_root)
    out_dir = (
        route_geo_path(args.out_dir) if args.out_dir
        else analysis_out_dir(run_dir, "cluster")
    )

    # Load classification table
    table_path = args.table
    if table_path is None:
        stem = f"{run_dir.name}_classification_table.parquet"
        table_path = out_dir / stem
        if not table_path.exists():
            table_path = table_path.with_suffix(".csv")
    if not table_path.exists():
        raise FileNotFoundError(
            f"Classification table not found at {table_path}. "
            "Run `python -m scripts.analysis.classification_table` first."
        )
    table = (
        pd.read_parquet(table_path) if table_path.suffix == ".parquet"
        else pd.read_csv(table_path, index_col="target_id")
    )
    logger.info("Loaded classification table: %s  shape=%s", table_path, table.shape)

    # Per-target stats
    stats = analyze(table)

    # Join with targets.csv for geographic metadata
    targets_path = args.targets_csv or _targets_csv(run_dir)
    if targets_path is not None:
        targets = pd.read_csv(targets_path).set_index("target_id")
        result = targets.join(stats, how="right")
        logger.info("Joined with targets.csv: %s", targets_path)
    else:
        logger.warning("No targets.csv found under %s; outputting stats only", run_dir)
        result = stats

    result = result.sort_values(["n_correct", "category"], ascending=[True, True])

    # Distribution summary
    counts = result["category"].value_counts()
    logger.info("Distribution:")
    for cat in ("all_false", "partial", "all_true"):
        n = counts.get(cat, 0)
        logger.info("  %-12s %3d  (%.1f%%)", cat, n, 100 * n / len(result))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_dir.name}_classification_analysis.csv"
    result.to_csv(out_path)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
