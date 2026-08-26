"""Per-target cluster classification truth table.

Produces a boolean matrix: rows = targets (indexed by target_id), columns =
shortest_ping baseline + all CBG combo variants (sorted alphabetically). Each
cell is True if the prediction snapped to the correct Voronoi centroid (nearest-
centroid Voronoi equality between prediction and ground truth), False otherwise.
Non-SUCCESS rows (FALLBACK / hard failures) are False, not omitted.

Writes two sibling files to the same directory as `plot_cluster_match_bars.py`:
  <run_id>_classification_table.csv
  <run_id>_classification_table.parquet

The per-column mean of the table equals the `accuracy` values in
`<run_id>_cluster_accuracy.csv` (same denominator: all targets).

CLI:
    python -m scripts.analysis.classification_table \\
        --config scripts/benchmark/v2/config/north_america_as7018_final_us.yaml \\
        --radius-km 50
    python -m scripts.analysis.classification_table \\
        --run-dir scripts/benchmark/v2/outputs/north_america_as7018_final_us
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from scripts.analysis._cluster_data import (
    build_answer_space,
)
from scripts.analysis._v2_io import (
    add_geo_filter_args,
    analysis_out_dir,
    discover_combos,
    group_combos_by_id,
    load_targets,
    load_combo_allowlist_from_config,
    resolve_run_dir,
    route_geo_path,
    set_geo_filter_from_args,
)

logger = logging.getLogger(__name__)


def _target_id_index(combo_dirs: list[Path]) -> pd.Index:
    """Return unique target_id index (as strings) for all discovered combo dirs."""
    if not combo_dirs:
        return pd.Index([], dtype="object")
    ids = []
    for d in combo_dirs:
        df = load_targets(d).to_pandas()[["target_id"]]
        ids.extend(df["target_id"].astype(str).tolist())
    return pd.Index(pd.unique(pd.Series(ids, dtype="object")), dtype="object")


def combo_match_series_topn(
    combo_dirs: list[Path],
    target_index: pd.Index,
    top_n: int = 1,
) -> pd.Series:
    """Return boolean series for top-N cluster matching.

    Uses pre-scored combo CSVs from cluster-score and reads:
    - top_n=1: match column (fallback to match_top1)
    - top_n>1: match_top{N} column
    """
    if not combo_dirs:
        raise ValueError("combo_dirs is empty")

    # .../<run_id>/<source>/<setup>/<slice>/<combo_id>
    first = combo_dirs[0]
    setup_dir = first.parents[1]
    scored_path = setup_dir / "cluster_scored" / f"{first.name}_scored.csv"
    if not scored_path.exists():
        raise FileNotFoundError(
            f"Missing scored CSV for combo {first.name}: {scored_path}. "
            "Run cluster-score first."
        )

    scored = pd.read_csv(scored_path, usecols=lambda c: c == "target_id" or c.startswith("match"))
    scored["target_id"] = scored["target_id"].astype(str)

    if top_n == 1:
        match_col = "match" if "match" in scored.columns else "match_top1"
    else:
        match_col = f"match_top{top_n}"
    if match_col not in scored.columns:
        raise ValueError(
            f"{scored_path} is missing {match_col}; rerun cluster-score with --top-n >= {top_n}"
        )

    series = scored.set_index("target_id")[match_col].astype(bool)
    return series.reindex(target_index, fill_value=False)


def build_table(
    run_dir: Path,
    radius_km: float,
    top_n: int = 1,
    source=None,
    slice_=None,
    combos: list[str] | None = None,
    clusters_dir: Path | None = None,
    inputs_dir: Path | None = None,
    inputs_root: Path = Path("scripts/benchmark/v2/inputs"),
) -> pd.DataFrame:
    """Return the boolean classification table (target_id × combo columns).

    For top_n=1 (default): checks if prediction matches the correct cluster exactly.
    For top_n>1: checks if correct cluster is in the top-N closest clusters.
    """
    centroid_index, n_centroids, n_targets = build_answer_space(
        run_dir, source, slice_, radius_km, clusters_dir=clusters_dir
    )
    logger.info("answer space: %d targets → %d centroids (R=%.0f km)",
                n_targets, n_centroids, radius_km)

    combo_dirs = discover_combos(run_dir, source, slice_, combos)
    if combos is not None and combos and not combo_dirs:
        raise ValueError(
            "No combo directories matched requested combos "
            f"{sorted(set(combos))} under {run_dir}"
        )
    grouped = group_combos_by_id(combo_dirs)
    target_index = _target_id_index(combo_dirs)

    series_list = []
    for combo_id, dirs in sorted(grouped.items()):
        s = combo_match_series_topn(dirs, target_index, top_n=top_n)
        s.name = combo_id
        if top_n == 1:
            logger.info("  %s: %.1f%% match (%d targets)", combo_id, 100 * s.mean(), len(s))
        else:
            logger.info("  %s: %.1f%% match (top-%d) (%d targets)", combo_id, 100 * s.mean(), top_n, len(s))
        series_list.append(s)

    baseline = None
    if combo_dirs:
        first = combo_dirs[0]
        setup_dir = first.parents[1]
        baseline_path = setup_dir / "cluster_scored" / "baseline.csv"
        if baseline_path.exists():
            bdf = pd.read_csv(
                baseline_path,
                usecols=lambda c: c == "target_id" or c.startswith("vp_matches_centroid"),
            )
            bdf["target_id"] = bdf["target_id"].astype(str)
            if top_n == 1:
                baseline_col = (
                    "vp_matches_centroid"
                    if "vp_matches_centroid" in bdf.columns
                    else "vp_matches_centroid_top1"
                )
            else:
                baseline_col = f"vp_matches_centroid_top{top_n}"
            if baseline_col not in bdf.columns:
                raise ValueError(
                    f"{baseline_path} is missing {baseline_col}; rerun cluster-score with --top-n >= {top_n}"
                )
            baseline = bdf.set_index("target_id")[baseline_col].astype(bool).reindex(target_index, fill_value=False)
            baseline.name = "shortest_ping"
            if top_n == 1:
                logger.info("shortest_ping baseline: %.1f%% match (%d targets)",
                            100 * baseline.mean(), len(baseline))
            else:
                logger.info("shortest_ping baseline: %.1f%% match (top-%d) (%d targets)",
                            100 * baseline.mean(), top_n, len(baseline))
        else:
            logger.warning("no baseline.csv in %s; skipping shortest-ping baseline", setup_dir / "cluster_scored")
    else:
        logger.warning("no combo directories discovered; skipping shortest-ping baseline")

    cols = ([baseline] if baseline is not None else []) + series_list
    table = pd.concat(cols, axis=1)
    table.index = target_index
    table.index.name = "target_id"
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None,
                        help="Benchmark config YAML; its run_id resolves the run dir.")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Explicit outputs/<run_id>/ (overrides --config).")
    parser.add_argument("--outputs-root", type=Path, default=None,
                        help="Override the outputs root used with --config.")
    parser.add_argument("--source", default=None, help="Filter combos by source name.")
    parser.add_argument("--slice", dest="slice_", default=None, help="Filter combos by slice id.")
    parser.add_argument(
        "--combos", nargs="*", default=None,
        help=(
            "Restrict to these combo_ids. When omitted, uses "
            "config.classification_combos if present; otherwise includes all combos."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir (default: scripts/analysis/outputs/<run_id>/cluster).")
    parser.add_argument("--radius-km", type=float, default=50.0,
                        help="Cluster centroid-radius cap defining the answer space. Default 50.")
    parser.add_argument("--top-n", type=int, default=1,
                        help="Top-N ranking to check. Default 1 (exact match).")
    parser.add_argument("--clusters-dir", type=Path, default=None,
                        help="Precomputed cluster-eval results dir (single source of truth). "
                             "Geo subset auto-resolved when a geo filter is active.")
    parser.add_argument("--inputs-dir", type=Path, default=None,
                        help="Materialized inputs dir for the shortest-ping VP baseline. "
                             "Auto-derived when omitted.")
    parser.add_argument("--inputs-root", type=Path,
                        default=Path("scripts/benchmark/v2/inputs"),
                        help="Root of materialized inputs, used to auto-derive --inputs-dir.")
    add_geo_filter_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_geo_filter_from_args(args)

    run_dir = resolve_run_dir(args.config, args.run_dir, args.outputs_root)
    combos = args.combos if args.combos is not None else load_combo_allowlist_from_config(args.config)
    if args.combos is not None:
        logger.info("combo filter source: --combos (%d ids)", len(args.combos))
    elif combos is not None:
        logger.info("combo filter source: config.classification_combos (%d ids)", len(combos))
    else:
        logger.info("combo filter source: all discovered combos")

    out_dir = (
        route_geo_path(args.out_dir) if args.out_dir
        else analysis_out_dir(run_dir, "cluster")
    )

    table = build_table(
        run_dir,
        radius_km=args.radius_km,
        top_n=args.top_n,
        source=args.source,
        slice_=args.slice_,
        combos=combos,
        clusters_dir=args.clusters_dir,
        inputs_dir=args.inputs_dir,
        inputs_root=args.inputs_root,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    # Include top-N in filename
    suffix = f"_top{args.top_n}"
    stem = f"{run_dir.name}_classification_table{suffix}"
    csv_path = out_dir / f"{stem}.csv"
    parquet_path = out_dir / f"{stem}.parquet"
    table.to_csv(csv_path)
    table.to_parquet(parquet_path)
    logger.info("Saved %s  shape=%s", csv_path, table.shape)
    logger.info("Saved %s", parquet_path)


if __name__ == "__main__":
    main()
