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
    combo_match_series,
    geo_allowed_ids,
    resolve_inputs_dir,
    shortest_ping_match_series,
)
from scripts.analysis._v2_io import (
    add_geo_filter_args,
    analysis_out_dir,
    discover_combos,
    group_combos_by_id,
    resolve_run_dir,
    route_geo_path,
    set_geo_filter_from_args,
)

logger = logging.getLogger(__name__)


def build_table(
    run_dir: Path,
    radius_km: float,
    source=None,
    slice_=None,
    clusters_dir: Path | None = None,
    inputs_dir: Path | None = None,
    inputs_root: Path = Path("scripts/benchmark/v2/inputs"),
) -> pd.DataFrame:
    """Return the boolean classification table (target_id × combo columns)."""
    index, n_centroids, n_targets = build_answer_space(
        run_dir, source, slice_, radius_km, clusters_dir=clusters_dir
    )
    logger.info("answer space: %d targets → %d centroids (R=%.0f km)",
                n_targets, n_centroids, radius_km)

    combo_dirs = discover_combos(run_dir, source, slice_)
    grouped = group_combos_by_id(combo_dirs)

    series_list = []
    for combo_id, dirs in sorted(grouped.items()):
        s = combo_match_series(dirs, index)
        s.name = combo_id
        logger.info("  %s: %.1f%% match (%d targets)", combo_id, 100 * s.mean(), len(s))
        series_list.append(s)

    resolved_inputs = resolve_inputs_dir(run_dir, combo_dirs, inputs_root, inputs_dir)
    baseline = None
    if resolved_inputs is not None:
        try:
            baseline = shortest_ping_match_series(
                resolved_inputs, index, geo_allowed_ids(combo_dirs)
            )
            logger.info("shortest_ping baseline: %.1f%% match (%d targets)",
                        100 * baseline.mean(), len(baseline))
        except FileNotFoundError:
            logger.warning("no eval_observations under %s; skipping shortest-ping baseline",
                           resolved_inputs)
    else:
        logger.warning("no inputs dir resolved; skipping shortest-ping baseline "
                       "(pass --inputs-dir to enable)")

    cols = ([baseline] if baseline is not None else []) + series_list
    table = pd.concat(cols, axis=1)
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
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir (default: scripts/analysis/outputs/<run_id>/cluster).")
    parser.add_argument("--radius-km", type=float, default=50.0,
                        help="Cluster centroid-radius cap defining the answer space. Default 50.")
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
    out_dir = (
        route_geo_path(args.out_dir) if args.out_dir
        else analysis_out_dir(run_dir, "cluster")
    )

    table = build_table(
        run_dir,
        radius_km=args.radius_km,
        source=args.source,
        slice_=args.slice_,
        clusters_dir=args.clusters_dir,
        inputs_dir=args.inputs_dir,
        inputs_root=args.inputs_root,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_dir.name}_classification_table"
    csv_path = out_dir / f"{stem}.csv"
    parquet_path = out_dir / f"{stem}.parquet"
    table.to_csv(csv_path)
    table.to_parquet(parquet_path)
    logger.info("Saved %s  shape=%s", csv_path, table.shape)
    logger.info("Saved %s", parquet_path)


if __name__ == "__main__":
    main()
