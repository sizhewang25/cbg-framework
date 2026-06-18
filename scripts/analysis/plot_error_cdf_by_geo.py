"""Per-geo-bucket error CDFs from v2 benchmark outputs.

Given one benchmark config (or a run dir), this slices the eval targets by the
`target_continent` / `target_country` columns that `geo-eval` appended to each
`targets.parquet` and writes **one overlaid-combo error CDF per geo bucket**.
It is the column-driven replacement for the continent split in
`plot_error_cdf.py`, which had to join `target_id` against
`filtered_anchors.json` and bbox-guard the source country codes. Here the labels
ride on `targets.parquet` itself (reverse-geocoded from the ground-truth coords),
so no external join is needed and overseas territories bucket correctly.

For each bucket at the chosen `--level`, the per-combo `error_km` (SUCCESS +
FALLBACK) is pooled across folds for targets whose `target_<level>` equals that
bucket, then rendered with the shared `plot_error_cdf` machinery (single panel,
combos overlaid, the same palette as the global plots). Buckets with fewer than
`--min-targets` eval targets are skipped (sparse per-country subsets make CDFs
noise); `--top-n` keeps only the N largest buckets.

CLI:
    python -m scripts.analysis.plot_error_cdf_by_geo \\
        --config scripts/benchmark/v2/config/global_as16509_final.yaml \\
        --level continent
    python -m scripts.analysis.plot_error_cdf_by_geo \\
        --run-dir scripts/benchmark/v2/outputs/global_as16509_final \\
        --level country --top-n 8 --min-targets 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.analysis._v2_io import (
    discover_combos,
    group_combos_by_id,
    load_summary,
    load_targets,
    palette,
    resolve_run_dir,
)
from scripts.analysis.plot_error_cdf import plot_error_cdf

logger = logging.getLogger(__name__)

_SCORED_STATUSES = ("SUCCESS", "FALLBACK")
_LEVEL_COLUMN = {"continent": "target_continent", "country": "target_country"}


def _bucketed(
    grouped: dict[str, list[Path]], geo_col: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, int]], dict[str, dict[str, int]], dict[str, int]]:
    """Pool per-combo error_km by geo bucket across folds.

    Returns (errors_by_bucket, success_by_bucket, total_by_bucket, bucket_size),
    where the first three map bucket -> {combo_id: ...} and `bucket_size` maps
    bucket -> total eval-target count (max over combos — combos share the target
    set per slice, so this is the bucket's target population).
    """
    errors: dict[str, dict[str, list[np.ndarray]]] = {}
    succ: dict[str, dict[str, int]] = {}
    total: dict[str, dict[str, int]] = {}

    for combo_id, dirs in grouped.items():
        for d in dirs:
            df = load_targets(d).to_pandas()
            if geo_col not in df.columns:
                raise KeyError(
                    f"{d/'targets.parquet'} has no '{geo_col}' column — run "
                    "`cli geo-eval --run-id <id>` first to annotate it."
                )
            for bucket, sub in df.groupby(geo_col, dropna=False):
                b = str(bucket)
                total.setdefault(b, {})[combo_id] = (
                    total.setdefault(b, {}).get(combo_id, 0) + len(sub)
                )
                scored = sub[sub["status"].isin(_SCORED_STATUSES)]
                succ.setdefault(b, {})[combo_id] = (
                    succ.setdefault(b, {}).get(combo_id, 0)
                    + int((sub["status"] == "SUCCESS").sum())
                )
                arr = scored["error_km"].to_numpy(dtype=float)
                arr = arr[~np.isnan(arr)]
                errors.setdefault(b, {}).setdefault(combo_id, []).append(arr)

    errors_np: dict[str, dict[str, np.ndarray]] = {
        b: {
            cid: (np.concatenate(parts) if parts else np.array([], dtype=float))
            for cid, parts in per_combo.items()
        }
        for b, per_combo in errors.items()
    }
    bucket_size = {b: max(per_combo.values()) for b, per_combo in total.items()}
    return errors_np, succ, total, bucket_size


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
    parser.add_argument("--level", choices=tuple(_LEVEL_COLUMN), default="continent",
                        help="Geo granularity to bucket by (default: continent).")
    parser.add_argument("--combos", nargs="*", default=None,
                        help="Restrict to these combo_ids (default: every combo found).")
    parser.add_argument("--min-targets", type=int, default=10,
                        help="Skip buckets with fewer than this many eval targets.")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Keep only the N largest buckets by target count.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir (default: scripts/analysis/outputs/<run_id>/geo/<level>).")
    parser.add_argument("--max-x-km", type=float, default=10000.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_dir = resolve_run_dir(args.config, args.run_dir, args.outputs_root)
    geo_col = _LEVEL_COLUMN[args.level]
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent / "outputs" / run_dir.name / "geo" / args.level
    )

    combo_dirs = discover_combos(run_dir, args.source, args.slice_, args.combos)
    if not combo_dirs:
        raise FileNotFoundError(f"No combos found under {run_dir}")
    grouped = group_combos_by_id(combo_dirs)

    errors_np, succ, total, bucket_size = _bucketed(grouped, geo_col)

    # Color over the full combo set so colors match the global plots.
    colors = palette(sorted(grouped))
    summary = load_summary(run_dir)
    combo_to_ltd = dict(zip(
        summary.column("combo_id").to_pylist(),
        summary.column("ltd").to_pylist(),
    ))

    buckets = [b for b, n in bucket_size.items() if n >= args.min_targets]
    buckets.sort(key=lambda b: bucket_size[b], reverse=True)
    if args.top_n is not None:
        buckets = buckets[: args.top_n]

    skipped = sorted(set(bucket_size) - set(buckets), key=lambda b: -bucket_size[b])
    if skipped:
        logger.info(
            "Skipped %d sparse/excluded buckets: %s",
            len(skipped),
            ", ".join(f"{b}({bucket_size[b]})" for b in skipped),
        )

    for b in buckets:
        safe = b.replace(" ", "_").replace("/", "_")
        fig = plot_error_cdf(
            errors_np[b],
            out_dir / f"{safe}.png",
            successes_by_combo=succ[b],
            totals_by_combo=total[b],
            group_by="ltd",
            combo_to_ltd=combo_to_ltd,
            max_x_km=args.max_x_km,
            colors=colors,
            title=f"Error CDF — {args.level}={b} (n={bucket_size[b]})",
        )
        plt.close(fig)

    logger.info("Wrote %d %s CDF figures to %s", len(buckets), args.level, out_dir)


if __name__ == "__main__":
    main()
