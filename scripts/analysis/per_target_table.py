"""Build a flat per-target parquet joining VP metadata and per-combo error_km.

Output schema (one row per unique target_id):
  target_id              string
  target_lat             float64
  target_lon             float64
  closest_vp_id          string    — VP with minimum haversine distance to target
  closest_vp_dist_km     float64
  sping_vp_id            string    — VP with minimum latency_ms to target
  sping_vp_dist_km       float64   — haversine(target, sping_vp), not latency
  error_km_<combo_id>    float64 | null  (one column per combo, alphabetical order;
                                          null where status == ERROR)

K-fold test sets are disjoint by construction, so each target_id appears in
exactly one fold and can be used as a unique key without a fold qualifier.

Usage:
  python -m scripts.analysis.per_target_table \\
    --run-dir    scripts/benchmark/v2/outputs/<run_id>/ \\
    --inputs-dir scripts/benchmark/v2/inputs/<source>/<run_id>/<setup>/ \\
    --source     ripe_atlas_asn_corpora \\
    --out        scripts/analysis/outputs/<run_id>/<source>/<setup>/merged/per_target_table.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.analysis._v2_io import (
    discover_combos,
    group_combos_by_id,
    load_targets,
)
from scripts.libs.cbg.rtt_model import haversine_distance

logger = logging.getLogger(__name__)


def _load_eval_observations(inputs_dir: Path) -> pd.DataFrame:
    """Load eval_observations across all folds under inputs_dir.

    Mirrors the discovery logic in inspect_cbg_vs_shortest_ping._load_nearest_ping_full
    but returns the full flat DataFrame rather than a per-target dict.
    """
    direct = inputs_dir / "eval_observations.parquet"
    if direct.exists():
        paths = [direct]
    else:
        paths = sorted(inputs_dir.glob("*/eval_observations.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"No eval_observations.parquet at {inputs_dir} or under "
                f"{inputs_dir}/*/. Pass --inputs-dir pointing at a fold "
                "input dir or at its parent (for merged-fold mode)."
            )

    frames = [pq.read_table(p).to_pandas() for p in paths]
    df = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded %d eval_observations rows from %d file(s) under %s",
        len(df), len(paths), inputs_dir,
    )
    return df


def _compute_vp_columns(obs: pd.DataFrame) -> pd.DataFrame:
    """Return a per-target DataFrame with closest and shortest-ping VP columns.

    Input obs must have: target_id, target_lat, target_lon,
                         vp_id, vp_lat, vp_lon, latency_ms.
    """
    obs = obs.copy()
    obs["dist_km"] = haversine_distance(
        obs["target_lat"].to_numpy(dtype=float),
        obs["target_lon"].to_numpy(dtype=float),
        obs["vp_lat"].to_numpy(dtype=float),
        obs["vp_lon"].to_numpy(dtype=float),
    )

    # Closest VP by geographic distance
    idx_closest = obs.groupby("target_id")["dist_km"].idxmin()
    closest = obs.loc[idx_closest, ["target_id", "target_lat", "target_lon",
                                    "vp_id", "dist_km"]].copy()
    closest = closest.rename(columns={
        "vp_id":   "closest_vp_id",
        "dist_km": "closest_vp_dist_km",
    })

    # Shortest-ping VP by minimum latency
    idx_sping = obs.groupby("target_id")["latency_ms"].idxmin()
    sping = obs.loc[idx_sping, ["target_id", "vp_id", "vp_lat", "vp_lon"]].copy()
    sping["sping_vp_dist_km"] = haversine_distance(
        obs.loc[idx_sping, "target_lat"].to_numpy(dtype=float),
        obs.loc[idx_sping, "target_lon"].to_numpy(dtype=float),
        sping["vp_lat"].to_numpy(dtype=float),
        sping["vp_lon"].to_numpy(dtype=float),
    )
    sping = sping.rename(columns={"vp_id": "sping_vp_id"})
    sping = sping[["target_id", "sping_vp_id", "sping_vp_dist_km"]]

    result = closest.merge(sping, on="target_id", how="inner")
    result = result[["target_id", "target_lat", "target_lon",
                     "closest_vp_id", "closest_vp_dist_km",
                     "sping_vp_id", "sping_vp_dist_km"]]
    return result.reset_index(drop=True)


def _load_error_by_combo(
    run_dir: Path,
    source: Optional[str],
    combos: Optional[list[str]],
) -> dict[str, pd.Series]:
    """Return {combo_id: Series(target_id → error_km | NaN)} across all folds.

    ERROR-status rows produce NaN; SUCCESS and FALLBACK rows carry a valid error_km.
    """
    combo_dirs = discover_combos(run_dir, source, slice_=None, combos=combos)
    if not combo_dirs:
        raise FileNotFoundError(f"No combos found under {run_dir} (source={source})")
    grouped = group_combos_by_id(combo_dirs)
    logger.info("Discovered %d combos: %s", len(grouped), sorted(grouped))

    out: dict[str, pd.Series] = {}
    for combo_id, fold_dirs in sorted(grouped.items()):
        tables = [
            load_targets(d).select(["target_id", "error_km", "status"])
            for d in fold_dirs
        ]
        tbl = pa.concat_tables(tables)
        df = tbl.to_pandas()
        # ERROR rows have no valid prediction; set error_km to NaN
        df.loc[df["status"] == "ERROR", "error_km"] = np.nan
        series = df.set_index("target_id")["error_km"]
        if series.index.duplicated().any():
            # Disjoint K-fold guarantee violated — log a warning but continue
            logger.warning(
                "combo %s: duplicate target_ids found — K-fold sets may overlap",
                combo_id,
            )
        out[combo_id] = series
    return out


def build_per_target_table(
    run_dir: Path,
    inputs_dir: Path,
    source: Optional[str] = None,
    combos: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Core logic — exposed for testing / notebook use."""
    obs = _load_eval_observations(inputs_dir)
    vp_df = _compute_vp_columns(obs)

    error_by_combo = _load_error_by_combo(run_dir, source, combos)

    result = vp_df.copy()
    for combo_id, series in sorted(error_by_combo.items()):
        col = f"error_km_{combo_id}"
        result[col] = result["target_id"].map(series)

    logger.info(
        "Per-target table: %d rows × %d columns", len(result), len(result.columns)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a flat per-target parquet with VP metadata and per-combo "
            "error_km columns. One row per unique target_id across all K folds."
        ),
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Path to outputs/<run_id>/ (contains summary.parquet and combo subdirs).",
    )
    parser.add_argument(
        "--inputs-dir", type=Path, required=True,
        help=(
            "inputs/<source>/<run_id>/<setup>/ — parent of fold_*/ dirs, each "
            "containing eval_observations.parquet. Or a single fold dir for "
            "single-slice mode."
        ),
    )
    parser.add_argument(
        "--source", default=None,
        help="Filter combo discovery to this source name (e.g. 'ripe_atlas_asn_corpora').",
    )
    parser.add_argument(
        "--combos", nargs="*", default=None,
        help="Restrict to these combo_ids (default: all combos found on disk).",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output parquet path.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = build_per_target_table(
        run_dir=args.run_dir,
        inputs_dir=args.inputs_dir,
        source=args.source,
        combos=args.combos or None,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), args.out)
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
