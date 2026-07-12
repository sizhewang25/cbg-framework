"""Build per-target shortest-ping feature table for a benchmark run.

Outputs one row per target with these columns:
  target_id
  cls_result
  n_obs
  target_cell_gap_km
  closest_vp_dist_km
  shortest_ping_vp_dist_km
    shortest_ping_centroid_dist_km
    shortest_ping_centroid_dist_margin
  shortest_ping_rtt_ms
  target_distinguishable_vp_margin_km
  sign_of_margin

CLI examples:
  python -m scripts.analysis.shortest_ping.per_target_features \
      --config scripts/benchmark/v2/config/north_america_as7018_final_us.yaml

  python -m scripts.analysis.shortest_ping.per_target_features \
      --run-dir scripts/benchmark/v2/outputs/config-test02 \
      --clusters-dir scripts/benchmark/v2/outputs/config-test02/generic_csv/anchors_to_probes/clusters
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from scripts.analysis._cluster_data import (
    build_answer_space,
    discover_combos,
    geo_allowed_ids,
    resolve_inputs_dir,
)
from scripts.analysis._fleet_geometry import compute_fleet_geometry
from scripts.analysis._v2_io import (
    add_geo_filter_args,
    analysis_out_dir,
    resolve_run_dir,
    route_geo_path,
    set_geo_filter_from_args,
)
from scripts.libs.cbg.rtt_model import haversine_distance

logger = logging.getLogger(__name__)


def _shortest_ping_rows(
    inputs_dir: Path,
    allowed_ids: set[str] | None = None,
) -> pd.DataFrame:
    """One row per target for the min-latency VP observation."""
    direct = inputs_dir / "eval_observations.parquet"
    paths = ([direct] if direct.exists()
             else sorted(inputs_dir.glob("*/eval_observations.parquet")))
    if not paths:
        raise FileNotFoundError(f"no eval_observations.parquet under {inputs_dir}")

    obs = pd.concat([pq.read_table(p).to_pandas() for p in paths], ignore_index=True)
    if allowed_ids is not None:
        obs = obs[obs["target_id"].isin(allowed_ids)]
    if obs.empty:
        return pd.DataFrame(columns=[
            "target_id", "target_lat", "target_lon", "vp_lat", "vp_lon", "shortest_ping_rtt_ms",
        ])

    idx = obs.groupby("target_id")["latency_ms"].idxmin()
    rows = obs.loc[idx, ["target_id", "target_lat", "target_lon", "vp_lat", "vp_lon", "latency_ms"]].copy()
    rows = rows.rename(columns={"latency_ms": "shortest_ping_rtt_ms"})
    return rows


def _margin_sign(margin: pd.Series) -> pd.Series:
    out = pd.Series("zero", index=margin.index, dtype=object)
    out[margin > 0] = "positive"
    out[margin < 0] = "negative"
    return out


def build_features(
    run_dir: Path,
    radius_km: float,
    source=None,
    slice_=None,
    clusters_dir: Path | None = None,
    inputs_dir: Path | None = None,
    inputs_root: Path = Path("scripts/benchmark/v2/inputs"),
) -> pd.DataFrame:
    combo_dirs = discover_combos(run_dir, source, slice_)
    if not combo_dirs:
        raise FileNotFoundError(f"no combos found under {run_dir}")

    index, n_centroids, n_targets = build_answer_space(
        run_dir, source, slice_, radius_km, clusters_dir=clusters_dir
    )
    logger.info("answer space: %d targets -> %d centroids", n_targets, n_centroids)

    resolved_inputs = resolve_inputs_dir(run_dir, combo_dirs, inputs_root, inputs_dir)
    if resolved_inputs is None:
        raise FileNotFoundError(
            "could not resolve inputs dir (pass --inputs-dir or --inputs-root)"
        )

    allowed_ids = geo_allowed_ids(combo_dirs)
    sp = _shortest_ping_rows(resolved_inputs, allowed_ids=allowed_ids)
    if sp.empty:
        raise RuntimeError("no shortest-ping rows available")

    # Shortest-ping correctness under nearest-centroid classification.
    v_idx, _ = index.query(sp["vp_lat"].to_numpy(), sp["vp_lon"].to_numpy())
    t_idx, _ = index.query(sp["target_lat"].to_numpy(), sp["target_lon"].to_numpy())
    sp["cls_result"] = (v_idx == t_idx) & (v_idx >= 0)
    sp["truth_centroid_lat"] = index.lat[t_idx]
    sp["truth_centroid_lon"] = index.lon[t_idx]

    sp["shortest_ping_vp_dist_km"] = haversine_distance(
        sp["target_lat"].to_numpy(dtype=float),
        sp["target_lon"].to_numpy(dtype=float),
        sp["vp_lat"].to_numpy(dtype=float),
        sp["vp_lon"].to_numpy(dtype=float),
    )
    sp["shortest_ping_centroid_dist_km"] = haversine_distance(
        sp["truth_centroid_lat"].to_numpy(dtype=float),
        sp["truth_centroid_lon"].to_numpy(dtype=float),
        sp["vp_lat"].to_numpy(dtype=float),
        sp["vp_lon"].to_numpy(dtype=float),
    )

    fleet = compute_fleet_geometry(resolved_inputs, index, allowed_ids=allowed_ids)
    fleet = fleet.rename(columns={"cell_gap_km": "target_cell_gap_km"})

    out = sp[[
        "target_id",
        "cls_result",
        "shortest_ping_vp_dist_km",
        "shortest_ping_centroid_dist_km",
        "shortest_ping_rtt_ms",
    ]].merge(
        fleet[[
            "target_id",
            "n_obs",
            "target_cell_gap_km",
            "closest_vp_dist_km",
            "target_distinguishable_vp_margin_km",
        ]],
        on="target_id",
        how="left",
    )

    out["shortest_ping_centroid_dist_margin"] = (
        out["shortest_ping_centroid_dist_km"] - out["closest_vp_dist_km"]
    )
    out["shortest_ping_centroid_dist_margin"] = out[
        "shortest_ping_centroid_dist_margin"
    ].clip(lower=0.0)

    out["sign_of_margin"] = _margin_sign(out["target_distinguishable_vp_margin_km"])

    out = out[out["n_obs"].fillna(0) > 1].copy()

    numeric_cols = [
        "n_obs",
        "target_cell_gap_km",
        "closest_vp_dist_km",
        "shortest_ping_vp_dist_km",
        "shortest_ping_centroid_dist_km",
        "shortest_ping_centroid_dist_margin",
        "shortest_ping_rtt_ms",
        "target_distinguishable_vp_margin_km",
    ]
    out[numeric_cols] = out[numeric_cols].round(2)

    # Keep exactly the requested feature set (plus target_id for identity).
    out = out[[
        "target_id",
        "cls_result",
        "n_obs",
        "target_cell_gap_km",
        "closest_vp_dist_km",
        "shortest_ping_vp_dist_km",
        "shortest_ping_centroid_dist_km",
        "shortest_ping_centroid_dist_margin",
        "shortest_ping_rtt_ms",
        "target_distinguishable_vp_margin_km",
        "sign_of_margin",
    ]].sort_values("target_id")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None,
                        help="Benchmark config YAML; run dir is resolved from run_id.")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Explicit outputs/<run_id>/ (overrides --config).")
    parser.add_argument("--outputs-root", type=Path, default=None,
                        help="Override outputs root used with --config.")
    parser.add_argument("--source", default=None, help="Filter combos by source name.")
    parser.add_argument("--slice", dest="slice_", default=None, help="Filter combos by slice id.")
    parser.add_argument("--radius-km", type=float, default=50.0,
                        help="Cluster radius for answer space. Default 50.")
    parser.add_argument("--clusters-dir", type=Path, default=None,
                        help="Precomputed cluster-eval dir. Recommended when available.")
    parser.add_argument("--inputs-dir", type=Path, default=None,
                        help="Materialized inputs dir. Auto-derived when omitted.")
    parser.add_argument("--inputs-root", type=Path,
                        default=Path("scripts/benchmark/v2/inputs"),
                        help="Root for auto-deriving --inputs-dir.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir (default: scripts/analysis/outputs/<run_id>/shortest_ping).")
    add_geo_filter_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_geo_filter_from_args(args)

    run_dir = resolve_run_dir(args.config, args.run_dir, args.outputs_root)
    out_dir = route_geo_path(args.out_dir) if args.out_dir else analysis_out_dir(run_dir, "shortest_ping")

    out = build_features(
        run_dir=run_dir,
        radius_km=args.radius_km,
        source=args.source,
        slice_=args.slice_,
        clusters_dir=args.clusters_dir,
        inputs_dir=args.inputs_dir,
        inputs_root=args.inputs_root,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_dir.name}_shortest_ping_features"
    csv_path = out_dir / f"{stem}.csv"
    parquet_path = out_dir / f"{stem}.parquet"
    out.to_csv(csv_path, index=False)
    out.to_parquet(parquet_path, index=False)

    logger.info("Saved %s  shape=%s", csv_path, out.shape)
    logger.info("Saved %s", parquet_path)


if __name__ == "__main__":
    main()
