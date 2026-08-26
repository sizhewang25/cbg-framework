"""Reciprocal clustering: combine both datasets, cluster, keep mixed clusters.

Workflow:
2. Cluster all together at radius R (default 50 km)
4. Per cluster, ensure equal (or balanced) counts from each source

This maximizes VP preservation while ensuring every cluster represents both datasets.

Example:
    python scripts/benchmark/v2/sources/reciprocal_clustering.py \
        --ripe datasets/ripe-as7018-US-slice/vps.csv \
        --radius-km 50 \
        --out-dir datasets/as7018_common_50km_reciprocal \
        --balance-method per_cluster
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.benchmark.v2.sources.cluster_ground_truth import (
        cluster_ground_truth,
    )
except ModuleNotFoundError:
    # Fallback for standalone execution
    _mod_path = Path(__file__).parent / "cluster_ground_truth.py"
    _spec = importlib.util.spec_from_file_location("cluster_ground_truth_local", _mod_path)
    if _spec is None or _spec.loader is None:
        raise
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    cluster_ground_truth = _mod.cluster_ground_truth


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ripe", type=Path, required=True, help="RIPE VP CSV")
    p.add_argument("--radius-km", type=float, default=50.0, help="Cluster radius")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    p.add_argument(
        "--balance-method",
        choices=["per_cluster", "min_per_cluster"],
        default="per_cluster",
        help="Balance strategy within kept clusters",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    np.random.seed(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    ripe_raw = pd.read_csv(args.ripe)

    # Prepare columns
    ripe_t = ripe_raw[["vp_id", "vp_lat", "vp_lon"]].copy()
    ripe_t.columns = ["vp_id", "target_lat", "target_lon"]
    ripe_t["source"] = "RIPE"


    # Combine

    # Cluster all together
    res = cluster_ground_truth(
        combined["target_lat"].to_numpy(),
        combined["target_lon"].to_numpy(),
        radius_km=float(args.radius_km),
    )

    combined["cluster_id"] = res.labels

    # Find clusters with both sources
    cluster_sources = combined.groupby("cluster_id")["source"].apply(set)
    mixed_clusters = cluster_sources[cluster_sources.apply(len) == 2].index.tolist()

    print(f"Total clusters: {res.n_clusters}")
    print(f"Mixed clusters (both sources): {len(mixed_clusters)}")

    # Filter to mixed clusters
    combined_kept = combined[combined["cluster_id"].isin(mixed_clusters)].copy()

    # Balance per cluster
    if args.balance_method == "per_cluster":
        kept_rows = []
        for cid in mixed_clusters:
            cluster_df = combined_kept[combined_kept["cluster_id"] == cid]
            ripe_rows = cluster_df[cluster_df["source"] == "RIPE"]


            if n_target > 0:
                ripe_keep = ripe_rows.sample(n=n_target, random_state=args.seed)
                kept_rows.append(ripe_keep)

        combined_balanced = pd.concat(kept_rows, ignore_index=True)
    else:
        combined_balanced = combined_kept

    # Split back into datasets
    ripe_final = combined_balanced[combined_balanced["source"] == "RIPE"].copy()

    # Map back to original schema
    ripe_out = ripe_raw[ripe_raw["vp_id"].astype(str).isin(ripe_final["vp_id"].astype(str))].copy()

    # Cluster summary
    clusters_df = pd.DataFrame(
        {
            "cluster_id": np.arange(res.n_clusters),
            "centroid_lat": res.centroid_lat,
            "centroid_lon": res.centroid_lon,
            "n_members": res.member_counts,
            "is_mixed": np.isin(np.arange(res.n_clusters), mixed_clusters),
            "radius_km": np.round(res.radius_km, 3),
        }
    )

    # Assignments with cluster membership
    combined_out = combined_balanced.copy()
    combined_out = combined_out.sort_values("cluster_id").reset_index(drop=True)

    # Write outputs
    ripe_out.to_csv(out_dir / "ripe_reciprocal.csv", index=False)
    clusters_df.to_csv(out_dir / "clusters_reciprocal.csv", index=False)
    combined_out.to_csv(out_dir / "combined_assignments.csv", index=False)

    summary = {
        "radius_km": float(args.radius_km),
        "balance_method": args.balance_method,
        "seed": args.seed,
        "input": {
            "ripe_n": len(ripe_raw),
            "total_n": len(combined),
        },
        "clustering": {
            "total_clusters": int(res.n_clusters),
            "mixed_clusters": int(len(mixed_clusters)),
        },
        "output": {
            "ripe_preserved_n": len(ripe_out),
            "ripe_preserved_fraction": float(len(ripe_out) / len(ripe_raw)),
        },
    }

    with open(out_dir / "summary_reciprocal.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'ripe_reciprocal.csv'}")
    print(f"wrote {out_dir / 'clusters_reciprocal.csv'}")
    print(f"wrote {out_dir / 'combined_assignments.csv'}")
    print(f"wrote {out_dir / 'summary_reciprocal.json'}")


if __name__ == "__main__":
    main()
