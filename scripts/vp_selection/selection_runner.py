"""Stage d.1 — per-(strategy, seed) selection runner.

Dispatches to `sample_vps` (sampling strategies) or `select_vps` (sequence
strategies), iterating K over a configurable grid, and writes one parquet
per (strategy, seed) to the snakemake selections directory.

Output schemas:
  - Sampling: columns `strategy, seed, k, vp_id`  (K rows per K-value)
  - Sequence: columns `strategy, seed, position, vp_id`  (N rows total)

The sweep stage (Stage d.2) reads both schemas and dispatches the verifier
accordingly: sampling → per-(K, target) verifier call; sequence → one
first-violator scan per (target, claim).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import default
from scripts.benchmark.v2.sources.base import DataSource
from scripts.benchmark.v2.sources.ripe_atlas import RipeAtlasSource
from scripts.vp_selection.pair_distances import (
    compute_geodesic_distances,
    load_parquet as load_pair_distances_parquet,
)
from scripts.vp_selection.strategies import (
    VpMeta,
    sample_vps,
    select_vps,
)

logger = logging.getLogger(__name__)

_SAMPLING = ("random", "cluster_as", "cluster_city")
_SEQUENCE = ("dist_geo", "h1_as", "h1_city", "h2_as")


def build_k_grid(pool_size: int, k_min: int = 100, k_step: int = 100) -> list[int]:
    """K values to sweep, descending. Starts at `pool_size` and steps down
    by `k_step`, stopping at the largest K ≥ `k_min`.

    If `pool_size < k_min`, returns `[pool_size]` so callers still get at
    least one valid K to evaluate.
    """
    k_min = max(1, k_min)
    if pool_size <= 0:
        return []
    if pool_size < k_min:
        return [pool_size]
    grid: list[int] = []
    k = pool_size
    while k >= k_min:
        grid.append(k)
        k -= k_step
    return grid


def run_one_sampling(
    pool: dict[str, VpMeta],
    strategy: str,
    seed: int,
    k_grid: list[int],
) -> list[dict[str, Any]]:
    """Run a sampling strategy at each K in `k_grid`, return flat rows."""
    rows: list[dict[str, Any]] = []
    for k in k_grid:
        subset = sample_vps(pool, strategy=strategy, K=k, seed=seed)  # type: ignore[arg-type]
        for vp_id in subset:
            rows.append({
                "strategy": strategy,
                "seed": seed,
                "k": k,
                "vp_id": vp_id,
            })
    return rows


def run_one_sequence(
    pool: dict[str, VpMeta],
    distances: dict[tuple[str, str], float],
    strategy: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Run a sequence strategy once, return rows with 1-indexed positions."""
    seq = select_vps(pool, distances, strategy=strategy, seed=seed)  # type: ignore[arg-type]
    return [
        {
            "strategy": strategy,
            "seed": seed,
            "position": pos,
            "vp_id": vp_id,
        }
        for pos, vp_id in enumerate(seq, start=1)
    ]


def _load_pool() -> dict[str, VpMeta]:
    source = RipeAtlasSource(
        slice="all_anchors",
        setup=DataSource.PROBES_TO_ANCHORS,
        sanitize=True,
    )
    return {
        vp.vp_id: VpMeta(
            lat=vp.lat, lon=vp.lon,
            asn=vp.asn, city=vp.city, country=vp.country,
        )
        for vp in source.iter_vp_configs()
    }


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        if rows:
            table = pa.Table.from_pylist(rows)
        else:
            table = pa.table({k: [] for k in
                              ("strategy", "seed", "k", "vp_id")})
        pq.write_table(table, path)
    except ImportError:
        import csv
        if not rows:
            return
        with open(path.with_suffix(".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--strategy", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k-min", type=int, default=100)
    p.add_argument("--k-step", type=int, default=100)
    p.add_argument("--pair-distances", type=Path, default=None,
                   help="Cached pair-distance parquet "
                        "(required for sequence strategies; ignored for sampling).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    pool = _load_pool()
    logger.info("loaded pool of %d probes", len(pool))

    if args.strategy in _SAMPLING:
        k_grid = build_k_grid(len(pool), k_min=args.k_min, k_step=args.k_step)
        logger.info("sampling strategy %s — K grid: %d values (%d..%d)",
                    args.strategy, len(k_grid),
                    k_grid[-1] if k_grid else 0,
                    k_grid[0] if k_grid else 0)
        rows = run_one_sampling(pool, args.strategy, args.seed, k_grid)
    elif args.strategy in _SEQUENCE:
        if args.pair_distances is not None and args.pair_distances.exists():
            logger.info("loading cached pair distances from %s",
                        args.pair_distances)
            distances = load_pair_distances_parquet(args.pair_distances)
        else:
            logger.info("computing pair distances inline "
                        "(pass --pair-distances for caching)")
            coords = {vp_id: (m.lat, m.lon) for vp_id, m in pool.items()}
            distances = compute_geodesic_distances(coords)
        logger.info("sequence strategy %s — computing N-element ordering",
                    args.strategy)
        rows = run_one_sequence(pool, distances, args.strategy, args.seed)
    else:
        raise SystemExit(f"unknown strategy: {args.strategy}")

    _write_parquet(rows, args.output)
    logger.info("wrote %d rows to %s", len(rows), args.output)


if __name__ == "__main__":
    main()
