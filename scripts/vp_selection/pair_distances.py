"""Pair-distance generators for VP selection.

`compute_geodesic_distances` produces the haversine distance graph the Prim-style
selection strategies (`scripts/vp_selection/strategies.py`) maximize over.

CLI usage: writes a parquet `{vp_a, vp_b, distance_km}` (one row per
canonical pair with `vp_a < vp_b`) so the snakemake pipeline can compute
the O(N²) distances once and reuse them across all sequence-strategy runs.
"""

from __future__ import annotations

import argparse
import logging
from itertools import combinations
from pathlib import Path
from typing import Mapping, Tuple

from scripts.libs.cbg.rtt_model import haversine_distance

logger = logging.getLogger(__name__)


def compute_geodesic_distances(
    coords: Mapping[str, Tuple[float, float]],
) -> dict[tuple[str, str], float]:
    """Pairwise great-circle distances (km) over a VP pool.

    Returns `{(a, b): km}` with `a < b` lexicographic for every unordered pair.
    No self-pair entries. Empty or single-element pools return an empty dict.
    """
    pairs: dict[tuple[str, str], float] = {}
    for a, b in combinations(sorted(coords), 2):
        lat_a, lon_a = coords[a]
        lat_b, lon_b = coords[b]
        pairs[(a, b)] = haversine_distance(lat_a, lon_a, lat_b, lon_b)
    return pairs


def write_parquet(
    pairs: dict[tuple[str, str], float],
    path: Path,
) -> None:
    """Write `(vp_a, vp_b, distance_km)` rows to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"vp_a": a, "vp_b": b, "distance_km": d}
        for (a, b), d in pairs.items()
    ]
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        if rows:
            table = pa.Table.from_pylist(rows)
        else:
            table = pa.table({"vp_a": [], "vp_b": [], "distance_km": []})
        pq.write_table(table, path)
    except ImportError:
        import csv
        with open(path.with_suffix(".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["vp_a", "vp_b", "distance_km"])
            w.writeheader()
            w.writerows(rows)


def load_parquet(path: Path) -> dict[tuple[str, str], float]:
    """Read a pair-distance parquet back into the canonical dict shape."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        rows = table.to_pylist()
    except ImportError:
        import csv
        with open(path.with_suffix(".csv")) as f:
            rows = [
                {"vp_a": r["vp_a"], "vp_b": r["vp_b"],
                 "distance_km": float(r["distance_km"])}
                for r in csv.DictReader(f)
            ]
    return {(r["vp_a"], r["vp_b"]): float(r["distance_km"]) for r in rows}


def _load_probe_pool() -> dict[str, tuple[float, float]]:
    import default
    from scripts.benchmark.v2.sources.base import DataSource
    from scripts.benchmark.v2.sources.ripe_atlas import RipeAtlasSource
    source = RipeAtlasSource(
        slice="all_anchors",
        setup=DataSource.PROBES_TO_ANCHORS,
        sanitize=True,
    )
    return {vp.vp_id: (vp.lat, vp.lon) for vp in source.iter_vp_configs()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, required=True,
                   help="Parquet output path.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    coords = _load_probe_pool()
    logger.info("loaded %d probes; computing %d pair distances",
                len(coords), len(coords) * (len(coords) - 1) // 2)
    pairs = compute_geodesic_distances(coords)
    write_parquet(pairs, args.output)
    logger.info("wrote %d pairs to %s", len(pairs), args.output)


if __name__ == "__main__":
    main()
