"""Stage c — precompute (probe, country) → great-circle border distance table.

Two modes:

  * `--shard I --n-shards N`: precompute for the I-th shard of the probe pool
    (probes hash-bucketed by index modulo N). Writes a single parquet shard.
    Snakemake fans these out across cores.
  * `--merge --inputs <shard.parquet> ... --output <merged.parquet>`: concatenate
    the per-shard parquets into the canonical lookup table.

Inputs (shard mode):
  * `claims.parquet` (from Stage b) — defines the set of claimed countries
    for which we actually need distances.
  * RipeAtlasSource probe pool — same loader used by `calibrate_speed.py`.
  * Natural Earth shapefile — see README. Auto-detects ISO_A2 vs ISO_A3.

Output: parquet with columns `vp_id`, `country` (whatever ISO kind the
shapefile uses), `distance_km` (float, 0.0 if the probe is inside the
country polygon).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import default
from scripts.benchmark.v2.sources.base import DataSource
from scripts.benchmark.v2.sources.ripe_atlas import RipeAtlasSource
from scripts.vp_selection.country_borders import (
    load_country_polygons,
    nearest_border_distance_km,
)

logger = logging.getLogger(__name__)


def _default_naturalearth_path() -> Path:
    """Lookup chain: user-staged path, then pyogrio test fixture."""
    user_path = (
        Path(default.STATIC_PATH)
        / "naturalearth_lowres"
        / "ne_110m_admin_0_countries.shp"
    )
    if user_path.exists():
        return user_path
    for entry in sys.path:
        candidate = (
            Path(entry)
            / "pyogrio" / "tests" / "fixtures"
            / "naturalearth_lowres" / "naturalearth_lowres.shp"
        )
        if candidate.exists():
            return candidate
    return user_path  # let load fail with a clear error


def _load_iso3166_a2_to_a3() -> dict[str, str]:
    import csv
    mapping: dict[str, str] = {}
    path = Path(__file__).resolve().parent / "upstream_csv" / "iso3166.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            a2 = row["ISO_A2"].strip()
            a3 = row["ISO_A3"].strip()
            if a2 and a3:
                mapping[a2] = a3
    return mapping


def _build_a2_translator(iso_kind: str):
    if iso_kind == "ISO_A2":
        return lambda a2: a2
    a2_to_a3 = _load_iso3166_a2_to_a3()
    return lambda a2: a2_to_a3.get(a2)


def _load_probe_pool() -> dict[str, tuple[float, float]]:
    """Probe pool with coords, post-SOI."""
    source = RipeAtlasSource(
        slice="all_anchors",
        setup=DataSource.PROBES_TO_ANCHORS,
        sanitize=True,
    )
    return {
        vp.vp_id: (vp.lat, vp.lon)
        for vp in source.iter_vp_configs()
    }


def _load_claimed_countries(claims_path: Path) -> list[str]:
    """Read the unique claimed_country values out of `claims.parquet`."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(claims_path, columns=["claimed_country"])
        return sorted(set(table.column("claimed_country").to_pylist()))
    except ImportError:
        import csv
        seen: set[str] = set()
        with open(claims_path.with_suffix(".csv")) as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen.add(row["claimed_country"])
        return sorted(seen)


def run_shard(
    shard_idx: int,
    n_shards: int,
    claims_path: Path,
    output_path: Path,
    shapefile_path: Path,
) -> None:
    """Compute (probe, country) distances for the shard-i probes."""
    probes_all = _load_probe_pool()
    probe_ids = sorted(probes_all)
    shard_probes = {
        vp_id: probes_all[vp_id]
        for i, vp_id in enumerate(probe_ids)
        if i % n_shards == shard_idx
    }
    logger.info("shard %d/%d: %d probes", shard_idx, n_shards, len(shard_probes))

    polygons, iso_kind = load_country_polygons(shapefile_path)
    translate = _build_a2_translator(iso_kind)

    claimed_a2 = _load_claimed_countries(claims_path)
    polygon_keys: list[str] = []
    for a2 in claimed_a2:
        key = translate(a2)
        if key is not None and key in polygons:
            polygon_keys.append(key)
    polygon_keys = sorted(set(polygon_keys))
    logger.info("shard %d/%d: %d distinct country polygons",
                shard_idx, n_shards, len(polygon_keys))

    rows: list[dict] = []
    for vp_id, coord in shard_probes.items():
        for country in polygon_keys:
            d = nearest_border_distance_km(coord, country, polygons)
            rows.append({"vp_id": vp_id, "country": country, "distance_km": d})

    _write_parquet(rows, output_path)
    logger.info("shard %d/%d: wrote %d rows to %s",
                shard_idx, n_shards, len(rows), output_path)


def run_merge(inputs: list[Path], output_path: Path) -> None:
    """Concatenate per-shard parquets into one canonical lookup table."""
    try:
        import pyarrow.parquet as pq
        tables = [pq.read_table(p) for p in inputs]
        if not tables:
            raise RuntimeError("no input shards to merge")
        import pyarrow as pa
        merged = pa.concat_tables(tables)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(merged, output_path)
        logger.info("merged %d shards (%d rows) → %s",
                    len(inputs), merged.num_rows, output_path)
    except ImportError:
        # CSV fallback
        import csv
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path.with_suffix(".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["vp_id", "country", "distance_km"])
            for p in inputs:
                with open(Path(p).with_suffix(".csv")) as fin:
                    reader = csv.reader(fin)
                    next(reader, None)
                    for row in reader:
                        w.writerow(row)
        logger.info("merged %d shards → CSV fallback", len(inputs))


def _write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        if rows:
            table = pa.Table.from_pylist(rows)
        else:
            table = pa.table({"vp_id": [], "country": [], "distance_km": []})
        pq.write_table(table, path)
    except ImportError:
        import csv
        with open(path.with_suffix(".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["vp_id", "country", "distance_km"])
            w.writeheader()
            w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shard", type=int, default=None,
                   help="Shard index (0..n_shards-1).")
    p.add_argument("--n-shards", type=int, default=None)
    p.add_argument("--claims", type=Path, default=None,
                   help="Path to claims.parquet (Stage b output).")
    p.add_argument("--merge", action="store_true",
                   help="Merge mode: concat input shards into one parquet.")
    p.add_argument("--inputs", nargs="*", type=Path, default=[],
                   help="(merge mode) shard parquet paths.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shapefile-path", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.merge:
        if not args.inputs:
            raise SystemExit("--merge requires --inputs")
        run_merge(args.inputs, args.output)
        return

    if args.shard is None or args.n_shards is None or args.claims is None:
        raise SystemExit("shard mode requires --shard, --n-shards, --claims")
    shapefile_path = args.shapefile_path or _default_naturalearth_path()
    run_shard(args.shard, args.n_shards, args.claims, args.output, shapefile_path)


if __name__ == "__main__":
    main()
