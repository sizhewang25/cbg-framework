"""Stage b — anchor-country claim assignment with fake injection.

Each anchor gets exactly one (claimed_country, is_real) row. With probability
`fake_fraction` we overwrite the claimed_country to a uniform-random *wrong*
country (from `all_countries` excluding the real one); otherwise we keep the
real country.

Writes a parquet with columns: `target_id, real_country, claimed_country,
is_real`. Consumed downstream by `borders_precompute` (needs the union of
claimed countries to know which polygons to project) and `agreement` (needs
the (target, claim, is_real) triples for the sweep).
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import default
from scripts.benchmark.v2.sources.base import DataSource
from scripts.benchmark.v2.sources.ripe_atlas import RipeAtlasSource

logger = logging.getLogger(__name__)


def assign_claims(
    target_countries: dict[str, str],
    all_countries: list[str],
    fake_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Assign one claim per target. `fake_fraction` controls the rate of
    fake-country overwrites; `seed` makes the assignment reproducible.

    Returns a list of `{target_id, real_country, claimed_country, is_real}`
    dicts in the input-target-id sort order.
    """
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for target_id in sorted(target_countries):
        real_cc = target_countries[target_id]
        is_fake = rng.random() < fake_fraction
        if is_fake:
            wrong_pool = [c for c in all_countries if c != real_cc]
            if not wrong_pool:
                # No alternative country available — degrade to real claim.
                claim = real_cc
                is_real = True
            else:
                claim = rng.choice(sorted(wrong_pool))
                is_real = False
        else:
            claim = real_cc
            is_real = True
        rows.append({
            "target_id": target_id,
            "real_country": real_cc,
            "claimed_country": claim,
            "is_real": is_real,
        })
    return rows


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    """Write claim rows to parquet. Falls back to CSV if pyarrow unavailable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pylist(rows) if rows else pa.table({
            "target_id": [], "real_country": [],
            "claimed_country": [], "is_real": [],
        })
        pq.write_table(table, path)
    except ImportError:
        import csv
        with open(path.with_suffix(".csv"), "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["target_id", "real_country",
                               "claimed_country", "is_real"],
            )
            w.writeheader()
            w.writerows(rows)


def _load_target_countries() -> dict[str, str]:
    """Load anchor target countries from RipeAtlasSource (probes_to_anchors)."""
    source = RipeAtlasSource(
        slice="all_anchors",
        setup=DataSource.PROBES_TO_ANCHORS,
        sanitize=True,
    )
    target_countries: dict[str, str] = {}
    for tg in source.iter_tg_configs():
        if tg.country:
            target_countries[tg.tg_id] = tg.country
    return target_countries


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, required=True,
                   help="Output parquet path.")
    p.add_argument("--fake-fraction", type=float, default=0.30)
    p.add_argument("--fake-seed", type=int, default=9001)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    target_countries = _load_target_countries()
    all_countries = sorted(set(target_countries.values()))
    logger.info("loaded %d target anchors covering %d distinct countries",
                len(target_countries), len(all_countries))

    rows = assign_claims(
        target_countries=target_countries,
        all_countries=all_countries,
        fake_fraction=args.fake_fraction,
        seed=args.fake_seed,
    )
    n_fake = sum(1 for r in rows if not r["is_real"])
    logger.info("assigned %d claims: %d real, %d fake (%.1f%%)",
                len(rows), len(rows) - n_fake, n_fake,
                100 * n_fake / max(1, len(rows)))

    write_parquet(rows, args.output)
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
