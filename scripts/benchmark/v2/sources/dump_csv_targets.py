"""Dump a canonical CSV's target catalog to `targets.json`.

Sibling of `scripts/benchmark/v2/sources/dump_csv_vps.py`. That script
materializes the *VP*-role entities (`vps.json`) from the `vp_*` columns; this
one materializes the *target*-role entities (`targets.json`, the anchors in
RIPE Atlas cases) from the `target_*` columns.

Target parsing is delegated to `GenericCSVSource.iter_tg_configs`, so the same
canonical-schema contract applies — required `target_id, target_lat,
target_lon`, optional `target_asn, target_country, target_continent,
target_region, target_city` — with identical ASN normalization and NA-safe
string handling. Targets are deduped by `target_id`.

Inputs:
    --csv       canonical-schema CSV (see scripts/benchmark/v2/sources/generic_csv.py)
    --stratify  also write a DistGeo K-fold stratification of the targets
    --k                 fold count                          (default 5)
    --seed              DistGeo RNG seed                     (default 42)
    --asn-bucket-top-n  DistGeo bucket cap                   (default 20)

Output (default landing pad: `datasets/<csv stem>/`):
    targets.json — one entry per unique target_id, with the canonical CSV's own
                   field names (no RIPE-Atlas renaming):
                     {target_id, target_lat, target_lon, target_asn,
                      target_country,
                      target_continent/target_region/target_city when present
                      in the CSV}
    stratification/stratification.json — only with `--stratify`; fold_assignments
                   + policy metadata in the shape `stratify.py` writes (the same
                   file `legacy_dump_csv_stratification.py` produces).

Run::

    python -m scripts.benchmark.v2.sources.dump_csv_targets \\
        --csv datasets/vultr_pings_us_canonical.csv --stratify
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from scripts.benchmark.v2.sources.base import TgConfig
from scripts.benchmark.v2.sources.legacy_dump_csv_stratification import _stratification_json
from scripts.benchmark.v2.sources.generic_csv import GenericCSVSource
from scripts.processing.ripe_atlas.stratification import (
    AnchorInfo,
    DistGeoStratification,
)

logger = logging.getLogger(__name__)


def _target_json(tg: TgConfig) -> dict:
    """Canonical-schema target record: same field names the CSV uses."""
    out: dict = {
        "target_id": tg.tg_id,
        "target_lat": tg.lat,
        "target_lon": tg.lon,
        "target_asn": tg.asn,
        "target_country": tg.country,
    }
    # Pass through the extra geo columns only when the CSV supplied them, so
    # the output stays minimal for the common asn/country/coords-only case.
    if tg.continent is not None:
        out["target_continent"] = tg.continent
    if tg.region is not None:
        out["target_region"] = tg.region
    if tg.city is not None:
        out["target_city"] = tg.city
    return out


def _write_stratification(
    tgs: list[TgConfig],
    strat_path: Path,
    *,
    csv_path: Path,
    k: int,
    seed: int,
    asn_bucket_top_n: int,
) -> None:
    """Compute the DistGeo K-fold assignment over the target catalog and write
    it in the same shape `stratify.py` / `legacy_dump_csv_stratification.py` produce."""
    infos = [
        AnchorInfo(ip=tg.tg_id, lat=tg.lat, lon=tg.lon, country=tg.country, asn=tg.asn)
        for tg in tgs
    ]
    algo = DistGeoStratification(
        k=k,
        fold_index=0,  # full assignment is fold-index-independent
        seed=seed,
        asn_bucket_top_n=asn_bucket_top_n,
    )
    assignments = algo.compute_fold_assignments(infos)
    strat_path.parent.mkdir(parents=True, exist_ok=True)
    strat_path.write_text(json.dumps(
        _stratification_json(
            assignments,
            k=k, seed=seed, asn_bucket_top_n=asn_bucket_top_n,
            csv_source=str(csv_path),
        ),
        indent=2,
    ))
    logger.info("wrote stratification (%d targets, k=%d) to %s",
                len(assignments), k, strat_path)


def dump(
    csv_path: Path,
    out_path: Path,
    *,
    stratify: bool = False,
    k: int = 5,
    seed: int = 42,
    asn_bucket_top_n: int = 20,
) -> Path:
    source = GenericCSVSource(slice="all", csv_path=csv_path)
    tgs = list(source.iter_tg_configs())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([_target_json(tg) for tg in tgs], indent=2))
    logger.info("wrote %d targets to %s", len(tgs), out_path)

    if stratify:
        strat_path = out_path.parent / "stratification" / "stratification.json"
        _write_stratification(
            tgs, strat_path,
            csv_path=csv_path, k=k, seed=seed, asn_bucket_top_n=asn_bucket_top_n,
        )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True,
                        help="Canonical-schema CSV (vp_*, target_*, rtt_ms columns).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path. Defaults to datasets/<csv-stem>/targets.json.")
    parser.add_argument("--stratify", action="store_true",
                        help="Also write stratification/stratification.json next to targets.json.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--asn-bucket-top-n", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_path = args.out
    if out_path is None:
        repo_root = Path(__file__).resolve().parents[4]
        out_path = repo_root / "datasets" / args.csv.stem / "targets.json"

    dump(
        args.csv, out_path,
        stratify=args.stratify,
        k=args.k, seed=args.seed, asn_bucket_top_n=args.asn_bucket_top_n,
    )


if __name__ == "__main__":
    main()
