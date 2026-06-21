"""Dump a canonical CSV's K-fold stratification + anchor metadata to JSON.

Produces the two files the
`scripts/processing/ripe_atlas/visualize_stratification.ipynb` notebook
expects, so the same notebook works against an in-memory generic_csv
stratification (no need to run the RIPE-Atlas-specific `stratify.py`).

Inputs:
    --csv     canonical-schema CSV (see scripts/benchmark/v2/sources/generic_csv.py)
    --k       fold count                                                 (default 5)
    --seed    DistGeo RNG seed                                           (default 42)
    --asn-bucket-top-n  DistGeo bucket cap                               (default 20)

Outputs (default landing pad: `datasets/<csv stem>/stratification/`):
    stratification.json   — fold_assignments + policy metadata, in the
                            shape that `stratify.py` writes and the
                            notebook expects at `PARTITION_PATH`.
    anchors.json          — one entry per unique target_id, shaped like a
                            RIPE Atlas anchor record:
                              {address_v4, geometry.coordinates:[lon,lat],
                               country_code, asn_v4}
                            (notebook expects this at `ANCHORS_FILE`).

Run::

    python -m scripts.benchmark.v2.sources.dump_csv_stratification \\
        --csv datasets/vultr_pings_us_canonical.csv

Then in the notebook, set the two parameter cells::

    PARTITION_PATH = "datasets/vultr_pings_us_canonical/stratification/stratification.json"
    ANCHORS_FILE   = "datasets/vultr_pings_us_canonical/stratification/anchors.json"

and Run All.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.processing.ripe_atlas.stratification import (
    AnchorInfo,
    DistGeoStratification,
    normalize_asn,
)

logger = logging.getLogger(__name__)


def _build_anchor_infos(df: pd.DataFrame) -> list[AnchorInfo]:
    """Dedupe by target_id; map canonical CSV columns to AnchorInfo fields."""
    unique = df.drop_duplicates("target_id")
    out: list[AnchorInfo] = []
    for _, row in unique.iterrows():
        asn = row.get("target_asn")
        country = row.get("target_country")
        out.append(AnchorInfo(
            ip=str(row["target_id"]),
            lat=float(row["target_lat"]),
            lon=float(row["target_lon"]),
            country=str(country) if pd.notna(country) else None,
            asn=normalize_asn(asn) if pd.notna(asn) else None,
        ))
    return out


def _stratification_json(
    assignments: dict[str, int],
    *,
    k: int,
    seed: int,
    asn_bucket_top_n: int,
    csv_source: str,
) -> dict:
    fold_sizes = [sum(1 for f in assignments.values() if f == i) for i in range(k)]
    return {
        "policy": {
            "class": "DistGeoStratification",
            "kind": "dist_geo_kfold",
            "k": k,
            "seed": seed,
            "asn_bucket_top_n": asn_bucket_top_n,
        },
        "corpus": {
            "source": csv_source,
            "n_anchors_yielded": len(assignments),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fold_sizes": fold_sizes,
        "fold_assignments": assignments,
    }


def _anchors_json(infos: list[AnchorInfo]) -> list[dict]:
    """RIPE-Atlas anchor-record shape: the notebook reads
    `geometry.coordinates[0]` as lon and `[1]` as lat."""
    return [
        {
            "address_v4": a.ip,
            "geometry": {"coordinates": [a.lon, a.lat]},
            "country_code": a.country,
            "asn_v4": a.asn,
        }
        for a in infos
    ]


def dump(
    csv_path: Path,
    out_dir: Path,
    *,
    k: int,
    seed: int,
    asn_bucket_top_n: int,
) -> tuple[Path, Path]:
    df = pd.read_csv(csv_path)
    infos = _build_anchor_infos(df)
    algo = DistGeoStratification(
        k=k,
        fold_index=0,  # vestigial in compute_fold_assignments; see stratification.py:517
        seed=seed,
        asn_bucket_top_n=asn_bucket_top_n,
    )
    assignments = algo.compute_fold_assignments(infos)

    out_dir.mkdir(parents=True, exist_ok=True)
    strat_path = out_dir / "stratification.json"
    anchors_path = out_dir / "anchors.json"

    strat_path.write_text(json.dumps(
        _stratification_json(
            assignments,
            k=k, seed=seed, asn_bucket_top_n=asn_bucket_top_n,
            csv_source=str(csv_path),
        ),
        indent=2,
    ))
    anchors_path.write_text(json.dumps(_anchors_json(infos), indent=2))
    return strat_path, anchors_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True,
                        help="Canonical-schema CSV (vp_*, target_*, rtt_ms columns).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory. Defaults to datasets/<csv-stem>/stratification/.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--asn-bucket-top-n", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_dir = args.out_dir
    if out_dir is None:
        repo_root = Path(__file__).resolve().parents[4]
        out_dir = repo_root / "datasets" / args.csv.stem / "stratification"

    strat_path, anchors_path = dump(
        args.csv, out_dir,
        k=args.k, seed=args.seed, asn_bucket_top_n=args.asn_bucket_top_n,
    )
    logger.info("wrote %s", strat_path)
    logger.info("wrote %s", anchors_path)


if __name__ == "__main__":
    main()
