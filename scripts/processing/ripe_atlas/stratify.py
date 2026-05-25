"""Compute + serialize an anchor-level K-fold stratification.

Standalone tool — does NOT depend on ClickHouse. Loads the anchor metadata
directly from `reproducibility_anchors.json` (723 anchors — the IMC 2023
paper's canonical count) and computes a fold assignment with the chosen
stratification algorithm.

`RipeAtlasSource` consumes the output JSON via `LoadedStratification`
(from `stratification.py`) at materialize time, intersecting the loaded
assignments with the post-sanitize active anchor set. See `report.md` of
the leakage-free eval protocol task for context.

Usage:
  python -m scripts.processing.ripe_atlas.stratify \\
      --algo distgeo --k 5 --seed 42 --asn-bucket-top-n 20

  python -m scripts.processing.ripe_atlas.stratify \\
      --algo sechidis --k 5 --seed 42 --spatial-clusters 30

Output: `datasets/ripe_atlas/stratifications/<algo>/<param-tag>.json`
with structure:
  {
    "policy": {...},                 # algo class + params
    "corpus": {...},                 # source + anchor count + extraction flags
    "generated_at": "...",
    "fold_assignments": {ip: fold},
    "fold_sizes": [n0, n1, ...],
  }
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make `default` importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import default  # noqa: E402

from scripts.processing.ripe_atlas.stratification import (  # noqa: E402
    AnchorInfo,
    DistGeoStratification,
    SechidisStratification,
)


def load_anchors(anchors_file: Path) -> list[AnchorInfo]:
    """Read `reproducibility_anchors.json` and return the entries as
    AnchorInfo records.

    The anchors-only file (723 entries) is the canonical input — same
    count the IMC 2023 paper reports. Drops entries missing
    `address_v4` or `geometry.coordinates`. No sanitization or RTT
    filtering — that's what `RipeAtlasSource` does. This script captures
    the raw anchor set so stratification output is reproducible without a
    database.
    """
    with anchors_file.open() as fh:
        entries = json.load(fh)

    anchors: list[AnchorInfo] = []
    for e in entries:
        ip = e.get("address_v4")
        geom = (e.get("geometry") or {}).get("coordinates")
        if not ip or not geom or len(geom) < 2:
            continue
        lon, lat = geom[0], geom[1]
        anchors.append(AnchorInfo(
            ip=ip,
            lat=float(lat),
            lon=float(lon),
            country=e.get("country_code"),
            asn=e.get("asn_v4"),
        ))
    return anchors


def build_algo(args: argparse.Namespace):
    """Return a SechidisStratification or DistGeoStratification from CLI args."""
    if args.algo == "sechidis":
        return SechidisStratification(
            k=args.k,
            fold_index=0,  # full assignment is fold_index-independent
            seed=args.seed,
            spatial_clusters=args.spatial_clusters,
            asn_bucket_top_n=args.asn_bucket_top_n,
        )
    if args.algo == "distgeo":
        return DistGeoStratification(
            k=args.k,
            fold_index=0,
            seed=args.seed,
            asn_bucket_top_n=args.asn_bucket_top_n,
        )
    raise ValueError(f"unknown algo {args.algo!r}")


def param_tag(args: argparse.Namespace) -> str:
    """Filename-safe encoding of the salient algo params."""
    if args.algo == "sechidis":
        spatial = "none" if args.spatial_clusters is None else str(args.spatial_clusters)
        return f"k{args.k}_seed{args.seed}_spatial{spatial}_top{args.asn_bucket_top_n}"
    if args.algo == "distgeo":
        return f"k{args.k}_seed{args.seed}_top{args.asn_bucket_top_n}"
    raise ValueError(f"unknown algo {args.algo!r}")


def algo_to_dict(algo) -> dict[str, Any]:
    """Serialize a frozen-dataclass algo to plain JSON-able dict.
    `fold_index` is omitted — it's not part of stratification identity."""
    d = asdict(algo)
    d.pop("fold_index", None)
    return d


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute + save an anchor-level K-fold stratification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo", choices=["sechidis", "distgeo"], required=True,
        help="which stratification algorithm to apply",
    )
    parser.add_argument("--k", type=int, default=5, help="number of folds")
    parser.add_argument("--seed", type=int, default=42, help="determinism seed")
    parser.add_argument(
        "--asn-bucket-top-n", type=int, default=20,
        help="top-N ASNs each get their own bucket; rest collapse to other_AS",
    )
    parser.add_argument(
        "--spatial-clusters", type=int, default=30,
        help="(sechidis only) k-means cluster count; pass --spatial-none to disable",
    )
    parser.add_argument(
        "--spatial-none", action="store_true",
        help="(sechidis only) disable spatial blocking",
    )
    parser.add_argument(
        "--anchors-file", type=Path,
        default=Path(default.REPRO_ANCHORS_FILE),
        help="path to reproducibility_anchors.json (the 723-anchor canonical set)",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("datasets/ripe_atlas/stratifications"),
        help="root dir; final path is <root>/<algo>/<param-tag>.json",
    )
    args = parser.parse_args()

    if args.spatial_none:
        args.spatial_clusters = None

    print(f"loading anchors from {args.anchors_file} ...")
    anchors = load_anchors(args.anchors_file)
    print(f"  {len(anchors)} anchors with v4 IP + geometry")

    algo = build_algo(args)
    print(f"applying {algo.__class__.__name__}({algo_to_dict(algo)}) ...")
    fold_by_ip = algo.compute_fold_assignments(anchors)
    fold_sizes = [
        sum(1 for f in fold_by_ip.values() if f == i) for i in range(args.k)
    ]
    print(f"  fold sizes: {fold_sizes}")

    out_payload = {
        "policy": {
            "class": algo.__class__.__name__,
            **algo_to_dict(algo),
        },
        "corpus": {
            "source": "ripe_atlas",
            "anchors_file": str(args.anchors_file),
            "n_anchors_yielded": len(anchors),
            "sanitize_applied": False,
            "rtt_filter_applied": False,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fold_sizes": fold_sizes,
        "fold_assignments": fold_by_ip,
    }

    out_dir = args.output_root / args.algo
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{param_tag(args)}.json"
    with out_path.open("w") as fh:
        json.dump(out_payload, fh, indent=2, sort_keys=True)
    print(f"wrote stratification → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
