"""Compute + serialize an anchor-level K-fold partition.

Standalone tool — does NOT depend on ClickHouse. Loads the anchor metadata
directly from `reproducibility_anchors.json` (723 anchors — the IMC 2023
paper's canonical count) and computes a fold assignment with the chosen
holdout policy.

The benchmark (`scripts/benchmark/v2/sources/ripe_atlas.py`) computes its
own partition internally over the post-sanitization active anchor set, so
this output is for inspection / visualization / param sweeps, not for
direct consumption by the benchmark. See `report.md` of the leakage-free
eval protocol task for context.

Usage:
  python -m scripts.processing.ripe_atlas.partition \\
      --policy distgeo --k 5 --seed 42 --asn-bucket-top-n 20

  python -m scripts.processing.ripe_atlas.partition \\
      --policy sechidis --k 5 --seed 42 --spatial-clusters 30

Output: `datasets/ripe_atlas/<policy>/<param-tag>.json` with structure:
  {
    "policy": {...},                 # policy class + params
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

from scripts.processing.ripe_atlas.holdout import (  # noqa: E402
    AnchorInfo,
    DistGeoKFoldPolicy,
    HoldoutPolicy,
)


def load_anchors(anchors_file: Path) -> list[AnchorInfo]:
    """Read `reproducibility_anchors.json` and return the entries as
    AnchorInfo records.

    The anchors-only file (723 entries) is the canonical input — same
    count the IMC 2023 paper reports. Drops entries missing
    `address_v4` or `geometry.coordinates`. No sanitization or RTT
    filtering — that's what `RipeAtlasSource` does. This script captures
    the raw anchor set so partition output is reproducible without a
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


def build_policy(args: argparse.Namespace):
    """Return a HoldoutPolicy or DistGeoKFoldPolicy from CLI args."""
    if args.policy == "sechidis":
        return HoldoutPolicy(
            k=args.k,
            fold_index=0,  # full assignment is fold_index-independent
            seed=args.seed,
            spatial_clusters=args.spatial_clusters,
            asn_bucket_top_n=args.asn_bucket_top_n,
        )
    if args.policy == "distgeo":
        return DistGeoKFoldPolicy(
            k=args.k,
            fold_index=0,
            seed=args.seed,
            asn_bucket_top_n=args.asn_bucket_top_n,
        )
    raise ValueError(f"unknown policy {args.policy!r}")


def param_tag(args: argparse.Namespace) -> str:
    """Filename-safe encoding of the salient policy params."""
    if args.policy == "sechidis":
        spatial = "none" if args.spatial_clusters is None else str(args.spatial_clusters)
        return f"k{args.k}_seed{args.seed}_spatial{spatial}_top{args.asn_bucket_top_n}"
    if args.policy == "distgeo":
        return f"k{args.k}_seed{args.seed}_top{args.asn_bucket_top_n}"
    raise ValueError(f"unknown policy {args.policy!r}")


def policy_to_dict(policy) -> dict[str, Any]:
    """Serialize a frozen-dataclass policy to plain JSON-able dict.
    `slice_suffix_fmt` is omitted — it's a serialization knob, not a
    decision-affecting param."""
    d = asdict(policy)
    d.pop("slice_suffix_fmt", None)
    d.pop("fold_index", None)  # not part of partition identity
    return d


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute + save an anchor-level K-fold partition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--policy", choices=["sechidis", "distgeo"], required=True,
        help="which holdout policy to apply",
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
        default=Path("datasets/ripe_atlas"),
        help="root dir; final path is <root>/<policy>/<param-tag>.json",
    )
    args = parser.parse_args()

    if args.spatial_none:
        args.spatial_clusters = None

    print(f"loading anchors from {args.anchors_file} ...")
    anchors = load_anchors(args.anchors_file)
    print(f"  {len(anchors)} anchors with v4 IP + geometry")

    policy = build_policy(args)
    print(f"applying {policy.__class__.__name__}({policy_to_dict(policy)}) ...")
    fold_by_ip = policy.compute_fold_assignments(anchors)
    fold_sizes = [
        sum(1 for f in fold_by_ip.values() if f == i) for i in range(args.k)
    ]
    print(f"  fold sizes: {fold_sizes}")

    out_payload = {
        "policy": {
            "class": policy.__class__.__name__,
            **policy_to_dict(policy),
        },
        "corpus": {
            "source": "ripe_atlas",
            "slice": "all_anchors",
            "anchors_file": str(args.anchors_file),
            "n_anchors_yielded": len(anchors),
            "sanitize_applied": False,
            "rtt_filter_applied": False,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fold_sizes": fold_sizes,
        "fold_assignments": fold_by_ip,
    }

    out_dir = args.output_root / args.policy
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{param_tag(args)}.json"
    with out_path.open("w") as fh:
        json.dump(out_payload, fh, indent=2, sort_keys=True)
    print(f"wrote partition → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
