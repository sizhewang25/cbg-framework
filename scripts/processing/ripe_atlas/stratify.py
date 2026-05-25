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

Outputs — two layouts:

  Pipeline / single-config mode (--output-dir <D>):
    <D>/stratification.json
    <D>/kfolds/anchor_fold_<n>.json

  Standalone / multi-config mode (--output-root <R>):
    <R>/<algo>/<param-tag>.json
    <R>/<algo>/<param-tag>/anchor_fold_<n>.json

The pipeline mode is the one wired into `process_probes_and_anchors.smk`
(one canonical stratification colocated with the eval anchor corpus). The
standalone mode lets researchers compare algos/params side-by-side under
`datasets/ripe_atlas/stratifications/`.

Per-fold anchor JSONs carry the full anchor records (one JSON list per fold)
for direct downstream consumption without re-parsing the assignments map.

Stratification JSON structure:
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


def load_anchors(anchors_file: Path) -> tuple[list[AnchorInfo], dict[str, dict]]:
    """Read an anchor JSON file and return (AnchorInfo records, raw entries by IP).

    Works on `reproducibility_anchors.json` (723 entries, IMC 2023 canonical)
    and on the post-sanitize / per-pipeline anchor JSONs (e.g. the 721-anchor
    `asn_corpora/anchors.json`) which share the schema. Drops entries missing
    `address_v4` or `geometry.coordinates`. The raw entry dict is returned
    alongside so callers can write per-fold full-anchor-info files without
    re-parsing the source.
    """
    with anchors_file.open() as fh:
        entries = json.load(fh)

    anchors: list[AnchorInfo] = []
    raw_by_ip: dict[str, dict] = {}
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
        raw_by_ip[ip] = e
    return anchors, raw_by_ip


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
        help="(multi-config mode) root dir; final path is "
             "<root>/<algo>/<param-tag>.json with anchor_fold_<n>.json "
             "files in a sibling <param-tag>/ folder. Ignored if --output-dir is set.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="(single-config mode) flat output dir: writes <dir>/stratification.json "
             "+ <dir>/kfolds/anchor_fold_<n>.json. Overrides --output-root. Use this "
             "when one canonical stratification belongs alongside its corpus.",
    )
    args = parser.parse_args()

    if args.spatial_none:
        args.spatial_clusters = None

    print(f"loading anchors from {args.anchors_file} ...")
    anchors, raw_by_ip = load_anchors(args.anchors_file)
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

    if args.output_dir is not None:
        # Single-config mode: <dir>/stratification.json + <dir>/kfolds/anchor_fold_<n>.json
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "stratification.json"
        fold_dir = args.output_dir / "kfolds"
    else:
        # Multi-config mode: <root>/<algo>/<param-tag>.json + <root>/<algo>/<param-tag>/anchor_fold_<n>.json
        tag = param_tag(args)
        out_root = args.output_root / args.algo
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / f"{tag}.json"
        fold_dir = out_root / tag

    with out_path.open("w") as fh:
        json.dump(out_payload, fh, indent=2, sort_keys=True)
    print(f"wrote stratification → {out_path}")

    # Per-fold full-anchor-info JSONs: one JSON list of raw anchor records per
    # fold, so downstream code can load fold n directly without parsing the
    # assignments map.
    fold_dir.mkdir(parents=True, exist_ok=True)
    for n in range(args.k):
        ips = [ip for ip, f in fold_by_ip.items() if f == n]
        records = [raw_by_ip[ip] for ip in ips if ip in raw_by_ip]
        fold_path = fold_dir / f"anchor_fold_{n}.json"
        with fold_path.open("w") as fh:
            json.dump(records, fh, indent=2, sort_keys=True)
        print(f"  wrote fold {n} ({len(records)} anchors) → {fold_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
