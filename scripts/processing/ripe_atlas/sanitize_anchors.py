"""Run SOI (speed-of-Internet) sanitization on RIPE Atlas anchors via ClickHouse.

Anchor-mesh only — does *not* run the probe→anchor phase from
`create_datasets.ipynb` cell 32 / `_compute_soi_removed_ips` phase 2.
Rationale: phase 2 greedily removes the IP with the most SOI violations,
which can blame the anchor when the *probe* is the wrongly-geolocated
side. For anchor-only sanitization we want both endpoints of every
flagged pair to be anchors with curated GT — which means restricting to
the anchor-mesh table.

  Query: `anchors_meshed_pings` (threshold=300). Iteratively remove the
         anchor with the most SOI violations until none remain.

Output: filtered anchors JSON (subset of input with SOI-violators removed)
plus a side-by-side list of removed anchor IPs for audit.

Requires ClickHouse reachable per CLICKHOUSE_HOST / CLICKHOUSE_PASSWORD from
`.env`. Sister script to `stratify.py`, which then consumes the filtered
file (no DB needed at stratification time).

Usage:
  python -m scripts.processing.ripe_atlas.sanitize_anchors
  python -m scripts.processing.ripe_atlas.sanitize_anchors --threshold 300 \\
      --output datasets/ripe_atlas/filtered_anchors.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make `default` importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import default  # noqa: E402
from scripts.analysis.analysis import (  # noqa: E402
    compute_remove_wrongly_geolocated_probes,
    compute_rtts_per_dst_src,
)
from scripts.utils.helpers import haversine  # noqa: E402

logger = logging.getLogger(__name__)


def load_coords(path: Path) -> dict[str, tuple[float, float]]:
    """Read a probes / anchors JSON, return `{ip: (lat, lon)}` for v4 entries.

    Drops entries missing `address_v4` or coordinates. Does not look at
    `is_anchor` — caller decides which file to feed in."""
    with path.open() as fh:
        entries = json.load(fh)
    coords: dict[str, tuple[float, float]] = {}
    for e in entries:
        ip = e.get("address_v4")
        geom = (e.get("geometry") or {}).get("coordinates")
        if not ip or not geom or len(geom) < 2:
            continue
        lon, lat = geom[0], geom[1]
        coords[ip] = (float(lat), float(lon))
    return coords


def add_pairwise_dist(
    rtt_data: dict[str, dict[str, list[float]]],
    coords: dict[str, tuple[float, float]],
    dist: dict[str, dict[str, float]],
) -> None:
    """Populate `dist` in place: for every (dst, src) pair appearing in
    `rtt_data`, compute the great-circle km via haversine. Skips pairs
    missing coords. Used by `compute_remove_wrongly_geolocated_probes`."""
    for dst, srcs in rtt_data.items():
        if dst not in coords:
            continue
        row = dist.setdefault(dst, {})
        for src in srcs.keys():
            if src in row or src not in coords or src == dst:
                continue
            row[src] = float(haversine(coords[dst], coords[src]))


def run_sanitization(
    anchors_file: Path,
    anchor_mesh_table: str,
    threshold: int,
) -> tuple[set[str], list[dict]]:
    """Anchor-mesh SOI removal. Returns `(removed_ips, kept_anchors)`.

    Greedy: iteratively drops the anchor with the most SOI violations
    until none remain. Both endpoints of every queried pair are anchors,
    so a flagged IP must be a wrongly-geolocated anchor (no probe-side
    ambiguity).
    """
    anchor_coords = load_coords(anchors_file)
    print(f"  loaded {len(anchor_coords)} anchor coords")

    print(f"query {anchor_mesh_table} (threshold={threshold}) ...")
    rtt_anchors = compute_rtts_per_dst_src(
        anchor_mesh_table, "", threshold, is_per_prefix=False,
    )
    dist: dict[str, dict[str, float]] = {}
    add_pairwise_dist(rtt_anchors, anchor_coords, dist)
    removed = compute_remove_wrongly_geolocated_probes(
        rtt_anchors, anchor_coords, dist, set(),
    )
    print(f"  flagged {len(removed)} anchor IPs as SOI-violators")

    with anchors_file.open() as fh:
        entries = json.load(fh)
    kept = [e for e in entries if e.get("address_v4") not in removed]
    print(
        f"input anchors: {len(entries)} → kept: {len(kept)} "
        f"(removed: {len(entries) - len(kept)})"
    )
    return removed, kept


def main() -> int:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Drop SOI-violating RIPE Atlas anchors via ClickHouse.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--anchors-file", type=Path,
        default=Path(default.REPRO_ANCHORS_FILE),
        help="input anchors JSON (canonical 723-anchor file)",
    )
    parser.add_argument(
        "--anchor-mesh-table", default=default.ANCHORS_MESHED_PING_TABLE,
        help="ClickHouse anchor-mesh ping table",
    )
    parser.add_argument(
        "--threshold", type=int, default=300,
        help="SOI RTT threshold (matches create_datasets.ipynb cell 29)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("datasets/ripe_atlas/filtered_anchors.json"),
        help="output filtered anchors JSON",
    )
    args = parser.parse_args()

    print(f"loading anchor coords from {args.anchors_file}")
    removed, kept = run_sanitization(
        args.anchors_file,
        args.anchor_mesh_table,
        args.threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(kept, fh, indent=2)
    print(f"wrote {len(kept)} anchors → {args.output}")

    # Side-by-side audit trail: which anchor IPs got removed.
    removed_path = args.output.parent / "removed_anchor_ips.json"
    with removed_path.open("w") as fh:
        json.dump(sorted(removed), fh, indent=2)
    print(f"wrote {len(removed)} removed anchor IPs → {removed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
