"""Validate that the SOI-violation filter applied by the IMC 2023 pipeline
(materialized at `reproducibility_filtered_probes.json`) matches what the
paper's algorithm would produce against the live ClickHouse tables today.

Replicates exactly the two phases from `datasets/create_datasets.ipynb`:
  Phase 1 — anchors_meshed_pings  → removed_anchors
  Phase 2 — ping_10k_to_anchors   → removed_probes (with removed_anchors filtered out)

Then diffs (removed_anchors ∪ removed_probes) against the static filter file.
Run it with `python -m scripts.benchmark.v2.sources.validate_sanitization`.
"""

from __future__ import annotations

import json
from pathlib import Path

import default
from scripts.analysis.analysis import (
    compute_geo_info,
    compute_remove_wrongly_geolocated_probes,
    compute_rtts_per_dst_src,
)
from scripts.utils.file_utils import load_json


def main() -> None:
    probes_and_anchors = load_json(default.REPRO_PROBES_AND_ANCHORS_FILE)
    (
        vp_coordinates_per_ip,
        _ip_per_coordinates,
        _country_per_vp,
        _asn_per_vp,
        vp_distance_matrix,
        _anchors_per_ip,
    ) = compute_geo_info(probes_and_anchors, default.REPRO_PAIRWISE_DISTANCE_FILE)

    print(f"Loaded {len(vp_coordinates_per_ip)} VP coords; "
          f"distance matrix outer keys: {len(vp_distance_matrix)}")

    # --- Phase 1: anchor-anchor sanitization ----------------------------------
    print("\n[Phase 1] anchors_meshed_pings → removed_anchors")
    rtt_anchors = compute_rtts_per_dst_src(
        default.ANCHORS_MESHED_PING_TABLE, "", threshold=300
    )
    removed_anchors = compute_remove_wrongly_geolocated_probes(
        rtt_anchors, vp_coordinates_per_ip, vp_distance_matrix, set()
    )
    print(f"  removed_anchors: {len(removed_anchors)}")

    # --- Phase 2: probe→anchor sanitization -----------------------------------
    print("\n[Phase 2] ping_10k_to_anchors → removed_probes")
    if removed_anchors:
        in_clause = ",".join(f"toIPv4('{ip}')" for ip in removed_anchors)
        filter_clause = f"AND dst not in ({in_clause}) AND src not in ({in_clause}) "
    else:
        filter_clause = ""
    coords_filtered = {
        ip: c for ip, c in vp_coordinates_per_ip.items() if ip not in removed_anchors
    }
    rtt_probes = compute_rtts_per_dst_src(
        default.PROBES_TO_ANCHORS_PING_TABLE, filter_clause, threshold=300
    )
    removed_probes = compute_remove_wrongly_geolocated_probes(
        rtt_probes, coords_filtered, vp_distance_matrix, removed_anchors
    )
    print(f"  removed_probes (this phase only): {len(removed_probes)}")

    union = set(removed_anchors) | set(removed_probes)
    print(f"\nTotal removed by paper's algorithm: {len(union)}")

    # --- Compare against the static filter file -------------------------------
    static = set(load_json(default.REPRO_FILTERED_PROBES_FILE))
    print(f"Static filter file size:           {len(static)}")

    overlap = union & static
    only_alg = union - static
    only_static = static - union
    print(f"  overlap (in both):               {len(overlap)}")
    print(f"  only in algorithm output:        {len(only_alg)}")
    print(f"  only in static filter:           {len(only_static)}")

    if only_alg:
        print("\nNew flags (algorithm but not in static filter):")
        for ip in sorted(only_alg):
            print(f"  + {ip}")
    if only_static:
        print("\nStatic-only entries (in filter but not re-flagged):")
        for ip in sorted(only_static):
            print(f"  - {ip}")

    out = Path("/tmp/sanitize_validation.json")
    out.write_text(json.dumps({
        "removed_anchors": sorted(removed_anchors),
        "removed_probes": sorted(removed_probes),
        "union": sorted(union),
        "static": sorted(static),
        "only_in_algorithm": sorted(only_alg),
        "only_in_static": sorted(only_static),
    }, indent=2))
    print(f"\nSaved details to {out}")


if __name__ == "__main__":
    main()
