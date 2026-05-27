"""Select per-ASN VP corpora from the sanitized RIPE Atlas probe + anchor lists.

Six setups (two NA eyeball telcos, two EU eyeball telcos, two global CDN/cloud)
each materialize as a separate probe JSON plus a stats JSON. Anchors are
*shared* across all setups: a single eval corpus formed by intersecting the
anchor sets each setup's deduped probes can reach in `ping_10k_to_anchors`
within `--max-rtt-ms` (default 10000 ms), then dropping any anchor whose ASN
is one of the setup ASNs (institutional-proximity guard). This guarantees
every setup has ping data for every kept anchor → apples-to-apples eval and
no per-fold "this anchor isn't reachable from this AS" gaps.

Pipeline per setup:
  filtered_probes.json
    → ASN filter (probe.asn_v4 == setup.asn)
    → continent filter (country code + coord bbox; skipped for global setups)
    → million-scale CBG mislocation filter: drop probes whose speed-of-internet
      (2c/3) disk fails to contain any reachable anchor's true coord. The
      fiber-calibrated disk is physically incompatible with such an exclusion
      under correct metadata. Disable with `--no-cbg-sanitize`.
    → city-cell dedup (0.1° bins → one probe per cell, ranked by RTT-record
      count then median RTT in `ping_10k_to_anchors`)
    → probes_of_as_<asn>.json

Shared anchors:
  filtered_anchors.json
    → drop anchors in any setup ASN
    → keep only anchors reached by *every* setup's deduped probes at
      `min` ∈ (0, --max-rtt-ms) in ping_10k_to_anchors
    → anchors.json

Inputs:
  datasets/ripe_atlas/filtered_probes.json   (produced by sanitize_probes.py)
  datasets/ripe_atlas/filtered_anchors.json  (produced by sanitize_anchors.py)
  ClickHouse `ping_10k_to_anchors` table     (for dedup ranking + reachability)

Outputs (under datasets/ripe_atlas/asn_corpora/):
  anchors/anchors.json                          # shared eval set (intersection)
  anchors/anchors_stats.json                    # per-setup reach + dropped audit
  probes/north_america/probes_of_as_7922.json   + _stats.json
  probes/north_america/probes_of_as_7018.json   + _stats.json
  probes/europe/probes_of_as_3209.json          + _stats.json
  probes/europe/probes_of_as_3215.json          + _stats.json
  probes/global/probes_of_as_31898.json         + _stats.json
  probes/global/probes_of_as_16509.json         + _stats.json

Usage:
  python -m scripts.processing.ripe_atlas.select_probes_and_anchors
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Make `default` importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.processing.ripe_atlas.continents import (  # noqa: E402
    continent_bbox_contains,
    continent_of,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Setup:
    """One per-ASN deployment scenario."""

    asn: int
    operator: str
    setup_continent: str
    probe_continent_filter: Optional[str]  # None = global, no filter
    folder: str  # subfolder under the output dir


SETUPS: list[Setup] = [
    Setup(7922,  "Comcast",      "North America", "North America", "north_america"),
    Setup(7018,  "AT&T",         "North America", "North America", "north_america"),
    Setup(3209,  "Vodafone DE",  "Europe",        "Europe",        "europe"),
    Setup(3215,  "Orange FR",    "Europe",        "Europe",        "europe"),
    Setup(31898, "Oracle Cloud", "Global",        None,            "global"),
    Setup(16509, "Amazon AWS",   "Global",        None,            "global"),
]


def _load_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2)
    tmp.replace(path)


def _has_coords(entry: dict) -> bool:
    geom = (entry.get("geometry") or {}).get("coordinates")
    return bool(geom) and len(geom) >= 2


def _breakdown_by(entries: list[dict], key_fn) -> dict[str, int]:
    """Sorted-by-count descending dict for stats display."""
    counts: dict[str, int] = {}
    for e in entries:
        k = key_fn(e)
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def select_probes(setup: Setup, probes: list[dict]) -> tuple[list[dict], dict]:
    """Filter probes to those in `setup.asn`, then optionally by continent.

    Returns (kept_probes, stats_dict).
    """
    input_total = len(probes)
    matched_asn = [p for p in probes if p.get("asn_v4") == setup.asn]

    dropped_no_country = 0
    dropped_other_continent = 0
    dropped_coords_outside_continent = 0
    dropped_no_coords = 0
    kept: list[dict] = []

    for p in matched_asn:
        if not _has_coords(p):
            dropped_no_coords += 1
            continue
        cc = p.get("country_code")
        if setup.probe_continent_filter is not None:
            if not cc:
                dropped_no_country += 1
                continue
            if continent_of(cc) != setup.probe_continent_filter:
                dropped_other_continent += 1
                continue
            # country_code passed, but the parent-country ISO can mask an
            # overseas territory on a different continent (e.g. cc=FR for
            # Guadeloupe). Sanity-check the coords against the continent's
            # bounding box.
            geom = (p.get("geometry") or {}).get("coordinates")
            lon, lat = float(geom[0]), float(geom[1])
            if not continent_bbox_contains(setup.probe_continent_filter, lat, lon):
                dropped_coords_outside_continent += 1
                continue
        kept.append(p)

    stats = {
        "asn": setup.asn,
        "operator": setup.operator,
        "setup_continent": setup.setup_continent,
        "continent_filter": setup.probe_continent_filter,
        "input_total": input_total,
        "matched_asn": len(matched_asn),
        "kept": len(kept),
        "dropped_other_continent": dropped_other_continent,
        "dropped_coords_outside_continent": dropped_coords_outside_continent,
        "dropped_no_country_code": dropped_no_country,
        "dropped_no_coords": dropped_no_coords,
        "kept_by_country": _breakdown_by(kept, lambda e: e.get("country_code") or "Unknown"),
        "kept_by_continent": _breakdown_by(
            kept, lambda e: continent_of(e.get("country_code"))
        ),
    }
    return kept, stats


def drop_mislocated_probes(
    setup: Setup,
    kept_probes: list[dict],
    anchors: list[dict],
    *,
    table: str = "ping_10k_to_anchors",
    max_rtt_ms: float = 10000.0,
    speed_ratio: float = 2.0 / 3.0,
) -> tuple[list[dict], list[dict], dict]:
    """Drop probes whose declared coords are physically incompatible with their
    measured RTTs to anchors with known coords.

    Million-scale CBG (`speed_of_internet` LTD + `spherical_circle` MTL) over
    every reachable (probe, anchor) pair. The disk radius `(RTT/2) · (2c/3)` is
    fiber-calibrated: on a real terrestrial fiber path, `RTT ≥ 3d/c` ⇒ disk ≥
    geographic distance. So a correctly-located probe cannot have its disk
    exclude an anchor's true coord. Any (probe, anchor) pair where the
    declared probe→anchor distance exceeds the disk radius implies either bad
    probe metadata or a faster-than-fiber path (rare, mostly microwave). A
    probe with even one such exclusion is dropped.

    Mathematically equivalent to "run the spherical-circle MTL per anchor,
    collect probes that fail to bracket truth on the FALLBACK anchors": the
    MTL region is empty iff ≥1 probe excludes truth, and the per-pair check
    tags exactly those probes. Skipping the MTL stage saves O(N²) work per
    anchor without changing the result.

    Same-ASN anchors are excluded to avoid the institutional-proximity
    artifact (near-0 RTT → near-0 disk, trivially incompatible).

    Returns `(kept_filtered, dropped_records, stats)`.
    """
    from scripts.libs.cbg.rtt_model import haversine_distance
    from scripts.utils.clickhouse import Clickhouse

    # km of light travel per ms (2.998e5 km/s × 1e-3 s/ms).
    SPEED_OF_LIGHT_KM_PER_MS = 299.792458

    probe_coords: dict[str, tuple[float, float]] = {}
    src_ints: list[int] = []
    for p in kept_probes:
        ip = p.get("address_v4")
        geom = (p.get("geometry") or {}).get("coordinates")
        if not ip or not geom or len(geom) < 2:
            continue
        probe_coords[ip] = (float(geom[1]), float(geom[0]))  # (lat, lon)
        src_ints.append(int(ipaddress.IPv4Address(ip)))

    candidate_anchors = [
        a for a in anchors
        if a.get("address_v4") and a.get("asn_v4") != setup.asn
    ]
    anchor_coords: dict[str, tuple[float, float]] = {}
    dst_ints: list[int] = []
    for a in candidate_anchors:
        ip = a["address_v4"]
        geom = (a.get("geometry") or {}).get("coordinates")
        if not geom or len(geom) < 2:
            continue
        anchor_coords[ip] = (float(geom[1]), float(geom[0]))
        dst_ints.append(int(ipaddress.IPv4Address(ip)))

    if not src_ints or not dst_ints:
        # Nothing to evaluate against — pass through unchanged.
        return kept_probes, [], {
            "speed_ratio": speed_ratio,
            "rtt_table": table,
            "max_rtt_ms": max_rtt_ms,
            "n_probes_input": len(kept_probes),
            "n_anchors_considered": len(anchor_coords),
            "n_probe_anchor_pairs_observed": 0,
            "n_probe_anchor_pairs_excluding_truth": 0,
            "n_dropped_mislocated_probes": 0,
            "n_kept": len(kept_probes),
            "dropped_mislocated_probes": [],
            "note": "no probes or no eligible anchors with coords; filter skipped",
        }

    ch = Clickhouse()
    query = f"""
        SELECT IPv4NumToString(src) AS src_ip,
               IPv4NumToString(dst) AS dst_ip,
               min(`min`) AS rtt_ms
        FROM {ch.database}.{table}
        WHERE src IN ({','.join(map(str, src_ints))})
          AND dst IN ({','.join(map(str, dst_ints))})
          AND `min` > 0 AND `min` < {max_rtt_ms}
        GROUP BY src, dst
    """
    logger.info(
        "AS%d (%s): million-scale CBG sanitization — query %d probes × %d anchors from %s.%s",
        setup.asn, setup.operator, len(src_ints), len(dst_ints),
        ch.database, table,
    )
    rows = list(ch.client.execute_iter(query))
    ch.client.disconnect()

    mislocated_ips: dict[str, dict] = {}  # ip → {worst-case evidence}
    n_pairs_observed = 0
    n_pairs_excluding = 0
    for src_ip, dst_ip, rtt_ms in rows:
        pc = probe_coords.get(src_ip)
        ac = anchor_coords.get(dst_ip)
        if pc is None or ac is None:
            continue
        n_pairs_observed += 1
        rtt_ms_f = float(rtt_ms)
        disk_km = (rtt_ms_f / 2.0) * speed_ratio * SPEED_OF_LIGHT_KM_PER_MS
        dist_km = float(haversine_distance(pc[0], pc[1], ac[0], ac[1]))
        if dist_km > disk_km:
            n_pairs_excluding += 1
            gap_km = dist_km - disk_km
            prev = mislocated_ips.get(src_ip)
            if prev is None or gap_km > prev["worst_gap_km"]:
                mislocated_ips[src_ip] = {
                    "worst_gap_km": gap_km,
                    "worst_anchor_ip": dst_ip,
                    "worst_anchor_distance_km": dist_km,
                    "worst_disk_radius_km": disk_km,
                    "worst_rtt_ms": rtt_ms_f,
                }
            # accumulate counter
            mislocated_ips[src_ip].setdefault("n_excluding_pairs", 0)
            mislocated_ips[src_ip]["n_excluding_pairs"] += 1

    dropped = [p for p in kept_probes if p.get("address_v4") in mislocated_ips]
    kept_filtered = [
        p for p in kept_probes if p.get("address_v4") not in mislocated_ips
    ]

    dropped_compact = []
    for p in dropped:
        ip = p["address_v4"]
        ev = mislocated_ips[ip]
        geom = (p.get("geometry") or {}).get("coordinates") or [None, None]
        dropped_compact.append({
            "id": p.get("id"),
            "address_v4": ip,
            "country_code": p.get("country_code"),
            "declared_lat": float(geom[1]) if geom[1] is not None else None,
            "declared_lon": float(geom[0]) if geom[0] is not None else None,
            "n_excluding_pairs": ev["n_excluding_pairs"],
            "worst_anchor_ip": ev["worst_anchor_ip"],
            "worst_gap_km": round(ev["worst_gap_km"], 1),
            "worst_anchor_distance_km": round(ev["worst_anchor_distance_km"], 1),
            "worst_disk_radius_km": round(ev["worst_disk_radius_km"], 1),
            "worst_rtt_ms": round(ev["worst_rtt_ms"], 3),
        })
    # Sort the compact list by worst_gap_km descending — biggest metadata
    # errors first.
    dropped_compact.sort(key=lambda r: -r["worst_gap_km"])

    stats = {
        "speed_ratio": speed_ratio,
        "rtt_table": table,
        "max_rtt_ms": max_rtt_ms,
        "n_probes_input": len(kept_probes),
        "n_anchors_considered": len(anchor_coords),
        "n_probe_anchor_pairs_observed": n_pairs_observed,
        "n_probe_anchor_pairs_excluding_truth": n_pairs_excluding,
        "n_dropped_mislocated_probes": len(dropped),
        "n_kept": len(kept_filtered),
        "dropped_mislocated_probes": dropped_compact,
    }
    logger.info(
        "AS%d (%s): mislocation filter dropped %d / %d probes "
        "(%d / %d observed pairs excluded truth)",
        setup.asn, setup.operator,
        len(dropped), len(kept_probes),
        n_pairs_excluding, n_pairs_observed,
    )
    return kept_filtered, dropped, stats


def select_common_anchors(
    setups: list[Setup],
    setup_probe_ips: dict[int, list[str]],
    anchors: list[dict],
    *,
    table: str = "ping_10k_to_anchors",
    max_rtt_ms: float = 10000.0,
) -> tuple[list[dict], dict]:
    """Shared anchor corpus = ∩(anchors reachable from each setup's probes).

    Two filters compose:
      1. **Same-ASN exclusion** (institutional-proximity guard): drop any
         anchor whose `asn_v4` is one of the setup ASNs. Otherwise the setup
         that owns the anchor's host network would see ~0ms RTT and an
         unfairly tight CBG constraint.
      2. **Reachability intersection**: of the remaining candidates, keep
         only those reached by *every* setup's deduped probes with
         `min ∈ (0, max_rtt_ms)` in `table`. Guarantees the per-setup
         K-fold eval will have observations for every kept anchor.

    One ClickHouse round-trip pulls all (src, dst) pairs across all setups'
    probes; per-setup sets are formed in Python and intersected.
    """
    from scripts.utils.clickhouse import Clickhouse

    excluded_asns = {s.asn for s in setups}
    dropped_same_asn = [a for a in anchors if a.get("asn_v4") in excluded_asns]
    candidate_anchors = [a for a in anchors if a.get("asn_v4") not in excluded_asns]
    candidate_ips = [
        a["address_v4"] for a in candidate_anchors if a.get("address_v4")
    ]

    # Map probe_ip → setup ASN so we can split CH rows back into per-setup sets.
    probe_to_setup_asn: dict[str, int] = {}
    for asn, ips in setup_probe_ips.items():
        for ip in ips:
            probe_to_setup_asn[ip] = asn
    all_probe_ips = sorted(probe_to_setup_asn)

    src_ints = [int(ipaddress.IPv4Address(ip)) for ip in all_probe_ips]
    dst_ints = [int(ipaddress.IPv4Address(ip)) for ip in candidate_ips]
    if not src_ints or not dst_ints:
        raise RuntimeError(
            "select_common_anchors: empty probe or candidate set "
            f"(probes={len(src_ints)}, anchors={len(dst_ints)})"
        )

    ch = Clickhouse()
    query = f"""
        SELECT IPv4NumToString(src) AS src_ip,
               IPv4NumToString(dst) AS dst_ip
        FROM {ch.database}.{table}
        WHERE src IN ({','.join(map(str, src_ints))})
          AND dst IN ({','.join(map(str, dst_ints))})
          AND `min` > 0 AND `min` < {max_rtt_ms}
        GROUP BY src, dst
    """
    logger.info(
        "query reachable (src, dst) pairs from %s.%s: %d probes × %d candidate anchors at `min` < %.0f ms",
        ch.database, table, len(src_ints), len(dst_ints), max_rtt_ms,
    )
    pairs = list(ch.client.execute_iter(query))
    ch.client.disconnect()
    logger.info("  pulled %d distinct (probe, anchor) pairs", len(pairs))

    reached_per_setup: dict[int, set[str]] = {asn: set() for asn in setup_probe_ips}
    for src_ip, dst_ip in pairs:
        asn = probe_to_setup_asn.get(src_ip)
        if asn is not None:
            reached_per_setup[asn].add(dst_ip)

    # Intersection across all setups.
    common_ips: set[str] = (
        set.intersection(*reached_per_setup.values())
        if reached_per_setup else set()
    )
    kept = [a for a in candidate_anchors if a.get("address_v4") in common_ips]

    dropped_unreachable_count = len(candidate_anchors) - len(kept)

    dropped_same_asn_compact = [
        {
            "id": a.get("id"),
            "address_v4": a.get("address_v4"),
            "asn_v4": a.get("asn_v4"),
            "country_code": a.get("country_code"),
        }
        for a in dropped_same_asn
    ]

    stats = {
        "input_total": len(anchors),
        "excluded_asns": sorted(excluded_asns),
        "excluded_asn_operators": {s.asn: s.operator for s in setups},
        "dropped_same_asn_count": len(dropped_same_asn),
        "rtt_table": table,
        "max_rtt_ms": max_rtt_ms,
        "candidate_count": len(candidate_anchors),
        "reached_per_setup": {
            asn: len(reached_per_setup[asn]) for asn in sorted(reached_per_setup)
        },
        "dropped_unreachable_count": dropped_unreachable_count,
        "kept": len(kept),
        "kept_by_continent": _breakdown_by(
            kept, lambda e: continent_of(e.get("country_code"))
        ),
        "kept_by_country": _breakdown_by(
            kept, lambda e: e.get("country_code") or "Unknown"
        ),
        "dropped_same_asn": dropped_same_asn_compact,
    }
    return kept, stats


def compute_probe_quality(
    anchor_ips: list[str],
    table: str = "ping_10k_to_anchors",
    max_rtt_ms: float = 10000.0,
) -> dict[str, tuple[int, float]]:
    """Per-probe (n_records, median_rtt_ms) aggregated from `table`.

    Counts only rows where `dst` is one of the shared eval anchors and the
    min RTT is in (0, `max_rtt_ms`). Returns `{src_ip: (n_records, median_rtt)}`;
    probes absent from the result had zero qualifying records.

    Lazy-imports the Clickhouse client so the rest of this module stays
    DB-free for the `--no-city-dedup` path.
    """
    from scripts.utils.clickhouse import Clickhouse

    dst_ints = sorted({
        int(ipaddress.IPv4Address(ip)) for ip in anchor_ips if ip
    })
    if not dst_ints:
        return {}

    ch = Clickhouse()
    query = f"""
        SELECT IPv4NumToString(src) AS src_ip,
               count() AS n_records,
               quantile(0.5)(`min`) AS median_rtt
        FROM {ch.database}.{table}
        WHERE dst IN ({','.join(map(str, dst_ints))})
          AND `min` > 0 AND `min` < {max_rtt_ms}
        GROUP BY src
    """
    logger.info(
        "query probe quality from %s.%s against %d shared-eval anchors",
        ch.database, table, len(dst_ints),
    )
    rows = list(ch.client.execute_iter(query))
    ch.client.disconnect()
    return {ip: (int(n), float(med)) for ip, n, med in rows}


def _city_cell(lat: float, lon: float, grid_deg: float) -> tuple[float, float]:
    """Snap `(lat, lon)` to the center of its `grid_deg`° cell."""
    return (
        round(lat / grid_deg) * grid_deg,
        round(lon / grid_deg) * grid_deg,
    )


def city_dedup(
    kept_probes: list[dict],
    quality_map: dict[str, tuple[int, float]],
    grid_deg: float,
) -> tuple[list[dict], dict]:
    """Collapse probes sharing a `grid_deg`° lat/lon cell to one representative.

    Ranking per cell: highest `n_records`, tiebroken by lowest `median_rtt`,
    finally by probe `id` (deterministic). Probes without any qualifying
    records sort last (their median is set to +inf) but can still win a cell
    where every candidate is recordless.

    Returns `(dedup_kept, extra_stats)` where extra_stats is a dict to merge
    into the per-setup stats record.
    """
    # Group probes by city cell.
    cells: dict[tuple[float, float], list[dict]] = {}
    no_records = 0
    for p in kept_probes:
        geom = (p.get("geometry") or {}).get("coordinates")
        if not geom or len(geom) < 2:
            continue
        lat, lon = float(geom[1]), float(geom[0])
        cells.setdefault(_city_cell(lat, lon, grid_deg), []).append(p)

    def sort_key(p: dict) -> tuple[int, float, int]:
        ip = p.get("address_v4")
        nrec, med = quality_map.get(ip, (0, math.inf))
        # Negate n_records so larger counts sort *earlier* under the default
        # ascending tuple comparison.
        return (-nrec, med, p.get("id") or 0)

    dedup: list[dict] = []
    for cell, candidates in cells.items():
        candidates.sort(key=sort_key)
        winner = candidates[0]
        ip = winner.get("address_v4")
        nrec, _ = quality_map.get(ip, (0, math.inf))
        if nrec == 0:
            no_records += 1
        dedup.append(winner)

    extra = {
        "city_dedup": True,
        "city_grid_deg": grid_deg,
        "pre_dedup_kept": len(kept_probes),
        "post_dedup_kept": len(dedup),
        "dedup_dropped": len(kept_probes) - len(dedup),
        "n_city_cells": len(cells),
        "kept_without_rtt_records": no_records,
    }
    return dedup, extra


def run_setup(
    setup: Setup,
    probes: list[dict],
    out_root: Path,
    quality_map: Optional[dict[str, tuple[int, float]]] = None,
    grid_deg: float = 0.1,
    sanitize_anchors: Optional[list[dict]] = None,
    sanitize_table: str = "ping_10k_to_anchors",
    sanitize_max_rtt_ms: float = 10000.0,
) -> list[dict]:
    """Probe-only per-setup output. Anchors are shared (see select_common_anchors).

    When `sanitize_anchors` is provided, runs the million-scale CBG
    mislocation filter immediately after the metadata (ASN + continent)
    filter, before city dedup. This drops probes whose declared coords
    are physically incompatible with their measured RTTs to any reachable
    anchor; see `drop_mislocated_probes` for the physics. Filtering before
    dedup avoids the case where a mislocated probe wins its city cell on
    record-count and then survives downstream.

    Returns the city-deduped probe list so the caller can compute the
    cross-setup reachability intersection for the shared anchor corpus.
    """
    folder = out_root / "probes" / setup.folder
    folder.mkdir(parents=True, exist_ok=True)

    kept_probes, probe_stats = select_probes(setup, probes)

    if sanitize_anchors is not None:
        kept_probes, _dropped, cbg_stats = drop_mislocated_probes(
            setup,
            kept_probes,
            sanitize_anchors,
            table=sanitize_table,
            max_rtt_ms=sanitize_max_rtt_ms,
        )
        probe_stats["cbg_mislocation_filter"] = cbg_stats
        probe_stats["kept_after_cbg_filter"] = len(kept_probes)

    if quality_map is not None:
        kept_probes, dedup_extra = city_dedup(kept_probes, quality_map, grid_deg)
        probe_stats.update(dedup_extra)
        # Recompute the country / continent rollups on the post-dedup set so
        # the stats file reflects what was actually written.
        probe_stats["kept"] = len(kept_probes)
        probe_stats["kept_by_country"] = _breakdown_by(
            kept_probes, lambda e: e.get("country_code") or "Unknown"
        )
        probe_stats["kept_by_continent"] = _breakdown_by(
            kept_probes, lambda e: continent_of(e.get("country_code"))
        )

    probe_path = folder / f"probes_of_as_{setup.asn}.json"
    probe_stats_path = folder / f"probes_of_as_{setup.asn}_stats.json"
    _save_json(probe_path, kept_probes)
    _save_json(probe_stats_path, probe_stats)

    if quality_map is not None:
        logger.info(
            "AS%d (%s, %s): probes %d→%d after city-dedup over %d cells "
            "(matched %d, recordless %d)",
            setup.asn, setup.operator, setup.setup_continent,
            probe_stats["pre_dedup_kept"], probe_stats["post_dedup_kept"],
            probe_stats["n_city_cells"], probe_stats["matched_asn"],
            probe_stats["kept_without_rtt_records"],
        )
    else:
        logger.info(
            "AS%d (%s, %s): probes %d/%d (matched %d)",
            setup.asn, setup.operator, setup.setup_continent,
            probe_stats["kept"], probe_stats["input_total"], probe_stats["matched_asn"],
        )
    return kept_probes


def main() -> int:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Select per-ASN VP corpora for the deployment-scenario benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--probes-file", type=Path,
        default=Path("datasets/ripe_atlas/filtered_probes.json"),
        help="sanitized probes JSON (from sanitize_probes.py)",
    )
    parser.add_argument(
        "--anchors-file", type=Path,
        default=Path("datasets/ripe_atlas/filtered_anchors.json"),
        help="sanitized anchors JSON (from sanitize_anchors.py)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("datasets/ripe_atlas/asn_corpora"),
        help="root output directory; per-continent subfolders created beneath",
    )
    parser.add_argument(
        "--city-grid-deg", type=float, default=0.1,
        help="city-cell side length in degrees (~11 km at the equator)",
    )
    parser.add_argument(
        "--rtt-table", default="ping_10k_to_anchors",
        help="ClickHouse table queried for the dedup ranking",
    )
    parser.add_argument(
        "--max-rtt-ms", type=float, default=10000.0,
        help="upper bound on `min` RTT counted as a valid record",
    )
    parser.add_argument(
        "--cbg-sanitize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop probes whose speed-of-internet (2c/3) disk excludes any "
             "reachable anchor's true coord (post-continent filter, pre-dedup). "
             "Use --no-cbg-sanitize to skip.",
    )
    args = parser.parse_args()

    logger.info("loading probes from %s", args.probes_file)
    probes = _load_json(args.probes_file)
    logger.info("  %d probes", len(probes))

    logger.info("loading anchors from %s", args.anchors_file)
    anchors = _load_json(args.anchors_file)
    logger.info("  %d anchors", len(anchors))

    # Probe quality (n_records, median_rtt) is computed against *all*
    # sanitized anchor IPs — independent of the shared anchor set, since the
    # shared set is itself derived from the deduped probes (circular if we
    # used it here). The dedup ranking only cares about which probes are
    # active in ping_10k_to_anchors, so using the broader anchor pool is fine.
    all_anchor_ips = [a.get("address_v4") for a in anchors if a.get("address_v4")]
    quality_map = compute_probe_quality(
        all_anchor_ips,
        table=args.rtt_table,
        max_rtt_ms=args.max_rtt_ms,
    )
    logger.info("  probe quality records: %d src IPs", len(quality_map))

    # Per-setup probe selection + city dedup. Collect deduped probe IPs per
    # setup so we can intersect their reachable-anchor sets next.
    setup_probe_ips: dict[int, list[str]] = {}
    for setup in SETUPS:
        kept_probes = run_setup(
            setup, probes, args.output_dir,
            quality_map=quality_map,
            grid_deg=args.city_grid_deg,
            sanitize_anchors=anchors if args.cbg_sanitize else None,
            sanitize_table=args.rtt_table,
            sanitize_max_rtt_ms=args.max_rtt_ms,
        )
        setup_probe_ips[setup.asn] = [
            p["address_v4"] for p in kept_probes if p.get("address_v4")
        ]

    # Shared anchor corpus = ∩(reachable anchors per setup) with same-ASN guard.
    common_anchors, anchor_stats = select_common_anchors(
        SETUPS, setup_probe_ips, anchors,
        table=args.rtt_table,
        max_rtt_ms=args.max_rtt_ms,
    )
    anchors_dir = args.output_dir / "anchors"
    anchors_dir.mkdir(parents=True, exist_ok=True)
    _save_json(anchors_dir / "anchors.json", common_anchors)
    _save_json(anchors_dir / "anchors_stats.json", anchor_stats)
    logger.info(
        "shared anchors: kept=%d / %d candidates "
        "(dropped %d same-ASN, %d unreachable at `min` < %.0f ms)",
        anchor_stats["kept"], anchor_stats["candidate_count"],
        anchor_stats["dropped_same_asn_count"],
        anchor_stats["dropped_unreachable_count"],
        anchor_stats["max_rtt_ms"],
    )
    logger.info(
        "  per-setup reachable counts (pre-intersection): %s",
        anchor_stats["reached_per_setup"],
    )

    logger.info("done. wrote %d probe corpora + shared anchors under %s",
                len(SETUPS), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
