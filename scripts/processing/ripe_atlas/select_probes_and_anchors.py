"""Select per-ASN VP corpora from the sanitized RIPE Atlas probe + anchor lists.

Six setups (two NA eyeball telcos, two EU eyeball telcos, two global CDN/cloud)
each materialize as a separate probe JSON plus a stats JSON. Anchors are
*shared* across all setups: a single eval corpus that excludes the union of
all setup ASNs, which keeps cross-setup comparisons apples-to-apples and
removes institutional-proximity leakage in one go.

Each setup's probes go through a **city-cell deduplication** step after the
ASN + continent filter: probes binned by 0.1° lat/lon cell (~11 km) are
collapsed to one representative per cell. The winner per cell is the probe
with (a) the most RTT records to the shared anchor set in `ping_10k_to_anchors`,
(b) tiebroken by best (lowest) median RTT, (c) finally by probe id for
determinism. Rationale: extra probes inside one metro contribute redundant
geometry to CBG, but a probe with denser + lower-RTT measurements is a
materially better VP.

Inputs:
  datasets/ripe_atlas/filtered_probes.json   (produced by sanitize_probes.py)
  datasets/ripe_atlas/filtered_anchors.json  (produced by sanitize_anchors.py)
  ClickHouse `ping_10k_to_anchors` table     (for the dedup ranking)

Outputs (under datasets/ripe_atlas/asn_corpora/):
  anchors.json                                # shared eval set
  anchors_stats.json                          # union-exclusion audit trail
  north_america/probes_of_as_7922.json        + _stats.json
  north_america/probes_of_as_7018.json        + _stats.json
  europe/probes_of_as_3209.json               + _stats.json
  europe/probes_of_as_3215.json               + _stats.json
  global/probes_of_as_31898.json              + _stats.json
  global/probes_of_as_16509.json              + _stats.json

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


def select_shared_anchors(setups: list[Setup], anchors: list[dict]) -> tuple[list[dict], dict]:
    """Single anchor corpus shared across all setups.

    Excludes any anchor whose `asn_v4` matches *any* setup's ASN — so the same
    eval set is apples-to-apples across all six deployment scenarios. We pay
    the price of the union of dropped anchors once instead of per-setup, which
    keeps cross-setup comparisons honest and the downstream pipeline simpler
    (one anchor file, not six).
    """
    excluded_asns = {s.asn for s in setups}
    dropped = [a for a in anchors if a.get("asn_v4") in excluded_asns]
    kept = [a for a in anchors if a.get("asn_v4") not in excluded_asns]

    # Trim dropped entries to a small audit record (id + addr + asn + cc).
    dropped_compact = [
        {
            "id": a.get("id"),
            "address_v4": a.get("address_v4"),
            "asn_v4": a.get("asn_v4"),
            "country_code": a.get("country_code"),
        }
        for a in dropped
    ]

    stats = {
        "input_total": len(anchors),
        "excluded_asns": sorted(excluded_asns),
        "excluded_asn_operators": {s.asn: s.operator for s in setups},
        "dropped_count": len(dropped),
        "kept": len(kept),
        "kept_by_continent": _breakdown_by(
            kept, lambda e: continent_of(e.get("country_code"))
        ),
        "kept_by_country": _breakdown_by(
            kept, lambda e: e.get("country_code") or "Unknown"
        ),
        "dropped_anchors": dropped_compact,
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
) -> None:
    """Probe-only per-setup output. Anchors are shared (see select_shared_anchors)."""
    folder = out_root / setup.folder
    folder.mkdir(parents=True, exist_ok=True)

    kept_probes, probe_stats = select_probes(setup, probes)

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
    args = parser.parse_args()

    logger.info("loading probes from %s", args.probes_file)
    probes = _load_json(args.probes_file)
    logger.info("  %d probes", len(probes))

    logger.info("loading anchors from %s", args.anchors_file)
    anchors = _load_json(args.anchors_file)
    logger.info("  %d anchors", len(anchors))

    # One shared anchor eval set for all setups. Dropping the union of all
    # setup ASNs keeps cross-setup comparisons apples-to-apples.
    shared_anchors, anchor_stats = select_shared_anchors(SETUPS, anchors)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(args.output_dir / "anchors.json", shared_anchors)
    _save_json(args.output_dir / "anchors_stats.json", anchor_stats)
    logger.info(
        "shared anchors: %d/%d (excluded ASNs: %s; dropped %d)",
        anchor_stats["kept"], anchor_stats["input_total"],
        anchor_stats["excluded_asns"], anchor_stats["dropped_count"],
    )

    # One ClickHouse query gives us per-probe (n_records, median_rtt) against
    # the shared anchor set; reused across all six setups for the city dedup.
    anchor_ips = [a.get("address_v4") for a in shared_anchors if a.get("address_v4")]
    quality_map = compute_probe_quality(
        anchor_ips,
        table=args.rtt_table,
        max_rtt_ms=args.max_rtt_ms,
    )
    logger.info("  probe quality records: %d src IPs", len(quality_map))

    for setup in SETUPS:
        run_setup(
            setup, probes, args.output_dir,
            quality_map=quality_map,
            grid_deg=args.city_grid_deg,
        )

    logger.info("done. wrote %d probe corpora + shared anchors under %s",
                len(SETUPS), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
