"""RIPE Atlas dataset characterization — emit stats JSON.

Loads the IMC 2023 RIPE Atlas data via `RipeAtlasSource` (default
sanitize=True, PROBES_TO_ANCHORS setup) and writes a JSON snapshot of the
dataset statistics documented in
`scripts/benchmark/v2/sources/RIPE_ATLAS_DATA.md`:

  - raw and eval counts (probes + anchors)
  - unique countries / ASNs (raw + eval)
  - top-10 countries / ASNs (eval)
  - country / ASN overlap (eval)
  - paper-claim cross-check (IMC 2023)

Reaches out to ClickHouse on first iter (same as the benchmark). Run from
the repo root:

    python -m scripts.analysis.characterize_ripe_atlas \\
        --output scripts/analysis/outputs/ripe_atlas_characterization.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterable

from scripts.benchmark.v2.sources.base import DataSource
from scripts.benchmark.v2.sources.ripe_atlas import RipeAtlasSource

logger = logging.getLogger(__name__)


def _top_n(counts: Counter, n: int = 10) -> list[dict]:
    return [{"key": k, "count": v} for k, v in counts.most_common(n)]


def _city_clusters(coords, ips: Iterable[str], grid: float = 0.1) -> int:
    """Approximate 'cities' as unique lat/lon grid cells (~11 km @ 0.1°)."""
    cells: set[tuple[int, int]] = set()
    for ip in ips:
        c = coords.get(ip)
        if c is None:
            continue
        cells.add((round(c.lat / grid), round(c.lon / grid)))
    return len(cells)


def characterize(source: RipeAtlasSource) -> dict:
    # Run the actual iterators — this is the ground truth for what the
    # benchmark consumes. iter_vp_configs / iter_eval_targets both call
    # _ensure_loaded() internally and apply all coord / RTT / obs filters.
    vp_ids = {vp.vp_id for vp in source.iter_vp_configs()}
    target_ids = {t.target_id for t in source.iter_eval_targets()}

    coords = source._coords_by_ip
    anchor_ips = source._anchor_ips
    asn_by_ip = source._asn_by_ip
    country_by_ip = source._country_by_ip
    removed_ips = source._removed_ips
    assert coords is not None and anchor_ips is not None

    # Raw = entries loaded from JSON with v4 + geometry. Anchor / probe split
    # via the `is_anchor` flag.
    raw_anchor_ips = set(anchor_ips)
    raw_probe_ips = set(coords.keys()) - raw_anchor_ips

    # Eval = what the iterators yield. The setup decides which side is VP
    # vs target; project back to anchors/probes for the report.
    if source.setup_id() == DataSource.PROBES_TO_ANCHORS:
        eval_probe_ips = vp_ids
        eval_anchor_ips = target_ids
    else:  # ANCHORS_TO_PROBES
        eval_anchor_ips = vp_ids
        eval_probe_ips = target_ids

    def country_set(ips):
        return {country_by_ip.get(ip) for ip in ips if country_by_ip.get(ip)}

    def asn_set(ips):
        return {asn_by_ip.get(ip) for ip in ips if asn_by_ip.get(ip) is not None}

    def with_country(ips):
        return sum(1 for ip in ips if country_by_ip.get(ip))

    def with_asn(ips):
        return sum(1 for ip in ips if asn_by_ip.get(ip) is not None)

    raw = {
        "probes": {
            "n": len(raw_probe_ips),
            "with_country_code": with_country(raw_probe_ips),
            "with_asn_v4": with_asn(raw_probe_ips),
            "unique_countries": len(country_set(raw_probe_ips)),
            "unique_asns": len(asn_set(raw_probe_ips)),
        },
        "anchors": {
            "n": len(raw_anchor_ips),
            "with_country_code": with_country(raw_anchor_ips),
            "with_asn_v4": with_asn(raw_anchor_ips),
            "unique_countries": len(country_set(raw_anchor_ips)),
            "unique_asns": len(asn_set(raw_anchor_ips)),
            "city_clusters_0_1deg": _city_clusters(coords, raw_anchor_ips),
        },
    }
    eval_ = {
        "probes": {
            "n": len(eval_probe_ips),
            "unique_countries": len(country_set(eval_probe_ips)),
            "unique_asns": len(asn_set(eval_probe_ips)),
        },
        "anchors": {
            "n": len(eval_anchor_ips),
            "unique_countries": len(country_set(eval_anchor_ips)),
            "unique_asns": len(asn_set(eval_anchor_ips)),
            "city_clusters_0_1deg": _city_clusters(coords, eval_anchor_ips),
        },
    }

    probe_country_counter = Counter(
        country_by_ip[ip] for ip in eval_probe_ips if country_by_ip.get(ip)
    )
    anchor_country_counter = Counter(
        country_by_ip[ip] for ip in eval_anchor_ips if country_by_ip.get(ip)
    )
    probe_asn_counter = Counter(
        asn_by_ip[ip] for ip in eval_probe_ips if asn_by_ip.get(ip) is not None
    )
    anchor_asn_counter = Counter(
        asn_by_ip[ip] for ip in eval_anchor_ips if asn_by_ip.get(ip) is not None
    )

    probe_countries = country_set(eval_probe_ips)
    anchor_countries = country_set(eval_anchor_ips)
    probe_asns = asn_set(eval_probe_ips)
    anchor_asns = asn_set(eval_anchor_ips)

    overlap = {
        "countries": {
            "shared": len(probe_countries & anchor_countries),
            "probe_only": len(probe_countries - anchor_countries),
            "anchor_only": len(anchor_countries - probe_countries),
        },
        "asns": {
            "shared": len(probe_asns & anchor_asns),
            "probe_only": len(probe_asns - anchor_asns),
            "anchor_only": len(anchor_asns - probe_asns),
        },
    }

    paper_claims = {
        "probe_count": ">10K",
        "probe_countries": 172,
        "probe_asns": 3494,
        "anchor_count": 723,
        "anchor_countries": 96,
        "anchor_asns": 561,
        "anchor_cities": 441,
    }
    cross_check = {
        "probe_count": {
            "paper": paper_claims["probe_count"],
            "raw": raw["probes"]["n"],
            "eval": eval_["probes"]["n"],
        },
        "probe_countries": {
            "paper": paper_claims["probe_countries"],
            "raw": raw["probes"]["unique_countries"],
            "eval": eval_["probes"]["unique_countries"],
        },
        "probe_asns": {
            "paper": paper_claims["probe_asns"],
            "raw": raw["probes"]["unique_asns"],
            "eval": eval_["probes"]["unique_asns"],
        },
        "anchor_count": {
            "paper": paper_claims["anchor_count"],
            "raw": raw["anchors"]["n"],
            "eval": eval_["anchors"]["n"],
        },
        "anchor_countries": {
            "paper": paper_claims["anchor_countries"],
            "raw": raw["anchors"]["unique_countries"],
            "eval": eval_["anchors"]["unique_countries"],
        },
        "anchor_asns": {
            "paper": paper_claims["anchor_asns"],
            "raw": raw["anchors"]["unique_asns"],
            "eval": eval_["anchors"]["unique_asns"],
        },
        "anchor_cities_0_1deg_clusters": {
            "paper": paper_claims["anchor_cities"],
            "raw": raw["anchors"]["city_clusters_0_1deg"],
            "eval": eval_["anchors"]["city_clusters_0_1deg"],
        },
    }

    return {
        "source": source.name,
        "setup": source.setup_id(),
        "slice": source.slice_id(),
        "sanitize": source._sanitize,
        "config": {
            "ping_table": source._ping_table,
            "anchor_mesh_table": source._anchor_mesh_table,
            "threshold": source._threshold,
            "sanitize_threshold": source._sanitize_threshold,
            "probes_and_anchors_file": str(source._probes_and_anchors_file),
        },
        "removed_ips": {
            "count": len(removed_ips),
            "ips": sorted(removed_ips),
        },
        "raw": raw,
        "eval": eval_,
        "top10_countries_eval": {
            "probes": _top_n(probe_country_counter, 10),
            "anchors": _top_n(anchor_country_counter, 10),
        },
        "top10_asns_eval": {
            "probes": _top_n(probe_asn_counter, 10),
            "anchors": _top_n(anchor_asn_counter, 10),
        },
        "overlap_eval": overlap,
        "paper_claims": paper_claims,
        "cross_check": cross_check,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/analysis/outputs/ripe_atlas_characterization.json"),
        help="Where to write the JSON snapshot",
    )
    parser.add_argument(
        "--setup",
        default=DataSource.PROBES_TO_ANCHORS,
        choices=sorted(DataSource.ALLOWED_SETUPS),
        help="RipeAtlasSource setup id",
    )
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Disable SOI sanitization (matches sanitize=False).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    source = RipeAtlasSource(
        setup=args.setup,
        sanitize=not args.no_sanitize,
    )

    logger.info("loading RIPE Atlas data (setup=%s, sanitize=%s)…",
                args.setup, not args.no_sanitize)
    stats = characterize(source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(stats, fh, indent=2, default=str)
    logger.info("wrote %s (eval probes=%d, eval anchors=%d, removed=%d)",
                args.output,
                stats["eval"]["probes"]["n"],
                stats["eval"]["anchors"]["n"],
                stats["removed_ips"]["count"])


if __name__ == "__main__":
    main()
