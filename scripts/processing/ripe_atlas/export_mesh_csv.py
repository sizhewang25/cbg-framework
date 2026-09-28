"""Export a per-ASN RIPE Atlas VP mesh from ClickHouse as a canonical CBG CSV.

Bridges the RIPE Atlas corpora pipeline
(`process_probes_and_anchors.smk`) to the standardized raw-CSV preprocessing
pipeline (`scripts/processing/source/preprocess_cbg_raw_data.smk`), which
expects one measurement row per (VP, target) observation in the
`generic_csv` schema.

What it does:

  1. Load the per-ASN probe corpus
     (`asn_corpora/probes/<region>/probes_of_as_<asn>.json`) — these are the
     VPs, already continent-filtered, CBG-mislocation-filtered and
     city-deduped by `select_probes_and_anchors.py`.
  2. Load the shared anchor corpus (`asn_corpora/anchors/anchors.json`) —
     these are the targets, already same-ASN-guarded and reachability-
     intersected across all six setups.
  3. Query ClickHouse for `min(min)` RTT per (src, dst), restricted to exactly
     those probe IPs as `src` and anchor IPs as `dst`. The AS filter is applied
     on the *metadata* side (which probes are in the corpus) and pushed into
     the query as an explicit `src IN (...)` list — the ping table itself
     carries no ASN column.
  4. Join probe + anchor geo metadata (lat/lon, ASN, country, continent, city)
     onto every RTT row and write the canonical CSV.

The output is deliberately *unfiltered* geographically: mainland-US selection
and speed-of-Internet sanitization are the next pipeline's job.

Usage:
  python -m scripts.processing.ripe_atlas.export_mesh_csv \\
      --vp-asn 7018 --output datasets/raw/ripe-asmix-mesh.csv
"""

from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Make `default` importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.processing.ripe_atlas.continents import continent_of  # noqa: E402

logger = logging.getLogger(__name__)

ASN_CORPORA = Path("datasets/ripe_atlas/asn_corpora")

# Where `select_probes_and_anchors.py` files each setup's probe corpus. Used to
# resolve `--vp-asn` to a path when `--probes-file` is not given explicitly.
_REGION_OF_ASN: dict[int, str] = {
    7922: "north_america",
    7018: "north_america",
    3209: "europe",
    3215: "europe",
    31898: "global",
    16509: "global",
}

CSV_FIELDS = [
    "vp_id", "vp_lat", "vp_lon", "vp_asn", "vp_country", "vp_continent", "vp_city",
    "target_id", "target_lat", "target_lon", "target_asn", "target_country",
    "target_continent", "target_city",
    "rtt_ms",
]


def _load_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def _resolve_probes_file(vp_asn: int, corpora_root: Path) -> Path:
    """Locate `probes_of_as_<asn>.json` under the corpora root.

    Uses the known setup→region map first, then falls back to a glob so an
    ASN added to SETUPS later still resolves without editing this table.
    """
    region = _REGION_OF_ASN.get(vp_asn)
    if region is not None:
        candidate = corpora_root / "probes" / region / f"probes_of_as_{vp_asn}.json"
        if candidate.exists():
            return candidate
    matches = sorted((corpora_root / "probes").glob(f"*/probes_of_as_{vp_asn}.json"))
    if not matches:
        raise FileNotFoundError(
            f"no probes_of_as_{vp_asn}.json under {corpora_root / 'probes'} — "
            f"run process_probes_and_anchors.smk first, or pass --probes-file."
        )
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous corpus for AS{vp_asn}: {matches}")
    return matches[0]


def _geo_record(entry: dict, role: str) -> Optional[dict]:
    """Flatten one RIPE probe/anchor object into `<role>_*` CSV cells.

    Returns None when the entry lacks an IPv4 address or coordinates — such
    entries cannot participate in a measurement row. `continent` is taken from
    the enriched field written by `append_geo_info_to_probe_anchor.py`, falling
    back to deriving it from the country code.
    """
    ip = entry.get("address_v4")
    geom = (entry.get("geometry") or {}).get("coordinates")
    if not ip or not geom or len(geom) < 2:
        return None
    country = entry.get("country_code") or ""
    continent = entry.get("continent") or (continent_of(country) if country else "")
    return {
        f"{role}_id": ip,
        f"{role}_lat": float(geom[1]),
        f"{role}_lon": float(geom[0]),
        f"{role}_asn": entry.get("asn_v4") or "",
        f"{role}_country": country,
        f"{role}_continent": continent,
        f"{role}_city": entry.get("city") or "",
    }


def load_side(path: Path, role: str, *, require_asn: Optional[int] = None) -> dict[str, dict]:
    """Read a corpus JSON into `{ip: <role>_* record}`.

    When `require_asn` is set, entries whose `asn_v4` differs are dropped and
    counted — a guard against pointing `--vp-asn` at the wrong corpus file.
    """
    entries = _load_json(path)
    out: dict[str, dict] = {}
    skipped_no_geo = 0
    skipped_wrong_asn = 0
    collapsed_dupes = 0
    for e in entries:
        if require_asn is not None and e.get("asn_v4") != require_asn:
            skipped_wrong_asn += 1
            continue
        rec = _geo_record(e, role)
        if rec is None:
            skipped_no_geo += 1
            continue
        ip = rec[f"{role}_id"]
        if ip in out:
            # Several RIPE probes can sit behind one public IPv4 (home NAT).
            # The ping table keys on the address, so their measurements are
            # indistinguishable — keep the first entry (input order is stable)
            # and drop the rest rather than letting the last one silently win
            # with a coordinate up to a city's width away.
            collapsed_dupes += 1
            continue
        out[ip] = rec
    logger.info(
        "loaded %d %s from %s (skipped %d without coords/IPv4%s%s)",
        len(out), role, path, skipped_no_geo,
        f", {skipped_wrong_asn} outside AS{require_asn}" if require_asn is not None else "",
        f", collapsed {collapsed_dupes} sharing an IPv4" if collapsed_dupes else "",
    )
    return out


def query_min_rtts(
    src_ips: list[str],
    dst_ips: list[str],
    *,
    table: str,
    max_rtt_ms: float,
) -> list[tuple[str, str, float]]:
    """`min(min)` RTT per (src, dst) over the given IP lists.

    Both sides are pushed into the query as explicit integer IN-lists, so the
    scan is restricted to the AS's probes and the shared anchors. Rows with a
    non-positive or out-of-range `min` are excluded, matching the convention
    used throughout the corpora pipeline.
    """
    from scripts.utils.clickhouse import Clickhouse

    src_ints = sorted({int(ipaddress.IPv4Address(ip)) for ip in src_ips})
    dst_ints = sorted({int(ipaddress.IPv4Address(ip)) for ip in dst_ips})
    if not src_ints or not dst_ints:
        raise RuntimeError(
            f"empty query side (srcs={len(src_ints)}, dsts={len(dst_ints)})"
        )

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
        "query %s.%s: %d VPs x %d targets at 0 < min < %g ms",
        ch.database, table, len(src_ints), len(dst_ints), max_rtt_ms,
    )
    rows = [(s, d, float(r)) for s, d, r in ch.client.execute_iter(query)]
    ch.client.disconnect()
    logger.info("  %d (VP, target) observations returned", len(rows))
    return rows


def write_csv(
    rows: list[tuple[str, str, float]],
    vps: dict[str, dict],
    targets: dict[str, dict],
    output: Path,
) -> dict:
    """Join geo metadata onto each RTT row and write the canonical CSV.

    Rows are sorted by (target, VP) so the output is byte-stable across runs
    for a fixed database state. Returns a stats dict for the sidecar.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_vps: set[str] = set()
    seen_targets: set[str] = set()
    n_rows = 0
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for src_ip, dst_ip, rtt in sorted(rows, key=lambda r: (r[1], r[0])):
            vp = vps.get(src_ip)
            tg = targets.get(dst_ip)
            if vp is None or tg is None:
                # Cannot happen given the IN-lists, but a corpus edited between
                # the query and the join would silently emit half a row.
                continue
            writer.writerow({**vp, **tg, "rtt_ms": rtt})
            seen_vps.add(src_ip)
            seen_targets.add(dst_ip)
            n_rows += 1
    tmp.replace(output)

    target_asns = {targets[ip]["target_asn"] for ip in seen_targets}
    stats = {
        "output": str(output),
        "n_rows": n_rows,
        "n_vps_with_observations": len(seen_vps),
        "n_vps_in_corpus": len(vps),
        "n_targets_with_observations": len(seen_targets),
        "n_targets_in_corpus": len(targets),
        "n_distinct_target_asns": len(target_asns),
        "n_vps_without_observations": len(vps) - len(seen_vps),
        "n_targets_without_observations": len(targets) - len(seen_targets),
    }
    return stats


def main() -> int:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vp-asn", type=int, default=7018,
        help="ASN whose probe corpus supplies the VPs",
    )
    parser.add_argument(
        "--corpora-root", type=Path, default=ASN_CORPORA,
        help="root of the asn_corpora tree from process_probes_and_anchors.smk",
    )
    parser.add_argument(
        "--probes-file", type=Path, default=None,
        help="explicit probe corpus JSON (default: resolved from --vp-asn)",
    )
    parser.add_argument(
        "--anchors-file", type=Path, default=None,
        help="shared anchor corpus JSON (default: <corpora-root>/anchors/anchors.json)",
    )
    parser.add_argument(
        "--table", default="ping_10k_to_anchors",
        help="ClickHouse probe→anchor ping table",
    )
    parser.add_argument(
        "--max-rtt-ms", type=float, default=10000.0,
        help="upper bound on `min` RTT for a row to count",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/raw/ripe-asmix-mesh.csv"),
        help="canonical measurement CSV to write",
    )
    parser.add_argument(
        "--stats-output", type=Path, default=None,
        help="stats JSON sidecar (default: <output>.stats.json)",
    )
    args = parser.parse_args()

    probes_file = args.probes_file or _resolve_probes_file(args.vp_asn, args.corpora_root)
    anchors_file = args.anchors_file or (args.corpora_root / "anchors" / "anchors.json")

    vps = load_side(probes_file, "vp", require_asn=args.vp_asn)
    targets = load_side(anchors_file, "target")
    if not vps:
        raise RuntimeError(f"no usable VPs for AS{args.vp_asn} in {probes_file}")

    rows = query_min_rtts(
        list(vps), list(targets), table=args.table, max_rtt_ms=args.max_rtt_ms,
    )
    stats = write_csv(rows, vps, targets, args.output)
    stats.update({
        "vp_asn": args.vp_asn,
        "probes_file": str(probes_file),
        "anchors_file": str(anchors_file),
        "rtt_table": args.table,
        "max_rtt_ms": args.max_rtt_ms,
    })

    stats_path = args.stats_output or args.output.with_suffix(args.output.suffix + ".stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w") as fh:
        json.dump(stats, fh, indent=2)

    logger.info(
        "wrote %d rows → %s (%d VPs x %d targets over %d target ASNs)",
        stats["n_rows"], args.output,
        stats["n_vps_with_observations"], stats["n_targets_with_observations"],
        stats["n_distinct_target_asns"],
    )
    logger.info("wrote stats → %s", stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
