"""Select per-ASN VP corpora from the sanitized RIPE Atlas probe + anchor lists.

Six setups (two NA eyeball telcos, two EU eyeball telcos, two global CDN/cloud)
each materialize as a separate probe JSON plus a stats JSON. Anchors are
*shared* across all setups: a single eval corpus that excludes the union of
all setup ASNs, which keeps cross-setup comparisons apples-to-apples and
removes institutional-proximity leakage in one go.

Inputs:
  datasets/ripe_atlas/filtered_probes.json   (produced by sanitize_probes.py)
  datasets/ripe_atlas/filtered_anchors.json  (produced by sanitize_anchors.py)

Outputs (under datasets/ripe_atlas/asn_corpora/):
  anchors.json                                # shared eval set, 721 entries
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
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Make `default` importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.processing.ripe_atlas.continents import continent_of  # noqa: E402

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


def run_setup(setup: Setup, probes: list[dict], out_root: Path) -> None:
    """Probe-only per-setup output. Anchors are shared (see select_shared_anchors)."""
    folder = out_root / setup.folder
    folder.mkdir(parents=True, exist_ok=True)

    kept_probes, probe_stats = select_probes(setup, probes)

    probe_path = folder / f"probes_of_as_{setup.asn}.json"
    probe_stats_path = folder / f"probes_of_as_{setup.asn}_stats.json"
    _save_json(probe_path, kept_probes)
    _save_json(probe_stats_path, probe_stats)

    logger.info(
        "AS%d (%s, %s): probes %d/%d (matched %d)",
        setup.asn, setup.operator, setup.setup_continent,
        probe_stats["kept"], probe_stats["input_total"], probe_stats["matched_asn"],
    )


def main() -> int:
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

    for setup in SETUPS:
        run_setup(setup, probes, args.output_dir)

    logger.info("done. wrote %d probe corpora + shared anchors under %s",
                len(SETUPS), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
