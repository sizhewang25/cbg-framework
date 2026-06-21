"""Extract the RIPE Atlas ASN-corpora JSON into two canonical-schema CSVs.

`RipeAtlasASNCorporaSource` reads its inputs as RIPE-Atlas-shaped JSON:
  * probes  — `datasets/ripe_atlas/asn_corpora/probes/<region>/probes_of_as_<n>.json`
  * anchors — `datasets/ripe_atlas/asn_corpora/anchors/kfolds/anchor_fold_<i>.json`
each a list of probe/anchor objects with `address_v4`, `geometry.coordinates`
([lon, lat]), `asn_v4`, `country_code`, `continent`, `city`.

This one-time helper pools all of them into just two CSVs in the canonical
generic_csv field names (the same names `dump_csv_vps` / `dump_csv_targets`
emit), written directly under the corpora root:

    asn_corpora/vps.csv      — every probe, `vp_*` columns
    asn_corpora/targets.csv  — every anchor, `target_*` columns

The role mapping follows the source's own convention (see
ripe_atlas_asn_corpora.py:10-16): **probes play the VP role**, **anchors play
the target role**. Records are deduped by id (first occurrence wins, so an
anchor that recurs across folds is written once). Region (absent from the RIPE
data) is omitted; the optional `*_continent` / `*_city` columns ride along.
`_stats.json` sidecars are skipped. Idempotent — re-running overwrites the CSVs.

Run::

    python -m scripts.benchmark.v2.sources.canonicalize_asn_corpora

The default root is repo-relative; override with --root (or the per-role dirs).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from scripts.processing.ripe_atlas.stratification import normalize_asn

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = REPO_ROOT / "datasets" / "ripe_atlas" / "asn_corpora"

# Real corpus files only — exclude the `*_stats.json` sidecars.
_PROBE_FILE_RE = re.compile(r"^probes_of_as_\d+\.json$")
_ANCHOR_FILE_RE = re.compile(r"^anchor_fold_\d+\.json$")

# Canonical column order per role (region omitted — not in the RIPE data).
_FIELDS = ("id", "lat", "lon", "asn", "country", "continent", "city")


def _entry_record(entry: dict, role: str) -> Optional[dict]:
    """Map one RIPE-Atlas probe/anchor object to a canonical `<role>_*` record,
    or None when it lacks a usable IPv4 address / coordinate pair."""
    ip = entry.get("address_v4")
    geom = (entry.get("geometry") or {}).get("coordinates")
    if not ip or not geom or len(geom) < 2:
        return None
    lon, lat = geom[0], geom[1]
    return {
        f"{role}_id": ip,
        f"{role}_lat": float(lat),
        f"{role}_lon": float(lon),
        f"{role}_asn": normalize_asn(entry.get("asn_v4")),
        f"{role}_country": entry.get("country_code"),
        f"{role}_continent": entry.get("continent"),
        f"{role}_city": entry.get("city"),
    }


def _discover(base: Path, pattern: re.Pattern[str]) -> list[Path]:
    """Corpus JSONs under `base` whose filename matches `pattern` (excludes the
    `*_stats.json` sidecars)."""
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*.json") if pattern.match(p.name))


def _pool_to_csv(json_files: list[Path], role: str, out_path: Path) -> tuple[int, int]:
    """Pool every entry across `json_files` into one canonical CSV, deduped by
    `<role>_id` (first occurrence wins). Returns (rows_written, rows_skipped)."""
    id_col = f"{role}_id"
    seen: set[str] = set()
    records, skipped = [], 0
    for jp in json_files:
        for entry in json.loads(jp.read_text()):
            rec = _entry_record(entry, role)
            if rec is None:
                skipped += 1
                continue
            if rec[id_col] in seen:
                continue
            seen.add(rec[id_col])
            records.append(rec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [f"{role}_{f}" for f in _FIELDS]
    pd.DataFrame(records, columns=columns).to_csv(out_path, index=False)
    return len(records), skipped


def canonicalize_corpora(probes_dir: Path, anchors_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    """Write `out_dir/vps.csv` (all probes, `vp_*`) and `out_dir/targets.csv`
    (all anchors, `target_*`). Returns the two output paths."""
    probe_files = _discover(probes_dir, _PROBE_FILE_RE)
    anchor_files = _discover(anchors_dir, _ANCHOR_FILE_RE)

    vps_path = out_dir / "vps.csv"
    targets_path = out_dir / "targets.csv"
    n_vps, vps_skipped = _pool_to_csv(probe_files, "vp", vps_path)
    n_tgs, tgs_skipped = _pool_to_csv(anchor_files, "target", targets_path)

    logger.info("vps.csv     ← %d probe corpora → %d vps (%d skipped)",
                len(probe_files), n_vps, vps_skipped)
    logger.info("targets.csv ← %d anchor folds → %d targets (%d skipped)",
                len(anchor_files), n_tgs, tgs_skipped)
    return vps_path, targets_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="asn_corpora root holding probes/ and anchors/.")
    parser.add_argument("--probes-dir", type=Path, default=None,
                        help="Override the probes dir (default <root>/probes).")
    parser.add_argument("--anchors-dir", type=Path, default=None,
                        help="Override the anchors dir (default <root>/anchors).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where vps.csv / targets.csv land (default <root>).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    probes_dir = args.probes_dir or (args.root / "probes")
    anchors_dir = args.anchors_dir or (args.root / "anchors")
    out_dir = args.out_dir or args.root
    vps_path, targets_path = canonicalize_corpora(probes_dir, anchors_dir, out_dir)
    logger.info("done: wrote %s and %s", vps_path, targets_path)


if __name__ == "__main__":
    main()
