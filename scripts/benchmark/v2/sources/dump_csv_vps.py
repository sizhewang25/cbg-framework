"""Dump a canonical CSV's VP catalog to `vps.json`.

Sibling of `scripts/benchmark/v2/sources/legacy_dump_csv_stratification.py`. That
script materializes the *target*-role entities (`anchors.json`) from the
`target_*` columns; this one materializes the *VP*-role entities (`vps.json`)
from the `vp_*` columns.

VP parsing is delegated to `GenericCSVSource.iter_vp_configs`, so the same
canonical-schema contract applies — required `vp_id, vp_lat, vp_lon`, optional
`vp_asn, vp_country, vp_continent, vp_region, vp_city` — with identical ASN
normalization and NA-safe string handling. VPs are deduped by `vp_id`.

Inputs:
    --csv     canonical-schema CSV (see scripts/benchmark/v2/sources/generic_csv.py)

Output (default landing pad: `datasets/<csv stem>/vps.json`):
    vps.json   — one entry per unique vp_id, with the canonical CSV's own
                 field names (no RIPE-Atlas renaming):
                   {vp_id, vp_lat, vp_lon, vp_asn, vp_country,
                    vp_continent/vp_region/vp_city when present in the CSV}

Run::

    python -m scripts.benchmark.v2.sources.dump_csv_vps \\
        --csv datasets/vultr_pings_us_canonical.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from scripts.benchmark.v2.sources.base import VpConfig
from scripts.benchmark.v2.sources.generic_csv import GenericCSVSource

logger = logging.getLogger(__name__)


def _vp_json(vp: VpConfig) -> dict:
    """Canonical-schema VP record: same field names the CSV uses."""
    out: dict = {
        "vp_id": vp.vp_id,
        "vp_lat": vp.lat,
        "vp_lon": vp.lon,
        "vp_asn": vp.asn,
        "vp_country": vp.country,
    }
    # Pass through the extra geo columns only when the CSV supplied them, so
    # the output stays minimal for the common asn/country/coords-only case.
    if vp.continent is not None:
        out["vp_continent"] = vp.continent
    if vp.region is not None:
        out["vp_region"] = vp.region
    if vp.city is not None:
        out["vp_city"] = vp.city
    return out


def dump(csv_path: Path, out_path: Path) -> Path:
    source = GenericCSVSource(slice="all", csv_path=csv_path)
    vps = [_vp_json(vp) for vp in source.iter_vp_configs()]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(vps, indent=2))
    logger.info("wrote %d vps to %s", len(vps), out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True,
                        help="Canonical-schema CSV (vp_*, target_*, rtt_ms columns).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path. Defaults to datasets/<csv-stem>/vps.json.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_path = args.out
    if out_path is None:
        repo_root = Path(__file__).resolve().parents[4]
        out_path = repo_root / "datasets" / args.csv.stem / "vps.json"

    dump(args.csv, out_path)


if __name__ == "__main__":
    main()
