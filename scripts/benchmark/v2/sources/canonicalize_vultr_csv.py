"""One-time helper to rename the Vultr ping CSV to the canonical generic_csv schema.

Vultr's `vultr_pings_us_only.csv` uses its own column names (`prb_id`, `dst_ip`,
`probe_*`, `anchor_*`, `min_rtt`). The canonical generic_csv schema is
role-named (`vp_*` / `target_*`) — per the Vultr convention, anchors are the
VPs and probes are the targets, so the rename routes anchor data into the
`vp_*` columns and probe data into the `target_*` columns.

The script is idempotent: re-running it against an already-canonical CSV is a
no-op (pandas silently ignores renames for columns that don't exist).

Run once::

    python -m scripts.benchmark.v2.sources.canonicalize_vultr_csv

The default IN/OUT paths are repo-relative; both can be overridden via CLI.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_IN = REPO_ROOT / "datasets" / "vultr_pings_us_only.csv"
DEFAULT_OUT = REPO_ROOT / "datasets" / "vultr_pings_us_canonical.csv"

# Vultr's CSV columns → canonical generic_csv columns.
# Convention: anchor data plays the VP role, probe data plays the target role.
RENAME = {
    "dst_ip":           "vp_id",
    "anchor_latitude":  "vp_lat",
    "anchor_longitude": "vp_lon",
    "anchor_asn":       "vp_asn",
    "anchor_country":   "vp_country",
    "prb_id":           "target_id",
    "probe_latitude":   "target_lat",
    "probe_longitude":  "target_lon",
    "probe_asn":        "target_asn",
    "probe_country":    "target_country",
    "min_rtt":          "rtt_ms",
}


def canonicalize(in_path: Path, out_path: Path) -> int:
    """Read `in_path`, apply RENAME, write `out_path`. Returns row count."""
    df = pd.read_csv(in_path)
    df = df.rename(columns=RENAME)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n = canonicalize(args.in_path, args.out_path)
    logger.info("wrote %s (%d rows)", args.out_path, n)


if __name__ == "__main__":
    main()
