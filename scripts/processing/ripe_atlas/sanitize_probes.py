"""Filter RIPE Atlas probes to those active enough in ping_10k_to_anchors.

Pipeline:
  1. Load all probes from `reproducibility_probes.json` (or the file passed
     via `--probes-file`).
  2. Query the ClickHouse `ping_10k_to_anchors` table for unique src IPs
     that have at least N rows with `min` > 0 (i.e. successful pings within
     `--max-rtt-ms`). N is the `--min-measurements` knob.
  3. Keep only the input probes whose `address_v4` is in that set.
  4. Write the filtered list to `datasets/ripe_atlas/filtered_probes.json`.

Sister script to `sanitize_anchors.py` (same env wiring, same output dir).
Requires ClickHouse reachable per CLICKHOUSE_HOST / CLICKHOUSE_PASSWORD
from `.env`.

Usage:
  python -m scripts.processing.ripe_atlas.sanitize_probes
  python -m scripts.processing.ripe_atlas.sanitize_probes \\
      --min-measurements 200 \\
      --output datasets/ripe_atlas/filtered_probes_200.json
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
from scripts.utils.clickhouse import Clickhouse  # noqa: E402

logger = logging.getLogger(__name__)


def fetch_active_probe_ips(
    table: str,
    min_measurements: int,
    max_rtt_ms: float = 10000.0,
) -> set[str]:
    """Unique src IPv4 strings with at least `min_measurements` valid rows
    (`min` > 0 AND `min` < `max_rtt_ms`) in `table`.

    The 10000 ms default matches the wide upper bound used elsewhere in the
    codebase — narrow it if you want to count only fast pings.
    """
    ch = Clickhouse()
    query = f"""
        SELECT IPv4NumToString(src) AS src_ip, count() AS n_measurements
        FROM {ch.database}.{table}
        WHERE `min` > 0 AND `min` < {max_rtt_ms}
        GROUP BY src
        HAVING n_measurements >= {min_measurements}
    """
    logger.info("query %s.%s (min_measurements ≥ %d, rtt < %g ms)",
                ch.database, table, min_measurements, max_rtt_ms)
    rows = list(ch.client.execute_iter(query))
    ch.client.disconnect()
    return {row[0] for row in rows}


def load_probes(path: Path) -> list[dict]:
    with path.open() as fh:
        return json.load(fh)


def filter_probes(probes: list[dict], valid_ips: set[str]) -> list[dict]:
    """Keep input probe entries whose `address_v4` is in `valid_ips`."""
    return [p for p in probes if p.get("address_v4") in valid_ips]


def main() -> int:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Filter RIPE Atlas probes to those with ≥ N measurements in "
            "ping_10k_to_anchors."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--probes-file", type=Path,
        default=Path(default.REPRO_PROBES_FILE),
        help="input probes JSON (full reproducibility_probes.json)",
    )
    parser.add_argument(
        "--table", default="ping_10k_to_anchors",
        help="ClickHouse table with probe→anchor pings",
    )
    parser.add_argument(
        "-n", "--min-measurements", type=int, default=100,
        help="N — keep probes with at least this many valid rows in the table",
    )
    parser.add_argument(
        "--max-rtt-ms", type=float, default=10000.0,
        help="upper bound on `min` RTT to count a row",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("datasets/ripe_atlas/filtered_probes.json"),
        help="output filtered probes JSON",
    )
    args = parser.parse_args()

    logger.info("loading probes from %s", args.probes_file)
    probes = load_probes(args.probes_file)
    logger.info("  %d probes in input", len(probes))

    valid_ips = fetch_active_probe_ips(
        args.table,
        args.min_measurements,
        max_rtt_ms=args.max_rtt_ms,
    )
    logger.info("  %d probe IPs meet the threshold", len(valid_ips))

    filtered = filter_probes(probes, valid_ips)
    pct = 100.0 * len(filtered) / max(1, len(probes))
    logger.info("filtered: %d / %d probes match (%.1f%%)",
                len(filtered), len(probes), pct)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(filtered, fh, indent=2)
    logger.info("wrote %d probes → %s", len(filtered), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
