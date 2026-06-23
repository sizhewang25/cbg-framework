"""Filter US anchors from a CSV and write summary statistics.

Example:
  python -m scripts.processing.filter_us_anchors \
      datasets/ripe_atlas/asn_corpora/targets.csv \
      --out-dir /tmp/ripe-us-anchors

Outputs:
  <out-dir>/us_anchors.csv
  <out-dir>/us_city_stats.csv
  <out-dir>/us_asn_stats.csv
  <out-dir>/us_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MISSING_LABEL = "<missing>"

COUNTRY_COLUMNS = ("target_country", "anchor_country", "country_code", "country")
CITY_COLUMNS = ("target_city", "anchor_city", "city")
ASN_COLUMNS = ("target_asn", "anchor_asn", "asn_v4", "asn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter US anchors from a CSV and save subset + stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path", type=Path, help="input anchor CSV")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("scripts/processing/outputs/us_anchors"),
        help="directory for filtered CSV and stats files",
    )
    parser.add_argument(
        "--country",
        default="US",
        help="country code to keep; defaults to US because this tool is US-focused",
    )
    parser.add_argument(
        "--country-column",
        help="override country column name if it cannot be inferred",
    )
    parser.add_argument(
        "--city-column",
        help="override city column name if it cannot be inferred",
    )
    parser.add_argument(
        "--asn-column",
        help="override ASN column name if it cannot be inferred",
    )
    return parser.parse_args()


def infer_column(
    fieldnames: list[str],
    override: str | None,
    candidates: tuple[str, ...],
    label: str,
) -> str:
    if override:
        if override not in fieldnames:
            raise SystemExit(
                f"{label} column {override!r} is not present. "
                f"Available columns: {', '.join(fieldnames)}"
            )
        return override

    for candidate in candidates:
        if candidate in fieldnames:
            return candidate

    raise SystemExit(
        f"Could not infer {label} column. Pass --{label}-column. "
        f"Available columns: {', '.join(fieldnames)}"
    )


def clean_value(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def value_or_missing(value: str | None) -> str:
    cleaned = clean_value(value)
    return cleaned if cleaned else MISSING_LABEL


def normalize_asn(value: str | None) -> str:
    cleaned = clean_value(value)
    if not cleaned:
        return MISSING_LABEL
    if cleaned.upper().startswith("AS"):
        cleaned = cleaned[2:].strip()
    return cleaned or MISSING_LABEL


def sort_key(value: str) -> tuple[int, int, int, str]:
    if value == MISSING_LABEL:
        return (1, 1, 0, value)
    try:
        return (0, 0, int(value), "")
    except ValueError:
        return (0, 1, 0, value.casefold())


def sorted_counts(counter: Counter[str]) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], sort_key(item[0])))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_stats_csv(
    path: Path,
    key_field: str,
    counts: Counter[str],
    total: int,
) -> None:
    rows = []
    for key, count in sorted_counts(counts):
        rows.append({
            key_field: key,
            "count": str(count),
            "percent": f"{(count / total * 100) if total else 0:.3f}",
        })
    write_csv(path, [key_field, "count", "percent"], rows)


def main() -> int:
    args = parse_args()
    country = args.country.upper()

    with args.csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise SystemExit(f"{args.csv_path} has no CSV header")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    country_column = infer_column(
        fieldnames, args.country_column, COUNTRY_COLUMNS, "country"
    )
    city_column = infer_column(fieldnames, args.city_column, CITY_COLUMNS, "city")
    asn_column = infer_column(fieldnames, args.asn_column, ASN_COLUMNS, "asn")

    subset = [
        row for row in rows
        if clean_value(row.get(country_column)).upper() == country
    ]
    city_counts = Counter(value_or_missing(row.get(city_column)) for row in subset)
    asn_counts = Counter(normalize_asn(row.get(asn_column)) for row in subset)

    country_slug = country.lower()
    subset_path = args.out_dir / f"{country_slug}_anchors.csv"
    city_stats_path = args.out_dir / f"{country_slug}_city_stats.csv"
    asn_stats_path = args.out_dir / f"{country_slug}_asn_stats.csv"
    summary_path = args.out_dir / f"{country_slug}_summary.json"

    write_csv(subset_path, fieldnames, subset)
    write_stats_csv(city_stats_path, "city", city_counts, len(subset))
    write_stats_csv(asn_stats_path, "asn", asn_counts, len(subset))

    summary = {
        "input_csv": str(args.csv_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "country_filter": country,
        "columns": {
            "country": country_column,
            "city": city_column,
            "asn": asn_column,
        },
        "input_total_anchors": len(rows),
        "filtered_anchor_count": len(subset),
        "filtered_anchor_percent": (
            round(len(subset) / len(rows) * 100, 3) if rows else 0.0
        ),
        "unique_cities": len(city_counts),
        "missing_city_count": city_counts.get(MISSING_LABEL, 0),
        "unique_asns": len(asn_counts),
        "missing_asn_count": asn_counts.get(MISSING_LABEL, 0),
        "outputs": {
            "subset_csv": str(subset_path),
            "city_stats_csv": str(city_stats_path),
            "asn_stats_csv": str(asn_stats_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"Read {len(rows)} anchors from {args.csv_path}")
    print(f"Kept {len(subset)} anchors where {country_column} == {country}")
    print(f"Wrote {subset_path}")
    print(f"Wrote {city_stats_path}")
    print(f"Wrote {asn_stats_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
