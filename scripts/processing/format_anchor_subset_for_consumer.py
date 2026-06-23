"""Convert an anchor subset CSV to the consumer-compatible schema.

Example:
  python -m scripts.processing.format_anchor_subset_for_consumer \
      scripts/processing/outputs/us_anchors/us_anchors.csv

Default output:
  scripts/processing/outputs/us_anchors/us_anchors_consumer.csv
"""

from __future__ import annotations

import argparse
import csv
import ipaddress
from pathlib import Path

INPUT_COLUMNS = (
    "target_id",
    "target_asn",
    "target_country",
    "target_city",
    "target_lat",
    "target_lon",
)

OUTPUT_COLUMNS = (
    "IP",
    "IP_VER",
    "ASN",
    "LOC_COUNTRY",
    "LOC_CITY",
    "LOC_LAT",
    "LOC_LON",
    "UNLOCODE",
    "LOC_TYPE",
    "TRAFFIC_RANK",
    "CITY_RANK",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename anchor subset columns for consumer compatibility.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="filtered anchor subset CSV, e.g. us_anchors.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="consumer-compatible CSV path; defaults beside input",
    )
    parser.add_argument(
        "--null-value",
        default="",
        help="placeholder for consumer columns that have no source value",
    )
    return parser.parse_args()


def require_columns(fieldnames: list[str], required: tuple[str, ...]) -> None:
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise SystemExit(
            "Input CSV is missing required columns: "
            f"{', '.join(missing)}. Available columns: {', '.join(fieldnames)}"
        )


def ip_version(ip_value: str, line_number: int) -> str:
    try:
        return str(ipaddress.ip_address(ip_value).version)
    except ValueError as exc:
        raise SystemExit(
            f"Invalid IP address on CSV line {line_number}: {ip_value!r}"
        ) from exc


def convert_row(row: dict[str, str], line_number: int, null_value: str) -> dict[str, str]:
    ip = (row.get("target_id") or "").strip()
    return {
        "IP": ip,
        "IP_VER": ip_version(ip, line_number),
        "ASN": (row.get("target_asn") or "").strip(),
        "LOC_COUNTRY": (row.get("target_country") or "").strip(),
        "LOC_CITY": (row.get("target_city") or "").strip(),
        "LOC_LAT": (row.get("target_lat") or "").strip(),
        "LOC_LON": (row.get("target_lon") or "").strip(),
        "UNLOCODE": null_value,
        "LOC_TYPE": null_value,
        "TRAFFIC_RANK": null_value,
        "CITY_RANK": null_value,
    }


def default_output_path(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_consumer.csv")


def main() -> int:
    args = parse_args()
    output_csv = args.output_csv or default_output_path(args.input_csv)

    with args.input_csv.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise SystemExit(f"{args.input_csv} has no CSV header")
        require_columns(list(reader.fieldnames), INPUT_COLUMNS)
        rows = [
            convert_row(row, line_number, args.null_value)
            for line_number, row in enumerate(reader, start=2)
        ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Read {len(rows)} anchors from {args.input_csv}")
    print(f"Wrote {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
