"""Enrich the sanitized RIPE Atlas anchor + probe corpora with geo metadata.

Reads `filtered_{anchors,probes}.json` and writes them back in-place with two
extra fields attached to each entry:

  continent  — derived from `country_code` via `continents.continent_of`.
               Always added (resolves to "Unknown" for missing/unknown codes).

  city       — derived from `{anchor,probe}_city.json` (Nominatim reverse-
               geocoded sidecar files produced by `fetch_city_for_probe_anchor.py`).
               Looked up by entry `id` at the path
               `data[id].features[0].properties.address.city`. If the sidecar
               file is missing the city phase is skipped entirely; if an entry
               is missing from the sidecar (or its features list is empty / has
               no `city` key) that entry is left without a `city` field rather
               than dropped.

The two sidecar files share the same shape across the online and offline
Nominatim backends, so this consumer is backend-agnostic.

Usage:
  python -m scripts.processing.ripe_atlas.append_geo_info_to_probe_anchor
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Make the `scripts.*` package importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.processing.ripe_atlas.continents import continent_of  # noqa: E402

logger = logging.getLogger(__name__)

DATASETS_DIR = Path("datasets/ripe_atlas")


def _load_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def _save_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2)
    tmp.replace(path)


def append_continent_field(items: list[dict[str, Any]]) -> int:
    """Attach `continent` to each item based on its `country_code`. Returns
    the number of items that resolved to a known continent (not "Unknown")."""
    n_known = 0
    for it in items:
        cont = continent_of(it.get("country_code"))
        it["continent"] = cont
        if cont != "Unknown":
            n_known += 1
    return n_known


def _extract_city(record: Any) -> str | None:
    """Pull the city string out of a Nominatim FeatureCollection record.

    Matches the shape produced by both `fetch_city_for_probe_anchor.py` backends:
    `record.features[0].properties.address.city`.
    """
    if not isinstance(record, dict):
        return None
    features = record.get("features") or []
    if not features:
        return None
    props = (features[0] or {}).get("properties") or {}
    address = props.get("address") or {}
    city = address.get("city")
    if isinstance(city, str) and city.strip():
        return city
    return None


def append_city_field(
    items: list[dict[str, Any]], city_data: dict[str, Any]
) -> tuple[int, int]:
    """Attach `city` to each item that has a usable record in `city_data`.

    Items whose id is missing from `city_data` (or whose record has no city
    string) are left untouched. Returns `(n_with_city, n_missing)`.
    """
    n_with = 0
    n_miss = 0
    for it in items:
        key = str(it.get("id"))
        record = city_data.get(key)
        if record is None:
            n_miss += 1
            continue
        city = _extract_city(record)
        if city is None:
            n_miss += 1
            continue
        it["city"] = city
        n_with += 1
    return n_with, n_miss


def process_file(
    input_path: Path, city_path: Path, label: str
) -> None:
    items = _load_json(input_path)
    if not isinstance(items, list):
        raise ValueError(f"{input_path} is not a JSON list — got {type(items).__name__}")

    n_known_cont = append_continent_field(items)
    logger.info(
        "%s: continent attached — %d / %d resolved (%d 'Unknown')",
        label, n_known_cont, len(items), len(items) - n_known_cont,
    )

    if city_path.exists():
        city_data = _load_json(city_path)
        if not isinstance(city_data, dict):
            raise ValueError(
                f"{city_path} is not a JSON object — got {type(city_data).__name__}"
            )
        n_with, n_miss = append_city_field(items, city_data)
        logger.info(
            "%s: city attached — %d with city, %d skipped (no record or empty city) [source: %s]",
            label, n_with, n_miss, city_path.name,
        )
    else:
        logger.info("%s: city sidecar not found at %s — skipping city phase", label, city_path)

    _save_json(input_path, items)
    logger.info("%s: wrote %d enriched entries → %s", label, len(items), input_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-dir", type=Path, default=DATASETS_DIR,
        help="Directory containing filtered_{anchors,probes}.json and the city sidecars.",
    )
    parser.add_argument(
        "--side", choices=("anchors", "probes", "both"), default="both",
        help="Which corpus to enrich (default: both). Snakemake rules pin a single side.",
    )
    parser.add_argument(
        "--anchor-city", type=Path, default=None,
        help="Override path to the anchor city sidecar (default: <datasets-dir>/anchor_city.json).",
    )
    parser.add_argument(
        "--probe-city", type=Path, default=None,
        help="Override path to the probe city sidecar (default: <datasets-dir>/probe_city.json).",
    )
    parser.add_argument(
        "--sentinel", type=Path, default=None,
        help="If set, touch this file after a successful run (used by snakemake for DAG ordering "
             "since the enrichment is in-place).",
    )
    args = parser.parse_args()

    datasets_dir: Path = args.datasets_dir
    anchor_city_path: Path = args.anchor_city or (datasets_dir / "anchor_city.json")
    probe_city_path: Path = args.probe_city or (datasets_dir / "probe_city.json")

    if args.side in ("anchors", "both"):
        process_file(datasets_dir / "filtered_anchors.json", anchor_city_path, "anchors")
    if args.side in ("probes", "both"):
        process_file(datasets_dir / "filtered_probes.json", probe_city_path, "probes")

    if args.sentinel is not None:
        args.sentinel.parent.mkdir(parents=True, exist_ok=True)
        args.sentinel.touch()


if __name__ == "__main__":
    main()
