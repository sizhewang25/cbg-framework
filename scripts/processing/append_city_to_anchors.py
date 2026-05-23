"""One-shot: reverse-geocode RIPE Atlas anchor coordinates via OSM Nominatim.

Reads `reproducibility_anchors.json`, queries Nominatim's reverse endpoint for
each anchor's (lat, lon), and writes `anchor_city.json` keyed by anchor `id`
with the raw GeoJSON response stored as the value.

Resumable: if the output file already exists, anchors already present are
skipped — re-run after a network hiccup to fill in the gaps.

Nominatim usage policy (https://operations.osmfoundation.org/policies/nominatim/):
  - Max ~1 request/second. We sleep `--sleep` seconds between calls (default 1.0).
  - Custom User-Agent identifying the application is mandatory.
  - Bulk geocoding (>thousands) should use a self-hosted instance. ~700 anchors
    is well below that bar.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

# Make `default` importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import default  # noqa: E402


NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "geoloc-imc-2023-cbg/1.0 (research enrichment; one-shot anchor city lookup)"


def _load_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fh:
        return json.load(fh)


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2)
    tmp.replace(path)


def _reverse_geocode(session: requests.Session, lat: float, lon: float, timeout: float) -> dict[str, Any]:
    resp = session.get(
        NOMINATIM_URL,
        params={"format": "geojson", "lat": lat, "lon": lon},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=default.REPRO_ANCHORS_FILE,
        help="Path to reproducibility_anchors.json (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default.REPRO_ATLAS_PATH / "anchor_city.json",
        help="Where to write the city map (default: %(default)s)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds between Nominatim requests (default: 1.0 — do not lower)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Per-request timeout in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N new lookups (for smoke testing)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Flush partial results to disk every N successful lookups (default: 25)",
    )
    args = parser.parse_args()

    with args.input.open() as fh:
        anchors = json.load(fh)

    out: dict[str, Any] = _load_existing(args.output)
    print(f"input anchors: {len(anchors)}; already cached: {len(out)}")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    fetched = 0
    failed: list[tuple[int, str]] = []
    for i, anchor in enumerate(anchors):
        anchor_id = anchor.get("id")
        if anchor_id is None:
            continue
        key = str(anchor_id)
        if key in out:
            continue

        geom = (anchor.get("geometry") or {}).get("coordinates")
        if not geom or len(geom) < 2:
            failed.append((anchor_id, "missing coordinates"))
            continue
        lon, lat = float(geom[0]), float(geom[1])

        try:
            out[key] = _reverse_geocode(session, lat, lon, args.timeout)
        except Exception as exc:
            failed.append((anchor_id, str(exc)))
            print(f"  [{i+1}/{len(anchors)}] id={anchor_id} ({lat:.4f},{lon:.4f}): FAILED — {exc}")
            time.sleep(args.sleep)
            continue

        fetched += 1
        if fetched % 10 == 0 or fetched <= 3:
            print(f"  [{i+1}/{len(anchors)}] id={anchor_id} ({lat:.4f},{lon:.4f}): ok")
        if fetched % args.checkpoint_every == 0:
            _save(args.output, out)

        if args.limit is not None and fetched >= args.limit:
            print(f"--limit reached ({args.limit}); stopping.")
            break

        time.sleep(args.sleep)

    _save(args.output, out)
    print(f"done. cached entries: {len(out)}; new this run: {fetched}; failures: {len(failed)}")
    if failed:
        print("first 10 failures:")
        for fid, msg in failed[:10]:
            print(f"  id={fid}: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
