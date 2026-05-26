"""Reverse-geocode RIPE Atlas anchor / probe coordinates to a per-entry city.

Reads a RIPE-Atlas-shaped JSON list (any file with id-keyed entries that have
`geometry.coordinates = [lon, lat]`) and writes `{id: city_record}` to the
output file. Works for both anchors and probes — the input schema is the same.

Two backends (`--method`):

  online   OSM Nominatim reverse endpoint, one request per entry, ~1 req/s
           (Nominatim usage policy). Returns the full GeoJSON FeatureCollection
           from Nominatim under each id, so consumers get the rich address
           hierarchy.

  offline  reverse_geocoder (Ajay Thampi) — kdtree over GeoNames cities1000.
           Bulk lookup, sub-second for ~10K entries. Returns only the city
           name, but mirrors the online structure exactly at the path
           `data[id]["features"][0]["properties"]["address"]["city"]` so
           downstream code can read either output uniformly.

Resumable: in both modes, entries already present in the output file are
skipped, so a re-run after a crash or rate-limit hiccup fills in the gaps.

Nominatim usage policy (https://operations.osmfoundation.org/policies/nominatim/):
  - Max ~1 request/second. The online path sleeps `--sleep` seconds between
    calls (default 1.0).
  - Custom User-Agent is mandatory; identifying string set below.
  - Bulk geocoding (>thousands) should use a self-hosted instance — that's
    when you want `--method offline` instead.
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


def _reverse_geocode_online(
    session: requests.Session, lat: float, lon: float, timeout: float
) -> dict[str, Any]:
    resp = session.get(
        NOMINATIM_URL,
        params={"format": "geojson", "lat": lat, "lon": lon},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _offline_record(city: str) -> dict[str, Any]:
    """Minimal GeoJSON-shaped dict that exposes `city` at the same access path
    as the Nominatim online response."""
    return {
        "features": [
            {"properties": {"address": {"city": city}}}
        ]
    }


def _run_online(
    entries: list[dict],
    out: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    fetched = 0
    failed: list[tuple[int, str]] = []
    for i, entry in enumerate(entries):
        entry_id = entry.get("id")
        if entry_id is None:
            continue
        key = str(entry_id)
        if key in out:
            continue

        geom = (entry.get("geometry") or {}).get("coordinates")
        if not geom or len(geom) < 2:
            failed.append((entry_id, "missing coordinates"))
            continue
        lon, lat = float(geom[0]), float(geom[1])

        try:
            out[key] = _reverse_geocode_online(session, lat, lon, args.timeout)
        except Exception as exc:
            failed.append((entry_id, str(exc)))
            print(f"  [{i+1}/{len(entries)}] id={entry_id} ({lat:.4f},{lon:.4f}): FAILED — {exc}")
            time.sleep(args.sleep)
            continue

        fetched += 1
        if fetched % 10 == 0 or fetched <= 3:
            print(f"  [{i+1}/{len(entries)}] id={entry_id} ({lat:.4f},{lon:.4f}): ok")
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


def _run_offline(
    entries: list[dict],
    out: dict[str, Any],
    output_path: Path,
) -> int:
    # Lazy import: only required when --method offline.
    import reverse_geocoder as rg

    pending_ids: list[str] = []
    pending_coords: list[tuple[float, float]] = []
    skipped_no_coords = 0
    for entry in entries:
        entry_id = entry.get("id")
        if entry_id is None:
            continue
        key = str(entry_id)
        if key in out:
            continue
        geom = (entry.get("geometry") or {}).get("coordinates")
        if not geom or len(geom) < 2:
            skipped_no_coords += 1
            continue
        lon, lat = float(geom[0]), float(geom[1])
        pending_ids.append(key)
        pending_coords.append((lat, lon))

    if not pending_coords:
        print("nothing to do (all entries already cached or lack coords)")
        _save(output_path, out)
        return 0

    print(f"reverse_geocoder lookup: {len(pending_coords)} entries (kdtree)")
    results = rg.search(pending_coords)
    for key, hit in zip(pending_ids, results):
        out[key] = _offline_record(hit.get("name") or "")

    _save(output_path, out)
    print(
        f"done. cached entries: {len(out)}; new this run: {len(pending_coords)}; "
        f"skipped (no coords): {skipped_no_coords}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("online", "offline"),
        default="online",
        help="online: Nominatim HTTP (1 req/s); offline: reverse_geocoder kdtree (sub-second)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default.REPRO_ANCHORS_FILE,
        help="Path to anchors/probes JSON (default: %(default)s)",
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
        help="[online only] Seconds between Nominatim requests (default: 1.0 — do not lower)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="[online only] Per-request timeout in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="[online only] Stop after N new lookups (for smoke testing)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="[online only] Flush partial results every N successful lookups (default: 25)",
    )
    args = parser.parse_args()

    with args.input.open() as fh:
        entries = json.load(fh)

    out = _load_existing(args.output)
    print(f"input: {len(entries)} entries; already cached: {len(out)}; method: {args.method}")

    if args.method == "offline":
        return _run_offline(entries, out, args.output)
    return _run_online(entries, out, args)


if __name__ == "__main__":
    raise SystemExit(main())
