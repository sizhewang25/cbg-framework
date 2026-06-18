"""Airport reference layer for the closest-airport eval metric.

Operators care less about exact lat/lon error than about whether a geolocation
estimate resolves to the right airport (IATA metro code). This module provides:

- `filter_airports` — distil the raw OurAirports CSV to the operator-facing set
  (large hubs with scheduled service that carry an IATA code and a municipality).
- `build_slim_airports` — write that slim set to a committed parquet.
- `AirportIndex` — a haversine `BallTree` over the slim set, returning the
  nearest IATA code + great-circle distance (km) for query points, vectorized
  and NaN-safe.

Design decisions live in notes/2026-06-18-closest-airport-eval-decisions.md.
"""

from __future__ import annotations

import tempfile
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM

# Public-domain OurAirports CSV (continuously updated).
OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"

# OurAirports `type` values we keep. Large airports are the major hubs whose
# IATA codes operators reference in PoP/router rDNS hostnames and where data
# centers colocate; ~82% of our targets already snap to one. Medium airports
# are intentionally excluded (hub-level, not metro-faithful) — see the decision
# note. Everything else (small/heliport/seaplane/closed) is noise.
AIRPORT_TYPES = ("large_airport",)

# Columns carried into the slim parquet.
SLIM_COLUMNS = (
    "iata_code",
    "name",
    "municipality",
    "iso_country",
    "latitude_deg",
    "longitude_deg",
    "type",
    "scheduled_service",
)

# Default location of the committed slim parquet, relative to the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AIRPORTS_PARQUET = (
    _REPO_ROOT / "datasets" / "static_datasets" / "ourairports_iata.parquet"
)


def _nonblank(series: pd.Series) -> pd.Series:
    """True where the string cell is present and not blank/whitespace."""
    return series.notna() & (series.astype("string").str.strip() != "")


def download_ourairports_csv(dest: Optional[Path] = None) -> Path:
    """Fetch the latest OurAirports airports.csv (to a temp file by default)."""
    dest = dest or (Path(tempfile.mkdtemp()) / "airports.csv")
    urllib.request.urlretrieve(OURAIRPORTS_URL, dest)
    return dest


def filter_airports(
    raw: pd.DataFrame, types: Sequence[str] = AIRPORT_TYPES
) -> pd.DataFrame:
    """Distil the raw OurAirports frame to the operator-facing airport set.

    Keeps rows whose `type` is in `types` (default: large hubs only — the eval
    reference set), that carry a non-blank IATA code and municipality, *and*
    that have scheduled commercial service. The scheduled-service gate is the
    in-dataset proxy for "codes operators actually reference" (the IATA codes
    that appear in PoP/router rDNS hostnames) — it drops GA/military fields like
    PAO (Palo Alto) and NUQ (Moffett) while keeping real metro hubs. See the
    decision note for the resulting count (~1,158 for large-only).

    `types` is widened (e.g. to include medium airports) by the distribution
    visualization, which contrasts large vs. medium scheduled-service airports.
    """
    keep = (
        raw["type"].isin(types)
        & _nonblank(raw["iata_code"])
        & _nonblank(raw["municipality"])
        & (raw["scheduled_service"] == "yes")
    )
    slim = raw.loc[keep, list(SLIM_COLUMNS)].copy()
    slim["iata_code"] = slim["iata_code"].astype("string").str.strip()
    slim["municipality"] = slim["municipality"].astype("string").str.strip()
    return slim.reset_index(drop=True)


def build_slim_airports(src_csv: Path, out_parquet: Path) -> pd.DataFrame:
    """Read the raw OurAirports CSV, filter it, write the slim parquet."""
    raw = pd.read_csv(src_csv, low_memory=False)
    slim = filter_airports(raw)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    slim.to_parquet(out_parquet, index=False)
    return slim


class AirportIndex:
    """Nearest-airport lookup over a (lat, lon) airport set.

    Uses a `BallTree(metric="haversine")` so neighbour search is exact
    great-circle, no chord→arc correction. Coordinates are stored in radians;
    query distances come back in radians and are scaled by `EARTH_RADIUS_KM`.
    """

    def __init__(self, airports: pd.DataFrame) -> None:
        if airports.empty:
            raise ValueError("AirportIndex requires a non-empty airport set")
        self.iata = airports["iata_code"].astype(str).to_numpy()
        self.lat = airports["latitude_deg"].to_numpy(dtype=float)
        self.lon = airports["longitude_deg"].to_numpy(dtype=float)
        self._tree = BallTree(
            np.radians(np.column_stack([self.lat, self.lon])), metric="haversine"
        )

    @classmethod
    def from_parquet(cls, path: Path = DEFAULT_AIRPORTS_PARQUET) -> "AirportIndex":
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Slim airport parquet not found at {path}. Build it first with "
                "`scripts/benchmark/v2/airports.py` / the build step in the "
                "decision note (notes/2026-06-18-closest-airport-eval-decisions.md)."
            )
        return cls(pd.read_parquet(path))

    def _query(self, lats, lons) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Core k=1 query. Returns `(idx, km, valid)` where `idx` is the nearest
        airport row index (−1 where invalid) and `valid` masks NaN-free points."""
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        n = lats.shape[0]
        idx_out = np.full(n, -1, dtype=int)
        km_out = np.full(n, np.nan, dtype=float)

        valid = ~(np.isnan(lats) | np.isnan(lons))
        if valid.any():
            pts = np.radians(np.column_stack([lats[valid], lons[valid]]))
            dist, idx = self._tree.query(pts, k=1)
            idx_out[valid] = idx[:, 0]
            km_out[valid] = dist[:, 0] * EARTH_RADIUS_KM
        return idx_out, km_out, valid

    def query_many(self, lats, lons) -> tuple[np.ndarray, np.ndarray]:
        """Nearest airport per query point.

        Returns `(iata, km)` arrays aligned to the input. Points with a missing
        (NaN/None) latitude or longitude get `iata=None` and `km=NaN`; valid
        points are unaffected by NaN siblings.
        """
        idx, km, valid = self._query(lats, lons)
        iata_out = np.full(idx.shape[0], None, dtype=object)
        iata_out[valid] = self.iata[idx[valid]]
        return iata_out, km

    def query_full(
        self, lats, lons
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Like `query_many` but also returns the matched airport's coordinates.

        Returns `(iata, km, ap_lat, ap_lon)`. The coordinate arrays carry the
        nearest *airport's* lat/lon (NaN where the query point was invalid),
        which the postprocessor uses to measure the airport-to-airport gap.
        """
        idx, km, valid = self._query(lats, lons)
        n = idx.shape[0]
        iata_out = np.full(n, None, dtype=object)
        ap_lat = np.full(n, np.nan, dtype=float)
        ap_lon = np.full(n, np.nan, dtype=float)
        iata_out[valid] = self.iata[idx[valid]]
        ap_lat[valid] = self.lat[idx[valid]]
        ap_lon[valid] = self.lon[idx[valid]]
        return iata_out, km, ap_lat, ap_lon


@lru_cache(maxsize=1)
def load_airport_index(
    path: Optional[str] = None,
) -> AirportIndex:
    """Process-wide cached index built once from the slim parquet."""
    return AirportIndex.from_parquet(Path(path) if path else DEFAULT_AIRPORTS_PARQUET)
