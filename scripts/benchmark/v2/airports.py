"""Airport reference layer for the closest-airport eval metric.

Operators care less about exact lat/lon error than about whether a geolocation
estimate resolves to the right airport (IATA metro code). This module provides:

- `filter_airports` — distil the raw OurAirports CSV to the operator-facing set
  (large/medium airports that carry both an IATA code and a municipality).
- `build_slim_airports` — write that slim set to a committed parquet.
- `AirportIndex` — a haversine `BallTree` over the slim set, returning the
  nearest IATA code + great-circle distance (km) for query points, vectorized
  and NaN-safe.

Design decisions live in notes/2026-06-18-closest-airport-eval-decisions.md.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM

# OurAirports `type` values we keep — the recognizable commercial fields. The
# rest (small_airport, heliport, seaplane_base, closed, balloonport) are noise
# for an operator reasoning in metro codes.
AIRPORT_TYPES = ("large_airport", "medium_airport")

# Columns carried into the slim parquet.
SLIM_COLUMNS = (
    "iata_code",
    "name",
    "municipality",
    "iso_country",
    "latitude_deg",
    "longitude_deg",
    "type",
)

# Default location of the committed slim parquet, relative to the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AIRPORTS_PARQUET = (
    _REPO_ROOT / "datasets" / "static_datasets" / "ourairports_iata.parquet"
)


def _nonblank(series: pd.Series) -> pd.Series:
    """True where the string cell is present and not blank/whitespace."""
    return series.notna() & (series.astype("string").str.strip() != "")


def filter_airports(raw: pd.DataFrame) -> pd.DataFrame:
    """Distil the raw OurAirports frame to the operator-facing airport set.

    Keeps rows whose `type` is large/medium *and* that carry a non-blank IATA
    code *and* a non-blank municipality (city). See the decision note for the
    rationale and the resulting count (~4,441 worldwide).
    """
    keep = (
        raw["type"].isin(AIRPORT_TYPES)
        & _nonblank(raw["iata_code"])
        & _nonblank(raw["municipality"])
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
        lat = airports["latitude_deg"].to_numpy(dtype=float)
        lon = airports["longitude_deg"].to_numpy(dtype=float)
        self._tree = BallTree(np.radians(np.column_stack([lat, lon])), metric="haversine")

    @classmethod
    def from_parquet(cls, path: Path = DEFAULT_AIRPORTS_PARQUET) -> "AirportIndex":
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Slim airport parquet not found at {path}. Build it first with "
                "`scripts/benchmark/v2/airports.py` / the build step in the "
                "decision note (notes/2026-06-18-closest-airport-eval-decisions.md)."
            )
        return cls(pd.read_parquet(path))

    def query_many(
        self, lats, lons
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nearest airport per query point.

        Returns `(iata, km)` arrays aligned to the input. Points with a missing
        (NaN/None) latitude or longitude get `iata=None` and `km=NaN`; valid
        points are unaffected by NaN siblings.
        """
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        n = lats.shape[0]
        iata_out = np.full(n, None, dtype=object)
        km_out = np.full(n, np.nan, dtype=float)

        valid = ~(np.isnan(lats) | np.isnan(lons))
        if valid.any():
            pts = np.radians(np.column_stack([lats[valid], lons[valid]]))
            dist, idx = self._tree.query(pts, k=1)
            dist = dist[:, 0]
            idx = idx[:, 0]
            iata_out[valid] = self.iata[idx]
            km_out[valid] = dist * EARTH_RADIUS_KM
        return iata_out, km_out


@lru_cache(maxsize=1)
def load_airport_index(
    path: Optional[str] = None,
) -> AirportIndex:
    """Process-wide cached index built once from the slim parquet."""
    return AirportIndex.from_parquet(Path(path) if path else DEFAULT_AIRPORTS_PARQUET)
