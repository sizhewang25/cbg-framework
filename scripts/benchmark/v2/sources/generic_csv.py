"""GenericCSVSource — adapt any well-formed CSV into the v2 benchmark.

How to use:
  1. Produce a CSV with the required columns listed below (one row per
     (vp, target, RTT) observation).
  2. Point `DEFAULT_CSV` (see CONFIG block) at it.
  3. Run:
        python -m scripts.benchmark.v2.cli materialize-inputs \\
            --source generic_csv --slice all

Required columns (every row):
  vp_id          : str    — stable VP identifier (probe id, IP, hostname, …)
  vp_lat         : float  — VP latitude in degrees
  vp_lon         : float  — VP longitude in degrees
  target_id      : str    — stable target identifier (must have hard-GT coords)
  target_lat     : float  — target latitude in degrees
  target_lon     : float  — target longitude in degrees
  rtt_ms         : float  — strictly positive RTT in milliseconds

Optional columns (auto-detected, only used when present):
  vp_asn, vp_country, target_asn, target_country

Slicing (`--slice`):
  all       — every row (after column validation + NaN drop)
  head<k>   — keep the k targets that sort first by target_id (deterministic,
              cheap smoke-test slice)

To benchmark several different CSVs in parallel, subclass and set a
different `name` per CSV — the on-disk inputs/outputs tree is keyed off
`<name>/<setup>/<slice>/`, so distinct names get parallel directory trees.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from scripts.benchmark.v2.sources.base import DataSource, EvalTarget, VpConfig
from scripts.framework.v2 import FitSample
from scripts.framework.v2.types import Coord, Latency, VpId

# ============================================================================
# CONFIG — edit this block to point at your CSV.
# ============================================================================
#
# DEFAULT_CSV is the only thing most users need to change. It's relative to
# the repo root by default; absolute paths work too.

DEFAULT_CSV = (
    Path(__file__).resolve().parents[4]
    / "datasets" / "smoke-test.csv"
)

# ============================================================================

_REQUIRED = (
    "vp_id", "vp_lat", "vp_lon",
    "target_id", "target_lat", "target_lon",
    "rtt_ms",
)

_OPTIONAL = ("vp_asn", "vp_country", "target_asn", "target_country")


class GenericCSVSource(DataSource):
    """CSV-backed source with a fixed canonical schema (see module docstring).

    Both setups are supported by swapping which side of each pair acts as the
    VP — column names stay the same; only the role assignment flips.
    """

    name = "generic_csv"

    def __init__(
        self,
        slice: str = "all",
        setup: str = DataSource.PROBES_TO_ANCHORS,
        csv_path: Optional[Path] = None,
    ) -> None:
        if setup not in DataSource.ALLOWED_SETUPS:
            raise ValueError(
                f"unknown setup {setup!r}; expected one of {DataSource.ALLOWED_SETUPS}"
            )
        self._slice = slice
        self._setup = setup
        self._csv_path = Path(csv_path) if csv_path is not None else DEFAULT_CSV
        self._df: Optional[pd.DataFrame] = None  # lazy-loaded

    # ---- DataSource API ------------------------------------------------------

    def slice_id(self) -> str:
        return self._slice

    def setup_id(self) -> str:
        return self._setup

    def iter_vp_configs(self) -> Iterator[VpConfig]:
        df = self._load()
        vp_id_col, lat_col, lon_col, asn_col, country_col = self._vp_columns()
        for _, row in df.drop_duplicates(vp_id_col).iterrows():
            yield VpConfig(
                vp_id=str(row[vp_id_col]),
                lat=float(row[lat_col]),
                lon=float(row[lon_col]),
                asn=_opt_int(row.get(asn_col)) if asn_col else None,
                country=_opt_str(row.get(country_col)) if country_col else None,
            )

    def iter_fit_samples(self) -> Iterator[FitSample]:
        df = self._load()
        vp_id_col, vp_lat, vp_lon, _, _ = self._vp_columns()
        _, tg_lat, tg_lon = self._target_columns()
        for row in df.itertuples(index=False):
            yield FitSample(
                vp_id=VpId(str(getattr(row, vp_id_col))),
                vp_coord=Coord(lat=float(getattr(row, vp_lat)), lon=float(getattr(row, vp_lon))),
                probe_coord=Coord(lat=float(getattr(row, tg_lat)), lon=float(getattr(row, tg_lon))),
                latency=Latency(float(row.rtt_ms)),
            )

    def iter_eval_targets(self) -> Iterator[EvalTarget]:
        df = self._load()
        vp_id_col, vp_lat, vp_lon, _, _ = self._vp_columns()
        tg_id_col, tg_lat, tg_lon = self._target_columns()
        for tg_id, group in df.groupby(tg_id_col, sort=True):
            first = group.iloc[0]
            true_coord = Coord(lat=float(first[tg_lat]), lon=float(first[tg_lon]))
            obs: list[tuple[VpId, Coord, Latency]] = [
                (
                    VpId(str(getattr(r, vp_id_col))),
                    Coord(lat=float(getattr(r, vp_lat)), lon=float(getattr(r, vp_lon))),
                    Latency(float(r.rtt_ms)),
                )
                for r in group.itertuples(index=False)
            ]
            yield EvalTarget(target_id=str(tg_id), true_coord=true_coord, obs=obs)

    # ---- internals -----------------------------------------------------------

    def _vp_columns(self) -> tuple[str, str, str, Optional[str], Optional[str]]:
        """Resolve which CSV columns play the VP role for this setup.

        Returns (id, lat, lon, asn-or-None, country-or-None). The asn/country
        slots are None when the corresponding optional columns aren't present.
        """
        df = self._load()
        if self._setup == DataSource.PROBES_TO_ANCHORS:
            asn = "vp_asn" if "vp_asn" in df.columns else None
            country = "vp_country" if "vp_country" in df.columns else None
            return "vp_id", "vp_lat", "vp_lon", asn, country
        # ANCHORS_TO_PROBES: target side becomes VP.
        asn = "target_asn" if "target_asn" in df.columns else None
        country = "target_country" if "target_country" in df.columns else None
        return "target_id", "target_lat", "target_lon", asn, country

    def _target_columns(self) -> tuple[str, str, str]:
        """Resolve which CSV columns play the target role for this setup."""
        if self._setup == DataSource.PROBES_TO_ANCHORS:
            return "target_id", "target_lat", "target_lon"
        return "vp_id", "vp_lat", "vp_lon"

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            df = pd.read_csv(self._csv_path)
            missing = [c for c in _REQUIRED if c not in df.columns]
            if missing:
                raise ValueError(
                    f"CSV {self._csv_path} missing required columns: {missing}. "
                    f"Required: {list(_REQUIRED)}; optional: {list(_OPTIONAL)}."
                )
            df = df.dropna(subset=list(_REQUIRED))
            df = df[df["rtt_ms"] > 0].copy()
            df = self._apply_slice(df, self._slice)
            self._df = df.reset_index(drop=True)
        return self._df

    @staticmethod
    def _apply_slice(df: pd.DataFrame, slice_name: str) -> pd.DataFrame:
        if slice_name == "all":
            return df
        if not slice_name.startswith("head"):
            raise ValueError(
                f"unknown slice {slice_name!r}; expected 'all' or 'head<k>'"
            )
        try:
            k = int(slice_name.removeprefix("head"))
        except ValueError as e:
            raise ValueError(f"invalid head-k slice: {slice_name!r}") from e
        if k < 1:
            raise ValueError(f"head-k must be >=1, got {k}")
        keep = sorted(df["target_id"].astype(str).unique())[:k]
        if not keep:
            raise ValueError(f"slice {slice_name!r} selected no targets (empty CSV?)")
        return df[df["target_id"].astype(str).isin(keep)].copy()


def _opt_int(value: object) -> Optional[int]:
    return int(value) if pd.notna(value) else None  # type: ignore[arg-type]


def _opt_str(value: object) -> Optional[str]:
    return str(value) if pd.notna(value) else None  # type: ignore[arg-type]
