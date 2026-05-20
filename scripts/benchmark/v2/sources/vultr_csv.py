"""VultrCSVSource — feeds the v2 benchmark from the Vultr US-only ping CSV.

Maps the v1 CSV schema into v2 abstractions:

  CSV column       →  v2 role
  ---------------------------
  prb_id           →  vp_id (stable RIPE Atlas probe identifier)
  probe_lat/lon    →  vp_coord
  dst_ip           →  target_id (anchor IP)
  anchor_lat/lon   →  target / probe_coord (anchors are hard ground truth)
  min_rtt          →  latency_ms

The ASN top-k slice logic is ported (not imported) from
[scripts/benchmark/v1/dataset.py](scripts/benchmark/v1/dataset.py) so v2 stays
independent of v1's combination-pipeline internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from scripts.benchmark.v2.sources.base import DataSource, EvalTarget, VpConfig
from scripts.framework.v2 import FitSample
from scripts.framework.v2.types import Coord, Latency, VpId

DEFAULT_CSV = (
    Path(__file__).resolve().parents[4]
    / "datasets" / "cbg_test" / "vultr_pings_us_only.csv"
)

# Columns we actually need. The CSV has more, but we don't depend on the rest.
_REQUIRED = (
    "prb_id", "dst_ip", "min_rtt",
    "probe_latitude", "probe_longitude", "probe_asn", "probe_country",
    "anchor_latitude", "anchor_longitude",
)


class VultrCSVSource(DataSource):
    """Vultr-US CBG measurements, sliced by top-k probe ASNs.

    slice values:
      - "all_us"    : every row (after column validation + NaN drop)
      - "top<k>"    : rows from the k ASNs with the most unique probes
                      (k ∈ 1..10). Mirrors v1's ranking deterministically.
    """

    name = "vultr_csv"

    def __init__(self, slice: str = "all_us", csv_path: Optional[Path] = None) -> None:
        self._slice = slice
        self._csv_path = Path(csv_path) if csv_path is not None else DEFAULT_CSV
        self._df: Optional[pd.DataFrame] = None  # lazy-loaded

    # ---- DataSource API ------------------------------------------------------

    def slice_id(self) -> str:
        return self._slice

    def iter_vp_configs(self) -> Iterator[VpConfig]:
        df = self._load()
        for _, row in df.drop_duplicates("prb_id").iterrows():
            yield VpConfig(
                vp_id=str(int(row["prb_id"])),
                lat=float(row["probe_latitude"]),
                lon=float(row["probe_longitude"]),
                asn=int(row["probe_asn"]) if pd.notna(row["probe_asn"]) else None,
                country=str(row["probe_country"]) if pd.notna(row["probe_country"]) else None,
            )

    def iter_fit_samples(self) -> Iterator[FitSample]:
        df = self._load()
        for row in df.itertuples(index=False):
            yield FitSample(
                vp_id=VpId(str(int(row.prb_id))),
                vp_coord=Coord(lat=float(row.probe_latitude), lon=float(row.probe_longitude)),
                probe_coord=Coord(lat=float(row.anchor_latitude), lon=float(row.anchor_longitude)),
                latency=Latency(float(row.min_rtt)),
            )

    def iter_eval_targets(self) -> Iterator[EvalTarget]:
        df = self._load()
        # Group by target anchor IP. Anchor coordinates are constant within a
        # group (same anchor across all probes), so we read true_coord from the
        # first row of each group.
        for dst_ip, group in df.groupby("dst_ip", sort=True):
            first = group.iloc[0]
            true_coord = Coord(
                lat=float(first["anchor_latitude"]),
                lon=float(first["anchor_longitude"]),
            )
            obs: list[tuple[VpId, Coord, Latency]] = [
                (
                    VpId(str(int(r.prb_id))),
                    Coord(lat=float(r.probe_latitude), lon=float(r.probe_longitude)),
                    Latency(float(r.min_rtt)),
                )
                for r in group.itertuples(index=False)
            ]
            yield EvalTarget(target_id=str(dst_ip), true_coord=true_coord, obs=obs)

    # ---- internals -----------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            df = pd.read_csv(self._csv_path)
            missing = [c for c in _REQUIRED if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Vultr CSV {self._csv_path} missing required columns: {missing}"
                )
            df = df.dropna(subset=list(_REQUIRED))
            # Strictly positive RTTs only — predict() rejects RTT<=0 anyway.
            df = df[df["min_rtt"] > 0].copy()
            df = self._apply_slice(df, self._slice)
            self._df = df.reset_index(drop=True)
        return self._df

    @staticmethod
    def _apply_slice(df: pd.DataFrame, slice_name: str) -> pd.DataFrame:
        if slice_name == "all_us":
            return df
        if not slice_name.startswith("top"):
            raise ValueError(f"unknown slice {slice_name!r}; expected 'all_us' or 'top<k>'")
        try:
            k = int(slice_name.removeprefix("top"))
        except ValueError as e:
            raise ValueError(f"invalid top-k slice: {slice_name!r}") from e
        if k < 1:
            raise ValueError(f"top-k must be >=1, got {k}")

        # Rank ASNs by unique probe count then row count (deterministic).
        ranked = (
            df.groupby("probe_asn")
              .agg(probes=("prb_id", "nunique"), rows=("prb_id", "size"))
              .reset_index()
              .sort_values(["probes", "rows", "probe_asn"], ascending=[False, False, True])
        )
        selected = ranked.head(k)["probe_asn"].tolist()
        if not selected:
            raise ValueError(f"slice {slice_name!r} selected no ASNs (empty CSV?)")
        return df[df["probe_asn"].isin(selected)].copy()
