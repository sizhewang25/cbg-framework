"""VultrCSVSource — feeds the v2 benchmark from the Vultr US-only ping CSV.

Aligned with `RipeAtlasASNCorporaSource`: only the `anchors_to_probes` setup
is supported, and the slice grammar is `fold_N` driven by a DistGeo K-fold
stratification of the *target* probes (eval = fold N, fit = union of the
other K-1 folds). Stratification runs in-memory at load time — there's no
external JSON artifact since the CSV is single-file and colocated with
its own eval set.

  CSV column        →  v2 role
  ----------------------------
  prb_id            →  target_id (probe IP — the entity being geolocated)
  probe_lat/lon     →  target / probe_coord (probes are hard ground truth)
  dst_ip            →  vp_id (anchor IP — anchors are the VPs)
  anchor_lat/lon    →  vp_coord
  min_rtt           →  latency_ms·
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from scripts.benchmark.v2.sources.base import DataSource, EvalTarget, TgConfig, VpConfig
from scripts.framework.v2 import FitSample
from scripts.framework.v2.types import Coord, Latency, VpId
from scripts.processing.ripe_atlas.stratification import (
    AnchorInfo,
    DistGeoStratification,
)

logger = logging.getLogger(__name__)

DEFAULT_CSV = (
    Path(__file__).resolve().parents[4]
    / "datasets" / "vultr_pings_us_only.csv"
)

# Columns we actually need. The CSV has more, but we don't depend on the rest.
_REQUIRED = (
    "prb_id", "dst_ip", "min_rtt",
    "probe_latitude", "probe_longitude", "probe_asn", "probe_country",
    "anchor_latitude", "anchor_longitude",
)

_FOLD_SLICE_RE = re.compile(r"^fold_(\d+)$")


class VultrCSVSource(DataSource):
    """Vultr-US measurements, K-fold-stratified on the probe-target set.

    Slice grammar:
      - "fold_N" where N ∈ [0, k): fold N is the eval slice; folds 0..k-1 \\ {N}
        form the fit corpus. Stratification is deterministic in (k, seed,
        asn_bucket_top_n) via `DistGeoStratification`.

    Only `setup=anchors_to_probes` is supported (anchors are VPs, probes are
    targets). For the probes→anchors direction, use `ripe_atlas_asn_corpora`.
    """

    name = "vultr_csv"

    def __init__(
        self,
        slice: str,
        setup: str = DataSource.ANCHORS_TO_PROBES,
        csv_path: Optional[Path] = None,
        *,
        k: int = 5,
        seed: int = 42,
        asn_bucket_top_n: int = 20,
    ) -> None:
        if setup != DataSource.ANCHORS_TO_PROBES:
            raise ValueError(
                f"{self.name!r} only supports setup="
                f"{DataSource.ANCHORS_TO_PROBES!r}, got {setup!r}. "
                "For the probes→anchors direction use 'ripe_atlas_asn_corpora'."
            )
        match = _FOLD_SLICE_RE.match(slice)
        if not match:
            raise ValueError(
                f"slice must match 'fold_N' for {self.name!r} (got {slice!r}). "
                "Each fold of the probe stratification is a separate slice."
            )
        fold_index = int(match.group(1))
        if fold_index >= k:
            raise ValueError(
                f"slice fold index {fold_index} >= k={k} "
                f"(available: fold_0..fold_{k - 1})"
            )

        self._slice = slice
        self._setup = setup
        self._fold_index = fold_index
        self._k = k
        self._seed = seed
        self._asn_bucket_top_n = asn_bucket_top_n
        self._csv_path = Path(csv_path) if csv_path is not None else DEFAULT_CSV

        # Lazily populated by `_ensure_loaded`.
        self._df: Optional[pd.DataFrame] = None
        self._eval_targets: Optional[set[str]] = None
        self._fit_targets: Optional[set[str]] = None

    # ---- DataSource API ------------------------------------------------------

    def slice_id(self) -> str:
        return self._slice

    def setup_id(self) -> str:
        return self._setup

    def iter_vp_configs(self) -> Iterator[VpConfig]:
        # anchors_to_probes: VPs are anchors (dst_ip).
        df = self._ensure_loaded()
        for _, row in df.drop_duplicates("dst_ip").iterrows():
            anchor_asn = row.get("anchor_asn")
            anchor_country = row.get("anchor_country")
            yield VpConfig(
                vp_id=str(row["dst_ip"]),
                lat=float(row["anchor_latitude"]),
                lon=float(row["anchor_longitude"]),
                asn=int(anchor_asn) if pd.notna(anchor_asn) else None,
                country=str(anchor_country) if pd.notna(anchor_country) else None,
            )

    def iter_tg_configs(self) -> Iterator[TgConfig]:
        # Static catalog of every target the source knows about (eval ∪ fit).
        # The eval/fit split is enforced inside iter_eval_targets /
        # iter_fit_samples. Matches the convention at
        # ripe_atlas_asn_corpora.py:137-158.
        df = self._ensure_loaded()
        for _, row in df.drop_duplicates("prb_id").iterrows():
            yield TgConfig(
                tg_id=str(int(row["prb_id"])),
                lat=float(row["probe_latitude"]),
                lon=float(row["probe_longitude"]),
                asn=int(row["probe_asn"]) if pd.notna(row["probe_asn"]) else None,
                country=str(row["probe_country"]) if pd.notna(row["probe_country"]) else None,
                city=None,  # Vultr CSV has no probe_city column
            )

    def iter_fit_samples(self) -> Iterator[FitSample]:
        df = self._ensure_loaded()
        assert self._fit_targets is not None
        for row in df.itertuples(index=False):
            prb_id = str(int(row.prb_id))
            if prb_id not in self._fit_targets:
                continue
            yield FitSample(
                vp_id=VpId(str(row.dst_ip)),
                vp_coord=Coord(lat=float(row.anchor_latitude), lon=float(row.anchor_longitude)),
                probe_coord=Coord(lat=float(row.probe_latitude), lon=float(row.probe_longitude)),
                latency=Latency(float(row.min_rtt)),
            )

    def iter_eval_targets(self) -> Iterator[EvalTarget]:
        df = self._ensure_loaded()
        assert self._eval_targets is not None
        for prb_id, group in df.groupby("prb_id", sort=True):
            tg_id = str(int(prb_id))
            if tg_id not in self._eval_targets:
                continue
            first = group.iloc[0]
            true_coord = Coord(
                lat=float(first["probe_latitude"]),
                lon=float(first["probe_longitude"]),
            )
            obs = [
                (
                    VpId(str(r.dst_ip)),
                    Coord(lat=float(r.anchor_latitude), lon=float(r.anchor_longitude)),
                    Latency(float(r.min_rtt)),
                )
                for r in group.itertuples(index=False)
            ]
            yield EvalTarget(target_id=tg_id, true_coord=true_coord, obs=obs)

    # ---- internals -----------------------------------------------------------

    def _ensure_loaded(self) -> pd.DataFrame:
        if self._df is None:
            self._load_csv()
            self._apply_stratification()
        assert self._df is not None
        return self._df

    def _load_csv(self) -> None:
        df = pd.read_csv(self._csv_path)
        missing = [c for c in _REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(
                f"Vultr CSV {self._csv_path} missing required columns: {missing}"
            )
        df = df.dropna(subset=list(_REQUIRED))
        # Strictly positive RTTs only — predict() rejects RTT<=0 anyway.
        df = df[df["min_rtt"] > 0].copy()
        self._df = df.reset_index(drop=True)

    def _apply_stratification(self) -> None:
        """Stratify the target probes (unique prb_ids) into K folds via
        DistGeo and cache the eval / fit target-id sets."""
        assert self._df is not None
        unique = self._df.drop_duplicates("prb_id")
        targets: list[AnchorInfo] = []
        for _, row in unique.iterrows():
            asn = row.get("probe_asn")
            country = row.get("probe_country")
            targets.append(AnchorInfo(
                ip=str(int(row["prb_id"])),
                lat=float(row["probe_latitude"]),
                lon=float(row["probe_longitude"]),
                country=str(country) if pd.notna(country) else None,
                asn=int(asn) if pd.notna(asn) else None,
            ))

        algo = DistGeoStratification(
            k=self._k,
            fold_index=0,  # full assignment is fold-index-independent
            seed=self._seed,
            asn_bucket_top_n=self._asn_bucket_top_n,
        )
        fold_by_id = algo.compute_fold_assignments(targets)

        eval_targets: set[str] = set()
        fit_targets: set[str] = set()
        for tg_id, fold in fold_by_id.items():
            (eval_targets if fold == self._fold_index else fit_targets).add(tg_id)
        self._eval_targets = eval_targets
        self._fit_targets = fit_targets
        logger.info(
            "stratified %d probe-targets into K=%d folds: eval=fold_%d "
            "(%d targets), fit=union of %d other folds (%d targets)",
            len(targets), self._k, self._fold_index,
            len(eval_targets), self._k - 1, len(fit_targets),
        )
