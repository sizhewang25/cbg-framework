"""DataSource ABC — the contract the benchmark's `materialize-inputs` step relies on.

A source surfaces the same (RTT, VP coord, target coord) triples in two views:

  * `iter_fit_samples()`   — flat stream of v2 `FitSample`s, used to call
                             `LTDModel.fit(samples)`. One sample per
                             (vp, target_with_known_coord, rtt) triple.

  * `iter_eval_targets()`  — per-target grouping of `(vp_id, vp_coord, rtt)`
                             observation tuples, used to call
                             `CBGModel.geolocate(obs)` once per target.

Splitting these two views is what lets one CSV row contribute simultaneously
to LTD training and to evaluation without the source needing to know whether
it's in a fit or eval phase.

Concrete subclasses currently:
  * GenericCSVSource         — canonical-schema CSV (vp_*, target_*, rtt_ms)
  * RipeAtlasSource          — ClickHouse, ping_10k_to_anchors (probes → anchors)
  * RipeAtlasASNCorporaSource — per-ASN VP corpus vs K-fold-stratified anchors
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

from scripts.framework.v2 import FitSample
from scripts.framework.v2.types import Coord, Latency, VpId


@dataclass(frozen=True)
class EvalTarget:
    """One target + the full set of VP observations available for it.

    target_id and true_coord are the geolocation ground truth (anchor coords
    for RIPE Atlas / Vultr — both have hard-GT anchors).
    """

    target_id: str
    true_coord: Coord
    obs: list[tuple[VpId, Coord, Latency]]


@dataclass(frozen=True)
class VpConfig:
    """One VP's static configuration as it should appear in vp_configs.parquet."""
    vp_id: str
    lat: float
    lon: float
    asn: int | None = None
    country: str | None = None
    city: str | None = None


@dataclass(frozen=True)
class TgConfig:
    """One target's static configuration as it should appear in tg_configs.parquet."""
    tg_id: str
    lat: float
    lon: float
    asn: int | None = None
    country: str | None = None
    city: str | None = None


class DataSource(ABC):
    """Interface every benchmark data adapter must satisfy."""

    name: str  # short identifier — also the subdirectory name under inputs/

    # Allowed values for the `setup` axis. Subclasses must accept all.
    #
    # PROBES_TO_ANCHORS / ANCHORS_TO_PROBES: VP and target sides differ —
    #   one is probes, one is anchors. Field mapping decides which.
    # ANCHORS_TO_ANCHORS: both sides are anchors (anchor mesh data). Field
    #   mapping is identical to PROBES_TO_ANCHORS (vp_id = src, target = dst);
    #   the separate name documents the data shape and lets callers (e.g.
    #   the speed-limit calibration) declare intent explicitly.
    PROBES_TO_ANCHORS = "probes_to_anchors"
    ANCHORS_TO_PROBES = "anchors_to_probes"
    ANCHORS_TO_ANCHORS = "anchors_to_anchors"
    ALLOWED_SETUPS = (PROBES_TO_ANCHORS, ANCHORS_TO_PROBES, ANCHORS_TO_ANCHORS)

    @abstractmethod
    def iter_vp_configs(self) -> Iterator[VpConfig]: ...

    @abstractmethod
    def iter_tg_configs(self) -> Iterator[TgConfig]: ...

    @abstractmethod
    def iter_fit_samples(self) -> Iterator[FitSample]: ...

    @abstractmethod
    def iter_eval_targets(self) -> Iterator[EvalTarget]: ...

    @abstractmethod
    def slice_id(self) -> str:
        """Short identifier of the dataset slice this source represents
        (e.g. 'top1', 'all_us', '723_anchors'). Used as the inputs/output
        directory name so different slices don't collide."""

    @abstractmethod
    def setup_id(self) -> str:
        """Which side of the (probe, anchor) pair is treated as the
        vantage point. One of `PROBES_TO_ANCHORS` (default, the IMC 2023
        primary direction) or `ANCHORS_TO_PROBES` (anchors-as-VPs pressure
        test). Becomes a directory level in inputs/outputs paths so the
        two configurations never collide."""
