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
  * VultrCSVSource   — datasets/cbg_test/vultr_pings_us_only.csv
  * RipeAtlasSource  — ClickHouse, ping_10k_to_anchors (probes → anchors)
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


class DataSource(ABC):
    """Interface every benchmark data adapter must satisfy."""

    name: str  # short identifier — also the subdirectory name under inputs/

    @abstractmethod
    def iter_vp_configs(self) -> Iterator[VpConfig]: ...

    @abstractmethod
    def iter_fit_samples(self) -> Iterator[FitSample]: ...

    @abstractmethod
    def iter_eval_targets(self) -> Iterator[EvalTarget]: ...

    @abstractmethod
    def slice_id(self) -> str:
        """Short identifier of the dataset slice this source represents
        (e.g. 'top1', 'all_us', '723_anchors'). Used as the inputs/output
        directory name so different slices don't collide."""
