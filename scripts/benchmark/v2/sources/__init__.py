"""DataSource adapters for v2 benchmark.

Each source maps an upstream measurement collection (CSV, ClickHouse, ...) into
the v2 framework's `FitSample` + per-target observation tuples. The benchmark's
materialize-inputs CLI command builds parquets from whichever source is named
via `--source`.
"""

from __future__ import annotations

from scripts.benchmark.v2.sources.base import DataSource, EvalTarget, VpConfig
from scripts.benchmark.v2.sources.ripe_atlas import RipeAtlasSource
from scripts.benchmark.v2.sources.vultr_csv import VultrCSVSource

SOURCES: dict[str, type[DataSource]] = {
    VultrCSVSource.name: VultrCSVSource,
    RipeAtlasSource.name: RipeAtlasSource,
}

__all__ = [
    "DataSource",
    "EvalTarget",
    "VpConfig",
    "VultrCSVSource",
    "RipeAtlasSource",
    "SOURCES",
]
