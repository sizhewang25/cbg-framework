"""DataSource adapters for v2 benchmark.

Each source maps an upstream measurement collection (CSV, ClickHouse, ...) into
the v2 framework's `FitSample` + per-target observation tuples. The benchmark's
materialize-inputs CLI command builds parquets from whichever source is named
via `--source`.
"""

from __future__ import annotations

from scripts.benchmark.v2.sources.base import DataSource, EvalTarget, VpConfig
from scripts.benchmark.v2.sources.generic_csv import GenericCSVSource
from scripts.benchmark.v2.sources.ripe_atlas import RipeAtlasSource
from scripts.benchmark.v2.sources.ripe_atlas_asn_corpora import (
    RipeAtlasASNCorporaSource,
)

SOURCES: dict[str, type[DataSource]] = {
    RipeAtlasSource.name: RipeAtlasSource,
    RipeAtlasASNCorporaSource.name: RipeAtlasASNCorporaSource,
    GenericCSVSource.name: GenericCSVSource,
}

__all__ = [
    "DataSource",
    "EvalTarget",
    "VpConfig",
    "RipeAtlasSource",
    "RipeAtlasASNCorporaSource",
    "GenericCSVSource",
    "SOURCES",
]
