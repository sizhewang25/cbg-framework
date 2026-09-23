"""The canonical `(vp, target, rtt)` CSV: its column contract and its reader.

**One owner of the schema, importable by both layers.** The benchmark reads
these CSVs to build a run's inputs; the analysis layer reads the same files to
describe the dataset and to resolve geometry. Before this module the reader
lived in `benchmark/v2/eval_source.py`, so five `analysis/v3` modules had to
reach down into the benchmark layer — with deferred, inside-function imports —
merely to parse their own input. Worse, the reader borrowed `raw_str` from
`sources/generic_csv.py`, which drags `scripts.framework.v2` (every CBG solver
the analysis layer exists to evaluate) in behind it: ~2 s and an inverted
dependency to read a CSV.

Nothing here may import `scripts.benchmark` or `scripts.analysis`. That is the
property the whole package exists to hold, and the one a test pins.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: The columns a CSV must have to be canonical. A row is one observation:
#: `(vp, target, rtt_ms)` with both endpoints' coordinates.
REQUIRED_COLUMNS = (
    "vp_id", "vp_lat", "vp_lon",
    "target_id", "target_lat", "target_lon",
    "rtt_ms",
)

#: Kept alongside the required columns, when present, so the eval-side filters
#: in `eval_filters` can be applied identically to how
#: TrafficWeightedCSVSource/GenericPresplitSource do it at materialize time.
#: `raw_str` opts target_city out of pandas' NA-sentinel coercion, same reason
#: as generic_csv.py's `_OPTIONAL_STR`.
OPTIONAL_FOR_FILTERS = ("weight", "target_city")

#: Optional metadata used for dataset characterization when available.
OPTIONAL_META = ("target_asn",)


def raw_str(value: str) -> str:
    """Identity converter — keeps a cell's literal text so pandas' default
    NA-sentinel coercion never fires on it (see generic_csv's `_OPTIONAL_STR`).

    Lives here rather than in `sources/generic_csv.py` because both the source
    classes and this module's reader need it, and importing it from there is
    what used to pull the whole benchmark stack into the analysis layer.
    """
    return value


def load_canonical_csv(csv_path: Path) -> pd.DataFrame:
    """Load the required canonical columns, case-insensitively, dropping rows
    with missing values or non-positive RTTs (mirrors TrafficWeightedCSVSource).

    Also keeps `weight` (normalized to a numeric >=0 column, defaulting to
    1.0 when absent — same two-default convention as generic_csv.py) and
    `target_city`, when either is present in the CSV, so
    `eval_filters.apply_eval_target_filters` can reproduce the materialize-time
    eval-side filters."""
    converters = {c: raw_str for c in ("target_city", "TARGET_CITY")}
    df = pd.read_csv(csv_path, converters=converters)
    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is not a canonical CSV — missing columns: {missing}"
        )
    keep = list(REQUIRED_COLUMNS)
    keep += [c for c in OPTIONAL_FOR_FILTERS if c in df.columns]
    keep += [c for c in OPTIONAL_META if c in df.columns]
    df = df[keep].copy()
    for col in ("vp_id", "target_id"):
        df[col] = df[col].astype(str)
    for col in ("vp_lat", "vp_lon", "target_lat", "target_lon", "rtt_ms"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    df = df[df["rtt_ms"] > 0].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"{csv_path}: no usable rows after dropping NaNs and non-positive RTTs"
        )
    if "weight" not in df.columns:
        df["weight"] = 1.0
    else:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
        if (df["weight"] < 0).any():
            raise ValueError(f"{csv_path}: weight must be >= 0")
    return df
