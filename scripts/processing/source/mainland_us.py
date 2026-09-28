"""Mainland-US row filter for canonical CBG measurement CSVs.

Replaces the import `scripts.benchmark.v2.sources.atnt_ant`, which the
preprocessing pipeline depended on but which is no longer in the tree.

A measurement row survives only when **both** endpoints sit inside the
continental-US window — the same extent the plotting and analysis modules use
(`scripts/analysis/v4/modules/mapping.py:37`,
`scripts/visualization/cluster/plot_targets_vps.py:36`), so a row that passes
here is a row the maps can frame. Alaska, Hawaii and the territories fall
outside the box, which is the point: a VP fleet spanning them turns the CBG
constraint geometry into a different problem.

Country codes are checked when present (`vp_country` / `target_country`), so a
non-US point that happens to fall in the box — southern Ontario, northern
Mexico — is dropped too. Rows with the column absent or blank are judged on
coordinates alone.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

#: (lon_min, lon_max, lat_min, lat_max) — the repo's one continental-US frame.
US_MAINLAND_EXTENT = (-125.0, -66.0, 24.0, 50.0)


def _within_mainland(lat: pd.Series, lon: pd.Series) -> pd.Series:
    lon_min, lon_max, lat_min, lat_max = US_MAINLAND_EXTENT
    lat_n = pd.to_numeric(lat, errors="coerce")
    lon_n = pd.to_numeric(lon, errors="coerce")
    return (
        lat_n.between(lat_min, lat_max)
        & lon_n.between(lon_min, lon_max)
    ).fillna(False)


def _country_ok(series: pd.Series | None, index: pd.Index) -> pd.Series:
    """US-or-unknown predicate. An absent column is no evidence, so it passes."""
    if series is None:
        return pd.Series(True, index=index)
    cc = series.astype("string").str.strip().str.upper()
    return (cc.isna() | (cc == "") | (cc == "US")).fillna(True)


def filter_non_mainland_us_targets_and_vps(
    df: pd.DataFrame,
    *,
    vp_lat_col: str = "vp_lat",
    vp_lon_col: str = "vp_lon",
    target_lat_col: str = "target_lat",
    target_lon_col: str = "target_lon",
) -> pd.DataFrame:
    """Drop rows whose VP or target lies outside the mainland US.

    Column names are matched case-insensitively so both the upper-case
    (`VP_LAT`) and lower-case (`vp_lat`) canonical schemas work.
    """
    lower = {str(c).strip().lower(): c for c in df.columns}
    missing = [
        c for c in (vp_lat_col, vp_lon_col, target_lat_col, target_lon_col)
        if c.lower() not in lower
    ]
    if missing:
        raise ValueError(f"Missing required coordinate columns: {missing}")

    def col(name: str) -> pd.Series | None:
        src = lower.get(name.lower())
        return df[src] if src is not None else None

    keep = (
        _within_mainland(col(vp_lat_col), col(vp_lon_col))
        & _within_mainland(col(target_lat_col), col(target_lon_col))
        & _country_ok(col("vp_country"), df.index)
        & _country_ok(col("target_country"), df.index)
    )
    logger.info(
        "mainland-US filter: kept %d / %d rows (extent %s)",
        int(keep.sum()), len(df), US_MAINLAND_EXTENT,
    )
    return df[keep].copy()
