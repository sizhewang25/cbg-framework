"""The site key: one spelling. Ported from v4's `sites.py`.

A **site** is a unique location of TGs. These meshes put ~20 TG replicas at one
operator facility; replicas of one site share every VP distance and are one
observation repeated, not twenty.

## A site is `(run_id, tg_lat, tg_lon)`

The run id is part of the key because operators geolocate ASN by ASN: one
facility serving two autonomous systems is two geolocation problems. So the
same coordinate in `as01` and `as03` is two sites. (Pooled, the three meshes
hold 65 sites at 43 distinct coordinates -- see v4's module for the counts.)

Replicas carry byte-identical coordinates; `SITE_DECIMALS` guards against a
future jittered dataset rather than tuning anything.

A site is not a seed. A seed groups sites that lie within one `grid_km` of each
other (complete linkage), so it is rung-dependent where a site is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The TG coordinate columns a site is keyed on, in every v5 frame.
SITE_COLUMNS = ("tg_lat", "tg_lon")

SITE_KEY_COL = "site_key"
SITE_COL = "site_id"

#: ~0.1 m: effectively exact. Every rounding from 2 to 6 agrees on today's data.
SITE_DECIMALS = 6

#: Given to a row with no usable coordinate. Such rows stay in the denominator.
MISSING_SITE = -1
_MISSING_KEY = "<missing>"


def _run_labels(df: pd.DataFrame, run_id: str | None) -> pd.Series:
    """Per-row run id: an explicit argument wins, else a `run_id` column.

    Raising when neither is available is deliberate: defaulting to one
    anonymous run would merge two runs' sites, the collapse the key prevents.
    """
    if run_id is not None:
        return pd.Series(str(run_id), index=df.index)
    if "run_id" in df.columns:
        return df["run_id"].astype(str)
    raise ValueError(
        "no run id: pass run_id=, or give the frame a 'run_id' column. A site "
        "key without one would merge the runs' sites."
    )


def site_key(
    df: pd.DataFrame, *, run_id: str | None = None, decimals: int = SITE_DECIMALS
) -> pd.Series:
    """`<run>|<lat>,<lon>` per row; `<missing>` where a coordinate is absent."""
    lat_col, lon_col = SITE_COLUMNS
    lat = pd.to_numeric(df[lat_col], errors="coerce").round(decimals)
    lon = pd.to_numeric(df[lon_col], errors="coerce").round(decimals)
    missing = lat.isna() | lon.isna()
    keys = _run_labels(df, run_id) + "|" + lat.astype(str) + "," + lon.astype(str)
    return pd.Series(
        np.where(missing, _MISSING_KEY, keys), index=df.index, name=SITE_KEY_COL
    )


def site_ids(
    df: pd.DataFrame, *, run_id: str | None = None, decimals: int = SITE_DECIMALS
) -> pd.Series:
    """Dense integer site id per row, assigned in **sorted** key order.

    Sorted rather than first-appearance order, so the same TG set yields the
    same ids however the frame was sorted upstream -- `seeds` relabels its
    groups by smallest site id and inherits this determinism.
    """
    keys = site_key(df, run_id=run_id, decimals=decimals)
    distinct = sorted(set(keys) - {_MISSING_KEY})
    lookup = {k: i for i, k in enumerate(distinct)}
    ids = [MISSING_SITE if k == _MISSING_KEY else lookup[k] for k in keys]
    return pd.Series(ids, index=df.index, dtype=int, name=SITE_COL)
