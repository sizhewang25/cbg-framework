"""The site key — one spelling, so three commands cannot drift apart.

## Why this module exists

A target is not an independent observation. These meshes put roughly twenty IP
replicas at one operator facility: replicas of one coordinate share a seed,
share every VP distance, and receive an identical Shortest-Ping verdict. The
recovery, significance and stability commands all need to group by *site*, and
before this module there were four different spellings of that grouping in the
tree — v3's `places.assign_region_ids` (6 dp), `map_answer_space.distinct_sites`
(raw float equality), `tg_seed_id` (a grid class, not a site), and per-script
coordinate rounding. All four agree on today's data, which is exactly how a
drift of this kind survives long enough to matter.

## A site is `(run_id, target_lat, target_lon)`

The run id is **part of the key**, and that is a modelling decision rather than
a convenience. Operators geolocate ASN by ASN: a facility that serves two
autonomous systems poses two distinct geolocation problems, with different
targets, different routing and a separately-fit calibration. So the same
coordinate appearing in `as01` and `as03` is two sites, not one.

The measured consequence has to be published wherever a site count is, because
it is easy to misread. On the three mesh runs:

* per-run distinct coordinates are **20 / 22 / 23**, summing to **65 sites**;
* pooled distinct *coordinates* are **43** — 7 appear in all three runs, 8 in
  exactly two, 28 in one.

So "65 unique site coordinates" is false: 65 counts site-*appearances* across
three runs, and 22 of them are repeat appearances of 15 physical places. Both
numbers go in the manifest via `site_diagnostics`, so a reader cannot take 65
for a count of places.

## There is no region id to use instead

`datasets/final/*.mainland.sanitized.csv` carries only vp/target coordinates,
`vp_asn`, `vp_country` and `rtt_ms`. The reconstruction manifests beside them
list `target_city` and `target_asn` as **unrecoverable** — the upstream
`tg_configs.parquet` is gone. The coordinate pair is the only stable identity
available, and it is enough: the parquet coordinates match those CSVs exactly,
so nothing here needs to join back to the dataset files.

## Rounding is a guard, not a tuning knob

The distinct-coordinate count is **identical at 2 through 6 decimal places**.
Replicas carry byte-identical coordinates; they do not merely sit close
together. `SITE_DECIMALS` therefore protects a future dataset whose replicas
are jittered, and `site_diagnostics` reports the count at neighbouring
roundings so such a dataset announces itself instead of silently collapsing
distinct sites.

## `solved_mask` is re-exported, not reimplemented

Every consumer of this module also needs to know which rows a method actually
answered, and getting that wrong is the single most expensive mistake available
here: a FALLBACK row carries a real `ring`, because the fallback point is the
shortest-ping VP rather than a null. 827 of the 1,100 FALLBACK rows on these
runs sit at `ring == 0`. A `groupby("ring")` without the mask credits Vanilla
16.9% at nside-128 instead of 6.3%. Re-exporting `classify.solved_mask` means a
consumer that imports this module has the right predicate already in hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.v4.modules.classify import solved_mask

__all__ = [
    "MISSING_SITE",
    "SITE_COL",
    "SITE_COLUMNS",
    "SITE_DECIMALS",
    "SITE_KEY_COL",
    "UNIT_NOTE",
    "site_diagnostics",
    "site_ids",
    "site_key",
    "solved_mask",
]

#: The coordinate columns a site is keyed on, in every v4 frame.
SITE_COLUMNS = ("target_lat", "target_lon")

#: Emitted column names. `site_key` is the human-readable `<run>|<lat>,<lon>`;
#: `site_id` is the dense integer a groupby is cheaper on.
SITE_KEY_COL = "site_key"
SITE_COL = "site_id"

#: Rounding used to test coordinate equality. Six decimals is ~0.1 m, far below
#: any plausible geolocation precision, so it is effectively exact. See the
#: module docstring: every rounding from 2 to 6 agrees on today's data.
SITE_DECIMALS = 6

#: Roundings reported beside the chosen one, so a dataset whose site count
#: depends on the cutoff is visible rather than silently resolved.
_SENSITIVITY_DECIMALS = (2, 3, 4, 5, 6)

#: Id and key given to a row with no usable coordinate. Such rows group
#: together as one bucket and **stay in the denominator** — dropping them would
#: quietly change the population every rate is taken over.
MISSING_SITE = -1
_MISSING_KEY = "<missing>"

UNIT_NOTE = (
    "A site is one distinct target coordinate within one run: the key is "
    "(run_id, target_lat, target_lon). The run id is part of the key because "
    "operators geolocate ASN by ASN, so one facility serving two autonomous "
    "systems is two geolocation problems. Replicas of a site share a seed and "
    "every VP distance, so they are one observation repeated, not ~20 "
    "observations. A site is not an answer-space class (tg_seed_id), which is "
    "rung-dependent and merges distinct sites at coarse rungs."
)


def _rounded(df: pd.DataFrame, decimals: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """The rounded coordinate pair and a mask of rows missing one."""
    lat_col, lon_col = SITE_COLUMNS
    lat = pd.to_numeric(df[lat_col], errors="coerce").round(decimals)
    lon = pd.to_numeric(df[lon_col], errors="coerce").round(decimals)
    return lat, lon, lat.isna() | lon.isna()


def _run_labels(df: pd.DataFrame, run_id: str | None) -> pd.Series:
    """Per-row run id: an explicit argument wins, else a `run_id` column.

    Raising when neither is available is deliberate. Defaulting to a single
    anonymous run would silently merge two runs' sites into one — the exact
    collapse the key exists to prevent, and one that shows up only as a site
    count that is too small.
    """
    if run_id is not None:
        return pd.Series(str(run_id), index=df.index)
    if "run_id" in df.columns:
        return df["run_id"].astype(str)
    raise ValueError(
        "no run id: pass run_id=, or give the frame a 'run_id' column. A site "
        "key without one would merge the runs' sites, which is what "
        f"(run_id, {', '.join(SITE_COLUMNS)}) exists to prevent."
    )


def site_key(
    df: pd.DataFrame,
    *,
    run_id: str | None = None,
    decimals: int = SITE_DECIMALS,
) -> pd.Series:
    """`<run>|<lat>,<lon>` per row — the readable form of the key.

    Rows with a missing coordinate get `<missing>` rather than a coordinate
    string, so they bucket together instead of each becoming its own site.
    """
    lat, lon, missing = _rounded(df, decimals)
    runs = _run_labels(df, run_id)
    keys = runs + "|" + lat.astype(str) + "," + lon.astype(str)
    return pd.Series(
        np.where(missing, _MISSING_KEY, keys), index=df.index, name=SITE_KEY_COL
    )


def site_ids(
    df: pd.DataFrame,
    *,
    run_id: str | None = None,
    decimals: int = SITE_DECIMALS,
) -> pd.Series:
    """Stable integer site id per row.

    Ids are assigned in **sorted** key order rather than in order of first
    appearance, so the same target set yields the same ids however the frame
    was sorted upstream — a property the regression tests depend on, and one
    that a `factorize` would not give. Missing coordinates get `MISSING_SITE`.
    """
    keys = site_key(df, run_id=run_id, decimals=decimals)
    distinct = sorted(set(keys) - {_MISSING_KEY})
    lookup = {k: i for i, k in enumerate(distinct)}
    ids = [MISSING_SITE if k == _MISSING_KEY else lookup[k] for k in keys]
    return pd.Series(ids, index=df.index, dtype=int, name=SITE_COL)


def _coordinate_overlap(frames: dict[str, pd.DataFrame], decimals: int) -> dict:
    """How many physical coordinates each number of runs shares.

    This is the number that stops `n_sites` being misread. Summing per-run site
    counts gives 65 on these meshes while only 43 physical places are involved,
    and nothing in a site-keyed table reveals that on its own.
    """
    per_run: dict[str, set[tuple[float, float]]] = {}
    for name, frame in frames.items():
        lat, lon, missing = _rounded(frame, decimals)
        per_run[name] = set(zip(lat[~missing], lon[~missing]))

    counts: dict[tuple[float, float], int] = {}
    for coords in per_run.values():
        for c in coords:
            counts[c] = counts.get(c, 0) + 1

    shared: dict[str, int] = {}
    for n in sorted(set(counts.values())):
        shared[str(n)] = sum(1 for v in counts.values() if v == n)
    return {
        "n_distinct_coordinates": len(counts),
        "coordinates_by_run_count": shared,
        "note": (
            "coordinates_by_run_count maps 'how many runs hold this coordinate' "
            "-> 'how many coordinates'. n_sites sums the per-run counts and is "
            "therefore larger than n_distinct_coordinates whenever runs share a "
            "facility; both are published so neither is mistaken for the other."
        ),
    }


def site_diagnostics(
    frames: dict[str, pd.DataFrame],
    *,
    decimals: int = SITE_DECIMALS,
) -> dict:
    """Everything a manifest must say about the site key, as a dict.

    `frames` maps run id -> a per-target frame. Pass one entry per run; passing
    a single pre-concatenated frame would lose the per-run split that
    `n_sites_by_run` reports.
    """
    by_run = {}
    sizes_all = []
    for name, frame in sorted(frames.items()):
        ids = site_ids(frame, run_id=name, decimals=decimals)
        sizes = ids[ids >= 0].value_counts()
        by_run[name] = int(len(sizes))
        sizes_all.extend(sizes.tolist())

    combined = pd.concat(
        [f.assign(run_id=n) for n, f in sorted(frames.items())], ignore_index=True
    )
    by_dec = {}
    for d in _SENSITIVITY_DECIMALS:
        by_dec[int(d)] = int(site_key(combined, decimals=d).nunique())

    sizes_arr = np.asarray(sizes_all, dtype=float)
    return {
        "key": f"(run_id, {', '.join(SITE_COLUMNS)})",
        "decimals": int(decimals),
        "n_runs": len(frames),
        "n_rows": int(len(combined)),
        "n_sites": int(sum(by_run.values())),
        "n_sites_by_run": by_run,
        "replicas_per_site_mean": float(sizes_arr.mean()) if sizes_arr.size else float("nan"),
        "replicas_per_site_min": int(sizes_arr.min()) if sizes_arr.size else 0,
        "replicas_per_site_max": int(sizes_arr.max()) if sizes_arr.size else 0,
        "accuracy_quantum": (1.0 / sum(by_run.values())) if by_run else float("nan"),
        "n_sites_by_decimals": by_dec,
        "site_count_is_rounding_stable": len(set(by_dec.values())) == 1,
        **_coordinate_overlap(frames, decimals),
        "key_rationale": UNIT_NOTE,
    }
