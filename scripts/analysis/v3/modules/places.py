"""Region ids and cluster-aware statistics: the independent unit is the site.

A library, not a command. `build-proximity` emits the `region_id` this assigns,
and the reporting modules use `region_level_accuracy` to print a clustered rate
beside every target-level one.

## Why this exists

The three operator runs report 399 / 412 / 458 `target_id`s but hold only
**20 / 22 / 23 distinct coordinates** — roughly twenty IP replicas per facility.
Replicas of one coordinate share a seed, share every VP distance, and receive an
identical Shortest-Ping verdict, so they are one observation repeated, not
twenty observations. Three consequences follow and all three are easy to miss:

* **Accuracy is quantized.** With 20 regions the attainable top-1 rates on as01
  are multiples of ~5%. A "1.000" or a "0.000" in a stratum is eight regions or
  five regions, not 160 or 100 targets.
* **Confidence intervals taken over targets are far too narrow.** The effective
  sample size is the region count. A binomial interval on `n = 1269` claims
  roughly `sqrt(1269/65) ~ 4.4x` more precision than the data carries.
* **A per-target correlation is a correlation over ~20 points**, whatever `n`
  the table prints.

None of this makes the datasets wrong — a CDN genuinely serves many addresses
per facility, and the replication is a property of the deployment rather than an
artifact of sampling. It makes the *unit of analysis* wrong, which is a
reporting fix.

## Region identity is exact equality, not clustering

Measured on all three runs, the distinct-coordinate count is **identical at 2
through 6 decimal places** (20 / 22 / 23 throughout). The replicas do not merely
sit close together; they carry byte-identical coordinates. So `decimals` is a
guard against a future dataset whose replicas are jittered, not a tuning knob
with a defensible value today. `assign_region_ids` reports the count at
neighbouring roundings in `region_diagnostics` so a dataset that *is* sensitive
announces itself rather than silently collapsing distinct sites.

This is deliberately **not** the answer-space clustering of prior work, and not
`pni_region` (a US state) from the PNI CSV. Here a region is one distinct target
coordinate and its IP replicas. See SCHEMA.md.

## Stance on violations

This layer reports a violation as a statistic and does not raise -- the same
stance `proximity.py` takes with its diamond implications. A region whose
replicas disagree, or a fold split that leaks every region, is a finding about
the data; refusing to compute would hide it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Column this module emits and every consumer groups by.
REGION_COL = "region_id"

#: Coordinate rounding used to test equality. Six decimal places is ~0.1 m, far
#: below any plausible geolocation precision, so it is effectively exact. See
#: the module docstring: on today's data every rounding from 2 to 6 agrees.
DEFAULT_DECIMALS = 6

#: Roundings reported beside the chosen one, so a dataset whose region count
#: depends on the cutoff is visible rather than silently resolved.
_SENSITIVITY_DECIMALS = (2, 3, 4, 5, 6)

_UNIT_NOTE = (
    "A region is one distinct target coordinate and its IP replicas. It is not "
    "an answer-space cluster and not pni_region (a US state). Replicas share a "
    "seed and every VP distance, so they are one observation repeated."
)


def assign_region_ids(
    df: pd.DataFrame,
    *,
    lat_col: str = "target_lat",
    lon_col: str = "target_lon",
    decimals: int = DEFAULT_DECIMALS,
) -> pd.Series:
    """Stable integer region id per row, keyed on the rounded coordinate.

    Ids are assigned in sorted `(lat, lon)` order rather than in order of first
    appearance, so the same target set yields the same ids however the frame was
    sorted upstream. Rows with a missing coordinate get `-1`, which groups them
    together as one bucket and keeps them in the denominator -- dropping them
    would quietly change the population a rate is taken over.
    """
    lat = pd.to_numeric(df[lat_col], errors="coerce").round(decimals)
    lon = pd.to_numeric(df[lon_col], errors="coerce").round(decimals)
    missing = lat.isna() | lon.isna()

    keys = pd.MultiIndex.from_arrays([lat, lon])
    distinct = sorted({k for k, m in zip(keys, missing) if not m})
    lookup = {k: i for i, k in enumerate(distinct)}

    ids = np.array([-1 if m else lookup[k] for k, m in zip(keys, missing)], dtype=int)
    return pd.Series(ids, index=df.index, name=REGION_COL)


def region_diagnostics(
    df: pd.DataFrame,
    *,
    lat_col: str = "target_lat",
    lon_col: str = "target_lon",
    decimals: int = DEFAULT_DECIMALS,
) -> dict:
    """F1, as a dict: how many regions, how many replicas, how stable the count.

    `n_distinct_by_decimals` is the sensitivity check the module docstring
    describes. When every entry agrees, region identity is exact equality and
    the rounding carries no assumption.
    """
    ids = assign_region_ids(df, lat_col=lat_col, lon_col=lon_col, decimals=decimals)
    sizes = ids[ids >= 0].value_counts()
    by_dec = {
        int(d): int(
            len(
                df[[lat_col, lon_col]]
                .round({lat_col: d, lon_col: d})
                .dropna()
                .drop_duplicates()
            )
        )
        for d in _SENSITIVITY_DECIMALS
    }
    return {
        "n_rows": int(len(df)),
        "n_regions": int(len(sizes)),
        "n_rows_missing_coords": int((ids < 0).sum()),
        "rows_per_region_mean": float(sizes.mean()) if len(sizes) else float("nan"),
        "rows_per_region_min": int(sizes.min()) if len(sizes) else 0,
        "rows_per_region_max": int(sizes.max()) if len(sizes) else 0,
        "n_distinct_by_decimals": by_dec,
        "region_count_is_rounding_stable": len(set(by_dec.values())) == 1,
        "decimals": int(decimals),
        "accuracy_quantum": (1.0 / len(sizes)) if len(sizes) else float("nan"),
        "note": _UNIT_NOTE,
    }


def region_level_accuracy(
    df: pd.DataFrame,
    *,
    correct_col: str,
    region_col: str = REGION_COL,
    n_boot: int = 2000,
    seed: int = 20260914,
    alpha: float = 0.05,
) -> dict:
    """Target-level and region-level accuracy with a clustered-bootstrap CI.

    Two estimators, both reported, because they answer different questions:

    * `target_accuracy` -- the unweighted mean over rows. What the existing
      tables print. Correct as a description of the measured addresses.
    * `region_accuracy` -- the mean over regions of each region's own rate. What
      generalizes to another deployment, where the replica counts would differ.

    They diverge only when replica counts are unequal *and* correlate with
    correctness, so printing both makes that dependence visible.

    The interval is a **clustered bootstrap over regions** -- resample regions
    with replacement, recompute. A cluster-robust sandwich estimator is the
    usual choice and is deliberately not used: with ~20 clusters its asymptotics
    do not hold, and it would report an interval whose nominal coverage the data
    cannot support. The bootstrap is also approximate at this cluster count;
    `n_regions` is emitted beside every interval so it is read with that in
    mind, and `ci_is_underpowered` flags the case outright.

    `region_homogeneity` is the share of regions whose replicas all agree. It
    should be 1.0 wherever the verdict is a pure function of the coordinate; a
    value below 1.0 means something target-specific (RTT noise, a per-IP
    fallback) is entering, and the region-level number is then a mean of
    fractions rather than a mean of verdicts.
    """
    work = df[[region_col, correct_col]].copy()
    work[correct_col] = work[correct_col].astype(float)
    work = work[work[region_col] >= 0]

    n_rows = int(len(work))
    if n_rows == 0:
        return {
            "n_rows": 0,
            "n_regions": 0,
            "target_accuracy": float("nan"),
            "region_accuracy": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "ci_method": "clustered bootstrap over regions",
            "ci_is_underpowered": True,
            "region_homogeneity": float("nan"),
            "note": _UNIT_NOTE,
        }

    per_region = work.groupby(region_col)[correct_col].agg(["mean", "size"])
    n_regions = int(len(per_region))

    means = per_region["mean"].to_numpy(dtype=float)
    sizes = per_region["size"].to_numpy(dtype=int)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_regions, size=(int(n_boot), n_regions))
    # Resampling clusters means resampling their rows too, so the bootstrap
    # replicate of the *target*-level rate is size-weighted; the region-level
    # replicate is not. Both are carried so each CI matches its own estimator.
    boot_region = means[draws].mean(axis=1)
    num = (means[draws] * sizes[draws]).sum(axis=1)
    den = sizes[draws].sum(axis=1)
    boot_target = num / den

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "n_rows": n_rows,
        "n_regions": n_regions,
        "target_accuracy": float(work[correct_col].mean()),
        "region_accuracy": float(means.mean()),
        "ci_low": float(np.percentile(boot_region, lo)),
        "ci_high": float(np.percentile(boot_region, hi)),
        "target_ci_low": float(np.percentile(boot_target, lo)),
        "target_ci_high": float(np.percentile(boot_target, hi)),
        "ci_method": f"clustered bootstrap over regions, {n_boot} draws, seed {seed}",
        "ci_is_underpowered": n_regions < 30,
        "region_homogeneity": float(np.isin(means, (0.0, 1.0)).mean()),
        "accuracy_quantum": 1.0 / n_regions,
        "note": _UNIT_NOTE,
    }


def region_rate(region_ids, correct) -> tuple[float, int]:
    """Unweighted mean of per-region rates, and the region count.

    The cheap form of `region_level_accuracy` for use inside a stratum loop,
    where a bootstrap per cell would dominate the runtime and the interval is
    not what the table prints anyway. Same estimator, no interval.

    Returns `(nan, 0)` on an empty stratum rather than raising: an empty
    stratum is a fact about the run -- a flag with no variance produces one on
    every method -- and the surrounding table already carries `n`.

    **Read with care inside a stratum.** Strata subdivide regions, so a region
    can contribute a single address to one stratum while its other nineteen sit
    elsewhere -- and the cluster mean then weights that sliver like a whole
    site. Measured on as02's top-1 `selection_hit` with SoI CBG: eight regions
    at 1.000 plus one region contributing one wrong address, giving
    `accuracy` 0.993 against `region_accuracy` 0.889. Both are right for their
    own estimator; what is wrong is reading the region figure as "the target
    rate with replication removed". Over a *whole* run the two agree closely
    (max divergence 0.001 / 0.028 / 0.003 on as01/02/03), because there every
    region contributes all of its addresses. `n_regions` is emitted beside the
    rate so a stratum resting on a one-address region is visible.
    """
    ids = np.asarray(region_ids)
    hits = np.asarray(correct, dtype=float)
    keep = ids >= 0
    ids, hits = ids[keep], hits[keep]
    if ids.size == 0:
        return float("nan"), 0
    order = np.unique(ids)
    means = np.array([hits[ids == r].mean() for r in order], dtype=float)
    return float(means.mean()), int(order.size)


def fold_region_overlap(
    df: pd.DataFrame,
    *,
    fold_col: str = "fold",
    region_col: str = REGION_COL,
) -> dict:
    """F2: how many folds each region's replicas are spread across.

    The K-fold protocol splits `target_id`. Where replicas of one coordinate
    land in several folds, every test fold's geometry also appears in its
    training folds, so a location-sensitive calibration -- LTD is one -- is fit
    on the geometry it is scored on. `share_regions_in_all_folds` at 1.0 is the
    complete-leakage case.

    Reported, not raised. Whether it *moves* a number is an empirical question
    that needs a re-cut run to answer, and this statistic is what decides
    whether that run is worth its cost.
    """
    work = df[[region_col, fold_col]].dropna()
    work = work[work[region_col] >= 0]
    if work.empty:
        return {
            "n_regions": 0,
            "n_folds": 0,
            "share_regions_in_all_folds": float("nan"),
            "note": _UNIT_NOTE,
        }

    n_folds = int(work[fold_col].nunique())
    per_region = work.groupby(region_col)[fold_col].nunique()
    return {
        "n_regions": int(len(per_region)),
        "n_folds": n_folds,
        "folds_per_region_min": int(per_region.min()),
        "folds_per_region_p50": float(per_region.median()),
        "folds_per_region_max": int(per_region.max()),
        "n_regions_in_all_folds": int((per_region == n_folds).sum()),
        "share_regions_in_all_folds": float((per_region == n_folds).mean()),
        "share_regions_in_one_fold": float((per_region == 1).mean()),
        "note": (
            "Folds split target_id, not coordinate. A region spanning several "
            "folds puts its geometry in both train and test, so a "
            "location-sensitive calibration is fit on what it is scored on."
        ),
    }
