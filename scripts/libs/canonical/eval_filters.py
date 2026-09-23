"""The eval-side subset: which rows a benchmark run would actually evaluate.

`min_obs` and the two traffic knobs are applied in three places — the source
classes at materialize time, and the dataset precheck that claims to describe
what those sources will score. They have to agree exactly, or the precheck
describes a different dataset than the benchmark runs on. That agreement used
to be a comment asking three files to stay in lockstep; the kernel lives here
so it is one implementation instead.

The *denominator* deliberately still varies by caller — `traffic_weighted_csv`
derives its threshold over the whole mesh so every fold gets the same cut,
while `generic_presplit` derives it over its own test file — so callers pass
the frame they mean and the numeric derivation is shared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_eval_target_filters(
    df: pd.DataFrame,
    *,
    min_obs: int | None = None,
    eval_pair_weight_min: float | None = None,
    eval_kept_traffic_fraction: float | None = None,
) -> tuple[pd.DataFrame, float | None]:
    """Restrict `df` to the rows a real benchmark run's eval_observations.parquet
    would actually contain, mirroring TrafficWeightedCSVSource/GenericPresplitSource's
    materialize-time eval-side filters (see sources/generic_csv.py's
    `_apply_min_obs_filter` / `_apply_eval_weight_filter` /
    `_derive_eval_weight_min_from_fraction`). Without this, the precheck
    silently scores every row in the CSV even when `source_kwargs.min_obs` or
    the top-level `eval_pair_weight_min` / `eval_kept_traffic_fraction` yaml
    keys shrink what the benchmark actually evaluates.

    `min_obs` drops targets with fewer than that many rows (raw CSV row
    count, matching the source classes' per-target-id row count). Then, at
    most one of `eval_pair_weight_min` / `eval_kept_traffic_fraction` narrows
    to eval-surviving obs: a target survives iff >= 1 of its rows has
    `weight >= threshold`, and only rows clearing the threshold are kept for
    surviving targets — exactly what `iter_eval_targets` would emit.

    Returns (filtered_df, resolved_eval_pair_weight_min) — the second value
    is the threshold actually used (derived from `eval_kept_traffic_fraction`
    when that's what was passed), so callers can record it for transparency.
    """
    if eval_pair_weight_min is not None and eval_kept_traffic_fraction is not None:
        raise ValueError(
            "pass only one of eval_pair_weight_min or eval_kept_traffic_fraction"
        )
    if eval_pair_weight_min is not None and eval_pair_weight_min < 0:
        raise ValueError(f"eval_pair_weight_min must be >= 0, got {eval_pair_weight_min}")
    if eval_kept_traffic_fraction is not None and not (0 < eval_kept_traffic_fraction <= 1):
        raise ValueError(
            f"eval_kept_traffic_fraction must be in (0, 1], got {eval_kept_traffic_fraction}"
        )

    if min_obs is not None:
        counts = df.groupby("target_id")["target_id"].transform("count")
        before = df["target_id"].nunique()
        df = df[counts >= min_obs].reset_index(drop=True)
        after = df["target_id"].nunique()
        print(f"min_obs={min_obs}: {before} -> {after} targets")
        if df.empty:
            raise ValueError(f"min_obs={min_obs} left zero targets")

    if eval_kept_traffic_fraction is not None:
        eval_pair_weight_min = derive_eval_pair_weight_min(
            df, eval_kept_traffic_fraction
        )

    if eval_pair_weight_min is not None:
        thr = eval_pair_weight_min
        before = df["target_id"].nunique()
        surviving_targets = set(df.loc[df["weight"] >= thr, "target_id"].astype(str))
        df = df[
            df["target_id"].astype(str).isin(surviving_targets)
            & (df["weight"] >= thr)
        ].reset_index(drop=True)
        after = df["target_id"].nunique()
        print(
            f"eval_pair_weight_min={thr}: {before} -> {after} targets "
            f"({len(df)} surviving obs)"
        )
        if df.empty:
            raise ValueError(f"eval_pair_weight_min={thr} left zero eval obs")

    return df, eval_pair_weight_min


def derive_eval_pair_weight_min(
    df: pd.DataFrame, frac: float, *, log: bool = True
) -> float:
    """The weight threshold keeping `frac` of the traffic in `df`.

    KEYLESS — one `(vp_id, target_id)` flow is one row, with no
    `(vp_id, target_city)` dedup, so no city column is needed — and computed
    over **the frame it is handed**, descending cumulative sum to the requested
    kept fraction.

    *Which* frame that is, is the caller's decision and is not the same for
    everyone: `TrafficWeightedCSVSource` passes the whole mesh so the threshold
    is fold-independent, `GenericPresplitSource` passes its own test file, and
    the precheck passes the post-`min_obs` frame to match whichever source it
    is describing. Sharing the arithmetic while leaving the denominator to the
    caller is what keeps those three honest about their differences instead of
    hiding them in three copies of this function.
    """
    if "weight" not in df.columns:
        raise ValueError(
            "eval_kept_traffic_fraction needs a 'weight' column; "
            f"columns present: {list(df.columns)}"
        )
    weights = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).to_numpy(float)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError(
            f"eval_kept_traffic_fraction={frac} needs a traffic signal, but the "
            f"total weight over {len(weights)} flows is {total} — all weightless"
        )
    weights_sorted = np.sort(weights)[::-1]
    target = frac * total
    cum = np.cumsum(weights_sorted)
    idx = int(np.searchsorted(cum, target, side="left"))
    # Load-bearing clamp: cum sums the sorted array, total the original, so at
    # frac=1.0 float error can push searchsorted past the end.
    idx = min(idx, len(weights_sorted) - 1)
    threshold = float(weights_sorted[idx])
    if log:
        kept = weights >= threshold
        print(
            f"eval_kept_traffic_fraction={frac}: derived eval_pair_weight_min="
            f"{threshold:.12g} over {len(weights)} whole-mesh flows "
            f"(total weight {total:.12g}); kept_flows={int(kept.sum())} "
            f"({100 * weights[kept].sum() / total:.2f}% traffic)"
        )
    return threshold
