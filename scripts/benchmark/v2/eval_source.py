"""Score a canonical CSV's CBG-friendliness from geographic topology + RTT quality.

Consumes a canonical-schema CSV (one row per `(vp, target, rtt_ms)` observation
— see sources/README.md) and scores each target over its *available* VPs only:
a VP counts for a target iff the CSV holds >= 1 RTT observation for the pair,
so sparse meshes are scored on what a benchmark run would actually see.

Multiple observations of the same (vp, target) pair are collapsed to the
min-RTT one before aggregation (same convention as inspect_source's flow map).

Per-target metrics, grouped by axis:

  availability  n_avail_vps          distinct VPs with >= 1 observation
  geography     closest_vp_km        min geodesic VP distance — the floor any
                                     measurement could achieve with ideal routing
  RTT           min_rtt_ms           min RTT over available VPs
                best_radius_km       min_rtt_ms / THEORETICAL_SLOPE — the
                                     tightest CBG constraint radius the data
                                     supports (rtt/2 x 2/3c = rtt x 100 km)
                min_inflation        min over pairs of rtt / (slope x gc_km),
                                     routing efficiency decoupled from proximity
                                     (NaN when every pair is colocated)
  combined      rtt_weighted_dist_km inverse-RTT weighted mean VP distance,
                                     sum(gc/rtt) / sum(1/rtt) — the fleet's
                                     effective geographic distance as CBG
                                     experiences it: low-RTT VPs dominate,
                                     high-RTT VPs fade out. Bounded by
                                     [min gc, max gc]; parameter-free.

The dataset-level summary is a percentile block per metric plus a
"resolvability" table: the share of targets whose closest_vp_km /
best_radius_km / rtt_weighted_dist_km falls within each km threshold
(defaults match THRESHOLD_DISTANCES minus the exact-match 0).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.libs.cbg.rtt_model import THEORETICAL_SLOPE, haversine_distance

_REQUIRED = (
    "vp_id", "vp_lat", "vp_lon",
    "target_id", "target_lat", "target_lon",
    "rtt_ms",
)

DEFAULT_THRESHOLDS_KM: tuple[float, ...] = (40.0, 100.0, 500.0, 1000.0)

PER_TARGET_METRICS = (
    "n_avail_vps",
    "closest_vp_km",
    "min_rtt_ms",
    "best_radius_km",
    "min_inflation",
    "rtt_weighted_dist_km",
)

# Resolvability is reported on every distance-valued axis: the geography
# floor, the data-supported constraint radius, and the combined fleet view.
RESOLVABILITY_METRICS = ("closest_vp_km", "best_radius_km", "rtt_weighted_dist_km")

_PCTS = (5, 25, 50, 75, 95)


def load_canonical_csv(csv_path: Path) -> pd.DataFrame:
    """Load the required canonical columns, case-insensitively, dropping rows
    with missing values or non-positive RTTs (mirrors GenericCSVSource)."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is not a canonical CSV — missing columns: {missing}"
        )
    df = df[list(_REQUIRED)].copy()
    for col in ("vp_id", "target_id"):
        df[col] = df[col].astype(str)
    for col in ("vp_lat", "vp_lon", "target_lat", "target_lon", "rtt_ms"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(_REQUIRED))
    df = df[df["rtt_ms"] > 0].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"{csv_path}: no usable rows after dropping NaNs and non-positive RTTs"
        )
    return df


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (vp, target) pair — the min-RTT observation — with the
    derived per-pair geometry columns (gc_km, radius_km, inflation)."""
    pairs = (
        df.sort_values("rtt_ms", kind="stable")
        .drop_duplicates(["vp_id", "target_id"], keep="first")
        .reset_index(drop=True)
    )
    pairs["gc_km"] = haversine_distance(
        pairs["vp_lat"].to_numpy(), pairs["vp_lon"].to_numpy(),
        pairs["target_lat"].to_numpy(), pairs["target_lon"].to_numpy(),
    )
    pairs["radius_km"] = pairs["rtt_ms"] / THEORETICAL_SLOPE
    # Routing inflation vs the 2/3c physical floor; undefined for colocated
    # endpoints (same guard as partvp extract_features).
    ideal_ms = THEORETICAL_SLOPE * pairs["gc_km"]
    pairs["inflation"] = np.where(
        ideal_ms > 1e-9, pairs["rtt_ms"] / ideal_ms, np.nan
    )
    return pairs


def per_target_metrics(pairs: pd.DataFrame) -> pd.DataFrame:
    """Reduce the pair frame to one row per target (see module docstring)."""
    inv_rtt = 1.0 / pairs["rtt_ms"]
    work = pairs.assign(_w=inv_rtt, _wd=pairs["gc_km"] * inv_rtt)
    g = work.groupby("target_id", sort=True)
    out = pd.DataFrame({
        "target_lat": g["target_lat"].first(),
        "target_lon": g["target_lon"].first(),
        "n_avail_vps": g["vp_id"].nunique(),
        "closest_vp_km": g["gc_km"].min(),
        "min_rtt_ms": g["rtt_ms"].min(),
        "best_radius_km": g["radius_km"].min(),
        "min_inflation": g["inflation"].min(),
        "rtt_weighted_dist_km": g["_wd"].sum() / g["_w"].sum(),
    })
    return out.reset_index()


def _stat_block(values: pd.Series) -> dict[str, Any]:
    v = values.dropna().to_numpy(dtype=float)
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, _PCTS)
    return {
        "n": int(v.size),
        "min": round(float(v.min()), 3),
        "max": round(float(v.max()), 3),
        "mean": round(float(v.mean()), 3),
        "percentiles": {f"p{p}": round(float(x), 3) for p, x in zip(_PCTS, q)},
    }


def summarize(
    per_target: pd.DataFrame, thresholds: Sequence[float]
) -> dict[str, Any]:
    resolvability = {
        metric: {
            f"within_{t:g}km": round(float((per_target[metric] <= t).mean()), 4)
            for t in thresholds
        }
        for metric in RESOLVABILITY_METRICS
    }
    return {
        "n_targets": int(len(per_target)),
        "metrics": {m: _stat_block(per_target[m]) for m in PER_TARGET_METRICS},
        "resolvability": resolvability,
    }


def eval_source(
    csv_path: Path,
    out_dir: Path,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS_KM,
) -> dict[str, Any]:
    """Score one canonical CSV; write per-target CSV + stats JSON into out_dir
    (named after the CSV's stem) and return the stats dict with output paths."""
    df = load_canonical_csv(csv_path)
    pairs = build_pairs(df)
    per_target = per_target_metrics(pairs)

    stats: dict[str, Any] = {
        "csv": str(csv_path),
        "n_obs": int(len(df)),
        "n_pairs": int(len(pairs)),
        "n_vps": int(pairs["vp_id"].nunique()),
        **summarize(per_target, thresholds),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    per_target_path = out_dir / f"{csv_path.stem}_eval_per_target.csv"
    stats_path = out_dir / f"{csv_path.stem}_eval_stats.json"
    per_target.to_csv(per_target_path, index=False)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")

    stats["per_target_csv"] = str(per_target_path)
    stats["stats_json"] = str(stats_path)
    return stats
