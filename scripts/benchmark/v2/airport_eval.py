"""Closest-airport postprocessing over benchmark `targets.parquet` outputs.

This is intentionally **decoupled from the runner**: it annotates already-written
`targets.parquet` files in place, so it can be re-run any time the airport set,
filter, or match definition changes — and it backfills existing outputs without
re-running CBG. See notes/2026-06-18-closest-airport-eval-decisions.md.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmark.v2.airports import AirportIndex
from scripts.benchmark.v2.schema import SUMMARY_STATS
from scripts.libs.cbg.rtt_model import haversine_distance

# Columns appended to each targets.parquet (not part of TARGETS_SCHEMA — the
# runner stays untouched; these live only in postprocessed files).
AIRPORT_COLUMNS = (
    "truth_airport_iata",
    "truth_airport_km",
    "pred_airport_iata",
    "pred_airport_km",
    "pred_truth_airport_km",
    "airport_match",
)

# Rows that count as a real prediction for summary purposes.
_SCORED_STATUSES = ("SUCCESS", "FALLBACK")

# City-level threshold (km) for the forgiving match rate. 40 km matches this
# repo's THRESHOLD_DISTANCES and the geolocation literature's city-level bin.
DEFAULT_THRESHOLDS_KM = (40.0,)

_STAT_Q = {"p5": 0.05, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p95": 0.95}

# Continuous per-target distance columns that get the p5..p95/mean/std block.
_STAT_COLUMNS = ("pred_airport_km", "pred_truth_airport_km")


def annotate_targets(df: pd.DataFrame, index: AirportIndex) -> pd.DataFrame:
    """Append the five airport columns to a targets frame (idempotent).

    Truth columns are always populated (ground truth is always known); pred
    columns and `airport_match` are NULL where the prediction is missing.
    Re-running overwrites the columns in place, so the result is stable.
    """
    out = df.copy()

    truth_iata, truth_km, truth_lat, truth_lon = index.query_full(
        out["target_lat"], out["target_lon"]
    )
    pred_iata, pred_km, pred_lat, pred_lon = index.query_full(
        out["pred_lat"], out["pred_lon"]
    )

    out["truth_airport_iata"] = truth_iata
    out["truth_airport_km"] = truth_km
    out["pred_airport_iata"] = pred_iata
    out["pred_airport_km"] = pred_km

    # Great-circle gap between the two nearest airports. NaN propagates wherever
    # the prediction is missing (pred_lat/lon are NaN there), so this column is
    # NULL exactly when there is no prediction.
    out["pred_truth_airport_km"] = haversine_distance(
        pred_lat, pred_lon, truth_lat, truth_lon
    )

    match = pd.array([pd.NA] * len(out), dtype="boolean")
    has_pred = pd.notna(pred_iata)
    match[has_pred] = pred_iata[has_pred] == truth_iata[has_pred]
    out["airport_match"] = match

    return out


def _stat_block(series: pd.Series, prefix: str) -> dict:
    """p5..p95/mean/std for a distance series, NaN-filled when empty."""
    s = series.dropna()
    out: dict = {}
    for stat in SUMMARY_STATS:
        if len(s) == 0:
            out[f"{prefix}_{stat}"] = float("nan")
        elif stat == "mean":
            out[f"{prefix}_mean"] = float(s.mean())
        elif stat == "std":
            out[f"{prefix}_std"] = float(s.std())
        else:
            out[f"{prefix}_{stat}"] = float(s.quantile(_STAT_Q[stat]))
    return out


def summarize_airport(
    df: pd.DataFrame, thresholds=DEFAULT_THRESHOLDS_KM
) -> dict:
    """Per-combo airport summary over SUCCESS/FALLBACK rows.

    Reports two match rates — the strict `airport_match_rate` (exact nearest-IATA
    equality) and a forgiving `airport_match_rate_within_<T>km` per threshold
    (the pred/truth airports lie within `T` km, which absorbs multi-airport
    metros) — plus the p5..p95/mean/std block for both distance columns. The
    threshold is applied here, not baked into the per-target file. Empty input
    is safe.
    """
    sub = df[df["status"].isin(_SCORED_STATUSES)]
    n = len(sub)

    summary: dict = {"n": n}

    if n:
        rate = sub["airport_match"].mean()
        summary["airport_match_rate"] = float(rate) if not pd.isna(rate) else float("nan")
    else:
        summary["airport_match_rate"] = float("nan")

    gap = sub["pred_truth_airport_km"].dropna() if n else pd.Series([], dtype=float)
    for t in thresholds:
        key = f"airport_match_rate_within_{int(t)}km"
        summary[key] = float((gap <= t).mean()) if len(gap) else float("nan")

    for col in _STAT_COLUMNS:
        series = sub[col] if n else pd.Series([], dtype=float)
        summary.update(_stat_block(series, col))

    return summary


def process_parquet(
    path: Path, index: AirportIndex, thresholds=DEFAULT_THRESHOLDS_KM
) -> dict:
    """Annotate a single targets.parquet in place (atomic) and return its summary."""
    path = Path(path)
    df = pd.read_parquet(path)
    annotated = annotate_targets(df, index)

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".parquet")
    os.close(fd)
    try:
        annotated.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    return summarize_airport(annotated, thresholds=thresholds)
