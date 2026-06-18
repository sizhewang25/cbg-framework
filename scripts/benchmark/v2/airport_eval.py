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

# Columns appended to each targets.parquet (not part of TARGETS_SCHEMA — the
# runner stays untouched; these live only in postprocessed files).
AIRPORT_COLUMNS = (
    "truth_airport_iata",
    "truth_airport_km",
    "pred_airport_iata",
    "pred_airport_km",
    "airport_match",
)

# Rows that count as a real prediction for summary purposes.
_SCORED_STATUSES = ("SUCCESS", "FALLBACK")

_STAT_Q = {"p5": 0.05, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p95": 0.95}


def annotate_targets(df: pd.DataFrame, index: AirportIndex) -> pd.DataFrame:
    """Append the five airport columns to a targets frame (idempotent).

    Truth columns are always populated (ground truth is always known); pred
    columns and `airport_match` are NULL where the prediction is missing.
    Re-running overwrites the columns in place, so the result is stable.
    """
    out = df.copy()

    truth_iata, truth_km = index.query_many(out["target_lat"], out["target_lon"])
    pred_iata, pred_km = index.query_many(out["pred_lat"], out["pred_lon"])

    out["truth_airport_iata"] = truth_iata
    out["truth_airport_km"] = truth_km
    out["pred_airport_iata"] = pred_iata
    out["pred_airport_km"] = pred_km

    match = pd.array([pd.NA] * len(out), dtype="boolean")
    has_pred = pd.notna(pred_iata)
    match[has_pred] = pred_iata[has_pred] == truth_iata[has_pred]
    out["airport_match"] = match

    return out


def summarize_airport(df: pd.DataFrame) -> dict:
    """Per-combo airport summary over SUCCESS/FALLBACK rows.

    Reports the match rate (the operator-facing headline) plus the usual
    p5..p95/mean/std block for `pred_airport_km`. Empty input is safe.
    """
    sub = df[df["status"].isin(_SCORED_STATUSES)]
    n = len(sub)

    summary: dict = {"n": n}

    if n:
        rate = sub["airport_match"].mean()
        summary["airport_match_rate"] = float(rate) if not pd.isna(rate) else float("nan")
    else:
        summary["airport_match_rate"] = float("nan")

    km = sub["pred_airport_km"].dropna() if n else pd.Series([], dtype=float)
    for stat in SUMMARY_STATS:
        if len(km) == 0:
            summary[f"pred_airport_km_{stat}"] = float("nan")
        elif stat == "mean":
            summary["pred_airport_km_mean"] = float(km.mean())
        elif stat == "std":
            summary["pred_airport_km_std"] = float(km.std())
        else:
            summary[f"pred_airport_km_{stat}"] = float(km.quantile(_STAT_Q[stat]))

    return summary


def process_parquet(path: Path, index: AirportIndex) -> dict:
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

    return summarize_airport(annotated)
