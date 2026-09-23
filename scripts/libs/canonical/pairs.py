"""Observations collapsed to one row per `(vp, target)` flow, plus its geometry.

The step between the raw CSV and anything that reasons about a *flow*: multiple
observations of the same pair become the min-RTT one, and the derived columns
(`gc_km`, `radius_km`, `inflation`, `rtt_rank_norm`) are attached once so no
caller re-derives them against a second copy of `THEORETICAL_SLOPE`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.libs.cbg.rtt_model import THEORETICAL_SLOPE, haversine_distance

#: Below this the two endpoints are colocated and `inflation` is undefined —
#: dividing by an ideal time of ~0 would report an arbitrary multiple. Named
#: because `analysis/v3`'s per-VP inflation (map_mtl, pni) applies the same
#: guard and must not fork the number.
COLOCATED_IDEAL_MS = 1e-9


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (vp, target) pair — the min-RTT observation — with the
    derived per-pair columns (gc_km, radius_km, inflation, rtt_rank_norm)."""
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
        ideal_ms > COLOCATED_IDEAL_MS, pairs["rtt_ms"] / ideal_ms, np.nan
    )
    # Normalized RTT rank of each pair within its target: 0 = the target's
    # fastest VP, 1 = its slowest (0 for single-VP targets; ties share the
    # lower rank so a tied-fastest VP still ranks 0).
    grp = pairs.groupby("target_id")["rtt_ms"]
    n = grp.transform("size").to_numpy(dtype=float)
    rank = grp.rank(method="min").to_numpy(dtype=float) - 1.0
    pairs["rtt_rank_norm"] = np.where(n > 1, rank / np.maximum(n - 1, 1), 0.0)
    return pairs
