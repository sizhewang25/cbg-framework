"""Fleet geometry utilities for cluster-level VP proximity analysis.

Provides `compute_fleet_geometry`, which takes a centroid answer space and a
materialized inputs dir (eval_observations.parquet) and returns per-target
fleet geometry features:

  closest_vp_dist_km                — great-circle km from the closest available
                                      VP to the target's truth centroid
  target_distinguishable_vp_dist_km — cell_gap_km / 2: if closest_vp_dist_km is
                                      below this bound, the closest VP is
                                      guaranteed to favor the truth centroid over
                                      the nearest competing centroid
  target_distinguishable_vp_margin_km — bound minus closest_vp_dist_km; positive
                                      means the fleet has a target-distinguishing VP
  has_target_distinguishing_vp      — margin > 0
  cell_gap_km                       — distance to the nearest *other* centroid
  n_obs                             — unique VP count from eval_observations

Used by `classification_analysis.py` and `partvp/fleet_geometry_explainability.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.neighbors import BallTree

from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM

logger = logging.getLogger(__name__)


def _haversine_vec(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(
        lambda a: np.radians(np.asarray(a, dtype=float)),
        (lat1, lon1, lat2, lon2),
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _nearest_other_centroid_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """For each centroid, great-circle km to the nearest *other* centroid."""
    n = len(lat)
    if n < 2:
        return np.full(n, np.nan)
    tree = BallTree(np.radians(np.column_stack([lat, lon])), metric="haversine")
    d, _ = tree.query(np.radians(np.column_stack([lat, lon])), k=2)
    return d[:, 1] * EARTH_RADIUS_KM


def compute_fleet_geometry(
    inputs_dir: Path,
    index,
    allowed_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Per-target fleet geometry features.

    Parameters
    ----------
    inputs_dir : Path
        Directory containing ``eval_observations.parquet`` (or
        ``<fold>/eval_observations.parquet`` for k-fold layouts).
    index : _CentroidIndex
        Centroid answer space from ``build_answer_space``.
    allowed_ids : set[str] | None
        Restrict to these target_ids (e.g. a geo subset). None = all targets.

    Returns
    -------
    DataFrame indexed by target_id with columns:
        closest_vp_dist_km, cell_gap_km, target_distinguishable_vp_dist_km,
        target_distinguishable_vp_margin_km, has_target_distinguishing_vp, n_obs
    """
    _EMPTY = pd.DataFrame(columns=[
        "target_id", "closest_vp_dist_km", "cell_gap_km",
        "target_distinguishable_vp_dist_km", "target_distinguishable_vp_margin_km",
        "has_target_distinguishing_vp", "n_obs",
    ])

    direct = inputs_dir / "eval_observations.parquet"
    paths = ([direct] if direct.exists()
             else sorted(inputs_dir.glob("*/eval_observations.parquet")))
    if not paths:
        raise FileNotFoundError(f"no eval_observations.parquet under {inputs_dir}")

    obs = pd.concat([pq.read_table(p).to_pandas() for p in paths], ignore_index=True)
    obs = obs.drop_duplicates(["target_id", "vp_id"])

    if allowed_ids is not None:
        obs = obs[obs["target_id"].isin(allowed_ids)]
    if obs.empty:
        return _EMPTY

    # Truth centroid and cell gap per unique target
    unique_tgts = obs[["target_id", "target_lat", "target_lon"]].drop_duplicates("target_id").copy()
    t_idx, _ = index.query(unique_tgts["target_lat"].to_numpy(),
                           unique_tgts["target_lon"].to_numpy())
    unique_tgts["_t_idx"] = t_idx
    unique_tgts["truth_centroid_lat"] = index.lat[t_idx]
    unique_tgts["truth_centroid_lon"] = index.lon[t_idx]

    gap = _nearest_other_centroid_km(index.lat, index.lon)
    unique_tgts["cell_gap_km"] = gap[t_idx]

    obs = obs.merge(
        unique_tgts[["target_id", "truth_centroid_lat", "truth_centroid_lon", "cell_gap_km"]],
        on="target_id", how="inner",
    )
    obs["vp_to_centroid_km"] = _haversine_vec(
        obs["truth_centroid_lat"], obs["truth_centroid_lon"],
        obs["vp_lat"], obs["vp_lon"],
    )

    out = obs.groupby("target_id").agg(
        closest_vp_dist_km=("vp_to_centroid_km", "min"),
        cell_gap_km=("cell_gap_km", "first"),
        n_obs=("vp_id", "nunique"),
    ).reset_index()
    out["target_distinguishable_vp_dist_km"] = out["cell_gap_km"] / 2.0
    out["target_distinguishable_vp_margin_km"] = (
        out["target_distinguishable_vp_dist_km"] - out["closest_vp_dist_km"]
    )
    out["has_target_distinguishing_vp"] = out["target_distinguishable_vp_margin_km"] > 0
    return out
