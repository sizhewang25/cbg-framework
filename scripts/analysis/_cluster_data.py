"""Data-layer helpers for cluster-classification evaluation.

Extracted from `plot_cluster_cdf.py` so multiple visualization scripts can
import a single, consistent data layer without circular viz→viz dependencies.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.neighbors import BallTree

from scripts.analysis._v2_io import (
    active_geo_filter,
    discover_combos,
    geo_segment,
    group_combos_by_id,
    load_targets,
)
from scripts.benchmark.v2.sources.cluster_ground_truth import cluster_ground_truth
from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM, haversine_distance

logger = logging.getLogger(__name__)

_CBG_STATUSES = ("SUCCESS",)


class _CentroidIndex:
    """Nearest-centroid lookup (haversine `BallTree`), mirroring `AirportIndex`."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray) -> None:
        self.lat = np.asarray(lat, dtype=float)
        self.lon = np.asarray(lon, dtype=float)
        self._tree = BallTree(
            np.radians(np.column_stack([self.lat, self.lon])), metric="haversine"
        )

    def query(self, lats, lons) -> tuple[np.ndarray, np.ndarray]:
        """Nearest centroid index + km per point; idx=-1, km=NaN where invalid."""
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        n = lats.shape[0]
        idx = np.full(n, -1, dtype=int)
        km = np.full(n, np.nan, dtype=float)
        valid = ~(np.isnan(lats) | np.isnan(lons))
        if valid.any():
            d, i = self._tree.query(
                np.radians(np.column_stack([lats[valid], lons[valid]])), k=1
            )
            idx[valid] = i[:, 0]
            km[valid] = d[:, 0] * EARTH_RADIUS_KM
        return idx, km

    def distance_to(self, lats, lons, idx: np.ndarray) -> np.ndarray:
        """Great-circle km from each point to ``centroid[idx]`` (NaN where idx<0)."""
        idx = np.asarray(idx)
        ok = idx >= 0
        out = np.full(idx.shape[0], np.nan, dtype=float)
        if ok.any():
            out[ok] = haversine_distance(
                np.asarray(lats, dtype=float)[ok], np.asarray(lons, dtype=float)[ok],
                self.lat[idx[ok]], self.lon[idx[ok]],
            )
        return out


def _load_precomputed(clusters_dir: Path) -> tuple[_CentroidIndex, int, int]:
    """Read a `cluster-eval` results dir into (index, n_centroids, n_targets).

    When a geo filter is active, the matching per-geo subset
    (``<clusters_dir>/geo/<level>/<value>/``) is read instead of the global set."""
    cdir = Path(clusters_dir)
    seg = geo_segment()
    if seg is not None:
        cdir = cdir / seg
    cpath = cdir / "clusters.csv"
    if not cpath.exists():
        raise FileNotFoundError(
            f"{cpath} not found — run `python -m scripts.benchmark.v2.cli cluster-eval` "
            "first (with matching --geo-level/--geo-value if a geo filter is active)."
        )
    clusters = pd.read_csv(cpath)
    index = _CentroidIndex(
        clusters["centroid_lat"].to_numpy(), clusters["centroid_lon"].to_numpy()
    )
    meta = cdir / "meta.json"
    n_targets = (int(json.loads(meta.read_text())["n_targets"]) if meta.exists()
                 else int(clusters["n_members"].sum()))
    return index, len(clusters), n_targets


def build_answer_space(
    run_dir: Path, source, slice_, radius_km: float, clusters_dir: Path | None = None
) -> tuple[_CentroidIndex, int, int]:
    """The centroid answer space as (index, n_centroids, n_targets).

    With `clusters_dir`, loads a precomputed `cluster-eval` result (single source
    of truth, geo-subset aware). Otherwise clusters the run's pooled unique ground
    truth in process — the geo filter (if active) flows through `load_targets`, so
    the answer space matches the targets in scope."""
    if clusters_dir is not None:
        return _load_precomputed(clusters_dir)
    combo_dirs = discover_combos(run_dir, source, slice_)
    if not combo_dirs:
        raise FileNotFoundError(f"No combos found under {run_dir}")
    frames = [
        load_targets(d).to_pandas()[["target_id", "target_lat", "target_lon"]]
        for d in combo_dirs
    ]
    cat = pd.concat(frames, ignore_index=True).drop_duplicates("target_id")
    res = cluster_ground_truth(
        cat["target_lat"].to_numpy(), cat["target_lon"].to_numpy(), radius_km=radius_km
    )
    index = _CentroidIndex(res.centroid_lat, res.centroid_lon)
    return index, res.n_clusters, len(cat)


def combo_frame(combo_dirs: list[Path], index: _CentroidIndex) -> pd.DataFrame:
    """One row per EVAL target: success flag, match, error-to-centroid, floor.

    Pools **all** target rows across folds (disjoint K-fold test sets). A CBG
    prediction is only read for SUCCESS rows; non-SUCCESS rows (FALLBACK ≈ the
    nearest-VP/shortest-ping fallback, or hard failures) carry `match=False` and
    `error_to_centroid_km=NaN`. This makes the denominator the **total** target
    set — so `df["match"].mean()` is accuracy over all targets (failures count as
    inaccurate) and `(error_to_centroid_km <= R).mean()` is within-R over all
    targets, while NaN keeps non-SUCCESS rows out of the error-distance CDF.
    `match` is nearest-centroid Voronoi equality between prediction and truth;
    `error_to_centroid_km` is the gap from the prediction to the truth's centroid
    (the correct answer point); `truth_centroid_km` (the floor) is defined for
    every target."""
    frames = [load_targets(d).to_pandas() for d in combo_dirs]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(df) == 0:
        return pd.DataFrame(columns=[
            "success", "match", "error_to_centroid_km", "truth_centroid_km", "error_km"])

    n = len(df)
    success = df["status"].isin(_CBG_STATUSES).to_numpy()
    # Floor (truth → its own nearest centroid) is a property of the target, so
    # it is computed for every row regardless of CBG status.
    t_idx, t_km = index.query(df["target_lat"], df["target_lon"])

    match = np.zeros(n, dtype=bool)
    err_to_centroid = np.full(n, np.nan, dtype=float)
    if success.any():
        sub = df[success]
        p_idx, _p_km = index.query(sub["pred_lat"], sub["pred_lon"])
        ts = t_idx[success]
        match[success] = (ts == p_idx) & (p_idx >= 0)
        err_to_centroid[success] = index.distance_to(sub["pred_lat"], sub["pred_lon"], ts)

    return pd.DataFrame({
        "success": success,
        "match": match,
        "error_to_centroid_km": err_to_centroid,
        "truth_centroid_km": t_km,
        "error_km": df["error_km"].to_numpy(dtype=float),
    })


def _shortest_ping_rows(inputs_dir: Path) -> pd.DataFrame:
    """Per target, the row of its shortest-ping VP, from eval_observations.

    Reads ``<inputs_dir>/eval_observations.parquet`` directly, or globs
    ``<inputs_dir>/*/eval_observations.parquet`` (merged-fold mode), then keeps
    the min-`latency_ms` row per `target_id`. Columns: target_id, target_lat,
    target_lon, vp_lat, vp_lon."""
    direct = inputs_dir / "eval_observations.parquet"
    paths = [direct] if direct.exists() else sorted(inputs_dir.glob("*/eval_observations.parquet"))
    if not paths:
        raise FileNotFoundError(f"no eval_observations.parquet at {inputs_dir} or {inputs_dir}/*/")
    df = pd.concat([pq.read_table(p).to_pandas() for p in paths], ignore_index=True)
    if df.empty:
        return df
    idx = df.groupby("target_id")["latency_ms"].idxmin()
    cols = ["target_id", "target_lat", "target_lon", "vp_lat", "vp_lon"]
    return df.loc[idx, cols].reset_index(drop=True)


def shortest_ping_to_centroid(
    inputs_dir: Path, index: _CentroidIndex, allowed_ids: set[str] | None = None
) -> np.ndarray:
    """Baseline: great-circle distance from each target's shortest-ping VP to the
    centroid the truth snaps to — i.e. "use the closest-by-RTT VP as the estimate".

    `allowed_ids` restricts the target set (e.g. to a geo subset). Returns an
    array of distances (km), one per eligible target."""
    rows = _shortest_ping_rows(inputs_dir)
    if len(rows) == 0:
        return np.array([], dtype=float)
    if allowed_ids is not None:
        rows = rows[rows["target_id"].isin(allowed_ids)]
        if len(rows) == 0:
            return np.array([], dtype=float)
    t_idx, _ = index.query(rows["target_lat"], rows["target_lon"])
    return index.distance_to(rows["vp_lat"], rows["vp_lon"], t_idx)


def shortest_ping_baseline_rates(
    inputs_dir: Path, index: _CentroidIndex, radius_km: float,
    allowed_ids: set[str] | None = None,
) -> tuple[float, float, int]:
    """Scalar baseline for the bar chart: if the shortest-ping VP were the
    estimate, its (classification accuracy, within-R rate, n). Accuracy =
    fraction where the VP snaps to the truth's centroid; within-R = fraction
    where VP→truth-centroid ≤ R."""
    rows = _shortest_ping_rows(inputs_dir)
    if allowed_ids is not None:
        rows = rows[rows["target_id"].isin(allowed_ids)]
    n = len(rows)
    if n == 0:
        return float("nan"), float("nan"), 0
    v_idx, _ = index.query(rows["vp_lat"], rows["vp_lon"])
    t_idx, _ = index.query(rows["target_lat"], rows["target_lon"])
    acc = float(((v_idx == t_idx) & (v_idx >= 0)).mean())
    dist = index.distance_to(rows["vp_lat"], rows["vp_lon"], t_idx)
    finite = np.isfinite(dist)
    within = float((dist[finite] <= radius_km).mean()) if finite.any() else float("nan")
    return acc, within, n


def resolve_inputs_dir(
    run_dir: Path, combo_dirs: list[Path], inputs_root: Path, explicit: Path | None = None
) -> Path | None:
    """Inputs dir for the shortest-ping baseline: `explicit`, else auto-derived
    from the run layout (`<inputs_root>/<source>/<run_id>/<setup>/`, the fold
    parent). Returns None when nothing resolves."""
    if explicit is not None:
        return explicit
    if not combo_dirs:
        return None
    first = combo_dirs[0]
    cand = inputs_root / first.parents[2].name / run_dir.name / first.parents[1].name
    return cand if cand.exists() else None


def geo_allowed_ids(combo_dirs: list[Path]) -> set[str] | None:
    """Union of (geo-filtered) target_ids across combos, or None when no geo
    filter is active (so the baseline spans every target)."""
    if active_geo_filter() is None:
        return None
    allowed: set[str] = set()
    for d in combo_dirs:
        allowed |= set(load_targets(d).to_pandas()["target_id"].tolist())
    return allowed


def _read_meta(clusters_dir: Path) -> tuple[int, int]:
    """(n_centroids, n_targets) from a cluster-eval dir. Respects active geo filter."""
    cdir = Path(clusters_dir)
    seg = geo_segment()
    if seg is not None:
        cdir = cdir / seg
    meta_path = cdir / "meta.json"
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        return int(m["n_clusters"]), int(m["n_targets"])
    clusters = pd.read_csv(cdir / "clusters.csv")
    return len(clusters), int(clusters["n_members"].sum())


def _load_scored_baseline(scored_dir: Path) -> np.ndarray | None:
    """Load pre-computed baseline VP→centroid distances from scored_dir/baseline.csv."""
    path = Path(scored_dir) / "baseline.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)["vp_to_centroid_km"].to_numpy(dtype=float)
