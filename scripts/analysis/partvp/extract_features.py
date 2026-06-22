"""Per-(combo, target) feature + confidence-tier extraction for the participating-VP study.

For one benchmark run, joins each target's CBG outcome (from targets.parquet,
pooled across folds) to:
  * the full VP observation set (eval_observations.parquet) — the *available*
    geometry, combo-independent; and
  * the MTL **participating** VPs (the new mtl_participants column) — the VPs
    that actually decided the intersection, combo-specific.

It then labels each target with a confidence tier and emits a tidy table — one
row per (combo_id, target_id) — for downstream box-plot / correlation /
decision-tree analysis.

Confidence tiers (centroid answer space, R = 50 km):
  tier1_high   SUCCESS, snaps to the truth's centroid, AND ≤ R of it.
  tier2_med    SUCCESS, snaps to the truth's centroid, but > R of it
               (the "tolerance dividend": right answer, imprecise point).
  tier3_low    everything else — mismatched centroid, FALLBACK, or ERROR.

Features (see README in this dir):
  Available-geometry (combo-independent):
    avail_min_vp_km     min great-circle target→VP over all observed VPs
    avail_min_rtt_ms    min RTT over all observed VPs (the shortest-ping signal)
    n_obs               number of observed VPs
  Participating-VP (combo-specific, the VPs deciding the region):
    n_part              number of participating VPs
    part_min_dist_km    min target→participant distance
    part_mean_dist_km / part_med_dist_km
    part_min_rtt_ms
    part_mean_rtt_ms / part_med_rtt_ms
    part_max_gap_deg    max angular gap between consecutive participants as seen
                        from the target (large ⇒ one-sided; small ⇒ surrounded)
    part_circ_var       circular variance of participant bearings (0 concentrated,
                        →1 evenly surrounded)
    part_mean_infl / part_min_infl   RTT inflation = measured / (slope·dist),
                        decouples congestion from raw distance
  Answer-space (target-level, combo-independent):
    truth_centroid_km        target → its own centroid (the floor)
    nearest_other_centroid_km  truth centroid → nearest *other* centroid
                        (isolation; small ⇒ crowded ⇒ easy to misclassify)

CLI:
    python -m scripts.analysis.partvp.extract_features \\
        --run-dir scripts/benchmark/v2/outputs/global_as16509_final \\
        --out scripts/analysis/partvp/data/global_as16509_final.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.neighbors import BallTree

from scripts.analysis._v2_io import discover_combos, group_combos_by_id, load_targets
from scripts.analysis.plot_cluster_cdf import build_answer_space
from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM, THEORETICAL_SLOPE

logger = logging.getLogger(__name__)

RADIUS_KM = 50.0


def _haversine_vec(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized great-circle km between paired (lat1,lon1) and (lat2,lon2)."""
    lat1, lon1, lat2, lon2 = map(lambda a: np.radians(np.asarray(a, dtype=float)),
                                 (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _bearings_deg(tlat, tlon, vlats, vlons) -> np.ndarray:
    """Initial great-circle bearing (deg, [0,360)) from target to each VP."""
    tlat, tlon = np.radians(tlat), np.radians(tlon)
    vlats, vlons = np.radians(np.asarray(vlats)), np.radians(np.asarray(vlons))
    dlon = vlons - tlon
    x = np.sin(dlon) * np.cos(vlats)
    y = np.cos(tlat) * np.sin(vlats) - np.sin(tlat) * np.cos(vlats) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def _angular_features(bearings: np.ndarray) -> tuple[float, float]:
    """(max_gap_deg, circular_variance) of a set of bearings.

    max_gap_deg: largest arc with no VP (incl. wrap-around). 360 for a single
    VP; small when VPs ring the target. circular_variance: 1 - |mean unit
    vector|, 0 when all bearings coincide, →1 when evenly spread."""
    b = np.sort(np.asarray(bearings, dtype=float))
    n = len(b)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return 360.0, 0.0
    gaps = np.diff(b)
    wrap = 360.0 - (b[-1] - b[0])
    max_gap = float(max(gaps.max(), wrap))
    ang = np.radians(b)
    r = np.hypot(np.cos(ang).mean(), np.sin(ang).mean())
    return max_gap, float(1.0 - r)


def _nearest_other_centroid_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """For each centroid, great-circle km to the nearest *other* centroid."""
    n = len(lat)
    if n < 2:
        return np.full(n, np.nan)
    tree = BallTree(np.radians(np.column_stack([lat, lon])), metric="haversine")
    d, _ = tree.query(np.radians(np.column_stack([lat, lon])), k=2)
    return d[:, 1] * EARTH_RADIUS_KM


def _avail_geometry(inputs_dir: Path) -> pd.DataFrame:
    """Per target_id: avail_min_vp_km, avail_min_rtt_ms, n_obs — over ALL observed VPs."""
    direct = inputs_dir / "eval_observations.parquet"
    paths = [direct] if direct.exists() else sorted(inputs_dir.glob("*/eval_observations.parquet"))
    if not paths:
        raise FileNotFoundError(f"no eval_observations.parquet under {inputs_dir}")
    df = pd.concat([pq.read_table(p).to_pandas() for p in paths], ignore_index=True)
    df = df.drop_duplicates(["target_id", "vp_id"])
    df["vp_km"] = _haversine_vec(df["target_lat"], df["target_lon"], df["vp_lat"], df["vp_lon"])
    g = df.groupby("target_id")
    return pd.DataFrame({
        "avail_min_vp_km": g["vp_km"].min(),
        "avail_min_rtt_ms": g["latency_ms"].min(),
        "n_obs": g.size(),
    }).reset_index()


def _participant_features(row) -> dict:
    """Features over one target's participating-VP list."""
    parts = row["mtl_participants"]
    out = {k: np.nan for k in (
        "n_part", "part_min_dist_km", "part_mean_dist_km", "part_med_dist_km",
        "part_min_rtt_ms", "part_mean_rtt_ms", "part_med_rtt_ms",
        "part_max_gap_deg", "part_circ_var", "part_mean_infl", "part_min_infl")}
    if parts is None or len(parts) == 0:
        out["n_part"] = 0
        return out
    vlat = np.array([p["vp_lat"] for p in parts], dtype=float)
    vlon = np.array([p["vp_lon"] for p in parts], dtype=float)
    rtt = np.array([p["rtt_ms"] for p in parts], dtype=float)
    dist = _haversine_vec(np.full(len(parts), row["target_lat"]),
                          np.full(len(parts), row["target_lon"]), vlat, vlon)
    # RTT inflation vs theoretical 2/3c propagation (slope ms/km); guard dist→0.
    ideal = THEORETICAL_SLOPE * dist
    infl = np.where(ideal > 1e-9, rtt / ideal, np.nan)
    max_gap, circ_var = _angular_features(_bearings_deg(row["target_lat"], row["target_lon"], vlat, vlon))
    out.update({
        "n_part": len(parts),
        "part_min_dist_km": float(np.nanmin(dist)),
        "part_mean_dist_km": float(np.nanmean(dist)),
        "part_med_dist_km": float(np.nanmedian(dist)),
        "part_min_rtt_ms": float(np.nanmin(rtt)),
        "part_mean_rtt_ms": float(np.nanmean(rtt)),
        "part_med_rtt_ms": float(np.nanmedian(rtt)),
        "part_max_gap_deg": max_gap,
        "part_circ_var": circ_var,
        "part_mean_infl": float(np.nanmean(infl)),
        "part_min_infl": float(np.nanmin(infl)),
    })
    return out


def extract_run(run_dir: Path, inputs_dir: Path, clusters_dir: Path | None,
                radius_km: float = RADIUS_KM) -> pd.DataFrame:
    index, n_centroids, n_targets = build_answer_space(
        run_dir, None, None, radius_km, clusters_dir=clusters_dir)
    logger.info("answer space: %d targets → %d centroids", n_targets, n_centroids)
    near_other = _nearest_other_centroid_km(index.lat, index.lon)

    avail = _avail_geometry(inputs_dir).set_index("target_id")

    grouped = group_combos_by_id(discover_combos(run_dir, None, None))
    rows = []
    for combo_id, dirs in sorted(grouped.items()):
        df = pd.concat([load_targets(d).to_pandas() for d in dirs], ignore_index=True)
        if "mtl_participants" not in df.columns:
            logger.warning("%s: no mtl_participants column (stale) — skipping", combo_id)
            continue
        success = df["status"].eq("SUCCESS").to_numpy()
        t_idx, t_km = index.query(df["target_lat"], df["target_lon"])
        p_idx = np.full(len(df), -1)
        err_cen = np.full(len(df), np.nan)
        if success.any():
            sub = df[success]
            pi, _ = index.query(sub["pred_lat"], sub["pred_lon"])
            p_idx[success] = pi
            err_cen[success] = index.distance_to(sub["pred_lat"], sub["pred_lon"], t_idx[success])
        match = success & (p_idx == t_idx) & (p_idx >= 0)
        within = err_cen <= radius_km
        tier = np.where(match & within, "tier1_high",
                np.where(match & ~within, "tier2_med", "tier3_low"))

        for i, (_, r) in enumerate(df.iterrows()):
            feat = _participant_features(r)
            tid = r["target_id"]
            av = avail.loc[tid] if tid in avail.index else None
            rows.append({
                "run_id": run_dir.name,
                "combo_id": combo_id,
                "target_id": tid,
                "status": r["status"],
                "tier": tier[i],
                "match": bool(match[i]),
                "within_r": bool(within[i]) if np.isfinite(within[i]) else False,
                "error_km": float(r["error_km"]) if pd.notna(r["error_km"]) else np.nan,
                "error_to_centroid_km": float(err_cen[i]) if np.isfinite(err_cen[i]) else np.nan,
                "truth_centroid_km": float(t_km[i]),
                "nearest_other_centroid_km": float(near_other[t_idx[i]]) if t_idx[i] >= 0 else np.nan,
                "avail_min_vp_km": float(av["avail_min_vp_km"]) if av is not None else np.nan,
                "avail_min_rtt_ms": float(av["avail_min_rtt_ms"]) if av is not None else np.nan,
                "n_obs": int(av["n_obs"]) if av is not None else 0,
                **feat,
            })
    out = pd.DataFrame(rows)
    logger.info("extracted %d (combo,target) rows over %d combos", len(out), out["combo_id"].nunique())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--inputs-dir", type=Path, default=None,
                    help="Fold-parent inputs dir. Default: scripts/benchmark/v2/inputs/"
                         "ripe_atlas_asn_corpora/<run_id>/probes_to_anchors")
    ap.add_argument("--clusters-dir", type=Path, default=None,
                    help="Precomputed cluster-eval dir; if omitted, answer space is built in-process.")
    ap.add_argument("--radius-km", type=float, default=RADIUS_KM)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_dir = args.run_dir
    inputs_dir = args.inputs_dir or (
        Path("scripts/benchmark/v2/inputs/ripe_atlas_asn_corpora") / run_dir.name / "probes_to_anchors")
    clusters_dir = args.clusters_dir
    if clusters_dir is None:
        cand = run_dir / "ripe_atlas_asn_corpora" / "probes_to_anchors" / "clusters"
        clusters_dir = cand if (cand / "clusters.csv").exists() else None

    df = extract_run(run_dir, inputs_dir, clusters_dir, args.radius_km)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
