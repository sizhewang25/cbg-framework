"""Inference-observable confidence levels from MTL-region vs answer-space overlap.

For each target, reconstruct the CBG **MTL feasible region** offline from the
persisted participating-VP annuli (`mtl_participants`: `vp_lat/lon`,
`[echoed_lower_km, echoed_upper_km]`) via the octant unweighted region builder,
then count how many answer-space **cluster disks** (uniform R km) the region
overlaps (`n_hit`). Combined with the point-estimate's distance to its nearest
hub (`d_hub`), this assigns a confidence level an operator could compute
**without ground truth**:

  L1 highest : n_hit == 1                      (region in exactly one answer cell)
  L2 high    : n_hit > 1  AND d_hub <  R        (ambiguous region, point in a hub)
  L3 mid     : n_hit > 1  AND d_hub >= R        (region present, point far from hub)
  L0 low/fail: n_hit == 0 (empty / no overlap) OR status in {FALLBACK, ERROR}

Priority order: L1 wins on intersection-count regardless of d_hub (a region
touching a single cell is unambiguous). L0 is observably one bucket; ground
truth is used **only in validation** to split it into "snaps right anyway"
(the recoverable case) vs "fail".

The region geometry is reconstructed in octant's equirectangular (lon=x, lat=y)
frame — faithful for the annulus method and the lower=0 circle special case;
`n_hit` itself is computed with haversine on the *sampled lat/lon points*, so
only the region shape carries planar error.

CLI:
    python -m scripts.analysis.partvp.region_confidence \\
        --run-dir scripts/benchmark/v2/outputs/global_as16509_final \\
        --out-csv scripts/analysis/outputs/partvp/analysis/region_confidence.csv \\
        --out-parquet scripts/analysis/outputs/partvp/data/region_confidence_global_as16509_final.parquet
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np

# Sampling uses non-power-of-2 batch sizes; the balance-property warning is noise.
warnings.filterwarnings("ignore", message=".*Sobol.*")
import pandas as pd
from sklearn.neighbors import BallTree

from scripts.analysis._v2_io import discover_combos, group_combos_by_id, load_targets
from scripts.analysis.plot_cluster_cdf import build_answer_space
from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM
from scripts.libs.octant.octant_geolocation import (
    AnnularConstraint,
    compute_feasible_region_unweighted,
    sample_points_in_region,
)

logger = logging.getLogger(__name__)

RADIUS_KM = 50.0
TEXTBOOK = ["vanilla_cbg", "million_scale_cbg", "octant_cbg", "spotter_cbg"]
LEVELS = ["L1", "L2", "L3", "L0"]


def _constraints_from_participants(parts) -> list[AnnularConstraint]:
    """Build annular constraints from a target's participating-VP list.

    Circle methods echo lower=0 (a disk); spotter/annulus echo a band — both are
    handled by the annulus formulation. Rows with missing coords/bounds are
    skipped, as are degenerate (outer<=inner) bands.
    """
    if parts is None or len(parts) == 0:
        return []
    out: list[AnnularConstraint] = []
    for p in parts:
        vlat, vlon = p.get("vp_lat"), p.get("vp_lon")
        upper, lower = p.get("echoed_upper_km"), p.get("echoed_lower_km")
        if vlat is None or vlon is None or upper is None:
            continue
        inner = float(lower) if lower is not None else 0.0
        outer = float(upper)
        if not (outer > inner) or not np.isfinite(outer):
            continue
        out.append(AnnularConstraint(
            landmark_lat=float(vlat), landmark_lon=float(vlon), landmark_ip=str(p.get("vp_id")),
            rtt_ms=float(p.get("rtt_ms") or 0.0),
            inner_radius_km=max(0.0, inner), outer_radius_km=outer, weight=1.0,
        ))
    return out


def _region_n_hit(parts, tree: BallTree, radius_km: float, rng: np.random.Generator,
                  n_samples: int) -> int:
    """Distinct cluster disks (radius R) the reconstructed MTL region overlaps.

    Samples lat/lon points inside the region and counts distinct centroids with
    a sampled point within R (haversine, via BallTree.query_radius). 0 when the
    region is empty/None.
    """
    constraints = _constraints_from_participants(parts)
    if not constraints:
        return 0
    region = compute_feasible_region_unweighted(constraints)
    if region is None or region.is_empty:
        return 0
    pts = sample_points_in_region(region, n_samples=n_samples, rng=rng)  # [lat, lon]
    if len(pts) == 0:
        c = region.centroid
        pts = np.array([[c.y, c.x]])  # shapely y=lat, x=lon
    rad = np.radians(pts[:, :2])
    neigh = tree.query_radius(rad, r=radius_km / EARTH_RADIUS_KM)
    hit: set[int] = set()
    for arr in neigh:
        hit.update(int(i) for i in arr)
    return len(hit)


def _assign_level(success: bool, n_hit: int, d_hub: float, radius_km: float) -> str:
    if not success or n_hit == 0:
        return "L0"
    if n_hit == 1:
        return "L1"
    # n_hit > 1
    if np.isfinite(d_hub) and d_hub < radius_km:
        return "L2"
    return "L3"


def process_combo(df: pd.DataFrame, index, tree: BallTree, radius_km: float,
                  rng: np.random.Generator, n_samples: int) -> pd.DataFrame:
    """One row per target: observable level + truth tier + the inputs behind them."""
    success = df["status"].eq("SUCCESS").to_numpy()
    t_idx, _ = index.query(df["target_lat"].to_numpy(), df["target_lon"].to_numpy())

    p_idx = np.full(len(df), -1)
    d_hub = np.full(len(df), np.nan)
    err_cen = np.full(len(df), np.nan)
    if success.any():
        sub = df[success]
        pi, pkm = index.query(sub["pred_lat"].to_numpy(), sub["pred_lon"].to_numpy())
        p_idx[success] = pi
        d_hub[success] = pkm  # pred -> nearest centroid (the hub)
        err_cen[success] = index.distance_to(
            sub["pred_lat"].to_numpy(), sub["pred_lon"].to_numpy(), t_idx[success])

    match = success & (p_idx == t_idx) & (p_idx >= 0)
    within = err_cen <= radius_km
    tier = np.where(match & within, "tier1", np.where(match & ~within, "tier2", "tier3"))

    rows = []
    parts_col = df["mtl_participants"].to_list()
    for i in range(len(df)):
        n_hit = _region_n_hit(parts_col[i], tree, radius_km, rng, n_samples) if success[i] else 0
        level = _assign_level(bool(success[i]), n_hit, float(d_hub[i]), radius_km)
        rows.append({
            "target_id": df["target_id"].iat[i],
            "status": df["status"].iat[i],
            "n_hit": int(n_hit),
            "d_hub_km": float(d_hub[i]) if np.isfinite(d_hub[i]) else np.nan,
            "level": level,
            "match": bool(match[i]),
            "within_r": bool(within[i]) if np.isfinite(within[i]) else False,
            "tier": tier[i],
            "error_to_centroid_km": float(err_cen[i]) if np.isfinite(err_cen[i]) else np.nan,
        })
    return pd.DataFrame(rows)


def calibration_table(per_target: pd.DataFrame) -> pd.DataFrame:
    """Per run x combo x level: counts + P(correct), P(tier1), tier mix."""
    rows = []
    for (run, combo), g in per_target.groupby(["run_id", "combo_id"]):
        n_total = len(g)
        n_correct = int(g["match"].sum())
        for lvl in LEVELS:
            sub = g[g["level"] == lvl]
            n = len(sub)
            rows.append({
                "run_id": run, "combo_id": combo, "level": lvl,
                "n": n,
                "frac": n / n_total if n_total else np.nan,
                "p_correct": sub["match"].mean() if n else np.nan,
                "p_tier1": (sub["tier"] == "tier1").mean() if n else np.nan,
                "p_tier2": (sub["tier"] == "tier2").mean() if n else np.nan,
                "p_tier3": (sub["tier"] == "tier3").mean() if n else np.nan,
                # coverage: share of all truly-correct targets captured by this level
                "recall_of_correct": (int(sub["match"].sum()) / n_correct) if n_correct else np.nan,
            })
    return pd.DataFrame(rows)


def process_run(run_dir: Path, clusters_dir: Path | None, radius_km: float,
                combos: list[str], n_samples: int, seed: int,
                max_targets: int | None = None) -> pd.DataFrame:
    index, n_centroids, n_targets = build_answer_space(
        run_dir, None, None, radius_km, clusters_dir=clusters_dir)
    logger.info("[%s] answer space: %d targets -> %d centroids", run_dir.name, n_targets, n_centroids)
    tree = BallTree(np.radians(np.column_stack([index.lat, index.lon])), metric="haversine")

    grouped = group_combos_by_id(discover_combos(run_dir, None, None, combos=combos))
    rng = np.random.default_rng(seed)
    out = []
    for combo_id, dirs in sorted(grouped.items()):
        df = pd.concat([load_targets(d).to_pandas() for d in dirs], ignore_index=True)
        if "mtl_participants" not in df.columns:
            logger.warning("[%s] %s: no mtl_participants (stale) — skipping", run_dir.name, combo_id)
            continue
        if max_targets is not None:
            df = df.head(max_targets)
        pt = process_combo(df, index, tree, radius_km, rng, n_samples)
        pt.insert(0, "run_id", run_dir.name)
        pt.insert(1, "combo_id", combo_id)
        out.append(pt)
        logger.info("[%s] %s: %d targets, level mix %s", run_dir.name, combo_id, len(pt),
                    pt["level"].value_counts().to_dict())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, action="append", required=True,
                    help="Benchmark run dir (repeatable).")
    ap.add_argument("--clusters-dir", type=Path, default=None,
                    help="Precomputed cluster-eval dir; default: <run>/.../clusters if present.")
    ap.add_argument("--radius-km", type=float, default=RADIUS_KM)
    ap.add_argument("--combos", nargs="+", default=TEXTBOOK)
    ap.add_argument("--n-samples", type=int, default=600,
                    help="Region-interior samples per target for the overlap count.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-targets", type=int, default=None, help="Cap per combo (quick test).")
    ap.add_argument("--out-csv", type=Path, required=True, help="Calibration table.")
    ap.add_argument("--out-parquet", type=Path, default=None, help="Per-target rows (optional).")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    per_target_all = []
    for run_dir in args.run_dir:
        clusters_dir = args.clusters_dir
        if clusters_dir is None:
            cand = run_dir / "ripe_atlas_asn_corpora" / "probes_to_anchors" / "clusters"
            clusters_dir = cand if (cand / "clusters.csv").exists() else None
        per_target_all.append(process_run(
            run_dir, clusters_dir, args.radius_km, args.combos,
            args.n_samples, args.seed, args.max_targets))

    per_target = pd.concat(per_target_all, ignore_index=True)
    cal = calibration_table(per_target)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    cal.to_csv(args.out_csv, index=False)
    logger.info("wrote %s", args.out_csv)
    if args.out_parquet is not None:
        args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
        per_target.to_parquet(args.out_parquet, index=False)
        logger.info("wrote %s", args.out_parquet)

    # Quick-read summary: per run x combo, L1 and L1+L2 precision/coverage.
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        logger.info("\n=== calibration (level x run x combo) ===\n%s",
                    cal.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
