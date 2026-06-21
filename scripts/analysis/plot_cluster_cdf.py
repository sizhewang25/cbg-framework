"""Per-combo cluster-classification eval from v2 benchmark outputs.

The cluster counterpart of `plot_airport_cdf.py`: instead of snapping to airport
hubs, the finite answer space is the set of **ground-truth cluster centroids**
(`scripts.benchmark.v2.sources.cluster_ground_truth`, complete-linkage with a
centroid-radius cap). CBG accuracy is then read as a classification problem —
each prediction snaps to its nearest centroid, and it is *correct* when that is
the same centroid the truth snaps to (nearest-centroid Voronoi, so a perfect
prediction is always correct, exactly as the airport metric treats hubs).

The answer space is built once from the run's own ground truth: the unique
target coordinates pooled across every combo/fold (deduped by target_id) are
clustered at `--radius-km` (default 50). All combos share that single centroid
set, so their classification accuracies are directly comparable.

For each combo it writes one figure overlaying three CDFs of the **error
distance to the cluster centroid** — the great-circle gap from the prediction to
the centroid the *truth* snaps to (i.e. distance to the correct answer point):

  all         every scored prediction.
  matched     predictions that snap to the truth's centroid (correct class) —
              here this equals the prediction's own snap distance.
  mismatched  predictions that snap to a different centroid (wrong class) —
              strictly larger, the misclassification penalty.

A combo-independent **shortest-ping baseline** is overlaid as a dotted gray line:
the distance from each target's shortest-ping VP (min latency_ms in
eval_observations) to the centroid the truth snaps to — i.e. "use the closest-by-
RTT VP as the estimate". The inputs dir is auto-derived from the run layout (or
passed via --inputs-dir).

The legend carries each cohort's sample count and share, and a stats box reports
the headline classification accuracy (matched share), the within-R rate
(error-to-centroid ≤ R, the point-estimate scoring rule), per-cohort
percentiles, and the answer-space floor (truth→nearest-centroid distance).

CBG cohorts are **SUCCESS rows only**, pooled across folds (disjoint K-fold test
sets): a FALLBACK prediction is the nearest-VP fallback (≈ the shortest-ping VP),
so including it would conflate CBG with the baseline. The baseline itself spans
every target (it doesn't depend on CBG status). Honors the shared
`--geo-level/--geo-value` filter.

CLI:
    python -m scripts.analysis.plot_cluster_cdf \\
        --run-dir scripts/benchmark/v2/outputs/ripe-smoke-01 --radius-km 50
    python -m scripts.analysis.plot_cluster_cdf \\
        --config scripts/benchmark/v2/config/north_america_as7018_final.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.ticker import ScalarFormatter
from sklearn.neighbors import BallTree

from scripts.analysis._v2_io import (
    active_geo_filter,
    add_geo_filter_args,
    analysis_out_dir,
    discover_combos,
    geo_segment,
    group_combos_by_id,
    load_targets,
    resolve_run_dir,
    route_geo_path,
    set_geo_filter_from_args,
)
from scripts.benchmark.v2.sources.cluster_ground_truth import cluster_ground_truth
from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM, haversine_distance

logger = logging.getLogger(__name__)

# SUCCESS only — a FALLBACK prediction IS the nearest-VP fallback (≈ the
# shortest-ping VP), so counting it as a CBG result would conflate CBG with the
# shortest-ping baseline this plot compares against.
_CBG_STATUSES = ("SUCCESS",)
_PCTILES = (50, 90, 95, 99)


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
    """Per SUCCESS target: match flag, error-to-truth-centroid, truth→centroid floor.

    Pools SUCCESS rows only across folds (FALLBACK predictions are the nearest-VP
    fallback ≈ the shortest-ping baseline, so they would conflate CBG with it).
    `match` is nearest-centroid Voronoi equality between prediction and truth;
    `error_to_centroid_km` is the gap from the prediction to the truth's centroid
    (the correct answer point)."""
    frames = [load_targets(d).to_pandas() for d in combo_dirs]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    sub = df[df["status"].isin(_CBG_STATUSES)] if len(df) else df
    if len(sub) == 0:
        return pd.DataFrame(columns=["match", "error_to_centroid_km", "truth_centroid_km", "error_km"])

    t_idx, t_km = index.query(sub["target_lat"], sub["target_lon"])
    p_idx, _p_km = index.query(sub["pred_lat"], sub["pred_lon"])
    err_to_centroid = index.distance_to(sub["pred_lat"], sub["pred_lon"], t_idx)

    return pd.DataFrame({
        "match": (t_idx == p_idx) & (p_idx >= 0),
        "error_to_centroid_km": err_to_centroid,
        "truth_centroid_km": t_km,
        "error_km": sub["error_km"].to_numpy(dtype=float),
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


def _cdf(ax, arr: np.ndarray, **kw) -> None:
    s = np.sort(arr[np.isfinite(arr)])
    if len(s):
        ax.plot(s, np.arange(1, len(s) + 1) / len(s), **kw)


def _pcts(arr: np.ndarray) -> dict[int, float]:
    s = arr[np.isfinite(arr)]
    if not len(s):
        return {p: float("nan") for p in _PCTILES}
    return {p: float(v) for p, v in zip(_PCTILES, np.percentile(s, _PCTILES))}


def plot_combo_cluster_cdf(
    df: pd.DataFrame,
    out_path: Path,
    *,
    combo_id: str,
    radius_km: float,
    n_centroids: int,
    baseline_km: np.ndarray | None = None,
    max_x_km: float = 10000.0,
) -> plt.Figure:
    """Overlay all / matched / mismatched error-to-centroid CDFs for one combo.

    When `baseline_km` is given, the shortest-ping-VP→truth-centroid baseline is
    drawn as a dotted gray reference (combo-independent)."""
    fig, ax = plt.subplots(figsize=(9, 6.5))

    n = len(df)
    if n == 0:
        ax.text(0.5, 0.5, "no scored targets", transform=ax.transAxes,
                ha="center", va="center")
    else:
        err = df["error_to_centroid_km"].to_numpy(dtype=float)
        matched = df["match"].to_numpy(dtype=bool)
        n_m, n_mm = int(matched.sum()), int((~matched).sum())
        acc = n_m / n
        within_r = float((err <= radius_km).mean())

        _cdf(ax, err, color="#222222", linewidth=2.4, zorder=5,
             label=f"all (n={n})")
        _cdf(ax, err[matched], color="#2ca02c", linewidth=2.0, zorder=4,
             label=f"matched (n={n_m}, {acc:.1%})")
        _cdf(ax, err[~matched], color="#d62728", linewidth=2.0, zorder=3,
             label=f"mismatched (n={n_mm}, {1 - acc:.1%})")

        n_base = 0
        if baseline_km is not None:
            base = baseline_km[np.isfinite(baseline_km)]
            n_base = len(base)
            if n_base:
                _cdf(ax, base, color="#888888", linestyle=":", linewidth=1.8, zorder=2,
                     label=f"shortest-ping VP → centroid (n={n_base})")

        ax.axvline(radius_km, color="green", linestyle=":", alpha=0.6)
        ax.text(radius_km, 0.02, f" R={radius_km:.0f} km", color="green",
                fontsize=8, rotation=90, va="bottom", ha="left")

        p_all, p_m, p_mm = _pcts(err), _pcts(err[matched]), _pcts(err[~matched])
        floor = _pcts(df["truth_centroid_km"].to_numpy(dtype=float))
        box = (
            f"classification accuracy   {acc:6.1%}\n"
            f"within R ({radius_km:.0f} km)        {within_r:6.1%}\n"
            f"answer space: {n_centroids} centroids\n"
            f"truth→centroid p50/p90  {floor[50]:.0f}/{floor[90]:.0f} km\n"
            f"\n"
            f"{'err→centroid':<14} {'p50':>5} {'p90':>5} {'p95':>5}\n"
            f"{'all':<14} {p_all[50]:5.0f} {p_all[90]:5.0f} {p_all[95]:5.0f}\n"
            f"{'matched':<14} {p_m[50]:5.0f} {p_m[90]:5.0f} {p_m[95]:5.0f}\n"
            f"{'mismatched':<14} {p_mm[50]:5.0f} {p_mm[90]:5.0f} {p_mm[95]:5.0f}"
        )
        if baseline_km is not None and n_base:
            p_b = _pcts(baseline_km)
            box += f"\n{'shortest_ping':<14} {p_b[50]:5.0f} {p_b[90]:5.0f} {p_b[95]:5.0f}"
        ax.text(0.02, 0.98, box, transform=ax.transAxes, fontsize=7.5, va="top",
                ha="left", family="monospace",
                bbox=dict(boxstyle="round", facecolor="#eef7ee", alpha=0.95))

    ax.set_xscale("log")
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlim(1, max_x_km)
    ax.set_ylim(0, 1)
    ax.set_xlabel("error distance to cluster centroid (km)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(f"Cluster classification CDF — {combo_id}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", out_path)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None,
                        help="Benchmark config YAML; its run_id resolves the run dir.")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Explicit outputs/<run_id>/ (overrides --config).")
    parser.add_argument("--outputs-root", type=Path, default=None,
                        help="Override the outputs root used with --config.")
    parser.add_argument("--source", default=None, help="Filter combos by source name.")
    parser.add_argument("--slice", dest="slice_", default=None, help="Filter combos by slice id.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir (default: scripts/analysis/outputs/<run_id>/cluster/cdf).")
    parser.add_argument("--radius-km", type=float, default=50.0,
                        help="Cluster centroid-radius cap defining the answer space. Default 50.")
    parser.add_argument("--clusters-dir", type=Path, default=None,
                        help="Precomputed cluster-eval results dir (single source of truth). "
                             "Geo subset auto-resolved when a geo filter is active. "
                             "If omitted, the answer space is clustered in process.")
    parser.add_argument("--inputs-dir", type=Path, default=None,
                        help="Materialized inputs dir (or its fold parent) for the "
                             "shortest-ping VP baseline. Auto-derived from --inputs-root "
                             "+ the run layout when omitted.")
    parser.add_argument("--inputs-root", type=Path,
                        default=Path("scripts/benchmark/v2/inputs"),
                        help="Root of materialized inputs, used to auto-derive --inputs-dir.")
    parser.add_argument("--max-x-km", type=float, default=10000.0)
    add_geo_filter_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_geo_filter_from_args(args)

    run_dir = resolve_run_dir(args.config, args.run_dir, args.outputs_root)
    out_dir = (
        route_geo_path(args.out_dir) if args.out_dir
        else analysis_out_dir(run_dir, "cluster", "cdf")
    )

    index, n_centroids, n_targets = build_answer_space(
        run_dir, args.source, args.slice_, args.radius_km, clusters_dir=args.clusters_dir
    )
    logger.info("answer space: %d targets → %d centroids (R=%.0f km)",
                n_targets, n_centroids, args.radius_km)

    combo_dirs = discover_combos(run_dir, args.source, args.slice_)
    grouped = group_combos_by_id(combo_dirs)

    # Shortest-ping VP → truth-centroid baseline (combo-independent, computed
    # once). Inputs dir is explicit, or auto-derived from the run layout
    # (<inputs_root>/<source>/<run_id>/<setup>/, the fold parent).
    inputs_dir = resolve_inputs_dir(run_dir, combo_dirs, args.inputs_root, args.inputs_dir)
    baseline_km = None
    if inputs_dir is not None:
        allowed_ids = geo_allowed_ids(combo_dirs)
        try:
            baseline_km = shortest_ping_to_centroid(inputs_dir, index, allowed_ids)
            logger.info("baseline: shortest-ping VP → centroid over %d targets (inputs %s)",
                        int(np.isfinite(baseline_km).sum()), inputs_dir)
        except FileNotFoundError:
            logger.warning("no eval_observations under %s; skipping shortest-ping baseline", inputs_dir)
    else:
        logger.warning("no inputs dir resolved; skipping shortest-ping baseline "
                       "(pass --inputs-dir to enable)")

    logger.info("%-28s %8s %9s %9s", "combo", "n", "accuracy", "within_R")
    for combo_id, dirs in sorted(grouped.items()):
        df = combo_frame(dirs, index)
        fig = plot_combo_cluster_cdf(
            df, out_dir / f"{combo_id}_cluster_cdf.png",
            combo_id=combo_id, radius_km=args.radius_km,
            n_centroids=n_centroids, baseline_km=baseline_km, max_x_km=args.max_x_km,
        )
        plt.close(fig)
        if len(df):
            acc = float(df["match"].mean())
            within = float((df["error_to_centroid_km"] <= args.radius_km).mean())
            logger.info("%-28s %8d %8.1f%% %8.1f%%", combo_id, len(df), 100 * acc, 100 * within)

    logger.info("Wrote %d per-combo figures to %s", len(grouped), out_dir)


if __name__ == "__main__":
    main()
