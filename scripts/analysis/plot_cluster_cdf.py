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

For each combo it writes one figure overlaying CDFs of the **error distance to
the cluster centroid** — the great-circle gap from the prediction to the
centroid the *truth* snaps to (i.e. distance to the correct answer point):

  scored      every SUCCESS prediction (the curves cover scored rows only).
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
the headline classification accuracy, the within-R rate (error-to-centroid ≤ R,
the point-estimate scoring rule), coverage (scored vs failures), per-cohort
percentiles, and the answer-space floor (truth→nearest-centroid distance).

Accuracy and within-R are taken over the **total** target set: a CBG estimate is
only read for SUCCESS rows, and non-SUCCESS rows (FALLBACK ≈ the nearest-VP /
shortest-ping fallback, or hard failures) count as **inaccurate** rather than
being dropped — so a method that fails on hard targets is penalised, not
flattered. The error-distance CDF curves still cover scored (SUCCESS) rows only,
since failures have no real CBG prediction. The shortest-ping baseline spans
every target (it doesn't depend on CBG status), so it is directly comparable.
Honors the shared `--geo-level/--geo-value` filter.

Alongside each combo's PNG it writes a sibling machine-readable CSV
(``<combo_id>_cluster_cdf.csv`` in the same output dir) with one row per CDF line
drawn (cohorts ``scored``/``matched``/``mismatched``/``shortest_ping``) carrying
``combo_id, cohort, n, p5, p25, p50, p75, p95, frac_within_radius, radius_km`` —
so other tools can read the percentiles and the within-R crossing without
OCR-ing the figure. ``frac_within_radius`` is the CDF y-value where the curve
crosses x = ``radius_km``.

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
from matplotlib.ticker import ScalarFormatter

from scripts.analysis._v2_io import (
    add_geo_filter_args,
    analysis_out_dir,
    resolve_run_dir,
    route_geo_path,
    set_geo_filter_from_args,
)
from scripts.analysis._cluster_data import (  # noqa: E402
    _CentroidIndex,
    _load_scored_baseline,
    _read_meta,
    build_answer_space,
    combo_frame,
    geo_allowed_ids,
    resolve_inputs_dir,
    shortest_ping_baseline_rates,
    shortest_ping_to_centroid,
)

logger = logging.getLogger(__name__)

# A CBG prediction is only read for SUCCESS rows — a FALLBACK prediction IS the
# nearest-VP fallback (≈ the shortest-ping VP), so treating it as a CBG estimate
# would conflate CBG with the shortest-ping baseline this plot compares against.
# Non-SUCCESS rows are NOT dropped, though: they count as inaccurate in the
# total-target denominator (see `combo_frame`).
_PCTILES = (50, 90, 95, 99)
# Percentiles reported in the sibling per-combo CSV (one row per CDF line).
_CDF_CSV_PCTILES = (5, 25, 50, 75, 95)


def _cdf(ax, arr: np.ndarray, **kw) -> None:
    s = np.sort(arr[np.isfinite(arr)])
    if len(s):
        ax.plot(s, np.arange(1, len(s) + 1) / len(s), **kw)


def _pcts(arr: np.ndarray) -> dict[int, float]:
    s = arr[np.isfinite(arr)]
    if not len(s):
        return {p: float("nan") for p in _PCTILES}
    return {p: float(v) for p, v in zip(_PCTILES, np.percentile(s, _PCTILES))}


def _cohort_stats(arr: np.ndarray, radius_km: float) -> dict:
    """Summary of one CDF line for the sibling CSV: sample count, the
    `_CDF_CSV_PCTILES` percentiles, and `frac_within_radius` (the CDF y-value
    where the curve crosses x=radius_km, i.e. the share ≤ radius_km). All over
    the finite values of `arr`; percentiles and frac are NaN when empty."""
    s = np.asarray(arr, dtype=float)
    s = s[np.isfinite(s)]
    n = int(len(s))
    if n == 0:
        stats = {f"p{p}": float("nan") for p in _CDF_CSV_PCTILES}
        stats["frac_within_radius"] = float("nan")
    else:
        stats = {
            f"p{p}": float(v)
            for p, v in zip(_CDF_CSV_PCTILES, np.percentile(s, _CDF_CSV_PCTILES))
        }
        stats["frac_within_radius"] = float((s <= radius_km).mean())
    stats["n"] = n
    return stats


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

    n_total = len(df)
    if n_total == 0:
        ax.text(0.5, 0.5, "no targets", transform=ax.transAxes,
                ha="center", va="center")
    else:
        err = df["error_to_centroid_km"].to_numpy(dtype=float)
        matched = df["match"].to_numpy(dtype=bool)
        finite = np.isfinite(err)            # SUCCESS rows with a scored prediction
        n_scored = int(finite.sum())
        n_fail = n_total - n_scored          # FALLBACK / hard failures (no CBG estimate)
        n_m = int(matched.sum())             # matched ⊆ scored
        n_mm = n_scored - n_m
        # Denominator is the TOTAL target set: failures count as inaccurate.
        acc = n_m / n_total
        within_r = int((err[finite] <= radius_km).sum()) / n_total

        _cdf(ax, err, color="#222222", linewidth=2.4, zorder=5,
             label=f"scored (n={n_scored})")
        _cdf(ax, err[matched], color="#2ca02c", linewidth=2.0, zorder=4,
             label=f"matched (n={n_m}, {acc:.1%} of {n_total})")
        _cdf(ax, err[finite & ~matched], color="#d62728", linewidth=2.0, zorder=3,
             label=f"mismatched (n={n_mm}, {n_mm / n_total:.1%} of {n_total})")

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

        p_all, p_m, p_mm = _pcts(err), _pcts(err[matched]), _pcts(err[finite & ~matched])
        floor = _pcts(df["truth_centroid_km"].to_numpy(dtype=float))
        box = (
            f"classification accuracy   {acc:6.1%}   (of {n_total} targets)\n"
            f"within R ({radius_km:.0f} km)        {within_r:6.1%}\n"
            f"scored {n_scored} / {n_total}  (failures: {n_fail})\n"
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
    parser.add_argument("--scored-dir", type=Path, default=None,
                        help="Pre-scored directory from `cluster-score` (per-combo *_scored.csv "
                             "and baseline.csv). When given, skips BallTree construction and "
                             "combo_frame computation.")
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

    if args.scored_dir is not None:
        scored_dir = Path(args.scored_dir)
        n_centroids, n_targets = (
            _read_meta(args.clusters_dir) if args.clusters_dir else (0, 0)
        )
        logger.info("answer space: %d targets → %d centroids (R=%.0f km)",
                    n_targets, n_centroids, args.radius_km)

        baseline_km = _load_scored_baseline(scored_dir)
        if baseline_km is not None:
            logger.info("baseline: %d targets from %s",
                        int(np.isfinite(baseline_km).sum()), scored_dir)
        else:
            logger.warning("no baseline.csv in %s; skipping shortest-ping baseline", scored_dir)

        scored_csvs = sorted(scored_dir.glob("*_scored.csv"))
        if not scored_csvs:
            logger.warning("no *_scored.csv files in %s", scored_dir)
            return

        logger.info("%-28s %8s %9s %9s", "combo", "n", "accuracy", "within_R")
        for csv_path in scored_csvs:
            combo_id = csv_path.stem[: -len("_scored")]
            df = pd.read_csv(csv_path)
            png_path = out_dir / f"{combo_id}_cluster_cdf.png"
            fig = plot_combo_cluster_cdf(
                df, png_path,
                combo_id=combo_id, radius_km=args.radius_km,
                n_centroids=n_centroids, baseline_km=baseline_km,
                max_x_km=args.max_x_km,
            )
            plt.close(fig)

            err = (df["error_to_centroid_km"].to_numpy(dtype=float)
                   if len(df) else np.array([], dtype=float))
            matched = (df["match"].to_numpy(dtype=bool)
                       if len(df) else np.array([], dtype=bool))
            finite = np.isfinite(err)
            cohorts: list[tuple[str, np.ndarray]] = [
                ("scored", err[finite]),
                ("matched", err[finite & matched]),
                ("mismatched", err[finite & ~matched]),
            ]
            if baseline_km is not None and np.isfinite(baseline_km).any():
                cohorts.append(("shortest_ping", baseline_km[np.isfinite(baseline_km)]))
            csv_rows = []
            for cohort, arr in cohorts:
                stats = _cohort_stats(arr, args.radius_km)
                csv_rows.append({
                    "combo_id": combo_id, "cohort": cohort, "n": stats["n"],
                    "p5": stats["p5"], "p25": stats["p25"], "p50": stats["p50"],
                    "p75": stats["p75"], "p95": stats["p95"],
                    "frac_within_radius": stats["frac_within_radius"],
                    "radius_km": args.radius_km,
                })
            cdf_csv = png_path.with_suffix(".csv")
            pd.DataFrame(csv_rows, columns=[
                "combo_id", "cohort", "n", "p5", "p25", "p50", "p75", "p95",
                "frac_within_radius", "radius_km",
            ]).to_csv(cdf_csv, index=False)
            logger.info("Saved: %s", cdf_csv)

            if len(df):
                acc = float(df["match"].mean())
                within = float((df["error_to_centroid_km"] <= args.radius_km).mean())
                logger.info("%-28s %8d %8.1f%% %8.1f%%",
                            combo_id, len(df), 100 * acc, 100 * within)

        logger.info("Wrote %d per-combo figures to %s", len(scored_csvs), out_dir)
        return

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
        png_path = out_dir / f"{combo_id}_cluster_cdf.png"
        fig = plot_combo_cluster_cdf(
            df, png_path,
            combo_id=combo_id, radius_km=args.radius_km,
            n_centroids=n_centroids, baseline_km=baseline_km, max_x_km=args.max_x_km,
        )
        plt.close(fig)

        # Sibling machine-readable CSV: one row per CDF line drawn in the figure.
        err = (df["error_to_centroid_km"].to_numpy(dtype=float)
               if len(df) else np.array([], dtype=float))
        matched = (df["match"].to_numpy(dtype=bool)
                   if len(df) else np.array([], dtype=bool))
        finite = np.isfinite(err)
        cohorts: list[tuple[str, np.ndarray]] = [
            ("scored", err[finite]),
            ("matched", err[finite & matched]),
            ("mismatched", err[finite & ~matched]),
        ]
        if baseline_km is not None and np.isfinite(baseline_km).any():
            cohorts.append(("shortest_ping", baseline_km[np.isfinite(baseline_km)]))
        csv_rows = []
        for cohort, arr in cohorts:
            stats = _cohort_stats(arr, args.radius_km)
            csv_rows.append({
                "combo_id": combo_id, "cohort": cohort, "n": stats["n"],
                "p5": stats["p5"], "p25": stats["p25"], "p50": stats["p50"],
                "p75": stats["p75"], "p95": stats["p95"],
                "frac_within_radius": stats["frac_within_radius"],
                "radius_km": args.radius_km,
            })
        csv_path = png_path.with_suffix(".csv")
        pd.DataFrame(csv_rows, columns=[
            "combo_id", "cohort", "n", "p5", "p25", "p50", "p75", "p95",
            "frac_within_radius", "radius_km",
        ]).to_csv(csv_path, index=False)
        logger.info("Saved: %s", csv_path)

        if len(df):
            acc = float(df["match"].mean())
            within = float((df["error_to_centroid_km"] <= args.radius_km).mean())
            logger.info("%-28s %8d %8.1f%% %8.1f%%", combo_id, len(df), 100 * acc, 100 * within)

    logger.info("Wrote %d per-combo figures to %s", len(grouped), out_dir)


if __name__ == "__main__":
    main()
