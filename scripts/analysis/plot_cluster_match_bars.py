"""Cluster-classification accuracy bar chart across all combos of one run.

The cluster counterpart of `plot_airport_match_bars.py`. Builds the centroid
answer space once from the run's pooled ground truth (see
`plot_cluster_cdf.build_answer_space`), then ranks every combo by its
**classification accuracy** — the share of scored predictions that snap to the
same cluster centroid as the truth (nearest-centroid Voronoi).

One figure, horizontal bars sorted descending (best on top). Each bar carries a
marker for the **within-R rate** (prediction within R km of the truth's
centroid — the point-estimate scoring rule); the gap between bar and marker is
the slack between "right centroid" and "within R of it". Two dotted gray vertical
reference lines mark the **shortest-ping-VP baseline** — the same-centroid
accuracy and within-R rate of using each target's closest-by-RTT VP as the
estimate — so you can see which combos beat the trivial baseline.

Rates pool SUCCESS+FALLBACK rows across folds per combo (disjoint K-fold test
sets). Honors the shared `--geo-level/--geo-value` filter.

CLI:
    python -m scripts.analysis.plot_cluster_match_bars \\
        --run-dir scripts/benchmark/v2/outputs/global_as16509_final --radius-km 50
    python -m scripts.analysis.plot_cluster_match_bars \\
        --config scripts/benchmark/v2/config/north_america_as7018_final.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts.analysis._v2_io import (
    add_geo_filter_args,
    analysis_out_dir,
    discover_combos,
    group_combos_by_id,
    resolve_run_dir,
    route_geo_path,
    set_geo_filter_from_args,
)
from scripts.analysis.plot_cluster_cdf import (
    build_answer_space,
    combo_frame,
    geo_allowed_ids,
    resolve_inputs_dir,
    shortest_ping_baseline_rates,
)

logger = logging.getLogger(__name__)


def compute_rates(run_dir: Path, radius_km: float, index, source=None, slice_=None) -> pd.DataFrame:
    """One row per combo_id: n, accuracy (snap match), within_r, against `index`."""
    grouped = group_combos_by_id(discover_combos(run_dir, source, slice_))
    rows = []
    for combo_id, dirs in grouped.items():
        df = combo_frame(dirs, index)
        n = len(df)
        rows.append({
            "combo_id": combo_id,
            "n": n,
            "accuracy": float(df["match"].mean()) if n else float("nan"),
            "within_r": float((df["error_to_centroid_km"] <= radius_km).mean()) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def plot_bars(
    rates: pd.DataFrame, out_path: Path, *, title: str, radius_km: float,
    baseline_acc: float = float("nan"), baseline_within: float = float("nan"),
    n_base: int = 0,
) -> plt.Figure:
    """Horizontal accuracy bars sorted descending, within-R marker per combo, and
    (when given) the shortest-ping-VP baseline as dotted gray vertical lines."""
    df = rates.sort_values("accuracy", ascending=True)  # ascending → best on top
    y = range(len(df))

    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(df) + 1.5)))
    ax.barh(list(y), df["accuracy"], color="#4E79A7", alpha=0.85,
            label="classification accuracy (same centroid)", zorder=2)
    ax.scatter(df["within_r"], list(y), color="#d62728", marker="D", s=28,
               zorder=3, label=f"within R ({radius_km:.0f} km of centroid)")

    if n_base and pd.notna(baseline_acc):
        ax.axvline(baseline_acc, color="#666666", linestyle=":", linewidth=1.8, zorder=1,
                   label=f"shortest-ping VP: same centroid ({baseline_acc:.1%}, n={n_base})")
        if pd.notna(baseline_within):
            ax.axvline(baseline_within, color="#aaaaaa", linestyle=":", linewidth=1.8, zorder=1,
                       label=f"shortest-ping VP: within R ({baseline_within:.1%})")

    for yi, (val, n) in enumerate(zip(df["accuracy"], df["n"])):
        if pd.notna(val):
            ax.text(val + 0.005, yi, f"{val:.1%}  (n={n})", va="center", fontsize=8)

    xs = [float(df["accuracy"].max()) if len(df) else 0.0]
    if pd.notna(baseline_acc):
        xs.append(baseline_acc)
    if pd.notna(baseline_within):
        xs.append(baseline_within)
    ax.set_xlim(0, min(1.0, max(0.1, max(xs) * 1.25)))
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["combo_id"], fontsize=8)
    ax.set_xlabel("rate over SUCCESS+FALLBACK targets", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

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
                        help="Output dir (default: scripts/analysis/outputs/<run_id>/cluster).")
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
    add_geo_filter_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_geo_filter_from_args(args)

    run_dir = resolve_run_dir(args.config, args.run_dir, args.outputs_root)
    out_dir = (
        route_geo_path(args.out_dir) if args.out_dir
        else analysis_out_dir(run_dir, "cluster")
    )

    index, n_centroids, n_targets = build_answer_space(
        run_dir, args.source, args.slice_, args.radius_km, clusters_dir=args.clusters_dir
    )
    logger.info("answer space: %d targets → %d centroids (R=%.0f km)",
                n_targets, n_centroids, args.radius_km)

    combo_dirs = discover_combos(run_dir, args.source, args.slice_)
    rates = compute_rates(run_dir, args.radius_km, index, args.source, args.slice_)

    # Shortest-ping-VP baseline (combo-independent): same-centroid accuracy +
    # within-R rate of using the closest-by-RTT VP as the estimate.
    base_acc = base_within = float("nan")
    n_base = 0
    inputs_dir = resolve_inputs_dir(run_dir, combo_dirs, args.inputs_root, args.inputs_dir)
    if inputs_dir is not None:
        try:
            base_acc, base_within, n_base = shortest_ping_baseline_rates(
                inputs_dir, index, args.radius_km, geo_allowed_ids(combo_dirs)
            )
            logger.info("baseline: shortest-ping VP same-centroid=%.1f%% within_R=%.1f%% (n=%d)",
                        100 * base_acc, 100 * base_within, n_base)
        except FileNotFoundError:
            logger.warning("no eval_observations under %s; skipping shortest-ping baseline", inputs_dir)
    else:
        logger.warning("no inputs dir resolved; skipping shortest-ping baseline "
                       "(pass --inputs-dir to enable)")

    fig = plot_bars(
        rates, out_dir / f"{run_dir.name}_cluster_accuracy.png",
        title=f"Cluster classification accuracy — {run_dir.name} "
              f"({n_centroids} centroids, R={args.radius_km:.0f} km)",
        radius_km=args.radius_km,
        baseline_acc=base_acc, baseline_within=base_within, n_base=n_base,
    )
    plt.close(fig)
    logger.info("Ranked %d combos to %s", len(rates), out_dir)


if __name__ == "__main__":
    main()
