"""Plot shortest-ping distance boxplots by success/failure category.

Given the per-target features CSV from
`scripts.analysis.shortest_ping.per_target_features`, this script draws grouped
boxplots with:
  x-axis: classification category (success, failure)
  y-axis: distance (km)
  boxes per category:
    - closest_vp_dist_km
    - shortest_ping_vp_dist_km

CLI example:
  python -m scripts.analysis.shortest_ping.plot_distance_boxplots \
      --features-csv scripts/analysis/outputs/config-test02/shortest_ping/config-test02_shortest_ping_features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


METRIC_COLS = [
    "closest_vp_dist_km",
    "shortest_ping_centroid_dist_km",
    "shortest_ping_centroid_dist_margin",
]
REQUIRED_COLS = ["cls_result", *METRIC_COLS]


def _validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")


def _to_category(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        ok = series
    else:
        ok = series.astype(str).str.lower().map({"true": True, "false": False})
    return ok.map({True: "success", False: "failure"})


def _boxplot_percentiles(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for category in ["success", "failure"]:
        subset = data.loc[data["category"] == category]
        for metric in METRIC_COLS:
            vals = pd.to_numeric(subset[metric], errors="coerce").dropna()
            if vals.empty:
                rows.append({
                    "category": category,
                    "metric": metric,
                    "n": 0,
                    "min": np.nan,
                    "p5": np.nan,
                    "p25": np.nan,
                    "p50": np.nan,
                    "p75": np.nan,
                    "p95": np.nan,
                    "max": np.nan,
                })
                continue

            rows.append({
                "category": category,
                "metric": metric,
                "n": int(vals.shape[0]),
                "min": float(vals.min()),
                "p5": float(vals.quantile(0.05)),
                "p25": float(vals.quantile(0.25)),
                "p50": float(vals.quantile(0.50)),
                "p75": float(vals.quantile(0.75)),
                "p95": float(vals.quantile(0.95)),
                "max": float(vals.max()),
            })
    out = pd.DataFrame(rows)
    numeric_cols = ["min", "p5", "p25", "p50", "p75", "p95", "max"]
    out[numeric_cols] = out[numeric_cols].round(2)
    return out


def plot_boxplots(
    df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path | None,
    out_csv: Path | None,
    log_y: bool,
) -> None:
    data = df.copy()
    data["category"] = _to_category(data["cls_result"])

    categories = ["success", "failure"]
    metric_labels = {
        "closest_vp_dist_km": "Closest VP dist",
        "shortest_ping_centroid_dist_km": "Shortest-ping centroid dist",
        "shortest_ping_centroid_dist_margin": "Shortest-ping centroid minus closest",
    }
    metric_colors = {
        "closest_vp_dist_km": "#4e79a7",
        "shortest_ping_centroid_dist_km": "#f28e2b",
        "shortest_ping_centroid_dist_margin": "#59a14f",
    }

    box_data: list[np.ndarray] = []
    positions: list[float] = []
    colors: list[str] = []

    group_gap = 4.0
    for gi, cat in enumerate(categories):
        base = gi * group_gap
        for mi, metric in enumerate(METRIC_COLS):
            vals = pd.to_numeric(
                data.loc[data["category"] == cat, metric], errors="coerce"
            ).dropna().to_numpy()
            box_data.append(vals)
            positions.append(base + mi)
            colors.append(metric_colors[metric])

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.65,
        whis=(5, 95),
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.2},
    )

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    centers = [1.0, group_gap + 1.0]
    ax.set_xticks(centers)
    ax.set_xticklabels(["success", "failure"])
    ax.set_ylabel("Distance (km)")
    ax.set_title("VP Distance by Shortest-Ping Classification Result")
    if log_y:
        # Use symlog to allow a visible 0 tick while keeping base-10 spacing.
        ax.set_yscale("symlog", linthresh=1.0, linscale=1.0, base=10)
        tick_values = [0, 1, 10, 100, 1000, 10000]
        ax.set_yticks(tick_values)
        ax.set_yticklabels([str(v) for v in tick_values])
        ax.set_ylim(0, 10000)
    ax.grid(True, axis="y", alpha=0.3)

    legend_handles = [
        Patch(facecolor=metric_colors[m], alpha=0.7, label=metric_labels[m])
        for m in METRIC_COLS
    ]
    ax.legend(handles=legend_handles, title="Distance metric", loc="upper left")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
    if out_csv is not None:
        _boxplot_percentiles(data).to_csv(out_csv, index=False)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", type=Path, required=True,
                        help="Per-target features CSV produced by per_target_features.py")
    parser.add_argument("--out-png", type=Path, default=None,
                        help="Output PNG path (default: alongside input CSV)")
    parser.add_argument("--out-pdf", type=Path, default=None,
                        help="Optional output PDF path (default: alongside input CSV)")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="Optional percentile-summary CSV path (default: alongside input CSV)")
    parser.add_argument("--log-y", action="store_true",
                        help="Use log scale on y-axis.")
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv)
    _validate_columns(df, args.features_csv)

    if args.out_png is None:
        args.out_png = args.features_csv.with_name(
            args.features_csv.stem + "_distance_boxplots.png"
        )
    if args.out_pdf is None:
        args.out_pdf = args.features_csv.with_name(
            args.features_csv.stem + "_distance_boxplots.pdf"
        )
    if args.out_csv is None:
        args.out_csv = args.features_csv.with_name(
            args.features_csv.stem + "_distance_boxplots_percentiles.csv"
        )

    plot_boxplots(
        df,
        out_png=args.out_png,
        out_pdf=args.out_pdf,
        out_csv=args.out_csv,
        log_y=args.log_y,
    )
    print(f"Saved {args.out_png}")
    print(f"Saved {args.out_pdf}")
    print(f"Saved {args.out_csv}")


if __name__ == "__main__":
    main()
