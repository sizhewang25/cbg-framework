"""Airport match-rate bar charts across all combos of one benchmark run.

Given one benchmark config (or a run dir), this ranks every combo by its
airport match rate and writes two figures:

  1. <run_id>_airport_match_exact.png
       Exact nearest-IATA match rate (prediction and truth snap to the *same*
       hub), bars sorted descending.
  2. <run_id>_airport_match_within_<T>km.png
       Forgiving match rate: the prediction's hub lies within `T` km (default
       40, the city-level bin) of the truth's hub, which absorbs multi-airport
       metros (JFK/LGA/EWR). Same combos, re-sorted descending by this rate.

Each bar also carries a marker for the *raw-error* city-level rate
(error_km <= T) — the arbitrary-lat/lon baseline. The distance between the bar
and its marker is the interpretation gain from scoring against the limited
airport answer space instead of demanding exact coordinates.

Rates pool SUCCESS+FALLBACK rows across folds per combo (K-fold test sets are
disjoint). Exact match uses the boolean `airport_match` column; the threshold
rates are computed here from the continuous `pred_truth_airport_km` /
`error_km` columns, so `--threshold-km` is free to change without re-annotating.

CLI:
    python -m scripts.analysis.plot_airport_match_bars \\
        --config scripts/benchmark/v2/config/north_america_as7018_final.yaml
    python -m scripts.analysis.plot_airport_match_bars \\
        --run-dir scripts/benchmark/v2/outputs/north_america_as7018_final \\
        --threshold-km 40
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
    load_targets,
    resolve_run_dir,
    route_geo_path,
    set_geo_filter_from_args,
)

logger = logging.getLogger(__name__)

_SCORED_STATUSES = ("SUCCESS", "FALLBACK")


def compute_rates(run_dir: Path, threshold_km: float, source=None, slice_=None) -> pd.DataFrame:
    """One row per combo_id with exact / within-threshold / error-within rates.

    Columns: combo_id, n, exact, within, err_within. Rates pool SUCCESS+FALLBACK
    rows across folds. NaN gap/error rows (no prediction) are dropped from the
    respective threshold rates but still counted in `n`.
    """
    combo_dirs = discover_combos(run_dir, source, slice_)
    if not combo_dirs:
        raise FileNotFoundError(f"No combos found under {run_dir}")
    grouped = group_combos_by_id(combo_dirs)

    rows = []
    for combo_id, dirs in grouped.items():
        frames = [load_targets(d).to_pandas() for d in dirs]
        df = pd.concat(frames, ignore_index=True)
        sub = df[df["status"].isin(_SCORED_STATUSES)]
        n = len(sub)
        gap = sub["pred_truth_airport_km"].dropna()
        err = sub["error_km"].dropna()
        rows.append({
            "combo_id": combo_id,
            "n": n,
            "exact": float(sub["airport_match"].mean()) if n else float("nan"),
            "within": float((gap <= threshold_km).mean()) if len(gap) else float("nan"),
            "err_within": float((err <= threshold_km).mean()) if len(err) else float("nan"),
        })
    return pd.DataFrame(rows)


def _plot_bars(
    rates: pd.DataFrame, rate_col: str, out_path: Path,
    *, title: str, bar_label: str, threshold_km: float,
) -> plt.Figure:
    """Horizontal bars of `rate_col` sorted descending, with the raw-error
    city-level rate (err_within) overlaid as a reference marker per combo."""
    df = rates.sort_values(rate_col, ascending=True)  # ascending → best on top in barh
    y = range(len(df))

    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(df) + 1.5)))
    ax.barh(list(y), df[rate_col], color="#4E79A7", alpha=0.85, label=bar_label, zorder=2)
    ax.scatter(df["err_within"], list(y), color="#d62728", marker="D", s=28,
               zorder=3, label=f"error_km ≤{threshold_km:.0f}km (raw lat/lon)")

    for yi, (val, n) in enumerate(zip(df[rate_col], df["n"])):
        if pd.notna(val):
            ax.text(val + 0.01, yi, f"{val:.1%}  (n={n})", va="center", fontsize=8)

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["combo_id"], fontsize=8)
    ax.set_xlim(0, 1.0)
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
                        help="Output dir (default: scripts/analysis/outputs/<run_id>/airport).")
    parser.add_argument("--threshold-km", type=float, default=40.0)
    add_geo_filter_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_geo_filter_from_args(args)

    run_dir = resolve_run_dir(args.config, args.run_dir, args.outputs_root)
    out_dir = (
        route_geo_path(args.out_dir) if args.out_dir
        else analysis_out_dir(run_dir, "airport")
    )
    rates = compute_rates(run_dir, args.threshold_km, args.source, args.slice_)

    t = int(args.threshold_km)
    fig1 = _plot_bars(
        rates, "exact", out_dir / f"{run_dir.name}_airport_match_exact.png",
        title=f"Exact airport match rate — {run_dir.name}",
        bar_label="exact-IATA match", threshold_km=args.threshold_km,
    )
    plt.close(fig1)
    fig2 = _plot_bars(
        rates, "within", out_dir / f"{run_dir.name}_airport_match_within_{t}km.png",
        title=f"Airport match rate within {t} km — {run_dir.name}",
        bar_label=f"airport gap ≤{t}km", threshold_km=args.threshold_km,
    )
    plt.close(fig2)

    logger.info("Wrote 2 bar charts for %d combos to %s", len(rates), out_dir)


if __name__ == "__main__":
    main()
