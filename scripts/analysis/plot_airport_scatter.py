"""Per-combo scatter of raw error vs airport-level error.

Given one benchmark config (or a run dir), this writes **one figure per combo**
plotting, for every scored target, the raw great-circle error (`error_km`, x)
against the airport-level error (`pred_truth_airport_km`, y — the gap between
the airport nearest the prediction and the airport nearest the truth). It is the
per-target companion to the CDF and match-rate views: the CDFs show the marginal
distributions, this shows their *joint* structure.

How to read it:
  - The y=x diagonal is "snapping changed nothing". Points **below** it are
    targets the airport answer space made look better (the two nearest hubs are
    closer than the raw coordinates); points **above** it (rare) are made worse.
  - Points pinned to the bottom floor are exact-IATA matches (gap = 0): the
    prediction and truth snap to the *same* hub. A point far to the right
    (large raw error) sitting on the floor is a big interpretation win — wrong
    by hundreds of km yet "right metro".
  - The shaded quadrant (x > T, y ≤ T) is the operator-facing payoff zone:
    targets that fail the raw city-level bar but pass the airport one. The
    opposite quadrant (x ≤ T, y > T) is the discretization tax — fine
    predictions split across a Voronoi boundary into the wrong hub.

Points are coloured by exact-IATA match. Zero gaps (exact matches) and sub-1km
errors are clipped to a 1 km floor so they remain visible on log axes; the floor
is drawn as a guide line. Scored rows = SUCCESS+FALLBACK with both values
present, pooled across folds (disjoint K-fold test sets).

CLI:
    python -m scripts.analysis.plot_airport_scatter \\
        --config scripts/benchmark/v2/config/north_america_as7018_final.yaml
    python -m scripts.analysis.plot_airport_scatter \\
        --run-dir scripts/benchmark/v2/outputs/north_america_as7018_final
"""

from __future__ import annotations

import argparse
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
    discover_combos,
    group_combos_by_id,
    load_targets,
    resolve_run_dir,
    route_geo_path,
    set_geo_filter_from_args,
)

logger = logging.getLogger(__name__)

_SCORED_STATUSES = ("SUCCESS", "FALLBACK")
_FLOOR_KM = 1.0  # log-axis display floor for zero gaps / sub-km errors


def _scored_frame(combo_dirs: list[Path]) -> pd.DataFrame:
    """Pool error_km / pred_truth_airport_km / airport_match over SUCCESS+FALLBACK
    rows with both distances present, across the given (per-fold) combo dirs."""
    frames = []
    for d in combo_dirs:
        t = load_targets(d).to_pandas()
        frames.append(t[t["status"].isin(_SCORED_STATUSES)])
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df
    df = df[["error_km", "pred_truth_airport_km", "airport_match"]].dropna(
        subset=["error_km", "pred_truth_airport_km"]
    )
    return df


def plot_combo_scatter(
    df: pd.DataFrame, out_path: Path, *, combo_id: str,
    threshold_km: float = 40.0, max_km: float = 10000.0,
) -> plt.Figure:
    """Scatter error_km (x) vs pred_truth_airport_km (y) for one combo."""
    fig, ax = plt.subplots(figsize=(7.5, 7))

    if df.empty:
        ax.text(0.5, 0.5, "no scored targets", transform=ax.transAxes,
                ha="center", va="center")
    else:
        x = np.clip(df["error_km"].to_numpy(dtype=float), _FLOOR_KM, max_km)
        y = np.clip(df["pred_truth_airport_km"].to_numpy(dtype=float), _FLOOR_KM, max_km)
        match = df["airport_match"].fillna(False).to_numpy(dtype=bool)

        # Payoff quadrant: raw error fails city-level bar, airport error passes.
        ax.axvspan(threshold_km, max_km, ymin=0, ymax=1, color="#eef7ee", zorder=0)
        ax.add_patch(plt.Rectangle(
            (threshold_km, _FLOOR_KM), max_km - threshold_km, threshold_km - _FLOOR_KM,
            facecolor="#bfe3bf", edgecolor="none", alpha=0.6, zorder=0,
        ))

        ax.scatter(x[~match], y[~match], s=12, c="#d62728", alpha=0.45,
                   edgecolors="none", label=f"different hub (n={int((~match).sum())})")
        ax.scatter(x[match], y[match], s=14, c="#2ca02c", alpha=0.6,
                   edgecolors="none", label=f"same hub / exact (n={int(match.sum())})")

        ax.plot([_FLOOR_KM, max_km], [_FLOOR_KM, max_km], color="black",
                linestyle="--", linewidth=1, alpha=0.6, label="y = x")
        ax.axvline(threshold_km, color="green", linestyle=":", alpha=0.5)
        ax.axhline(threshold_km, color="green", linestyle=":", alpha=0.5)
        ax.axhline(_FLOOR_KM, color="gray", linestyle="-", alpha=0.3)

        n = len(df)
        gain = int(((df["error_km"] > threshold_km) &
                    (df["pred_truth_airport_km"] <= threshold_km)).sum())
        tax = int(((df["error_km"] <= threshold_km) &
                   (df["pred_truth_airport_km"] > threshold_km)).sum())
        txt = (
            f"n={n}\n"
            f"exact hub        {match.mean():6.1%}\n"
            f"payoff quadrant  {gain / n:6.1%}  (x>{threshold_km:.0f}, y≤{threshold_km:.0f})\n"
            f"discretiz. tax   {tax / n:6.1%}  (x≤{threshold_km:.0f}, y>{threshold_km:.0f})"
        )
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8, va="top",
                ha="left", family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    for setscale, setfmt in ((ax.set_xscale, ax.xaxis), (ax.set_yscale, ax.yaxis)):
        setscale("log")
        f = ScalarFormatter()
        f.set_scientific(False)
        setfmt.set_major_formatter(f)
    ax.set_xlim(_FLOOR_KM, max_km)
    ax.set_ylim(_FLOOR_KM, max_km)
    ax.set_aspect("equal")
    ax.set_xlabel("error_km (raw lat/lon error)", fontsize=11)
    ax.set_ylabel("pred↔truth airport gap (km)", fontsize=11)
    ax.set_title(f"Raw vs airport-level error — {combo_id}", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
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
                        help="Output dir (default: scripts/analysis/outputs/<run_id>/airport/scatter).")
    parser.add_argument("--threshold-km", type=float, default=40.0)
    parser.add_argument("--max-km", type=float, default=10000.0)
    add_geo_filter_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_geo_filter_from_args(args)

    run_dir = resolve_run_dir(args.config, args.run_dir, args.outputs_root)
    out_dir = (
        route_geo_path(args.out_dir) if args.out_dir
        else analysis_out_dir(run_dir, "airport", "scatter")
    )

    combo_dirs = discover_combos(run_dir, args.source, args.slice_)
    if not combo_dirs:
        raise FileNotFoundError(f"No combos found under {run_dir}")
    grouped = group_combos_by_id(combo_dirs)

    for combo_id, dirs in sorted(grouped.items()):
        df = _scored_frame(dirs)
        fig = plot_combo_scatter(
            df, out_dir / f"{combo_id}_airport_scatter.png",
            combo_id=combo_id, threshold_km=args.threshold_km, max_km=args.max_km,
        )
        plt.close(fig)

    logger.info("Wrote %d per-combo scatters to %s", len(grouped), out_dir)


if __name__ == "__main__":
    main()
