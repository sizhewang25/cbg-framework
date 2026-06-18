"""Per-combo airport-vs-error CDF from v2 benchmark outputs.

Given one benchmark config (or a run dir), this writes **one figure per combo**
overlaying four CDFs from that combo's annotated `targets.parquet`
(scripts/benchmark/v2/airport_eval.py appends the airport columns):

  1. error_km              — raw great-circle prediction error (continuous,
                             the city-level bar the default metric scores).
  2. pred_truth_airport_km — gap between the airport nearest the prediction and
                             the airport nearest the truth: the *airport-level*
                             error after both are snapped to the hub grid.
  3. truth_airport_km      — distance from each truth to its nearest hub. This
                             depends only on the target set + airport grid, not
                             on the predictor: it is the answer-space
                             quantization floor (the resolution the metric
                             throws away). Identical across combos by design,
                             drawn as a reference in every figure.
  4. pred_airport_km       — distance from each prediction to its nearest hub
                             (how "airport-like" the prediction lands).

Reading the figure: the spread between (1) error_km and (2)
pred_truth_airport_km is the interpretation shift from "exact lat/lon" to "right
metro hub". Where the error_km curve sits left of `truth_airport_km`, the
predictor is already finer than the airport grid can express, so airport
snapping cannot reward it further; where pred_truth_airport_km collapses toward
0 well before error_km does, many predictions land in the correct metro despite
non-trivial raw error — that gap is exactly the operator-facing leniency the
airport answer space buys.

CDFs pool SUCCESS+FALLBACK rows across folds per combo (K-fold test sets are
disjoint). NaNs are dropped per series, so the predictor-dependent curves cover
only rows that produced a prediction; `truth_airport_km` covers every target.

CLI:
    python -m scripts.analysis.plot_airport_cdf \\
        --config scripts/benchmark/v2/config/north_america_as7018_final.yaml
    python -m scripts.analysis.plot_airport_cdf \\
        --run-dir scripts/benchmark/v2/outputs/north_america_as7018_final \\
        --out-dir /tmp/airport_cdf
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

# (column, label, color, linestyle, linewidth, zorder). Order = legend order.
_SERIES: tuple[tuple[str, str, str, str, float, int], ...] = (
    ("error_km", "error_km (raw lat/lon error)", "#222222", "-", 2.2, 5),
    ("pred_truth_airport_km", "pred↔truth airport gap", "#1f77b4", "-", 2.2, 4),
    ("truth_airport_km", "truth→nearest hub (answer-space floor)", "#999999", "--", 1.8, 2),
    ("pred_airport_km", "pred→nearest hub", "#ff7f0e", ":", 1.8, 3),
)


def _scored_columns(combo_dirs: list[Path]) -> dict[str, np.ndarray]:
    """Pool the four airport/error columns over SUCCESS+FALLBACK rows across the
    given (per-fold) combo dirs. Returns {column: NaN-dropped float array}."""
    cols = [s[0] for s in _SERIES]
    pooled: dict[str, list[np.ndarray]] = {c: [] for c in cols}
    for d in combo_dirs:
        tbl = load_targets(d).to_pandas()
        sub = tbl[tbl["status"].isin(_SCORED_STATUSES)]
        for c in cols:
            arr = sub[c].to_numpy(dtype=float)
            pooled[c].append(arr[~np.isnan(arr)])
    return {
        c: (np.concatenate(parts) if parts else np.array([], dtype=float))
        for c, parts in pooled.items()
    }


def _match_rates(combo_dirs: list[Path], threshold_km: float) -> tuple[float, float, float, int]:
    """(exact_match_rate, within_threshold_rate, error_within_threshold_rate, n)
    pooled over SUCCESS+FALLBACK rows. The third is the raw-error city-level rate
    (error_km <= threshold) — the arbitrary-lat/lon counterpart to compare against."""
    frames = []
    for d in combo_dirs:
        t = load_targets(d).to_pandas()
        frames.append(t[t["status"].isin(_SCORED_STATUSES)])
    import pandas as pd

    sub = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    n = len(sub)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    exact = float(sub["airport_match"].mean())
    gap = sub["pred_truth_airport_km"].dropna()
    within = float((gap <= threshold_km).mean()) if len(gap) else float("nan")
    err = sub["error_km"].dropna()
    err_within = float((err <= threshold_km).mean()) if len(err) else float("nan")
    return exact, within, err_within, n


def plot_combo_airport_cdf(
    series: dict[str, np.ndarray],
    out_path: Path,
    *,
    combo_id: str,
    rates: tuple[float, float, float, int],
    threshold_km: float = 40.0,
    max_x_km: float = 10000.0,
) -> plt.Figure:
    """Draw the four overlaid CDFs + a stats box for one combo."""
    fig, ax = plt.subplots(figsize=(9, 6.5))

    stat_lines = [f"{'series':<26} {'p50':>6} {'p90':>6}"]
    for col, label, color, ls, lw, z in _SERIES:
        arr = series.get(col, np.array([], dtype=float))
        if len(arr) == 0:
            continue
        s = np.sort(arr)
        cdf = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, cdf, color=color, linestyle=ls, linewidth=lw, alpha=0.9,
                zorder=z, label=f"{label} (n={len(s)})")
        p50, p90 = np.percentile(s, [50, 90])
        stat_lines.append(f"{label[:26]:<26} {p50:6.0f} {p90:6.0f}")

    ax.axvline(threshold_km, color="green", linestyle=":", alpha=0.5)
    ax.text(threshold_km, 0.02, f" {threshold_km:.0f} km", color="green",
            fontsize=8, rotation=90, va="bottom", ha="left")
    ax.hlines(0.5, 1, max_x_km, color="gray", linestyle="--", alpha=0.25)

    exact, within, err_within, n = rates
    rate_txt = (
        f"n={n}\n"
        f"exact-IATA match      {exact:6.1%}\n"
        f"airport gap ≤{threshold_km:.0f}km    {within:6.1%}\n"
        f"error_km ≤{threshold_km:.0f}km       {err_within:6.1%}"
    )

    ax.set_xscale("log")
    x_fmt = ScalarFormatter()
    x_fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(x_fmt)
    ax.set_xlim(1, max_x_km)
    ax.set_ylim(0, 1)
    ax.set_xlabel("distance (km)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(f"Airport vs error CDF — {combo_id}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)

    ax.text(0.02, 0.98, "\n".join(stat_lines), transform=ax.transAxes,
            fontsize=7, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax.text(0.02, 0.62, rate_txt, transform=ax.transAxes,
            fontsize=8, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#eef7ee", alpha=0.95))

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
                        help="Output dir (default: scripts/analysis/outputs/<run_id>/airport/cdf).")
    parser.add_argument("--threshold-km", type=float, default=40.0)
    parser.add_argument("--max-x-km", type=float, default=10000.0)
    add_geo_filter_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_geo_filter_from_args(args)

    run_dir = resolve_run_dir(args.config, args.run_dir, args.outputs_root)
    out_dir = (
        route_geo_path(args.out_dir) if args.out_dir
        else analysis_out_dir(run_dir, "airport", "cdf")
    )

    combo_dirs = discover_combos(run_dir, args.source, args.slice_)
    if not combo_dirs:
        raise FileNotFoundError(f"No combos found under {run_dir}")
    grouped = group_combos_by_id(combo_dirs)

    for combo_id, dirs in sorted(grouped.items()):
        series = _scored_columns(dirs)
        rates = _match_rates(dirs, args.threshold_km)
        fig = plot_combo_airport_cdf(
            series, out_dir / f"{combo_id}_airport_cdf.png",
            combo_id=combo_id, rates=rates,
            threshold_km=args.threshold_km, max_x_km=args.max_x_km,
        )
        plt.close(fig)

    logger.info("Wrote %d per-combo figures to %s", len(grouped), out_dir)


if __name__ == "__main__":
    main()
