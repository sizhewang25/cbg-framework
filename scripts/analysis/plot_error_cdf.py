"""Error CDF plot from v2 benchmark outputs.

Reads `error_km` from each combo's `targets.parquet` (TARGETS_SCHEMA) and
draws per-combo CDFs, optionally split into panels by LTD (using the `ltd`
column from `summary.parquet`).

Two views are supported via --success-only:
  default       : CDF over SUCCESS + FALLBACK rows (error_km not null).
  --success-only: CDF over SUCCESS rows only.

In both views the stats panel renders a "succ/total" column so the
non-error fraction per combo stays visible.

When --inputs-dir points to the materialized inputs directory containing
eval_observations.parquet, a "shortest_ping" baseline is overlaid in
every panel: for each target, predict its location as the coordinates of
the VP with the smallest observed latency. The same all-targets baseline
is drawn on both views as a fixed reference.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scripts.analysis._v2_io import (
    discover_combos,
    group_combos_by_id,
    load_summary,
    load_targets,
    palette,
)
from scripts.libs.cbg.rtt_model import haversine_distance

logger = logging.getLogger(__name__)

# Panel order when group_by="ltd". Known LTDs follow this sequence (loosest
# CBG geometry → tightest stat model); unknown LTDs append in name order.
_LTD_PANEL_ORDER: tuple[str, ...] = (
    "speed_of_internet",
    "low_envelope",
    "bounded_spline",
    "normal_dist",
)


def _ltd_panel_sort_key(ltd: str) -> tuple[int, str]:
    try:
        return (_LTD_PANEL_ORDER.index(ltd), ltd)
    except ValueError:
        return (len(_LTD_PANEL_ORDER), ltd)


def _short_label(cid: str) -> str:
    """Compact form of `cid` for the stats panel. Only the
    `million_scale_` prefix is abbreviated — it's the single combo family
    long enough to truncate at the 16-char label width.
    """
    if cid.startswith("million_scale_"):
        return "ms_" + cid[len("million_scale_"):]
    return cid


def plot_error_cdf(
    errors_by_combo: dict[str, np.ndarray],
    output_path: Path,
    *,
    successes_by_combo: dict[str, int],
    totals_by_combo: dict[str, int],
    baseline_errors: Optional[np.ndarray] = None,
    baseline_label: str = "shortest_ping",
    group_by: Optional[str] = "ltd",
    combo_to_ltd: Optional[dict[str, str]] = None,
    thresholds: tuple[int, ...] = (100, 500, 1000),
    max_x_km: float = 10000.0,
    colors: Optional[dict[str, str]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot error CDFs from a {combo_id: error_km array} dict.

    Args:
        errors_by_combo: NaN-dropped error_km arrays per combo.
        output_path: Where to save the PNG.
        successes_by_combo: combo_id -> number of SUCCESS-status rows. Drives
            the numerator of the "succ/total" column — the same value on both
            views so the success rate column is comparable across plots.
        totals_by_combo: combo_id -> total number of eval-target rows.
        baseline_errors: Optional NaN-dropped errors from a baseline predictor
            (e.g. shortest_ping). Drawn as a single dashed line in every panel.
        baseline_label: Legend label for the baseline curve.
        group_by: "ltd" to split into one panel per LTD model, None for one panel.
        combo_to_ltd: Required iff group_by="ltd". Maps combo_id to its LTD name.
        thresholds: Vertical reference lines (km).
        max_x_km: X-axis upper bound.
        colors: Optional combo_id -> hex color. Defaults to tab20 by sorted id.
        title: Figure title.
    """
    if group_by == "ltd":
        if combo_to_ltd is None:
            raise ValueError("combo_to_ltd is required when group_by='ltd'")
        ltds = sorted(
            {combo_to_ltd[c] for c in errors_by_combo if c in combo_to_ltd},
            key=_ltd_panel_sort_key,
        )
        panels: list[tuple[str, list[str]]] = [
            (ltd, [c for c in errors_by_combo if combo_to_ltd.get(c) == ltd])
            for ltd in ltds
        ]
    elif group_by is None:
        panels = [("", list(errors_by_combo))]
    else:
        raise ValueError(f"unsupported group_by={group_by!r}")

    if colors is None:
        colors = palette(list(errors_by_combo))

    count_header = "succ/total"
    count_width = 11

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6 * n_panels, 7), sharey=True, squeeze=False,
    )
    axes = axes[0]
    threshold_colors = {100: "green", 500: "orange", 1000: "red"}

    baseline_sorted = None
    baseline_cdf = None
    if baseline_errors is not None and len(baseline_errors) > 0:
        baseline_sorted = np.sort(baseline_errors)
        baseline_cdf = np.arange(1, len(baseline_sorted) + 1) / len(baseline_sorted)

    for ax, (panel_title, panel_combos) in zip(axes, panels):
        panel_data: list[tuple[str, np.ndarray, str]] = []
        for cid in panel_combos:
            errors = errors_by_combo[cid]
            if len(errors) == 0:
                continue
            sorted_e = np.sort(errors)
            cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
            ax.plot(
                sorted_e, cdf,
                color=colors.get(cid, "#4E79A7"),
                linewidth=2,
                alpha=0.8,
                label=cid,
            )
            n_success = successes_by_combo.get(cid, len(errors))
            total = totals_by_combo.get(cid, len(errors))
            count_str = f"{n_success}/{total}"
            panel_data.append((cid, errors, count_str))

        if baseline_sorted is not None:
            ax.plot(
                baseline_sorted, baseline_cdf,
                color="black",
                linestyle="--",
                linewidth=2,
                alpha=0.9,
                label=baseline_label,
            )

        for thresh in thresholds:
            ax.axvline(
                x=thresh,
                color=threshold_colors.get(thresh, "gray"),
                linestyle=":",
                alpha=0.4,
            )
        ax.hlines(y=0.5, xmin=1, xmax=max_x_km, color="gray", linestyle="--", alpha=0.3)

        if panel_title:
            ax.set_title(panel_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Error distance (km)", fontsize=11)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xscale("log")
        ax.set_xlim(1, max_x_km)
        ax.set_ylim(0, 1)

        if panel_data or baseline_sorted is not None:
            lines = [f"{'':<16} {count_header:>{count_width}}    p5   p25   p50   p75   p95"]
            for cid, errors, count_str in panel_data:
                lines.append(
                    f"{_short_label(cid)[:16]:<16} {count_str:>{count_width}} "
                    f"{np.percentile(errors, 5):5.0f} "
                    f"{np.percentile(errors, 25):5.0f} "
                    f"{np.median(errors):5.0f} "
                    f"{np.percentile(errors, 75):5.0f} "
                    f"{np.percentile(errors, 95):5.0f}"
                )
            if baseline_sorted is not None:
                lines.append(
                    f"{baseline_label[:16]:<16} {str(len(baseline_sorted)):>{count_width}} "
                    f"{np.percentile(baseline_sorted, 5):5.0f} "
                    f"{np.percentile(baseline_sorted, 25):5.0f} "
                    f"{np.median(baseline_sorted):5.0f} "
                    f"{np.percentile(baseline_sorted, 75):5.0f} "
                    f"{np.percentile(baseline_sorted, 95):5.0f}"
                )
            ax.text(
                0.98, 0.02, "\n".join(lines),
                transform=ax.transAxes, fontsize=7,
                verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                family="monospace",
            )

    axes[0].set_ylabel("CDF", fontsize=12)
    fig.suptitle(
        title or ("Error CDF by LTD" if group_by == "ltd" else "Error CDF"),
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    return fig


def _load_from_run(
    run_dir: Path,
    source: Optional[str],
    slice_: Optional[str],
    *,
    success_only: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, int], dict[str, str]]:
    """Walk `run_dir`, return (errors_by_combo, successes_by_combo,
    totals_by_combo, combo_to_ltd).

    `successes_by_combo` counts SUCCESS-status rows (independent of
    `success_only`) so the stats column shows the same success rate on both
    views. `totals_by_combo` is the total number of target rows per combo.

    With `slice_=None` on a K-fold layout, error arrays and counts are
    concatenated across folds per combo_id — K-fold test sets are disjoint
    by construction, so this is one row per target.
    """
    combo_dirs = discover_combos(run_dir, source, slice_)
    if not combo_dirs:
        raise FileNotFoundError(f"No combos found under {run_dir}")

    errors_by_combo: dict[str, list[np.ndarray]] = {}
    successes_by_combo: dict[str, int] = {}
    totals_by_combo: dict[str, int] = {}
    for combo_dir in combo_dirs:
        cid = combo_dir.name
        tbl = load_targets(combo_dir)
        totals_by_combo[cid] = totals_by_combo.get(cid, 0) + tbl.num_rows
        success_mask = pc.equal(tbl.column("status"), "SUCCESS")
        successes_by_combo[cid] = (
            successes_by_combo.get(cid, 0) + int(pc.sum(success_mask).as_py() or 0)
        )
        if success_only:
            tbl = tbl.filter(success_mask)
        arr = tbl.column("error_km").to_numpy(zero_copy_only=False)
        arr = arr[~np.isnan(arr)]
        errors_by_combo.setdefault(cid, []).append(arr)

    errors_concat: dict[str, np.ndarray] = {
        cid: (np.concatenate(parts) if parts else np.array([], dtype=float))
        for cid, parts in errors_by_combo.items()
    }

    summary = load_summary(run_dir)
    combo_to_ltd = dict(zip(
        summary.column("combo_id").to_pylist(),
        summary.column("ltd").to_pylist(),
    ))
    return errors_concat, successes_by_combo, totals_by_combo, combo_to_ltd


def _load_nearest_ping_baseline(inputs_dir: Path) -> np.ndarray:
    """Read eval_observations.parquet and compute haversine error from each
    target to the location of its smallest-latency VP. One error per target.

    If `inputs_dir/eval_observations.parquet` exists, it's read directly
    (single-fold mode). Otherwise the directory is treated as the parent of
    per-fold input dirs and `**/eval_observations.parquet` is globbed and
    concatenated — the merged-folds counterpart of the merged-folds loader
    in `_load_from_run`.
    """
    direct = inputs_dir / "eval_observations.parquet"
    if direct.exists():
        paths = [direct]
    else:
        paths = sorted(inputs_dir.glob("*/eval_observations.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"No eval_observations.parquet at {inputs_dir} or under "
                f"{inputs_dir}/*/. Pass --inputs-dir pointing at a fold "
                "input dir or at its parent (for merged-fold mode)."
            )

    import pandas as pd
    df = pd.concat([pq.read_table(p).to_pandas() for p in paths], ignore_index=True)
    if df.empty:
        return np.array([], dtype=float)

    idx = df.groupby("target_id")["latency_ms"].idxmin()
    nearest = df.loc[idx]
    errors = haversine_distance(
        nearest["target_lat"].to_numpy(dtype=float),
        nearest["target_lon"].to_numpy(dtype=float),
        nearest["vp_lat"].to_numpy(dtype=float),
        nearest["vp_lon"].to_numpy(dtype=float),
    )
    return np.asarray(errors, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot error CDF from a v2 benchmark run.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to outputs/<run_id>/ (contains summary.parquet).",
    )
    parser.add_argument("--source", default=None, help="Filter combos by source name.")
    parser.add_argument("--slice", dest="slice_", default=None, help="Filter combos by slice id.")
    parser.add_argument(
        "--group-by",
        choices=("ltd", "none"),
        default="ltd",
        help="Panel layout. 'ltd' = one subplot per LTD model; 'none' = single panel.",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="CDF over SUCCESS rows only (default also keeps FALLBACK).",
    )
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=None,
        help="Path to inputs/<source>/<setup>/<slice>/ containing "
             "eval_observations.parquet. When given, a nearest-ping VP baseline "
             "is overlaid in every panel.",
    )
    parser.add_argument("--max-x-km", type=float, default=10000.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    errors_by_combo, successes_by_combo, totals_by_combo, combo_to_ltd = _load_from_run(
        args.run_dir, args.source, args.slice_, success_only=args.success_only,
    )
    group_by = None if args.group_by == "none" else args.group_by

    baseline_errors = None
    if args.inputs_dir is not None:
        baseline_errors = _load_nearest_ping_baseline(args.inputs_dir)
        logger.info(
            "shortest_ping baseline: n=%d, p50=%.0f km",
            len(baseline_errors),
            float(np.median(baseline_errors)) if len(baseline_errors) else 0.0,
        )

    title = args.title
    if title is None and args.success_only:
        title = "Error CDF — SUCCESS only" + (" by LTD" if group_by == "ltd" else "")

    fig = plot_error_cdf(
        errors_by_combo,
        args.out,
        successes_by_combo=successes_by_combo,
        totals_by_combo=totals_by_combo,
        baseline_errors=baseline_errors,
        group_by=group_by,
        combo_to_ltd=combo_to_ltd if group_by == "ltd" else None,
        max_x_km=args.max_x_km,
        title=title,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
