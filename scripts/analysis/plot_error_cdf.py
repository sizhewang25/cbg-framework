"""Error CDF plot from v2 benchmark outputs.

Reads `error_km` from each combo's `targets.parquet` (TARGETS_SCHEMA) and
draws per-combo CDFs, optionally split into panels by LTD (using the `ltd`
column from `summary.parquet`).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from scripts.analysis._v2_io import (
    discover_combos,
    load_summary,
    load_targets,
    palette,
)

logger = logging.getLogger(__name__)


def plot_error_cdf(
    errors_by_combo: dict[str, np.ndarray],
    output_path: Path,
    *,
    group_by: Optional[str] = "ltd",
    combo_to_ltd: Optional[dict[str, str]] = None,
    thresholds: tuple[int, ...] = (100, 500, 1000),
    max_x_km: float = 3000.0,
    colors: Optional[dict[str, str]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot error CDFs from a {combo_id: error_km array} dict.

    Args:
        errors_by_combo: NaN-dropped error_km arrays per combo.
        output_path: Where to save the PNG.
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
        ltds = sorted({combo_to_ltd[c] for c in errors_by_combo if c in combo_to_ltd})
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

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6 * n_panels, 7), sharey=True, squeeze=False,
    )
    axes = axes[0]
    threshold_colors = {100: "green", 500: "orange", 1000: "red"}

    for ax, (panel_title, panel_combos) in zip(axes, panels):
        panel_data: list[tuple[str, np.ndarray]] = []
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
            panel_data.append((cid, errors))

        for thresh in thresholds:
            ax.axvline(
                x=thresh,
                color=threshold_colors.get(thresh, "gray"),
                linestyle=":",
                alpha=0.4,
            )
        ax.hlines(y=0.5, xmin=0, xmax=max_x_km, color="gray", linestyle="--", alpha=0.3)

        if panel_title:
            ax.set_title(panel_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Error distance (km)", fontsize=11)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_x_km)
        ax.set_ylim(0, 1)

        if panel_data:
            lines = ["       count    p5   p25   p50   p75   p95"]
            for cid, errors in panel_data:
                lines.append(
                    f"{cid[:16]:<16} {len(errors):>5d} "
                    f"{np.percentile(errors, 5):5.0f} "
                    f"{np.percentile(errors, 25):5.0f} "
                    f"{np.median(errors):5.0f} "
                    f"{np.percentile(errors, 75):5.0f} "
                    f"{np.percentile(errors, 95):5.0f}"
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
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Walk `run_dir`, return (errors_by_combo, combo_to_ltd)."""
    combo_dirs = discover_combos(run_dir, source, slice_)
    if not combo_dirs:
        raise FileNotFoundError(f"No combos found under {run_dir}")

    errors_by_combo: dict[str, np.ndarray] = {}
    for combo_dir in combo_dirs:
        tbl = load_targets(combo_dir)
        arr = tbl.column("error_km").to_numpy(zero_copy_only=False)
        arr = arr[~np.isnan(arr)]
        errors_by_combo[combo_dir.name] = arr

    summary = load_summary(run_dir)
    combo_to_ltd = dict(zip(
        summary.column("combo_id").to_pylist(),
        summary.column("ltd").to_pylist(),
    ))
    return errors_by_combo, combo_to_ltd


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
    parser.add_argument("--max-x-km", type=float, default=3000.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    errors_by_combo, combo_to_ltd = _load_from_run(args.run_dir, args.source, args.slice_)
    group_by = None if args.group_by == "none" else args.group_by
    fig = plot_error_cdf(
        errors_by_combo,
        args.out,
        group_by=group_by,
        combo_to_ltd=combo_to_ltd if group_by == "ltd" else None,
        max_x_km=args.max_x_km,
        title=args.title,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
