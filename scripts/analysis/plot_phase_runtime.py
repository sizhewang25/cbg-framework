"""Per-target stage runtime per v2 combo: stacked LTD / MTL / CTR (median)
with a line marking the median total per-target runtime.

Reads `{ltd,mtl,ctr}_ms_<stat>` from `summary.parquet` for the bars and the
raw per-target totals from each combo's `targets.parquet` for the line.
Fit time is intentionally excluded — it's a one-time per-combo cost, not a
per-target cost. The line differs from the stack top because median doesn't
distribute over a sum (mean does); the gap is informative.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from scripts.analysis._v2_io import discover_combos, load_summary, load_targets

logger = logging.getLogger(__name__)

_PHASE_ORDER = ("ltd", "mtl", "ctr")
_PHASE_LABEL = {
    "ltd": "LTD (distance)",
    "mtl": "MTL (multilateration)",
    "ctr": "CTR (centroid)",
}
_PHASE_COLOR = {
    "ltd": "#4E79A7",
    "mtl": "#F28E2B",
    "ctr": "#E15759",
}

_STAT_TO_QUANTILE = {"p5": 0.05, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p95": 0.95}


def _total_runtime_stat(targets: pa.Table, stat: str) -> float:
    """Per-target total (ltd+mtl+ctr) aggregated by `stat`.

    Null mtl/ctr (skipped stages) coerce to 0 so the total still reflects
    the work that actually ran. Only rows with a successful LTD contribute.
    """
    ltd = pc.fill_null(targets.column("ltd_ms"), 0.0)
    mtl = pc.fill_null(targets.column("mtl_ms"), 0.0)
    ctr = pc.fill_null(targets.column("ctr_ms"), 0.0)
    total = pc.add(pc.add(ltd, mtl), ctr)
    arr = total.to_numpy(zero_copy_only=False).astype(float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0.0
    if stat == "mean":
        return float(np.mean(arr))
    if stat == "std":
        return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    q = _STAT_TO_QUANTILE.get(stat)
    if q is None:
        raise ValueError(f"unknown stat {stat!r}")
    return float(np.quantile(arr, q))


def plot_phase_runtime(
    summary: pa.Table,
    targets_by_combo: dict[str, pa.Table],
    output_path: Path,
    *,
    stat: str = "p50",
    title: Optional[str] = None,
) -> plt.Figure:
    """Stacked per-stage runtime bars (ms) + per-target total runtime line.

    Args:
        summary: SUMMARY_SCHEMA table, already filtered to one (source, slice).
        targets_by_combo: combo_id -> TARGETS_SCHEMA table, used to compute
            the actual per-target total (line). Combo order follows `summary`.
        output_path: Where to save the PNG.
        stat: Per-stage stat for the bars — "p5"/"p25"/"p50"/"p75"/"p95"/"mean".
            Default p50 (median) per the v2 benchmark convention for runtime.
        title: Figure title.
    """
    combo_ids = summary.column("combo_id").to_pylist()
    phase_ms: dict[str, list[float]] = {}
    for phase in _PHASE_ORDER:
        col = f"{phase}_ms_{stat}"
        if col not in summary.column_names:
            raise ValueError(
                f"Column {col!r} not in summary.parquet — check --stat. "
                f"Available stats: p5/p25/p50/p75/p95/mean."
            )
        values = summary.column(col).to_numpy(zero_copy_only=False).astype(float)
        phase_ms[phase] = [
            v if not np.isnan(v) else 0.0 for v in values
        ]

    totals = [
        _total_runtime_stat(targets_by_combo[cid], stat)
        for cid in combo_ids
    ]

    # Sort by true total so the slowest combo sits on the left — same ordering
    # convention as plot_phase_memory.py uses for RSS.
    order = sorted(range(len(combo_ids)), key=lambda i: (-totals[i], i))
    combo_ids = [combo_ids[i] for i in order]
    totals = [totals[i] for i in order]
    phase_ms = {p: [phase_ms[p][i] for i in order] for p in phase_ms}

    fig_width = max(10.0, len(combo_ids) * 0.85)
    fig, ax = plt.subplots(figsize=(fig_width, 7.0))
    x = np.arange(len(combo_ids))
    bottoms = np.zeros(len(combo_ids), dtype=float)

    for phase in _PHASE_ORDER:
        values = np.array(phase_ms[phase], dtype=float)
        if not np.any(values):
            continue
        ax.bar(
            x, values,
            bottom=bottoms,
            color=_PHASE_COLOR[phase],
            edgecolor="white",
            linewidth=0.4,
            label=_PHASE_LABEL[phase],
        )
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels(combo_ids, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"Per-target stage runtime ({stat}) (ms)")
    ax.set_title(
        title or f"Per-target stage runtime ({stat}) — bars stack LTD/MTL/CTR; line = total",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)

    ax.plot(
        x, totals,
        color="#D1495B",
        marker="s",
        linewidth=2.5,
        label=f"total ({stat}) per target",
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="upper left", bbox_to_anchor=(0.0, -0.18),
        ncol=4, fontsize=8, frameon=False,
    )

    fig.text(
        0.01, 0.01,
        "Bars: per-stage stat from summary.parquet. "
        "Line: stat of per-target total (ltd+mtl+ctr) from targets.parquet — "
        "for quantiles, line ≠ stack top because medians don't add.",
        fontsize=8, color="#495057",
    )

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    return fig


def _load_from_run(
    run_dir: Path,
    source: Optional[str],
    slice_: Optional[str],
) -> tuple[pa.Table, dict[str, pa.Table]]:
    """Return (filtered summary, combo_id -> targets table)."""
    summary = load_summary(run_dir)
    if source is not None:
        summary = summary.filter(pc.equal(summary.column("source"), source))
    if slice_ is not None:
        summary = summary.filter(pc.equal(summary.column("slice"), slice_))
    if summary.num_rows == 0:
        raise ValueError(
            f"No rows in summary.parquet at {run_dir} after filtering "
            f"source={source!r} slice={slice_!r}"
        )

    combo_dirs = discover_combos(run_dir, source, slice_)
    targets_by_combo = {d.name: load_targets(d) for d in combo_dirs}
    return summary, targets_by_combo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot stacked per-stage runtime + total per-target runtime line.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", default=None)
    parser.add_argument("--slice", dest="slice_", default=None)
    parser.add_argument(
        "--stat",
        choices=("p5", "p25", "p50", "p75", "p95", "mean"),
        default="p50",
        help="Per-stage stat for the bars (default p50 = median).",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary, targets_by_combo = _load_from_run(args.run_dir, args.source, args.slice_)
    fig = plot_phase_runtime(
        summary,
        targets_by_combo,
        args.out,
        stat=args.stat,
        title=args.title,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
