"""Phase-local peak memory per v2 combo: stacked LTD / MTL / CTR (+ FIT) with
a process-max-RSS line.

Reads `{ltd,mtl,ctr}_peak_bytes_<stat>` from `summary.parquet` (one row per
combo, SUMMARY_SCHEMA) and `fit_peak_bytes` / `run_peak_rss_bytes` from each
combo's `run.json`.
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

from scripts.analysis._v2_io import (
    discover_combos,
    group_combos_by_id,
    load_run_json,
    load_summary,
    load_targets,
)

logger = logging.getLogger(__name__)

_PHASE_ORDER = ("ltd", "mtl", "ctr", "fit")
_PHASE_LABEL = {
    "ltd": "LTD (distance)",
    "mtl": "MTL (multilateration)",
    "ctr": "CTR (centroid)",
    "fit": "FIT (one-time)",
}
_PHASE_COLOR = {
    "ltd": "#4E79A7",
    "mtl": "#F28E2B",
    "ctr": "#E15759",
    "fit": "#8D6E63",
}

_BYTES_PER_MB = 1024 * 1024


def plot_phase_memory(
    summary: pa.Table,
    run_jsons: dict[str, dict],
    output_path: Path,
    *,
    stat: str = "p95",
    include_fit: bool = True,
    title: Optional[str] = None,
) -> plt.Figure:
    """Stacked phase-memory bars (MB) + process max-RSS line.

    Args:
        summary: SUMMARY_SCHEMA table, already filtered to one run.
        run_jsons: combo_id -> run.json contents (for fit_peak_bytes and
            run_peak_rss_bytes).
        output_path: Where to save the PNG.
        stat: Which per-stage stat to plot — "p5"/"p25"/"p50"/"p75"/"p95"/"mean".
        include_fit: Whether to add a stacked FIT bar from run.json fit_peak_bytes.
        title: Figure title.
    """
    combo_ids = summary.column("combo_id").to_pylist()
    phase_mb: dict[str, list[float]] = {}
    for phase in ("ltd", "mtl", "ctr"):
        col = f"{phase}_peak_bytes_{stat}"
        if col not in summary.column_names:
            raise ValueError(
                f"Column {col!r} not in summary.parquet — check --stat. "
                f"Available: {sorted(c for c in summary.column_names if c.endswith(tuple(['_p5','_p25','_p50','_p75','_p95','_mean','_std'])))}"
            )
        values = summary.column(col).to_numpy(zero_copy_only=False).astype(float)
        phase_mb[phase] = [
            v / _BYTES_PER_MB if not np.isnan(v) else 0.0 for v in values
        ]

    if include_fit:
        phase_mb["fit"] = [
            float(run_jsons.get(cid, {}).get("fit_peak_bytes") or 0.0) / _BYTES_PER_MB
            for cid in combo_ids
        ]

    rss_mb = [
        float(run_jsons.get(cid, {}).get("run_peak_rss_bytes") or 0.0) / _BYTES_PER_MB
        for cid in combo_ids
    ]

    phases_to_plot = _PHASE_ORDER if include_fit else _PHASE_ORDER[:-1]
    stack_totals = [
        sum(phase_mb[p][i] for p in phases_to_plot)
        for i in range(len(combo_ids))
    ]
    order = sorted(range(len(combo_ids)), key=lambda i: (-stack_totals[i], i))
    combo_ids = [combo_ids[i] for i in order]
    rss_mb = [rss_mb[i] for i in order]
    phase_mb = {p: [phase_mb[p][i] for i in order] for p in phase_mb}

    fig_width = max(10.0, len(combo_ids) * 0.85)
    fig, ax = plt.subplots(figsize=(fig_width, 7.0))
    x = np.arange(len(combo_ids))
    bottoms = np.zeros(len(combo_ids), dtype=float)

    for phase in phases_to_plot:
        values = np.array(phase_mb[phase], dtype=float)
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
    ax.set_ylabel(f"Stacked phase peak ({stat}) (MB)")
    ax.set_title(
        title or f"Phase-local peak memory ({stat}) and max RSS",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)

    ax_rss = ax.twinx()
    ax_rss.plot(
        x, rss_mb,
        color="#D1495B",
        marker="s",
        linewidth=2.5,
        label="run_peak_rss (MB)",
    )
    ax_rss.set_ylabel("Process max RSS (MB)")

    bars_handles, bars_labels = ax.get_legend_handles_labels()
    rss_handles, rss_labels = ax_rss.get_legend_handles_labels()
    ax.legend(
        bars_handles, bars_labels,
        loc="upper left", bbox_to_anchor=(0.0, -0.20),
        ncol=4, fontsize=8, frameon=False,
    )
    ax_rss.legend(rss_handles, rss_labels, loc="upper right", fontsize=8, frameon=True)

    fig.text(
        0.01, 0.01,
        f"Bars: per-stage tracemalloc peak (summary {stat}). "
        f"Stacks are attribution aids, not concurrent totals.",
        fontsize=8, color="#495057",
    )

    fig.tight_layout(rect=(0, 0.12, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    return fig


_PHASE_PEAK_COLS = ("ltd_peak_bytes", "mtl_peak_bytes", "ctr_peak_bytes")
_STAT_NAMES = ("p5", "p25", "p50", "p75", "p95", "mean", "std")
_STAT_QS = {"p5": 0.05, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p95": 0.95}


def _stats_from_array(arr: np.ndarray) -> dict[str, float]:
    """Compute the SUMMARY_SCHEMA stat set (p5/p25/p50/p75/p95/mean/std)
    from a raw numeric array. NaNs are dropped first; empty input → all NaN.
    """
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {s: float("nan") for s in _STAT_NAMES}
    out: dict[str, float] = {s: float(np.quantile(arr, q)) for s, q in _STAT_QS.items()}
    out["mean"] = float(np.mean(arr))
    out["std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
    return out


def _synthesize_merged_summary(
    grouped: dict[str, list[Path]],
    columns: tuple[str, ...],
) -> pa.Table:
    """Build a one-row-per-combo summary by concatenating raw per-target
    values across folds and recomputing the SUMMARY_SCHEMA stat columns
    for `columns`. Returned table has `combo_id` + `<col>_<stat>` columns
    for each col in `columns` and each stat in _STAT_NAMES.
    """
    cids = sorted(grouped)
    table_data: dict[str, list] = {"combo_id": cids}
    for col in columns:
        for stat in _STAT_NAMES:
            table_data[f"{col}_{stat}"] = []
    for cid in cids:
        for col in columns:
            arrs = []
            for combo_dir in grouped[cid]:
                tbl = load_targets(combo_dir)
                arrs.append(tbl.column(col).to_numpy(zero_copy_only=False))
            concat = np.concatenate(arrs) if arrs else np.array([], dtype=float)
            stats = _stats_from_array(concat)
            for stat in _STAT_NAMES:
                table_data[f"{col}_{stat}"].append(stats[stat])
    return pa.table(table_data)


def _merged_run_jsons(grouped: dict[str, list[Path]]) -> dict[str, dict]:
    """Per-combo run.json synthesized from K fold runs.

    `fit_peak_bytes` and `run_peak_rss_bytes` are reduced with max across
    folds — both are process-level peaks and max is the worst-case bound,
    consistent with framing the merged plot as a worst-case characterization
    of the combo across the K cross-validation runs.
    """
    out: dict[str, dict] = {}
    for cid, dirs in grouped.items():
        fits: list[int] = []
        rsss: list[int] = []
        for d in dirs:
            rj = load_run_json(d)
            fp = rj.get("fit_peak_bytes")
            if fp is not None:
                fits.append(int(fp))
            rss = rj.get("run_peak_rss_bytes")
            if rss is not None:
                rsss.append(int(rss))
        out[cid] = {
            "fit_peak_bytes": max(fits) if fits else None,
            "run_peak_rss_bytes": max(rsss) if rsss else None,
        }
    return out


def _load_from_run(
    run_dir: Path,
    source: Optional[str],
    slice_: Optional[str],
    combos: Optional[list[str]] = None,
) -> tuple[pa.Table, dict[str, dict]]:
    """Return (summary table, run_jsons by combo_id).

    Single-fold mode (`slice_` given): filters `summary.parquet` to
    (source, slice) and uses each combo's run.json verbatim.

    Merged-folds mode (`slice_=None` on a K-fold layout): synthesizes a
    summary by concatenating per-target peak-bytes from each fold's
    targets.parquet and recomputing percentiles from raw, since percentiles
    don't aggregate from per-fold percentiles. `fit_peak_bytes` and
    `run_peak_rss_bytes` are reduced with max-of-folds.
    """
    combo_dirs = discover_combos(run_dir, source, slice_, combos)
    if not combo_dirs:
        raise FileNotFoundError(f"No combos found under {run_dir}")

    if slice_ is None:
        grouped = group_combos_by_id(combo_dirs)
        summary = _synthesize_merged_summary(grouped, _PHASE_PEAK_COLS)
        run_jsons = _merged_run_jsons(grouped)
        return summary, run_jsons

    summary = load_summary(run_dir)
    if source is not None:
        summary = summary.filter(pa.compute.equal(summary.column("source"), source))
    summary = summary.filter(pa.compute.equal(summary.column("slice"), slice_))
    if combos:
        summary = summary.filter(pa.compute.is_in(
            summary.column("combo_id"), value_set=pa.array(list(combos))
        ))
    if summary.num_rows == 0:
        raise ValueError(
            f"No rows in summary.parquet at {run_dir} after filtering "
            f"source={source!r} slice={slice_!r} combos={combos!r}"
        )
    run_jsons = {d.name: load_run_json(d) for d in combo_dirs}
    return summary, run_jsons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot stacked phase-local peak memory + max RSS from a v2 benchmark run.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", default=None)
    parser.add_argument("--slice", dest="slice_", default=None)
    parser.add_argument(
        "--stat",
        choices=("p5", "p25", "p50", "p75", "p95", "mean"),
        default="p95",
    )
    parser.add_argument("--no-fit", action="store_true", help="Hide the FIT bar.")
    parser.add_argument(
        "--combos", nargs="*", default=None,
        help="Restrict to these combo_ids (default: every combo found on disk).",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary, run_jsons = _load_from_run(
        args.run_dir, args.source, args.slice_, combos=args.combos,
    )
    fig = plot_phase_memory(
        summary,
        run_jsons,
        args.out,
        stat=args.stat,
        include_fit=not args.no_fit,
        title=args.title,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
