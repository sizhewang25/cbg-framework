"""2D accuracy-vs-cost box-and-whisker plot from v2 benchmark outputs.

For each combo, draws on a single panel:
  * an IQR rectangle from (cost_p25, error_p25) to (cost_p75, error_p75) —
    the joint inter-quartile region in both dimensions
  * dotted horizontal/vertical whiskers spanning p5–p95 on each axis,
    intersecting at (cost_p50, error_p50)
  * solid short caps at each whisker end (display-space sized so the caps
    stay visible on log axes)
A dashed black line connects the per-combo medians sorted by cost — a
visual Pareto-frontier through the median points.

Two cost axes are supported via --cost:
  runtime       : per-target ltd_ms + mtl_ms + ctr_ms
  memory_alloc  : per-target ltd_alloc + mtl_alloc + ctr_alloc (tracemalloc, MB)
  memory_rss    : per-target ltd_rss   + mtl_rss   + ctr_rss   (sampled RSS, MB)

Continent split is optional (--split-by-main-continent), keyed off the same
filtered_anchors.json target_id → country_code → continent lookup as
plot_error_cdf.

Combo colors follow the deterministic `_v2_io.palette` mapping so the same
combo gets the same hue across every analysis figure in the project.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import offset_copy

from scripts.analysis._v2_io import discover_combos, load_targets, palette
from scripts.processing.ripe_atlas.continents import continent_of

logger = logging.getLogger(__name__)


_COST_SPECS = {
    "runtime": {
        "stage_cols": ("ltd_ms", "mtl_ms", "ctr_ms"),
        "scale": 1.0,
        "label": "Per-target runtime (ms)",
    },
    "memory_alloc": {
        "stage_cols": ("ltd_alloc_peak_bytes", "mtl_alloc_peak_bytes", "ctr_alloc_peak_bytes"),
        "scale": 1.0 / (1024 ** 2),
        "label": "Per-target tracemalloc peak (MB)",
    },
    "memory_rss": {
        "stage_cols": ("ltd_rss_peak_bytes", "mtl_rss_peak_bytes", "ctr_rss_peak_bytes"),
        "scale": 1.0 / (1024 ** 2),
        "label": "Per-target sampled RSS peak (MB)",
    },
}


_CONTINENT_SLUG_TO_CANON: dict[str, str] = {
    "africa": "Africa",
    "antarctica": "Antarctica",
    "asia": "Asia",
    "europe": "Europe",
    "north_america": "North America",
    "oceania": "Oceania",
    "south_america": "South America",
}


# Same prefix-abbrev table as plot_error_cdf so legend labels stay compact.
_LABEL_PREFIX_ABBREV: dict[str, str] = {
    "vanilla_": "va_",
    "million_scale_": "ms_",
    "octant_": "oc_",
    "spotter_": "sp_",
}


def _short_label(cid: str) -> str:
    for prefix, short in _LABEL_PREFIX_ABBREV.items():
        if cid.startswith(prefix):
            return short + cid[len(prefix):]
    return cid


def _normalize_continent(name: str) -> str:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in _CONTINENT_SLUG_TO_CANON:
        raise ValueError(
            f"Unknown continent {name!r}. Valid: "
            f"{sorted(_CONTINENT_SLUG_TO_CANON.values())}"
        )
    return _CONTINENT_SLUG_TO_CANON[key]


def _load_ip_to_continent(filtered_anchors_path: Path) -> dict[str, str]:
    records = json.loads(filtered_anchors_path.read_text())
    return {
        r["address_v4"]: continent_of(r.get("country_code"))
        for r in records
        if r.get("address_v4")
    }


def _load_combo_xy(
    run_dir: Path,
    cost_spec: dict,
    *,
    source: Optional[str],
    slice_: Optional[str],
    combos: Optional[list[str]],
    target_continent: Optional[str],
    filtered_anchors_path: Optional[Path],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Walk per-combo targets.parquet and return {combo_id: (cost_arr, err_arr)}.

    cost_arr is the per-target stage sum (in the cost spec's unit), err_arr
    is per-target error_km. Both filter to SUCCESS+FALLBACK rows. With
    `target_continent` set, the row is additionally filtered to targets
    whose continent matches.
    """
    combo_dirs = discover_combos(run_dir, source, slice_, combos)
    if not combo_dirs:
        raise FileNotFoundError(f"No combos found under {run_dir}")

    ip_to_continent: dict[str, str] = {}
    canon: Optional[str] = None
    if target_continent is not None:
        if filtered_anchors_path is None:
            raise ValueError(
                "--filtered-anchors is required when --split-by-main-continent is set"
            )
        canon = _normalize_continent(target_continent)
        ip_to_continent = _load_ip_to_continent(filtered_anchors_path)

    out: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for combo_dir in combo_dirs:
        cid = combo_dir.name
        tbl = load_targets(combo_dir)
        mask = pc.is_in(tbl.column("status"), value_set=pa.array(["SUCCESS", "FALLBACK"]))
        tbl = tbl.filter(mask)
        if target_continent is not None:
            tids = tbl.column("target_id").to_pylist()
            keep_idx = np.fromiter(
                (i for i, t in enumerate(tids) if ip_to_continent.get(t) == canon),
                dtype=np.int64,
            )
            tbl = tbl.take(keep_idx)
        if tbl.num_rows == 0:
            continue
        err = tbl.column("error_km").to_numpy(zero_copy_only=False).astype(float)
        cost = np.zeros(len(err), dtype=float)
        for col in cost_spec["stage_cols"]:
            arr = tbl.column(col).to_numpy(zero_copy_only=False).astype(float)
            arr = np.where(np.isnan(arr), 0.0, arr)
            cost += arr
        cost *= cost_spec["scale"]
        # Drop rows with NaN error_km (FALLBACK without distance, etc.)
        finite = ~np.isnan(err)
        if not finite.any():
            continue
        out.setdefault(cid, ([], []))[0].append(cost[finite])
        out[cid][1].append(err[finite])

    return {
        cid: (np.concatenate(c_parts), np.concatenate(e_parts))
        for cid, (c_parts, e_parts) in out.items()
    }


def _percentile_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "p5":  float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


def _draw_combo_box(
    ax,
    *,
    cx: dict[str, float],
    cy: dict[str, float],
    color: str,
    label: str,
) -> None:
    """One combo's 2D box-and-whisker."""
    # IQR rectangle (cost p25→p75, error p25→p75) — filled, semi-transparent.
    rect = Rectangle(
        (cx["p25"], cy["p25"]),
        cx["p75"] - cx["p25"],
        cy["p75"] - cy["p25"],
        facecolor=color, edgecolor=color, alpha=0.25, linewidth=1.5, label=label,
    )
    ax.add_patch(rect)

    # Whiskers: dotted lines crossing at the (p50, p50) center.
    ax.plot([cx["p5"], cx["p95"]], [cy["p50"], cy["p50"]],
            color=color, linestyle=":", linewidth=1.5, alpha=0.9)
    ax.plot([cx["p50"], cx["p50"]], [cy["p5"], cy["p95"]],
            color=color, linestyle=":", linewidth=1.5, alpha=0.9)

    # Cap markers: short perpendicular segments at each whisker end, sized
    # in display-space points so they stay visible on log axes.
    cap_pts = 5.0  # half-length in points
    trans_x = offset_copy(ax.transData, fig=ax.figure, x=cap_pts, y=0, units="points")
    trans_x_neg = offset_copy(ax.transData, fig=ax.figure, x=-cap_pts, y=0, units="points")
    trans_y = offset_copy(ax.transData, fig=ax.figure, x=0, y=cap_pts, units="points")
    trans_y_neg = offset_copy(ax.transData, fig=ax.figure, x=0, y=-cap_pts, units="points")
    # X caps (horizontal whisker ends — vertical cap lines)
    for cap_x in (cx["p5"], cx["p95"]):
        ax.annotate("", xy=(cap_x, cy["p50"]), xycoords="data",
                    xytext=(cap_x, cy["p50"]), textcoords=trans_y,
                    arrowprops=dict(arrowstyle="-", color=color, linewidth=1.5))
        ax.annotate("", xy=(cap_x, cy["p50"]), xycoords="data",
                    xytext=(cap_x, cy["p50"]), textcoords=trans_y_neg,
                    arrowprops=dict(arrowstyle="-", color=color, linewidth=1.5))
    # Y caps (vertical whisker ends — horizontal cap lines)
    for cap_y in (cy["p5"], cy["p95"]):
        ax.annotate("", xy=(cx["p50"], cap_y), xycoords="data",
                    xytext=(cx["p50"], cap_y), textcoords=trans_x,
                    arrowprops=dict(arrowstyle="-", color=color, linewidth=1.5))
        ax.annotate("", xy=(cx["p50"], cap_y), xycoords="data",
                    xytext=(cx["p50"], cap_y), textcoords=trans_x_neg,
                    arrowprops=dict(arrowstyle="-", color=color, linewidth=1.5))


def plot_accuracy_cost_box(
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    *,
    x_label: str,
    title: Optional[str] = None,
    log_x: bool = True,
    log_y: bool = True,
    colors: Optional[dict[str, str]] = None,
    figsize: tuple[float, float] = (11.0, 7.5),
) -> plt.Figure:
    """Draw the 2D box-and-whisker plot.

    `data` is `{combo_id: (cost_array, error_km_array)}` over per-target rows.
    Empty cost or error arrays for a combo are skipped silently.
    """
    if colors is None:
        colors = palette(list(data))

    stats: dict[str, tuple[dict, dict]] = {}
    for cid, (cost_arr, err_arr) in data.items():
        if len(cost_arr) == 0 or len(err_arr) == 0:
            continue
        stats[cid] = (_percentile_stats(cost_arr), _percentile_stats(err_arr))

    if not stats:
        raise ValueError("No data to plot — every combo had zero rows after filtering")

    fig, ax = plt.subplots(figsize=figsize)

    # Pareto-style median line — sorted by cost p50.
    sorted_cids = sorted(stats, key=lambda c: stats[c][0]["p50"])
    pareto_x = [stats[c][0]["p50"] for c in sorted_cids]
    pareto_y = [stats[c][1]["p50"] for c in sorted_cids]
    ax.plot(pareto_x, pareto_y, color="black", linestyle="--", linewidth=1.5,
            alpha=0.6, zorder=1, label="median trajectory")

    # Per-combo boxes — sort by cost p50 too so legend order matches Pareto.
    for cid in sorted_cids:
        cx, cy = stats[cid]
        _draw_combo_box(ax, cx=cx, cy=cy,
                        color=colors.get(cid, "#4E79A7"),
                        label=_short_label(cid))

    if log_x:
        ax.set_xscale("log")
        fmt_x = ScalarFormatter()
        fmt_x.set_scientific(False)
        ax.xaxis.set_major_formatter(fmt_x)
    if log_y:
        ax.set_yscale("log")
        fmt_y = ScalarFormatter()
        fmt_y.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt_y)

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Error distance (km)", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    # Two-column legend keeps a 16-combo plot tractable.
    ncol = 2 if len(stats) <= 8 else 3
    ax.legend(loc="upper right", fontsize=8, ncol=ncol, framealpha=0.9)

    fig.suptitle(title or "Accuracy vs cost", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Path to outputs/<run_id>/ (contains per-combo targets.parquet).")
    p.add_argument("--source", default=None, help="Filter combos by source name.")
    p.add_argument("--slice", dest="slice_", default=None,
                   help="Filter combos by slice id (omit for merged-folds mode).")
    p.add_argument("--cost", choices=tuple(_COST_SPECS), default="runtime",
                   help="X-axis quantity to plot against error_km.")
    p.add_argument("--combos", nargs="*", default=None,
                   help="Restrict to these combo_ids (default: every combo found on disk).")
    p.add_argument("--split-by-main-continent", default=None,
                   help="Keep only targets in this continent (e.g. 'north_america').")
    p.add_argument("--filtered-anchors", type=Path,
                   default=Path("datasets/ripe_atlas/filtered_anchors.json"),
                   help="Anchor metadata for target→continent lookup.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--no-log-x", action="store_true")
    p.add_argument("--no-log-y", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cost_spec = _COST_SPECS[args.cost]
    data = _load_combo_xy(
        args.run_dir,
        cost_spec,
        source=args.source,
        slice_=args.slice_,
        combos=args.combos,
        target_continent=args.split_by_main_continent,
        filtered_anchors_path=args.filtered_anchors,
    )

    title = args.title
    if title is None:
        suffix = f" — in {args.split_by_main_continent}" if args.split_by_main_continent else ""
        title = f"Accuracy vs {args.cost}{suffix}"

    fig = plot_accuracy_cost_box(
        data, args.out, x_label=cost_spec["label"], title=title,
        log_x=not args.no_log_x, log_y=not args.no_log_y,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
