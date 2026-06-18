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
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from matplotlib.ticker import ScalarFormatter

from scripts.analysis._v2_io import (
    active_geo_filter,
    add_geo_filter_args,
    discover_combos,
    group_combos_by_id,
    load_summary,
    load_targets,
    palette,
    route_geo_path,
    set_geo_filter_from_args,
)
from scripts.libs.cbg.rtt_model import haversine_distance
from scripts.processing.ripe_atlas.continents import continent_of

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


# CBG variant prefix → two-letter abbreviation, applied to both the stats
# panel (which truncates at 16 chars) and legend labels. Order doesn't
# matter — prefixes are mutually exclusive on the existing combo set.
_LABEL_PREFIX_ABBREV: dict[str, str] = {
    "vanilla_": "va_",
    "million_scale_": "ms_",
    "octant_": "oc_",
    "spotter_": "sp_",
}


def _short_label(cid: str) -> str:
    """Compact form of `cid` for legends and the stats panel: replaces the
    leading CBG-variant family name with a two-letter abbreviation so labels
    fit inside the 16-char stats-panel column and keep the legend tidy."""
    for prefix, short in _LABEL_PREFIX_ABBREV.items():
        if cid.startswith(prefix):
            return short + cid[len(prefix):]
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
    figsize_per_panel: Optional[tuple[float, float]] = None,
    stats_box_loc: str = "upper left",
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
        figsize_per_panel: (width, height) inches per panel. Total figure
            width is `panels × width`. Default depends on the resolved panel
            count: (10, 7) when only one panel renders (either group_by=None
            or group_by="ltd" with a single LTD across all combos — wider so
            the upper-left stats box doesn't cover the CDF curves), and
            (6, 7) when multiple LTD panels render side-by-side.
        stats_box_loc: "upper left" (default — anchored just below the
            upper-left legend) or "lower right" (renders in the
            lower-right corner; useful when the legend area is crowded).

    The stats-panel combo rows are ordered by p50 (median) error ascending
    (best first), independent of curve/legend order. The baseline row (if
    any) stays pinned at the bottom as a fixed reference.
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

    if figsize_per_panel is None:
        figsize_per_panel = (10.0, 7.0) if len(panels) == 1 else (6.0, 7.0)

    if colors is None:
        colors = palette(list(errors_by_combo))

    count_header = "succ/total"
    count_width = 11

    n_panels = len(panels)
    pw, ph = figsize_per_panel
    fig, axes = plt.subplots(
        1, n_panels, figsize=(pw * n_panels, ph), sharey=True, squeeze=False,
    )
    axes = axes[0]
    threshold_colors = {100: "green", 500: "orange", 1000: "red"}
    # Deferred stats-box state, populated in the panel loop and rendered
    # AFTER tight_layout so the legend bbox we read off is the final one.
    deferred_stats: list[tuple] = []

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
                label=_short_label(cid),
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
        legend = ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xscale("log")
        # Plain-number tick labels (1, 10, 100, ...) instead of 10^k form.
        x_fmt = ScalarFormatter()
        x_fmt.set_scientific(False)
        ax.xaxis.set_major_formatter(x_fmt)
        ax.set_xlim(1, max_x_km)
        ax.set_ylim(0, 1)

        if panel_data or baseline_sorted is not None:
            # `method="nearest"` snaps each percentile to an existing sample
            # so values here align with what the MTL world-map viewer shows
            # when you pick the same percentile bookmark.
            def _pcts(xs):
                return np.percentile(xs, [5, 25, 50, 75, 95], method="nearest")

            lines = [f"{'':<16} {count_header:>{count_width}}    p5   p25   p50   p75   p95"]
            stats_rows = sorted(
                panel_data,
                key=lambda row: _pcts(row[1])[2],  # p50
            )
            for cid, errors, count_str in stats_rows:
                p5, p25, p50, p75, p95 = _pcts(errors)
                lines.append(
                    f"{_short_label(cid)[:16]:<16} {count_str:>{count_width}} "
                    f"{p5:5.0f} {p25:5.0f} {p50:5.0f} {p75:5.0f} {p95:5.0f}"
                )
            if baseline_sorted is not None:
                p5, p25, p50, p75, p95 = _pcts(baseline_sorted)
                lines.append(
                    f"{baseline_label[:16]:<16} {str(len(baseline_sorted)):>{count_width}} "
                    f"{p5:5.0f} {p25:5.0f} {p50:5.0f} {p75:5.0f} {p95:5.0f}"
                )

            deferred_stats.append((ax, legend, "\n".join(lines)))

    axes[0].set_ylabel("CDF", fontsize=12)
    fig.suptitle(
        title or ("Error CDF by LTD" if group_by == "ltd" else "Error CDF"),
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    # Render stats boxes after tight_layout so legend bboxes are final.
    fig.canvas.draw()
    for ax, legend, text in deferred_stats:
        if stats_box_loc == "upper left":
            leg_bbox = legend.get_window_extent().transformed(
                ax.transAxes.inverted()
            )
            # Generous gap (~3% of axes height) so the stat box sits cleanly
            # below the legend even with display-vs-render rounding.
            text_x, text_y = 0.02, leg_bbox.ymin - 0.03
            va, ha = "top", "left"
        elif stats_box_loc == "lower right":
            text_x, text_y, va, ha = 0.98, 0.02, "bottom", "right"
        else:
            raise ValueError(
                f"unsupported stats_box_loc={stats_box_loc!r}; "
                "use 'lower right' or 'upper left'"
            )
        ax.text(
            text_x, text_y, text,
            transform=ax.transAxes, fontsize=7,
            verticalalignment=va, horizontalalignment=ha,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            family="monospace",
        )

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
    combos: Optional[list[str]] = None,
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
    combo_dirs = discover_combos(run_dir, source, slice_, combos)
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


def _load_nearest_ping_baseline_by_target(inputs_dir: Path) -> dict[str, float]:
    """Like `_load_nearest_ping_baseline` but keyed by target_id so callers can
    subset per-target (e.g. by continent membership).
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
        return {}

    idx = df.groupby("target_id")["latency_ms"].idxmin()
    nearest = df.loc[idx]
    errors = haversine_distance(
        nearest["target_lat"].to_numpy(dtype=float),
        nearest["target_lon"].to_numpy(dtype=float),
        nearest["vp_lat"].to_numpy(dtype=float),
        nearest["vp_lon"].to_numpy(dtype=float),
    )
    return dict(zip(nearest["target_id"].tolist(), errors.tolist()))


def _load_nearest_ping_baseline_by_fold_target(inputs_dir: Path) -> dict[str, float]:
    """Like `_load_nearest_ping_baseline_by_target` but keys by
    `<fold>/<target_id>` to match the fold-prefixed convention used in
    diff-CDF joins (`plot_error_diff_cdf._load_from_run`).

    Single-fold mode (`<inputs_dir>/eval_observations.parquet` exists):
    fold name is `inputs_dir.name`. Merged-folds mode: globs
    `<inputs_dir>/*/eval_observations.parquet` and pulls each fold name
    from the parent dir.
    """
    direct = inputs_dir / "eval_observations.parquet"
    if direct.exists():
        per_fold = [(inputs_dir.name, direct)]
    else:
        paths = sorted(inputs_dir.glob("*/eval_observations.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"No eval_observations.parquet at {inputs_dir} or under "
                f"{inputs_dir}/*/. Pass --inputs-dir pointing at a fold "
                "input dir or at its parent (for merged-fold mode)."
            )
        per_fold = [(p.parent.name, p) for p in paths]

    import pandas as pd
    frames = []
    for fold, p in per_fold:
        df = pq.read_table(p).to_pandas()
        df["__fold"] = fold
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return {}

    idx = df.groupby(["__fold", "target_id"])["latency_ms"].idxmin()
    nearest = df.loc[idx]
    errors = haversine_distance(
        nearest["target_lat"].to_numpy(dtype=float),
        nearest["target_lon"].to_numpy(dtype=float),
        nearest["vp_lat"].to_numpy(dtype=float),
        nearest["vp_lon"].to_numpy(dtype=float),
    )
    return {
        f"{f}/{t}": float(e)
        for f, t, e in zip(
            nearest["__fold"].tolist(),
            nearest["target_id"].tolist(),
            errors.tolist(),
        )
    }


def _load_nearest_ping_baseline(
    inputs_dir: Path, allowed_target_ids: Optional[set[str]] = None
) -> np.ndarray:
    """Read eval_observations.parquet and compute haversine error from each
    target to the location of its smallest-latency VP. One error per target.

    If `inputs_dir/eval_observations.parquet` exists, it's read directly
    (single-fold mode). Otherwise the directory is treated as the parent of
    per-fold input dirs and `**/eval_observations.parquet` is globbed and
    concatenated — the merged-folds counterpart of the merged-folds loader
    in `_load_from_run`.

    `eval_observations.parquet` has no geo columns, so when a geo filter is
    active the baseline can't filter itself. `allowed_target_ids` (the target
    ids that survive the filter in `targets.parquet`) restricts the baseline to
    the same subset so the overlaid line matches the CDF curves.
    """
    by_target = _load_nearest_ping_baseline_by_target(inputs_dir)
    if allowed_target_ids is not None:
        by_target = {k: v for k, v in by_target.items() if k in allowed_target_ids}
    return np.asarray(list(by_target.values()), dtype=float)


def _geo_allowed_target_ids(
    run_dir: Path,
    source: Optional[str],
    slice_: Optional[str],
    combos: Optional[list[str]] = None,
) -> Optional[set[str]]:
    """Target ids surviving the active geo filter, or None when no filter is set.

    `load_targets` already applies the filter, so this just unions the surviving
    `target_id`s across the run's combo dirs — used to subset the shortest-ping
    baseline to the selected geography.
    """
    if active_geo_filter() is None:
        return None
    ids: set[str] = set()
    for d in discover_combos(run_dir, source, slice_, combos):
        ids.update(load_targets(d).column("target_id").to_pylist())
    return ids


_CONTINENT_SLUG_TO_CANON: dict[str, str] = {
    "africa": "Africa",
    "antarctica": "Antarctica",
    "asia": "Asia",
    "europe": "Europe",
    "north_america": "North America",
    "oceania": "Oceania",
    "south_america": "South America",
}


def _normalize_continent(name: str) -> str:
    """Accept slug (`north_america`) or canonical (`North America`) and return
    the canonical form. Raise `ValueError` listing valid names on miss.
    """
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in _CONTINENT_SLUG_TO_CANON:
        raise ValueError(
            f"Unknown continent {name!r}. Valid: "
            f"{sorted(_CONTINENT_SLUG_TO_CANON.values())}"
        )
    return _CONTINENT_SLUG_TO_CANON[key]


def _load_ip_to_continent(filtered_anchors_path: Path) -> dict[str, str]:
    """Read `filtered_anchors.json` and return `{address_v4: continent}` via
    `continent_of(country_code)`. Anchors with no `country_code` map to
    `"Unknown"`.
    """
    records = json.loads(filtered_anchors_path.read_text())
    out: dict[str, str] = {}
    for r in records:
        ip = r.get("address_v4")
        if not ip:
            continue
        out[ip] = continent_of(r.get("country_code"))
    return out


def plot_error_cdf_by_continent(
    run_dir: Path,
    target_continent: str,
    output_path: Path,
    *,
    filtered_anchors_path: Path,
    source: Optional[str] = None,
    slice_: Optional[str] = None,
    inputs_dir: Optional[Path] = None,
    max_x_km: float = 10000.0,
    title: Optional[str] = None,
    combos: Optional[list[str]] = None,
) -> tuple[plt.Figure, plt.Figure]:
    """SUCCESS-only error CDF split by whether each target sits in
    `target_continent`. Writes two figures derived from `output_path`:
    `<stem>_<slug>.png` (in-continent) and `<stem>_rest.png` (everything else).

    Rationale: per-ASN VP corpora are continent-bounded fleets; regional
    VPs are expected to recover well in their home continent and degrade
    outside it. A single overall CDF averages those two regimes together
    and hides the asymmetry — splitting the eval rows makes it visible.

    The lookup uses `target_id` (an IPv4 anchor address) joined against
    `filtered_anchors.json`'s `address_v4` field; the per-target continent
    comes from `continent_of(country_code)`. Anchor IPs that are missing
    from `filtered_anchors.json` or whose continent resolves to "Unknown"
    are dropped from both groups with a single warning naming the count.
    """
    canon = _normalize_continent(target_continent)
    ip_to_continent = _load_ip_to_continent(filtered_anchors_path)

    combo_dirs = discover_combos(run_dir, source, slice_, combos)
    if not combo_dirs:
        raise FileNotFoundError(f"No combos found under {run_dir}")

    in_errors: dict[str, list[float]] = {}
    in_succ: dict[str, int] = {}
    in_total: dict[str, int] = {}
    rest_errors: dict[str, list[float]] = {}
    rest_succ: dict[str, int] = {}
    rest_total: dict[str, int] = {}
    unknown_target_ids: set[str] = set()

    for combo_dir in combo_dirs:
        cid = combo_dir.name
        tbl = load_targets(combo_dir)
        target_ids = tbl.column("target_id").to_pylist()
        status_arr = tbl.column("status").to_pylist()
        error_arr = tbl.column("error_km").to_numpy(zero_copy_only=False)

        for tid, st, err in zip(target_ids, status_arr, error_arr):
            cont = ip_to_continent.get(tid, "Unknown")
            if cont == "Unknown":
                unknown_target_ids.add(tid)
                continue
            in_grp = cont == canon
            if in_grp:
                in_total[cid] = in_total.get(cid, 0) + 1
                if st == "SUCCESS":
                    in_succ[cid] = in_succ.get(cid, 0) + 1
                    if not np.isnan(err):
                        in_errors.setdefault(cid, []).append(float(err))
            else:
                rest_total[cid] = rest_total.get(cid, 0) + 1
                if st == "SUCCESS":
                    rest_succ[cid] = rest_succ.get(cid, 0) + 1
                    if not np.isnan(err):
                        rest_errors.setdefault(cid, []).append(float(err))

    if unknown_target_ids:
        logger.warning(
            "%d anchor IPs missing from %s or with unknown country_code; "
            "their eval rows were dropped",
            len(unknown_target_ids), filtered_anchors_path,
        )

    all_cids = sorted({d.name for d in combo_dirs})
    in_errors_np: dict[str, np.ndarray] = {
        cid: np.asarray(in_errors.get(cid, []), dtype=float) for cid in all_cids
    }
    rest_errors_np: dict[str, np.ndarray] = {
        cid: np.asarray(rest_errors.get(cid, []), dtype=float) for cid in all_cids
    }
    for cid in all_cids:
        in_succ.setdefault(cid, 0)
        in_total.setdefault(cid, 0)
        rest_succ.setdefault(cid, 0)
        rest_total.setdefault(cid, 0)

    summary = load_summary(run_dir)
    combo_to_ltd = dict(zip(
        summary.column("combo_id").to_pylist(),
        summary.column("ltd").to_pylist(),
    ))

    in_baseline: Optional[np.ndarray] = None
    rest_baseline: Optional[np.ndarray] = None
    if inputs_dir is not None:
        baseline_by_target = _load_nearest_ping_baseline_by_target(inputs_dir)
        in_b, rest_b = [], []
        for tid, err in baseline_by_target.items():
            cont = ip_to_continent.get(tid, "Unknown")
            if cont == "Unknown":
                continue
            (in_b if cont == canon else rest_b).append(err)
        in_baseline = np.asarray(in_b, dtype=float) if in_b else None
        rest_baseline = np.asarray(rest_b, dtype=float) if rest_b else None
        logger.info(
            "shortest_ping baseline split: in=%d, rest=%d",
            len(in_b), len(rest_b),
        )

    slug = canon.lower().replace(" ", "_")
    in_path = output_path.with_stem(f"{output_path.stem}_{slug}")
    rest_path = output_path.with_stem(f"{output_path.stem}_rest")
    base_title = title or "Error CDF — SUCCESS only by LTD"

    colors = palette(all_cids)
    fig_in = plot_error_cdf(
        in_errors_np,
        in_path,
        successes_by_combo=in_succ,
        totals_by_combo=in_total,
        baseline_errors=in_baseline,
        group_by="ltd",
        combo_to_ltd=combo_to_ltd,
        max_x_km=max_x_km,
        colors=colors,
        title=f"{base_title} — in {canon}",
    )
    fig_rest = plot_error_cdf(
        rest_errors_np,
        rest_path,
        successes_by_combo=rest_succ,
        totals_by_combo=rest_total,
        baseline_errors=rest_baseline,
        group_by="ltd",
        combo_to_ltd=combo_to_ltd,
        max_x_km=max_x_km,
        colors=colors,
        title=f"{base_title} — rest of world",
    )
    return fig_in, fig_rest


def plot_error_cdf_merge(
    run_dir: Path,
    combos_to_merge: list[str],
    output_path: Path,
    *,
    source: Optional[str] = None,
    slice_: Optional[str] = None,
    inputs_dir: Optional[Path] = None,
    success_only: bool = False,
    max_x_km: float = 10000.0,
    title: Optional[str] = None,
    allowed_target_ids: Optional[set[str]] = None,
) -> plt.Figure:
    """Plot error CDFs for `combos_to_merge` overlaid on a single panel
    (no LTD grouping).

    Driven by the `merge_pairs` config field: callers pass the resolved list of
    combo_ids they want stacked on one CDF for a tight head-to-head read. The
    palette is derived from *every* combo found under `run_dir` (not just the
    merged subset), so each combo keeps the same color as on the LTD-grouped
    `plot_error_cdf` figures.

    When `inputs_dir` is provided a shortest-ping baseline is overlaid (same
    semantics as `plot_error_cdf`).
    """
    errors_by_combo, successes_by_combo, totals_by_combo, _ = _load_from_run(
        run_dir, source, slice_,
        success_only=success_only, combos=combos_to_merge,
    )

    # Reorder loaded dicts to follow `combos_to_merge` order — this becomes the
    # plot order in both the legend and the stats box (Python dicts preserve
    # insertion order; the single-panel branch of plot_error_cdf iterates them
    # in that order).
    errors_by_combo = {
        c: errors_by_combo[c] for c in combos_to_merge if c in errors_by_combo
    }
    successes_by_combo = {
        c: successes_by_combo[c] for c in combos_to_merge if c in successes_by_combo
    }
    totals_by_combo = {
        c: totals_by_combo[c] for c in combos_to_merge if c in totals_by_combo
    }

    # Palette over the full combo set in the run so colors stay aligned with
    # the LTD-grouped plots.
    all_dirs = discover_combos(run_dir, source, slice_)
    colors = palette(sorted({d.name for d in all_dirs}))

    baseline_errors = None
    if inputs_dir is not None:
        baseline_errors = _load_nearest_ping_baseline(inputs_dir, allowed_target_ids)

    base_title = title or (
        "Error CDF — SUCCESS only (merged combos)" if success_only
        else "Error CDF (merged combos)"
    )
    return plot_error_cdf(
        errors_by_combo,
        output_path,
        successes_by_combo=successes_by_combo,
        totals_by_combo=totals_by_combo,
        baseline_errors=baseline_errors,
        group_by=None,
        max_x_km=max_x_km,
        colors=colors,
        title=base_title,
        figsize_per_panel=(8.0, 6.0),  # 4:3
        stats_box_loc="upper left",
    )


def plot_error_cdf_merge_by_continent(
    run_dir: Path,
    combos_to_merge: list[str],
    target_continent: str,
    output_path: Path,
    *,
    filtered_anchors_path: Path,
    source: Optional[str] = None,
    slice_: Optional[str] = None,
    inputs_dir: Optional[Path] = None,
    max_x_km: float = 10000.0,
    title: Optional[str] = None,
) -> tuple[plt.Figure, plt.Figure]:
    """Continent-split, SUCCESS-only single-panel CDF for `combos_to_merge`.

    Writes `<output_path stem>_<continent_slug>.png` and `<stem>_rest.png`.
    Colors are derived from every combo found under `run_dir`, matching the
    LTD-grouped `plot_error_cdf_by_continent` outputs.
    """
    canon = _normalize_continent(target_continent)
    ip_to_continent = _load_ip_to_continent(filtered_anchors_path)

    combo_dirs = discover_combos(run_dir, source, slice_, combos_to_merge)
    if not combo_dirs:
        raise FileNotFoundError(
            f"No combos in {combos_to_merge} found under {run_dir}"
        )

    in_errors: dict[str, list[float]] = {}
    in_succ: dict[str, int] = {}
    in_total: dict[str, int] = {}
    rest_errors: dict[str, list[float]] = {}
    rest_succ: dict[str, int] = {}
    rest_total: dict[str, int] = {}
    unknown_target_ids: set[str] = set()

    for combo_dir in combo_dirs:
        cid = combo_dir.name
        tbl = load_targets(combo_dir)
        target_ids = tbl.column("target_id").to_pylist()
        status_arr = tbl.column("status").to_pylist()
        error_arr = tbl.column("error_km").to_numpy(zero_copy_only=False)
        for tid, st, err in zip(target_ids, status_arr, error_arr):
            cont = ip_to_continent.get(tid, "Unknown")
            if cont == "Unknown":
                unknown_target_ids.add(tid)
                continue
            in_grp = cont == canon
            if in_grp:
                in_total[cid] = in_total.get(cid, 0) + 1
                if st == "SUCCESS":
                    in_succ[cid] = in_succ.get(cid, 0) + 1
                    if not np.isnan(err):
                        in_errors.setdefault(cid, []).append(float(err))
            else:
                rest_total[cid] = rest_total.get(cid, 0) + 1
                if st == "SUCCESS":
                    rest_succ[cid] = rest_succ.get(cid, 0) + 1
                    if not np.isnan(err):
                        rest_errors.setdefault(cid, []).append(float(err))

    if unknown_target_ids:
        logger.warning(
            "%d anchor IPs missing from %s or with unknown country_code; "
            "their eval rows were dropped",
            len(unknown_target_ids), filtered_anchors_path,
        )

    # Follow `combos_to_merge` order so the legend and stats box match the
    # config — single-panel rendering iterates dict order.
    discovered = {d.name for d in combo_dirs}
    requested_cids = [c for c in combos_to_merge if c in discovered]
    in_errors_np = {
        cid: np.asarray(in_errors.get(cid, []), dtype=float)
        for cid in requested_cids
    }
    rest_errors_np = {
        cid: np.asarray(rest_errors.get(cid, []), dtype=float)
        for cid in requested_cids
    }
    for cid in requested_cids:
        in_succ.setdefault(cid, 0)
        in_total.setdefault(cid, 0)
        rest_succ.setdefault(cid, 0)
        rest_total.setdefault(cid, 0)

    # Palette over the full combo set in the run for color consistency.
    all_dirs = discover_combos(run_dir, source, slice_)
    colors = palette(sorted({d.name for d in all_dirs}))

    in_baseline: Optional[np.ndarray] = None
    rest_baseline: Optional[np.ndarray] = None
    if inputs_dir is not None:
        baseline_by_target = _load_nearest_ping_baseline_by_target(inputs_dir)
        in_b, rest_b = [], []
        for tid, err in baseline_by_target.items():
            cont = ip_to_continent.get(tid, "Unknown")
            if cont == "Unknown":
                continue
            (in_b if cont == canon else rest_b).append(err)
        in_baseline = np.asarray(in_b, dtype=float) if in_b else None
        rest_baseline = np.asarray(rest_b, dtype=float) if rest_b else None

    slug = canon.lower().replace(" ", "_")
    in_path = output_path.with_stem(f"{output_path.stem}_{slug}")
    rest_path = output_path.with_stem(f"{output_path.stem}_rest")
    base_title = title or "Error CDF — SUCCESS only (merged combos)"

    fig_in = plot_error_cdf(
        in_errors_np,
        in_path,
        successes_by_combo=in_succ,
        totals_by_combo=in_total,
        baseline_errors=in_baseline,
        group_by=None,
        max_x_km=max_x_km,
        colors=colors,
        title=f"{base_title} — in {canon}",
        figsize_per_panel=(8.0, 6.0),  # 4:3
        stats_box_loc="upper left",
    )
    fig_rest = plot_error_cdf(
        rest_errors_np,
        rest_path,
        successes_by_combo=rest_succ,
        totals_by_combo=rest_total,
        baseline_errors=rest_baseline,
        group_by=None,
        max_x_km=max_x_km,
        colors=colors,
        title=f"{base_title} — rest of world",
        figsize_per_panel=(8.0, 6.0),
        stats_box_loc="upper left",
    )
    return fig_in, fig_rest


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
    parser.add_argument(
        "--split-by-main-continent",
        default=None,
        help="When set, write two SUCCESS-only CDF figures split by whether "
             "each target sits in this continent (e.g. 'north_america', "
             "'europe'). Output paths: <out_stem>_<slug>.png and "
             "<out_stem>_rest.png. Ignores --success-only (always SUCCESS).",
    )
    parser.add_argument(
        "--filtered-anchors",
        type=Path,
        default=Path("datasets/ripe_atlas/filtered_anchors.json"),
        help="Anchor metadata for target→continent lookup. Used only with "
             "--split-by-main-continent.",
    )
    parser.add_argument(
        "--combos", nargs="*", default=None,
        help="Restrict to these combo_ids (default: every combo found on disk).",
    )
    parser.add_argument(
        "--merge-combos", nargs="*", default=None,
        help="Overlay these combo_ids onto a single CDF panel (no LTD "
             "grouping). Drives the merged-CDF outputs. Colors stay aligned "
             "with the LTD-grouped plots. Combines with "
             "--split-by-main-continent.",
    )
    add_geo_filter_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_geo_filter_from_args(args)

    if active_geo_filter() is not None and args.split_by_main_continent is not None:
        raise SystemExit(
            "--geo-level/--geo-value and --split-by-main-continent both slice "
            "by geography; use one or the other. The geo filter reads the "
            "geo-eval columns on targets.parquet; --split-by-main-continent "
            "uses the external filtered_anchors.json join."
        )

    # Route the output under geo/<level>/<value>/ when a filter is active, and
    # subset the shortest-ping baseline to the surviving target ids.
    args.out = route_geo_path(args.out)
    allowed_ids = _geo_allowed_target_ids(
        args.run_dir, args.source, args.slice_,
        args.merge_combos or args.combos,
    )

    if args.merge_combos:
        if args.split_by_main_continent is not None:
            fig_in, fig_rest = plot_error_cdf_merge_by_continent(
                args.run_dir,
                args.merge_combos,
                args.split_by_main_continent,
                args.out,
                filtered_anchors_path=args.filtered_anchors,
                source=args.source,
                slice_=args.slice_,
                inputs_dir=args.inputs_dir,
                max_x_km=args.max_x_km,
                title=args.title,
            )
            plt.close(fig_in)
            plt.close(fig_rest)
        else:
            fig = plot_error_cdf_merge(
                args.run_dir,
                args.merge_combos,
                args.out,
                source=args.source,
                slice_=args.slice_,
                inputs_dir=args.inputs_dir,
                success_only=args.success_only,
                max_x_km=args.max_x_km,
                title=args.title,
                allowed_target_ids=allowed_ids,
            )
            plt.close(fig)
        return

    if args.split_by_main_continent is not None:
        fig_in, fig_rest = plot_error_cdf_by_continent(
            args.run_dir,
            args.split_by_main_continent,
            args.out,
            filtered_anchors_path=args.filtered_anchors,
            source=args.source,
            slice_=args.slice_,
            inputs_dir=args.inputs_dir,
            max_x_km=args.max_x_km,
            title=args.title,
            combos=args.combos,
        )
        plt.close(fig_in)
        plt.close(fig_rest)
        return

    errors_by_combo, successes_by_combo, totals_by_combo, combo_to_ltd = _load_from_run(
        args.run_dir, args.source, args.slice_,
        success_only=args.success_only, combos=args.combos,
    )
    group_by = None if args.group_by == "none" else args.group_by

    baseline_errors = None
    if args.inputs_dir is not None:
        baseline_errors = _load_nearest_ping_baseline(args.inputs_dir, allowed_ids)
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
