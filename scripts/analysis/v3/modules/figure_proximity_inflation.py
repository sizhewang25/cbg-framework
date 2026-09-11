"""§8.1's geometry-vs-routing scatter: how close a VP was, and how much the RTT
that reached the target was inflated.

The headline table and the outcome bars say *how well* each method did on each
dataset type. This figure is the first one that says **what the targets were
like**, which is what the traffic-weighted comparison turns on: the weighted
campaign is the mesh one filtered to the flows carrying the top 95% of traffic,
so its targets are expected to sit closer to a peering location and to be
reached over a better-routed path. Those are two different claims, and this
figure puts one on each axis.

## y is routing proximity, and it is one column

`min_inflation`: the smallest ratio of observed RTT to the speed-of-internet RTT
for that geodesic, over the target's measured VPs. 1.0 is a path as fast as ⅔c
allows; 1.5 means the best-routed VP still spends half again as long as its
distance requires. It is `eval_source`'s definition, carried through
`build-proximity` rather than recomputed, because the theoretical slope behind
it is v2's constant. Being a **minimum over VPs**, it describes the best path
available to the target rather than a typical one.

## x is geometric proximity, and it is two columns — the choice matters

Both are distances to the target's **seed**, never to the target, for the reason
`proximity.py` gives at length: classification labels a coordinate by its
nearest seed, so the seed is what a guarantee can be stated about.

* `--x-metric closest` -> `tg_seed_nearest_vp_km`, the nearest *measured* VP.
  **Geometric opportunity, with routing divided out.** It asks whether the
  campaign put a vantage point anywhere near the answer, and nothing about
  whether RTT found it.
* `--x-metric sping` -> `sping_vp_to_tg_seed_km`, the VP the lowest RTT
  selected. **Geometry after routing has chosen.** It is the baseline's own
  error in kilometres before snapping, so its left mode *is* Shortest-Ping's
  accuracy — but RTT picked that VP, so a low value already implies a
  well-behaved path.

The two are not interchangeable and they do not tell the same story on these
runs. Against `closest`, Spearman ρ with inflation is -0.02 pooled (-0.35 /
0.18 / 0.02 per AS): geometric opportunity and routing quality are **unrelated**,
and 95.3% of mesh targets have a VP inside their own seed margin. Against
`sping` it is 0.54 pooled — but that correlation is partly the axis reading its
own y, since inflation is what decides which VP the RTT ranking hands over.
Both are emitted; a claim about *geometry* has to be made on `closest`.

## Colour is the dataset type, and nothing else

Mesh grey, traffic-weighted red, as §8.1 asks. No variant hue appears, because
no variant runs in this figure — these are properties of the targets, fixed
before any method is applied. The red is dE 10.4 from Spotter's hue, which
would matter in a figure that drew both; here there is nothing for it to be
confused with, the same stance `figure_outcome_bars` takes with green and red.
Grey and red are dE 72.5 apart, both clear 3:1 on white (3.59 / 4.94), and the
worst simulated deuteran/protan separation is dE 30.1, so the pair survives
colour-vision deficiency without a second channel.

Weighted is drawn **over** mesh, and its marginals are density-normalised, since
the weighted campaign is a subset and will always carry fewer points. A count
histogram would show it as a low bump under the mesh curve and say nothing about
where its targets concentrate.

## Two reference marks, both of them thresholds the paper already uses

* **The SoI floor**, `y = 1`. Below it a measurement would be faster than ⅔c,
  which is why nothing is there — it is the physical bound the whole framework's
  calibration-free variant is built on, and the distance from it is the figure's
  y-axis story.
* **The seed margin band**, the p25-p50-p75 of `tg_seed_margin_km` over the
  panel's targets. Half the distance from a seed to its nearest neighbour, so a
  VP inside it is *guaranteed* to snap to the right class. It is a band and not
  a line because the threshold is per target: the answer space is denser in some
  metros than others, and drawing one number would promise a sharp boundary the
  data does not have.

## The traffic-weighted half is absent, not empty

No run carries traffic weights, so today only the mesh series is drawn. The
legend still names the weighted series and marks it uncollected — the same
stance as the table's `-` rows and the bars' ghost outlines. When a weighted run
lands, `--weighted-run-id <dataset>=<run_id>` fills it with no other change.

Command: `plot-proximity-inflation`. Writes to
`outputs/analysis/v3/_cross/target-geometry/<dataset-set>/`. Needs
`build-proximity` on every run named.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from scripts.analysis.v3.modules import headline_table as H
from scripts.analysis.v3.modules.bipartite import spearman
from scripts.analysis.v3.modules.cross import cross_dir
from scripts.analysis.v3.modules.diagram.common.draw import plt
from scripts.analysis.v3.modules.diagram.common.palette import (
    _C_AXIS,
    _C_GRID,
    _C_INK,
    _C_INK_2,
    _C_MUTED,
    _SURFACE,
)
from scripts.analysis.v3.modules.grid import (
    DEFAULT_GRID,
    GRID_HELP,
    RESOLUTION_HELP,
    resolve_cli_grid,
)
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    RunPaths,
    grid_slug,
    resolve_run,
)
from scripts.analysis.v3.modules.proximity import load_proximity

#: Its own sibling under `_cross/`: these are target properties, not accuracy,
#: and filing them under `accuracy-table/` would put a figure that runs no
#: method beside the tables that rank them.
CROSS_KIND = "target-geometry"

POOLED = "pooled"
COMPARE = "compare"
LAYOUTS: tuple[str, ...] = (POOLED, COMPARE)

CLOSEST = "closest"
SPING = "sping"

#: The y axis, and the one column that is *carried* rather than computed by this
#: pipeline — see the module docstring.
Y_COLUMN = "min_inflation"

#: The per-target threshold the band summarises.
MARGIN_COLUMN = "tg_seed_margin_km"

#: The two x axes, each with the `target_labels.csv` column behind it, the name
#: it takes in the points CSV, the prefix its quantiles take in the summary, and
#: the filename fragment that keeps the two figures apart on disk.
#:
#: Registered rather than branched on, because four things have to move together
#: whenever an axis is added or renamed — a column, a label, a CSV prefix and a
#: stem — and three of them are easy to forget.
X_METRICS: dict[str, dict[str, str]] = {
    CLOSEST: {
        "column": "tg_seed_nearest_vp_km",
        "points": "x_closest_km",
        "prefix": "closest_vp_to_seed_km",
        "axis_label": "Closest measured VP to the target's seed (km)",
        "title": "Closest VP distance against min-RTT inflation, per target",
        "stem": CLOSEST,
    },
    SPING: {
        "column": "sping_vp_to_tg_seed_km",
        "points": "x_sping_km",
        "prefix": "sping_vp_to_seed_km",
        "axis_label": "Shortest-ping VP to the target's seed (km)",
        "title": "Shortest-ping VP distance against min-RTT inflation, per target",
        "stem": SPING,
    },
}

#: `closest` first, because it is the one that answers a question about geometry
#: alone. `sping` mixes the two axes by construction (RTT picks the VP), so a
#: reader who takes the default gets the separable pair.
X_METRIC_ORDER: tuple[str, ...] = (CLOSEST, SPING)

#: Carried into the points CSV rather than derived from x. The four diamond
#: flags are the categorical form of the same two axes, and one of them
#: (`has_proximate_sping_vp`) *is* Shortest-Ping's top-1 accuracy
#: (`proximity.py`), so they let a reader tie a cluster here to the headline
#: table without re-deriving either.
CARRIED = (
    "tg_seed_id",
    MARGIN_COLUMN,
    "has_proximate_vp",
    "has_discriminative_vp",
    "has_proximate_sping_vp",
    "has_discriminative_sping_vp",
    "n_measured_vps",
)

#: The VP ids behind the two x metrics. Not plotted; used to derive
#: `sping_is_closest_vp`, which is the categorical form of the gap between the
#: two axes — did the RTT ranking hand over the VP geometry would have chosen.
ID_COLUMNS = ("tg_seed_nearest_vp_id", "sping_vp_id")

#: Mesh grey, traffic-weighted red — §8.1's own instruction. See the module
#: docstring for the three checks behind the pair.
KIND_INK = {H.MESH: _C_MUTED, H.WEIGHTED: "#d1352b"}

#: The weighted campaign is a subset of the mesh one, so it is drawn last and
#: sits on top; the reverse order would bury the series the comparison is about.
KIND_Z = {H.MESH: 2, H.WEIGHTED: 3}

#: A log x axis has no room for a zero. Nothing on as01/02/03 is anywhere near
#: it (the smallest VP-to-seed distance is 4.1 km), but a VP sitting exactly on
#: a seed is a legal measurement, so it is clamped to the floor and counted in
#: the manifest rather than dropped or left to blow up the transform.
MIN_X_KM = 0.1

#: The physical bound: inflation cannot go below 1 without beating ⅔c. Always in
#: view, even though the data starts at 1.18, because "how far above the floor"
#: is what the y axis is for and an axis cropped to the data hides it.
SOI_FLOOR = 1.0

Y_LINEAR = "linear"
Y_LOG = "log"
Y_SCALES: tuple[str, ...] = (Y_LINEAR, Y_LOG)

#: `linear` stays the default so the published §8.1 panels keep the shape they
#: were read in. `log` exists because inflation is a *ratio* with a hard floor
#: at 1 and no ceiling, and on real operator data it reaches ~200x: a linear
#: axis then spends 99% of its height on an empty upper region and packs the
#: mass every claim is about into the bottom 5%. Both are honest views of the
#: same column, so the choice is the caller's and it is recorded in the manifest
#: and the filename rather than being inferred from the data's range — an axis
#: that silently rescales itself per run makes two panels incomparable.
DPI = 200
POINT_SIZE = 13
POINT_ALPHA = 0.55
PANEL_WIDTH_INCHES = 7.4
PANEL_HEIGHT_INCHES = 5.4
#: Main axes : marginal, in both directions. 4.6:1 keeps the marginals readable
#: as shape without letting them take space from the joint view they annotate.
MARGINAL_RATIO = 4.6
MARGINAL_BINS = 26


def x_metric_spec(x_metric: str) -> dict[str, str]:
    """The registry entry, or a refusal naming the ones that exist."""
    try:
        return X_METRICS[x_metric]
    except KeyError:
        raise ValueError(
            f"unknown x metric {x_metric!r}; pick from {list(X_METRICS)}"
        ) from None


def x_column(x_metric: str) -> str:
    """The points-frame column for this axis."""
    return x_metric_spec(x_metric)["points"]


# ---- data -------------------------------------------------------------------


def load_points(
    plan: list[dict],
    *,
    analysis_root: Path | None = None,
    grid: str = DEFAULT_GRID,
    resolution: int,
) -> tuple[pd.DataFrame, dict]:
    """One row per (dataset, kind, target), from `build-proximity`'s labels.

    **Both** x metrics are loaded regardless of which is being drawn, so the
    points CSV is one artifact describing the target set rather than one per
    axis, and the summary can report the two rank correlations side by side —
    which is the comparison that says whether the axis choice changed the claim.

    Driven by `headline_table.row_plan` rather than by the run dicts directly,
    so this figure and the table beside it agree on which dataset types exist,
    which are pending, and how a run maps to a (kind, dataset) pair. The
    aggregate rows are skipped: pooling happens here by concatenation, and a
    scatter has no denominator to micro-average.
    """
    frames: list[pd.DataFrame] = []
    diagnostics: dict = {"runs": {}, "n_clamped_to_floor": 0, "n_dropped_non_finite": 0}
    x_columns = {m: spec["column"] for m, spec in X_METRICS.items()}
    for entry in plan:
        if entry["scope"] != H.DATASET or entry["run_id"] is None:
            continue
        run: RunPaths = entry["run"]
        labels = load_proximity(
            run.proximity_dir(root=analysis_root, grid=grid, resolution=resolution)
        ).labels
        missing = [
            c for c in (*x_columns.values(), Y_COLUMN) if c not in labels.columns
        ]
        if missing:
            raise ValueError(
                f"{run.run_id}'s target_labels.csv has no {missing}; it was written "
                "by an older build-proximity. Re-run `cli build-proximity` for this "
                "run and grid."
            )
        if not np.isfinite(labels[Y_COLUMN].to_numpy(dtype=float)).any():
            raise ValueError(
                f"{run.run_id} carries no {Y_COLUMN}: every value is NaN. That "
                "column is carried from eval_source's per-target frame, so "
                "build-proximity was run without a source CSV it could find. "
                "Re-run it with --source-csv pointing at the run's canonical CSV."
            )
        frame = pd.DataFrame(
            {
                "dataset": entry["dataset"],
                "kind": entry["kind"],
                "kind_label": H.KIND_LABELS[entry["kind"]],
                "run_id": run.run_id,
                "target_id": labels["target_id"],
                "y_inflation": labels[Y_COLUMN].astype(float),
            }
        )
        for metric, column in x_columns.items():
            frame[X_METRICS[metric]["points"]] = labels[column].astype(float)
        for column in CARRIED:
            if column in labels.columns:
                frame[column] = labels[column].to_numpy()
        if all(c in labels.columns for c in ID_COLUMNS):
            # The gap between the two x axes, as a boolean: did the RTT ranking
            # return the VP geometry would have picked?
            frame["sping_is_closest_vp"] = (
                labels[ID_COLUMNS[1]].to_numpy() == labels[ID_COLUMNS[0]].to_numpy()
            )
        n_raw = len(frame)
        drawn_x = [X_METRICS[m]["points"] for m in X_METRICS]
        finite = np.isfinite(frame["y_inflation"])
        for column in drawn_x:
            finite &= np.isfinite(frame[column])
        frame = frame.loc[finite].reset_index(drop=True)
        clamped = 0
        for column in drawn_x:
            clamped += int((frame[column] < MIN_X_KM).sum())
            frame[column] = frame[column].clip(lower=MIN_X_KM)
        diagnostics["runs"][run.run_id] = {
            "dataset": entry["dataset"],
            "kind": entry["kind"],
            "n_targets": n_raw,
            "n_plotted": len(frame),
            "n_dropped_non_finite": n_raw - len(frame),
            "n_clamped_to_floor": clamped,
        }
        diagnostics["n_clamped_to_floor"] += clamped
        diagnostics["n_dropped_non_finite"] += n_raw - len(frame)
        frames.append(frame)

    if not frames:
        raise ValueError(
            "no run in this figure has proximity labels to plot. Run "
            "`cli build-proximity` on the runs named by --run-id."
        )
    return pd.concat(frames, ignore_index=True), diagnostics


def _quantiles(values: pd.Series, prefix: str) -> dict:
    q = values.quantile([0.25, 0.5, 0.75, 0.9])
    return {
        f"{prefix}_p25": float(q.loc[0.25]),
        f"{prefix}_p50": float(q.loc[0.5]),
        f"{prefix}_p75": float(q.loc[0.75]),
        f"{prefix}_p90": float(q.loc[0.9]),
    }


def _spearman(x: pd.Series, y: pd.Series) -> float:
    """Rank correlation between two series, without a scipy dependency.

    Delegates to `bipartite.spearman`, which is the shared definition now that
    the table commands need it too and cannot import this module (it pulls
    matplotlib at module scope). The `len(x) < 3` floor stays here rather than
    moving down: two points always give |rho| == 1, which is meaningless for
    *this* figure's per-population clouds but is a legitimate answer for a
    per-target correlation over two VPs, where the caller flags it as
    underpowered instead.
    """
    if len(x) < 3:
        return float("nan")
    return spearman(x, y)


def summarize(points: pd.DataFrame) -> pd.DataFrame:
    """Per (kind, dataset) and per (kind, pooled): where each cloud sits.

    One row per population carrying **both** x metrics, rather than one row per
    (population, metric): the y quantiles and the proximity flags are shared, so
    splitting the rows would duplicate them and invite the two copies to be read
    as two measurements. The pair that matters is `spearman_rho_closest` against
    `spearman_rho_sping` — the same targets, the same y, and the difference is
    the axis choice alone.
    """
    rows: list[dict] = []
    for kind, by_kind in points.groupby("kind", sort=False):
        scopes = [(d, g) for d, g in by_kind.groupby("dataset", sort=True)]
        if len(scopes) > 1:
            scopes.append((H.AGGREGATE, by_kind))
        for dataset, group in scopes:
            row = {
                "kind": kind,
                "kind_label": H.KIND_LABELS[kind],
                "dataset": dataset,
                "n_targets": int(len(group)),
                **_quantiles(group["y_inflation"], "min_inflation"),
            }
            for metric in X_METRIC_ORDER:
                spec = X_METRICS[metric]
                values = group[spec["points"]]
                row.update(_quantiles(values, spec["prefix"]))
                row[f"spearman_rho_{metric}"] = _spearman(values, group["y_inflation"])
                row[f"share_inside_margin_{metric}"] = _inside_margin_share(
                    group, spec["points"]
                )
            for flag in (
                "has_proximate_vp",
                "has_discriminative_vp",
                "has_proximate_sping_vp",
                "has_discriminative_sping_vp",
                "sping_is_closest_vp",
            ):
                row[f"share_{flag[4:] if flag.startswith('has_') else flag}"] = (
                    float(group[flag].mean()) if flag in group.columns else float("nan")
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _inside_margin_share(group: pd.DataFrame, column: str) -> float:
    """Share of targets whose VP for this metric clears the seed margin.

    The scalar form of the band drawn on the figure, and the thing that makes
    the two x metrics comparable as claims: on `closest` it is the *ceiling* on
    any VP-coordinate answer, on `sping` it is what the baseline actually got.
    """
    if MARGIN_COLUMN not in group.columns:
        return float("nan")
    return float((group[column] < group[MARGIN_COLUMN]).mean())


# ---- drawing ----------------------------------------------------------------


def legend_kinds(drawn: list[str], pending: list[str]) -> list[str]:
    """Every dataset type the figure is *about*, in `H.KINDS` order.

    Separate from the drawn set on purpose: an uncollected type has no points,
    so it would fall out of any list derived from the data, and the legend is
    the only place left that can say the comparison is half finished.
    """
    wanted = set(drawn) | set(pending)
    return [k for k in H.KINDS if k in wanted]


def kind_handles(kinds: list[str], *, pending: list[str]) -> list:
    """One legend entry per dataset type, present or not.

    A pending type keeps its entry and says why it is empty, rather than
    vanishing from the key — six grey dots with no legend beside them would read
    as the whole comparison instead of half of it.
    """
    handles = []
    for kind in kinds:
        label = H.KIND_LABELS[kind].lower()
        if kind in pending:
            handles.append(
                Line2D(
                    [], [], marker="o", linestyle="none", markersize=5,
                    markerfacecolor="none", markeredgecolor=KIND_INK[kind],
                    markeredgewidth=1.0, label=f"{label} — not collected",
                )
            )
            continue
        handles.append(
            Line2D(
                [], [], marker="o", linestyle="none", markersize=5,
                markerfacecolor=KIND_INK[kind], markeredgecolor="none", label=label,
            )
        )
    handles.append(
        Patch(facecolor=_C_GRID, edgecolor="none", label="seed margin (p25–p75)")
    )
    return handles


def _margin_band(ax, points: pd.DataFrame) -> None:
    """The p25-p50-p75 of the per-target seed margin, as a band and a rule.

    Computed over the panel's own targets, mesh and weighted together: the
    margin is a property of the answer space rather than of the campaign, so
    both series are measured against the same threshold.
    """
    if MARGIN_COLUMN not in points.columns:
        return
    margins = points[MARGIN_COLUMN].dropna()
    if margins.empty:
        return
    lo, mid, hi = (float(v) for v in margins.quantile([0.25, 0.5, 0.75]))
    ax.axvspan(lo, hi, color=_C_GRID, alpha=0.65, zorder=0)
    ax.axvline(mid, color=_C_AXIS, linestyle=":", linewidth=1.2, zorder=1)


def _draw_panel(
    ax, points: pd.DataFrame, *, kinds: list[str], y_lim, x_metric: str,
    y_scale: str = Y_LINEAR,
) -> None:
    """The scatter itself. Shared by both layouts so they cannot drift."""
    column = x_column(x_metric)
    _margin_band(ax, points)
    ax.axhline(SOI_FLOOR, color=_C_AXIS, linestyle="--", linewidth=1.1, zorder=1)
    for kind in kinds:
        series = points[points["kind"] == kind]
        if series.empty:
            continue
        ax.scatter(
            series[column], series["y_inflation"],
            s=POINT_SIZE, c=KIND_INK[kind], alpha=POINT_ALPHA,
            linewidths=0, zorder=KIND_Z[kind],
        )
    ax.set_xscale("log")
    # `%g` rather than ScalarFormatter, which renders the sub-kilometre decade
    # as a bare "0" — see `figure_error_cdf` for the same fix.
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    if y_scale == Y_LOG:
        ax.set_yscale("log")
        # Same `%g` as the x axis rather than the log default, which prints
        # these decades as 10^0 / 10^1 / 10^2. The reader is comparing a ratio
        # against the floor at 1, and "1, 10, 100" is that comparison written
        # out; an exponent makes them do it.
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(*y_lim)
    ax.grid(True, which="major", color=_C_GRID, linewidth=0.7, alpha=0.9, zorder=0)
    # A log decade carries eight minor lines, and at full strength they read as
    # data. Kept, because reading 300 off a log axis needs them, but at a third
    # of the weight so the eye stops at the decades.
    ax.grid(True, which="minor", color=_C_GRID, linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_C_AXIS)
    ax.tick_params(colors=_C_MUTED, labelsize=9)


def n_above_cap(points: pd.DataFrame, y_max: float | None) -> int:
    """Targets the cap puts off-panel.

    Reported rather than assumed harmless. Capping is the one option here that
    removes observations from view, and this figure's whole subject is a tail —
    so the count goes in the manifest and the console line, next to the number
    of targets drawn, instead of leaving a reader to infer that the axis simply
    ends where the data does.
    """
    if y_max is None:
        return 0
    return int((points["y_inflation"] > float(y_max)).sum())


def _limits(
    points: pd.DataFrame, x_metric: str, *, y_scale: str = Y_LINEAR,
    y_max: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis ranges shared by every panel, so two panels compare by position.

    The x range is the drawn metric's own — `closest` spans two decades and
    `sping` nearly four, so a shared range would squeeze one of them into a
    stripe. The y floor is `SOI_FLOOR` even when no target comes near it: the
    axis is read as "how far above the physical bound", and a range cropped to
    the data silently rescales that reading per figure.
    """
    x = points[x_column(x_metric)]
    y = points["y_inflation"]
    floor = min(SOI_FLOOR, float(y.min()))
    # An explicit cap is honoured exactly, with none of the headroom the
    # data-driven bound gets: the caller picked 4 because they want to read
    # against 4, and quietly drawing to 4.12 would put the gridline they
    # chose somewhere other than the top of the panel.
    top = float(y.max()) if y_max is None else float(y_max)
    x_lim = (max(MIN_X_KM, float(x.min()) * 0.7), float(x.max()) * 1.4)
    if y_scale == Y_LOG:
        # Multiplicative margins, because on a log axis the additive ones below
        # are not merely ugly: `floor - 0.04 * span` is -6.96 once a single
        # target inflates 200x, and a negative bound is not a coordinate the
        # transform has. The floor is also clamped positive — inflation is
        # RTT over a positive slope so it cannot legally be <= 0, but a
        # degenerate labels file should lose the axis, not the whole figure.
        safe_floor = floor if floor > 0 else SOI_FLOOR
        return x_lim, (safe_floor / 1.08, top if y_max is not None else top * 1.12)
    span = top - floor
    # The floor gets its own margin below it: with `ylim` set *to* it, the
    # `y = 1` rule lands exactly on the bottom spine and disappears into it,
    # leaving the annotation pointing at nothing.
    return x_lim, (
        floor - 0.04 * span,
        top if y_max is not None else top + 0.03 * span,
    )


def _axis_labels(ax, x_metric: str) -> None:
    ax.set_xlabel(x_metric_spec(x_metric)["axis_label"], fontsize=10.5, color=_C_INK_2)
    ax.set_ylabel(
        "min-RTT inflation (observed / speed-of-internet)",
        fontsize=10.5,
        color=_C_INK_2,
    )


def _marginal(ax, points: pd.DataFrame, *, kinds: list[str], bins, orientation: str,
              column: str, uniform_bins: bool = True) -> None:
    """A histogram per dataset type, normalised within its own series.

    Normalised and not counts: the weighted campaign is the mesh one filtered, so
    it will always carry fewer targets, and a count histogram would answer "how
    many were kept" when the question is "where do the kept ones sit".

    Which normalisation depends on the bins. `density=True` divides by bin
    *width*, which is right for the equal-width bins the linear axes use and
    wrong for the log-spaced ones a log y axis needs: there the widest bins are
    the high-inflation ones, so dividing by width would shrink exactly the tail
    the log axis was chosen to show. With non-uniform bins each bar is therefore
    the share of that series' targets falling in it, which is the quantity the
    eye reads off a histogram anyway.
    """
    for kind in kinds:
        series = points.loc[points["kind"] == kind, column]
        if series.empty:
            continue
        values = np.log10(series) if orientation == "vertical" else series
        shared = dict(bins=bins, orientation=orientation)
        if uniform_bins:
            shared["density"] = True
        else:
            shared["weights"] = np.full(len(values), 1.0 / len(values))
        ax.hist(
            values, **shared,
            color=KIND_INK[kind], alpha=0.35, zorder=KIND_Z[kind],
        )
        ax.hist(
            values, **shared,
            histtype="step", color=KIND_INK[kind], linewidth=1.1,
            zorder=KIND_Z[kind] + 2,
        )
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def plot_scatter(
    points: pd.DataFrame, out_path: Path, *, kinds: list[str], pending: list[str],
    title: str, x_metric: str = CLOSEST, y_scale: str = Y_LINEAR,
    y_max: float | None = None,
) -> Path:
    """Pooled: one joint panel over every dataset, with both marginals."""
    x_lim, y_lim = _limits(points, x_metric, y_scale=y_scale, y_max=y_max)
    fig = plt.figure(
        figsize=(PANEL_WIDTH_INCHES, PANEL_HEIGHT_INCHES), dpi=DPI, facecolor=_SURFACE
    )
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=(MARGINAL_RATIO, 1), height_ratios=(1, MARGINAL_RATIO),
        wspace=0.04, hspace=0.04,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    ax.set_facecolor(_SURFACE)

    _draw_panel(
        ax, points, kinds=kinds, y_lim=y_lim, x_metric=x_metric, y_scale=y_scale
    )
    ax.set_xlim(*x_lim)
    _axis_labels(ax, x_metric)

    x_bins = np.linspace(np.log10(x_lim[0]), np.log10(x_lim[1]), MARGINAL_BINS)
    _marginal(ax_top, points, kinds=kinds, bins=x_bins, orientation="vertical",
              column=x_column(x_metric))
    ax_top.set_xlim(np.log10(x_lim[0]), np.log10(x_lim[1]))
    # `ax_right` shares the main y axis, so its bins live in data coordinates
    # and have to follow whatever scale that axis is on — log-spaced under
    # `Y_LOG`, or the histogram would put 26 equal-width bins on a log axis and
    # leave every one above the first decade unreadably thin.
    log_y = y_scale == Y_LOG
    y_bins = (
        np.logspace(np.log10(y_lim[0]), np.log10(y_lim[1]), MARGINAL_BINS)
        if log_y
        else np.linspace(y_lim[0], y_lim[1], MARGINAL_BINS)
    )
    _marginal(ax_right, points, kinds=kinds, bins=y_bins, orientation="horizontal",
              column="y_inflation", uniform_bins=not log_y)
    ax_right.set_ylim(*y_lim)

    legend = ax.legend(
        handles=kind_handles(legend_kinds(kinds, pending), pending=pending),
        loc="upper left", fontsize=8.5, frameon=False, handletextpad=0.6,
    )
    for text in legend.get_texts():
        text.set_color(_C_INK_2)
    ax.annotate(
        "speed-of-internet floor",
        xy=(x_lim[0], SOI_FLOOR), xytext=(3, 3), textcoords="offset points",
        fontsize=7.5, color=_C_MUTED, ha="left", va="bottom",
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", color=_C_INK, y=0.985)
    fig.savefig(out_path, dpi=DPI, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_compare(
    points: pd.DataFrame, out_path: Path, *, kinds: list[str], pending: list[str],
    title: str, x_metric: str = CLOSEST, y_scale: str = Y_LINEAR,
    y_max: float | None = None,
) -> Path:
    """One panel per dataset, both axes shared, so a cloud's position compares."""
    datasets = sorted(points["dataset"].unique())
    x_lim, y_lim = _limits(points, x_metric, y_scale=y_scale, y_max=y_max)
    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(PANEL_WIDTH_INCHES * len(datasets) * 0.56, PANEL_HEIGHT_INCHES * 0.86),
        dpi=DPI, facecolor=_SURFACE, sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets):
        panel = points[points["dataset"] == dataset]
        ax.set_facecolor(_SURFACE)
        _draw_panel(
            ax, panel, kinds=kinds, y_lim=y_lim, x_metric=x_metric, y_scale=y_scale
        )
        ax.set_xlim(*x_lim)
        ax.set_title(
            f"{dataset.upper()}  ·  n={len(panel):,}", fontsize=10, color=_C_INK, pad=8
        )
    _axis_labels(axes[0], x_metric)
    for ax in axes[1:]:
        ax.set_ylabel("")
    for ax in axes:
        ax.set_xlabel("")
    fig.supxlabel(
        x_metric_spec(x_metric)["axis_label"], fontsize=10.5, color=_C_INK_2
    )
    legend = axes[0].legend(
        handles=kind_handles(legend_kinds(kinds, pending), pending=pending),
        loc="upper left", fontsize=8.5, frameon=False, handletextpad=0.6,
    )
    for text in legend.get_texts():
        text.set_color(_C_INK_2)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=_C_INK, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---- build ------------------------------------------------------------------


def build(
    mesh_runs: dict[str, RunPaths],
    weighted_runs: dict[str, RunPaths],
    *,
    analysis_root: Path | None = None,
    grid: str = DEFAULT_GRID,
    resolution: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The points, their summary, and the manifest — before any rendering."""
    plan = H.row_plan(mesh_runs, weighted_runs)
    by_run = {**mesh_runs, **{r.run_id: r for r in weighted_runs.values()}}
    for entry in plan:
        entry["run"] = by_run.get(entry["run_id"]) if entry["run_id"] else None
    points, diagnostics = load_points(
        plan, analysis_root=analysis_root, grid=grid, resolution=resolution
    )
    summary = summarize(points)
    drawn = sorted(points["kind"].unique(), key=H.KINDS.index)
    manifest = {
        "mesh_runs": sorted(mesh_runs),
        "weighted_runs": {d: r.run_id for d, r in sorted(weighted_runs.items())},
        "grid": {"scheme": grid, "resolution": resolution},
        "n_points": int(len(points)),
        "kinds_drawn": drawn,
        "kinds_pending": [k for k in H.KINDS if k not in drawn],
        "axes": {
            "y": f"{Y_COLUMN} — min over measured VPs of RTT / (theoretical slope x "
                 "great-circle km); 1.0 is the speed-of-internet floor",
            "x": {m: X_METRICS[m]["column"] for m in X_METRIC_ORDER},
        },
        "x_metric_note": (
            "closest = tg_seed_nearest_vp_km, geometric opportunity with routing "
            "divided out. sping = sping_vp_to_tg_seed_km, geometry after routing "
            "has chosen the VP — a low value there already implies a well-behaved "
            "path, so a claim about geometry alone has to be made on closest"
        ),
        "encoding": (
            "colour = dataset type (mesh grey, traffic-weighted red). No variant "
            "hue appears: these are target properties, fixed before any method runs"
        ),
        "reference_marks": (
            "y = 1 is the speed-of-internet floor. The vertical band is the "
            "p25-p75 of tg_seed_margin_km over the panel's targets, with p50 "
            "ruled: a VP inside its own target's margin is guaranteed to snap to "
            "the right class"
        ),
        "diagnostics": diagnostics,
    }
    if H.WEIGHTED in manifest["kinds_pending"]:
        manifest["pending_note"] = (
            "no run carries traffic weights (has_weight: false), so the weighted "
            "series is uncollected rather than empty. Pass --weighted-run-id "
            "<dataset>=<run_id> once the campaign lands"
        )
    return points, summary, manifest


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("plot-proximity-inflation")
    def plot_proximity_inflation_cmd(
        run_id: list[str] = typer.Option(
            None, "--run-id", help="Mesh runs, one per dataset (repeatable)."
        ),
        weighted_run_id: list[str] = typer.Option(
            None,
            "--weighted-run-id",
            help="Traffic-weighted twin, as <dataset>=<run_id> (repeatable). None "
            "exist yet, so that series draws as a legend entry only.",
        ),
        x_metric: list[str] = typer.Option(
            [],
            "--x-metric",
            help=f"{CLOSEST} (nearest measured VP — geometry alone) or {SPING} "
            f"(the VP the lowest RTT picked). Repeatable; default {CLOSEST}.",
        ),
        layout: list[str] = typer.Option(
            [],
            "--layout",
            help=f"{POOLED} (one joint panel with marginals) or {COMPARE} (one "
            f"panel per dataset). Repeatable; default {POOLED}.",
        ),
        y_scale: str = typer.Option(
            Y_LINEAR,
            "--y-scale",
            help=f"Inflation axis: {Y_LINEAR} (default) or {Y_LOG}. Use {Y_LOG} "
            f"when the tail reaches tens or hundreds of x — a linear axis then "
            f"packs the bulk into the bottom few percent. {Y_LOG} adds `.y{Y_LOG}` "
            f"to the filename so both views can coexist.",
        ),
        y_max: float = typer.Option(
            None,
            "--y-max",
            help="Cap the inflation axis at this value instead of the data's own "
            "maximum, to zoom on the bulk. Targets above it go off-panel and are "
            "counted in the manifest and the console line. Adds `.ymax<v>` to the "
            "filename.",
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option(
            [], "--resolution", "-r", help=RESOLUTION_HELP
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """§8.1's geometry-vs-routing scatter: VP distance to the target's seed
        against min-RTT inflation, per target and dataset type.

        Writes proximity_inflation[_by_dataset].<x-metric>.<grid>.png plus one
        points CSV, one summary CSV and a manifest into
        _cross/target-geometry/<dataset-set>/. Needs `build-proximity` on every
        run named.
        """
        if not run_id:
            raise typer.BadParameter(
                "pass at least one --run-id. Like `table-headline`, this command "
                "has no --all-runs: which datasets belong in one figure is the "
                "caller's call, and §7.3 declines the operator/public head-to-head."
            )
        unknown = [x for x in layout if x not in LAYOUTS]
        if unknown:
            raise typer.BadParameter(
                f"unknown --layout {unknown}; pick from {list(LAYOUTS)}"
            )
        layouts = tuple(dict.fromkeys(layout)) or (POOLED,)
        unknown_metric = [x for x in x_metric if x not in X_METRICS]
        if unknown_metric:
            raise typer.BadParameter(
                f"unknown --x-metric {unknown_metric}; pick from {list(X_METRICS)}"
            )
        metrics = tuple(dict.fromkeys(x_metric)) or (CLOSEST,)
        if y_scale not in Y_SCALES:
            raise typer.BadParameter(
                f"unknown --y-scale {y_scale!r}; pick from {list(Y_SCALES)}"
            )
        if y_max is not None and y_max <= SOI_FLOOR:
            raise typer.BadParameter(
                f"--y-max {y_max} is at or below the speed-of-internet floor "
                f"({SOI_FLOOR}); inflation cannot go below it, so the panel would "
                "be empty. Pick a value above 1."
            )

        g, resolutions = resolve_cli_grid(grid, resolution, sweep=False)
        mesh_runs = {rid: resolve_run(rid, outputs_root) for rid in run_id}
        weighted_runs = {
            dataset: resolve_run(rid, outputs_root)
            for dataset, rid in H.parse_weighted_pairs(weighted_run_id or []).items()
        }

        for res in resolutions:
            points, summary, manifest = build(
                mesh_runs,
                weighted_runs,
                analysis_root=analysis_root,
                grid=g.name,
                resolution=res,
            )
            # `sorted(mesh_runs)`, never the merged dict: `weighted_runs` is keyed
            # by dataset, so merging the two key spaces would fork the output
            # directory the moment a weighted run is passed.
            out_dir = cross_dir(analysis_root, sorted(mesh_runs), kind=CROSS_KIND)
            slug = grid_slug(g.name, res)
            # Render-time, so it is set here rather than in `build`: the CSVs
            # are scale-free and one manifest covers every PNG of this grid.
            manifest["y_scale"] = y_scale
            manifest["y_max"] = y_max
            clipped = n_above_cap(points, y_max)
            manifest["n_targets_above_y_max"] = clipped
            stem = f"proximity_inflation.{slug}"
            # One points and one summary CSV per grid, not per metric: both x
            # columns are in every row, so a second copy would only differ in
            # which one a reader looked at.
            points.to_csv(out_dir / f"{stem}.points.csv", index=False)
            summary.to_csv(out_dir / f"{stem}.summary.csv", index=False)
            (out_dir / f"{stem}.manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            kinds = manifest["kinds_drawn"]
            pending = manifest["kinds_pending"]
            written: list[Path] = []
            for metric in metrics:
                spec = X_METRICS[metric]
                for name in layouts:
                    base = (
                        "proximity_inflation"
                        if name == POOLED
                        else "proximity_inflation_by_dataset"
                    )
                    # The default keeps its historical name; only the opt-in
                    # scale is marked, so `--y-scale log` adds a figure beside
                    # the published one instead of overwriting it with a
                    # differently-shaped panel under the same filename.
                    y_tag = "" if y_scale == Y_LINEAR else f".y{y_scale}"
                    # `.` is this scheme's field separator, so a fractional cap
                    # would read as one: 4.5 -> `ymax4p5`.
                    if y_max is not None:
                        y_tag += ".ymax" + f"{y_max:g}".replace(".", "p")
                    png = out_dir / f"{base}.{spec['stem']}{y_tag}.{slug}.png"
                    renderer = plot_scatter if name == POOLED else plot_compare
                    written.append(
                        renderer(
                            points, png, kinds=kinds, pending=pending,
                            title=spec["title"], x_metric=metric,
                            y_scale=y_scale, y_max=y_max,
                        )
                    )
            typer.echo(
                f"{len(points):,} targets over {len(mesh_runs)} dataset(s) at {slug} "
                f"· x {', '.join(metrics)} · y {y_scale}"
                + (f" capped at {y_max:g} ({clipped} off-panel)" if y_max else "")
                + f" · layouts {', '.join(layouts)} -> "
                f"{', '.join(p.name for p in written)}"
            )
