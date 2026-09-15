"""§8.1's co-location scatter: how far the shortest-ping VP sat from the site it
was assigned, against how far that site sat from the target.

One point per target, on the **shortest-ping edge only** — the (VP, TG) pair
whose min-RTT was lowest, which is the edge the Shortest-Ping baseline actually
acts on. Its two axes are the two legs of that pair's assigned path:

    x = d(sping VP, selected PNI)     `vp_to_sel_pni_km`
    y = d(selected PNI, target)       `sel_pni_to_tg_km`

Both are carried from `build-pni-graph`, never recomputed here, so this figure
and `pni_edges.csv` cannot disagree about a distance.

## What the corner means

§8.1's chain is that good interconnect design plus good content routing leaves a
traffic-weighted campaign holding targets that sit *at* interconnect sites,
reached by VPs that are *also* near those sites. That is a claim about a corner:
both legs small at once. So the figure reads as

* **bottom-left** — VP, site and target mutually co-located. Shortest-Ping's
  nearest-answer snapping lands on the right class here for a geometric reason,
  not a lucky one.
* **left, high** — the VP reaches the site cheaply but the site is nowhere near
  the target. The RTT ranking is measuring the wrong leg.
* **bottom, right** — the target sits on its site but no VP is near it.
* **top-right** — neither leg holds; nothing about the assignment is local.

x + y is the pair's routing distance (`vp_to_tg_via_pni_km`), so iso-cost lines
run anti-diagonally; the corner is a stronger statement than the sum, which is
why both legs get an axis instead of being added up.

## Why the argmin/nearest distinction stops mattering here, and how to check

`build-pni-graph` assigns each pair the site minimising `d(VP,p) + d(p,TG)`,
which need not be the target's own nearest site. When the premise above holds
the two collapse: a target sitting on a site makes that site the argmin for
almost any VP. The collapse is therefore a *prediction*, not an assumption, and
`sel_pni_is_tg_nearest_share` in the stats file measures it — 0.89 on as01's
approximate site list, 0.73 on as02's synthetic one. A weighted arm whose share
does not rise toward 1 has not concentrated onto sites, whatever the scatter
looks like.

## Replicas: why the legend prints two counts

The operator datasets carry roughly 20 IP replicas per distinct coordinate, so
as01's 399 targets occupy **26** distinct (x, y) positions. Drawn as 399
translucent dots that is 26 opaque ones, and the transparency communicates
nothing. The legend therefore always prints both counts, and `--marker count`
scales each mark's area by the replicas behind it so the multiplicity is visible
rather than implied. Neither mode invents or drops a point; they differ only in
whether coincident targets are stacked or summed.

## Driven from the traffic-weighted run

`--run-id` is the **traffic-weighted** run and `--mesh-run-id` its unweighted
parent. Both are required, and the output lands in the weighted run's
`pni-graph/`.

The figure's content is a subset drawn against the population it was filtered
from, so it has nothing to say about a campaign that was never weighted, and it
cannot exist before a weighted arm does. Driving it from the arm that must
exist *last* turns that into the command's own precondition instead of a
caller's discipline, and lets the pairing be declared once — in the weighted
config, beside the two CSVs the subset was derived from.

The arrangement used to be the reverse: `--run-id` was the mesh run and the
overlay an optional `--weighted-run-id`. Every mesh run was then a legal
invocation, and one without a twin emitted a grey-only panel under a filename
and a pair of axis labels that both promise two populations.

## No pooling across datasets

One weighted run and one mesh run. Two peer ASNs do not share a site list —
as01 runs against `as01-us-pni.approx.csv`, as02 against the synthetic
carrier-hotel set — so an x of 40 km means a different thing in each and a
pooled panel would average over that. Run the command once per dataset.

Command: `plot-pni-colocation`. Writes `pni_colocation.png`,
`pni_colocation_points.csv` and `pni_colocation_stats.json` into the
traffic-weighted run's `pni-graph/`, or into `--out-dir`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules.bipartite import truthy
from scripts.analysis.v3.modules.cross import short_dataset
from scripts.analysis.v3.modules.diagram.common.draw import plt
from scripts.analysis.v3.modules.diagram.common.palette import (
    _C_AXIS,
    _C_GRID,
    _C_INK,
    _C_MUTED,
)
from scripts.analysis.v3.modules.paths import (
    DEFAULT_OUTPUTS_ROOT,
    resolve_run,
)
from scripts.analysis.v3.modules.pni import PNI_EDGES_CSV

#: Grey backdrop, red highlight — the same pair `figure_distance_rtt` uses, and
#: for the same reason: these are not two peer categories but a subset drawn
#: against the population it was filtered from. Identity never rests on the hue
#: alone; the legend names both layers and the stats JSON is the table view.
_C_MESH = "#8a8a8a"
_C_TW = "#d32f2f"

#: The two legs, as they are named in `pni_edges.csv`. Carried, not recomputed.
X_COLUMN = "vp_to_sel_pni_km"
Y_COLUMN = "sel_pni_to_tg_km"

#: The flag that reduces the edge list to one row per target.
SPING_FLAG = "is_sping_vp"

#: Carried into the points CSV so a cluster in the figure can be traced back to
#: the rows that made it without re-deriving anything.
CARRIED = (
    "target_id",
    "vp_id",
    "sel_pni_id",
    "rtt_ms",
    "vp_to_tg_km",
    "vp_to_tg_via_pni_km",
    "sel_pni_is_tg_nearest",
)

#: Distances are rounded to this many km before coincident targets are counted
#: as one position. `build-pni-graph` already writes km at 3 decimals, so this
#: matches its precision rather than imposing a coarser one.
_ROUND_KM_DIGITS = 3

MARKER_FIXED = "fixed"
MARKER_COUNT = "count"
MARKERS: tuple[str, ...] = (MARKER_FIXED, MARKER_COUNT)

#: Marker area at each end of the replica range under `--marker count`. Area is
#: linear in the count so that two stacked targets read as twice one, which is
#: the only mapping under which the eye's "how much" matches the data's.
_AREA_MIN = 10.0
_AREA_MAX = 190.0

#: The overlay's marks are scaled to this fraction of the backdrop's area.
#:
#: A weighted arm is a SUBSET of the mesh, so the two layers coincide almost
#: everywhere and equal marks let red hide grey completely -- the figure then
#: cannot show which targets the filter *dropped*, which is the overlay's whole
#: purpose. Scaling the whole array preserves the within-layer count encoding
#: while leaving a grey rim; a grey mark with no red centre is a dropped target.
#: Same remedy, same reason as `figure_distance_rtt`'s two point sizes.
_OVERLAY_AREA_SCALE = 0.34

#: Log axes have no zero to put the corner at, and a VP sitting exactly on its
#: site is a legal measurement. Marks below this are clamped to it for *drawing*
#: only — the points CSV and every statistic keep the real value — and the
#: clamped count is reported so a floor pile-up is never mistaken for data.
#: Nothing on as01 is near it (the smallest leg is 0.578 km).
_MIN_LOG_KM = 0.1

PNG_NAME = "pni_colocation.png"
POINTS_NAME = "pni_colocation_points.csv"
STATS_NAME = "pni_colocation_stats.json"

_QUANTILES = (0.25, 0.5, 0.75, 0.9)


def load_sping_legs(pni_graph_dir: Path) -> tuple[pd.DataFrame, dict]:
    """The two legs of every target's shortest-ping edge, plus a diagnostics block.

    Reads `pni_edges.csv` rather than `target_nodes.csv` because only the edge
    table carries the legs separately — the node table pre-sums them into
    `sping_vp_to_tg_via_pni_km`, and this figure is about the corner rather than
    the sum.
    """
    path = Path(pni_graph_dir) / PNI_EDGES_CSV
    if not path.exists():
        raise typer.BadParameter(
            f"{path} does not exist; run `cli build-pni-graph --run-id <run> "
            "--pni-csv <sites.csv>` for this run first."
        )
    edges = pd.read_csv(path)
    missing = [c for c in (X_COLUMN, Y_COLUMN, SPING_FLAG) if c not in edges.columns]
    if missing:
        raise typer.BadParameter(
            f"{path} lacks {missing}; it was written by an older build-pni-graph. "
            "Re-run that command for this run."
        )

    sping = edges.loc[truthy(edges[SPING_FLAG])].copy()
    if sping.empty:
        raise typer.BadParameter(
            f"{path} flags no shortest-ping edge ({SPING_FLAG} is false on every "
            "row), so there is nothing to plot one point per target from."
        )

    frame = pd.DataFrame(
        {
            "x_km": pd.to_numeric(sping[X_COLUMN], errors="coerce"),
            "y_km": pd.to_numeric(sping[Y_COLUMN], errors="coerce"),
        }
    )
    for column in CARRIED:
        if column in sping.columns:
            frame[column] = sping[column].to_numpy()

    n_raw = len(frame)
    frame = frame.loc[np.isfinite(frame["x_km"]) & np.isfinite(frame["y_km"])]
    frame = frame.reset_index(drop=True)
    diagnostics = {
        "n_edges": int(len(edges)),
        "n_sping_edges": int(n_raw),
        "n_plotted": int(len(frame)),
        "n_dropped_non_finite": int(n_raw - len(frame)),
    }
    return frame, diagnostics


def collapse(points: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct (x, y), with the number of targets stacked there.

    The operator datasets put ~20 IP replicas on one coordinate, so this is the
    difference between 399 marks and the 26 positions they occupy. Separate from
    the points CSV, which stays one row per target: collapsing is a *drawing*
    decision and must not change what the artifact records.
    """
    keys = points[["x_km", "y_km"]].round(_ROUND_KM_DIGITS)
    grouped = keys.groupby(["x_km", "y_km"], sort=True).size()
    return grouped.reset_index(name="n_targets")


def series_stats(points: pd.DataFrame) -> dict:
    """Quantiles of both legs, of their sum, and of their elementwise maximum.

    The maximum is the co-location radius: "both legs within Q km" is exactly
    its Q-quantile, which states the corner claim without inventing a threshold
    to count points inside. The sum is the routing distance, reported beside it
    so a reader can see when a small sum is one short leg rather than two.
    """
    x = points["x_km"].to_numpy(dtype=float)
    y = points["y_km"].to_numpy(dtype=float)
    distinct = collapse(points)
    out: dict = {
        "n_targets": int(len(points)),
        "n_distinct_points": int(len(distinct)),
        "max_targets_per_point": int(distinct["n_targets"].max()) if len(distinct) else 0,
    }
    series = {
        "vp_to_sel_pni_km": x,
        "sel_pni_to_tg_km": y,
        "routing_km": x + y,
        "colocation_radius_km": np.maximum(x, y),
    }
    for name, values in series.items():
        for q in _QUANTILES:
            out[f"{name}_p{int(q * 100)}"] = round(float(np.quantile(values, q)), 3)
    if "sel_pni_is_tg_nearest" in points.columns:
        flag = truthy(points["sel_pni_is_tg_nearest"])
        out["sel_pni_is_tg_nearest_share"] = round(float(flag.mean()), 4)
    if "sel_pni_id" in points.columns:
        out["n_distinct_sel_pni"] = int(points["sel_pni_id"].nunique())
    return out


def marker_sizes(distinct: pd.DataFrame, *, point_size: float, marker: str) -> np.ndarray:
    """Per-mark area: constant, or linear in the replica count.

    Linear in the *count*, not in its square root: the quantity being shown is
    "how many targets are here", and area is the channel the eye integrates, so
    a square-root mapping would understate a 20x stack as a ~4.5x mark.
    """
    counts = distinct["n_targets"].to_numpy(dtype=float)
    if marker == MARKER_FIXED:
        return np.full(counts.shape, float(point_size))
    hi = counts.max()
    if hi <= 1:
        return np.full(counts.shape, _AREA_MIN)
    return _AREA_MIN + (_AREA_MAX - _AREA_MIN) * (counts - 1.0) / (hi - 1.0)


#: Run ids ending in one of these state the layer's ROLE, which the label
#: already carries as its own argument. Stripped so a legend reads "as01 mesh"
#: rather than "as01-260728-260802-mesh mesh".
_ROLE_SUFFIXES = ("-mesh", "-weighted")

#: A run id carrying this was built from the `.randweight` fixture, whose
#: weights are synthetic uniform-random. See `_weighted_label`.
_SYNTHETIC_MARKER = "randweight"


def _layer_label(run_id: str, role: str) -> str:
    """`as01-260728-260802-weighted` + `traffic-weighted` -> `as01 traffic-weighted`.

    `short_dataset` alone is not enough here. It strips a trailing all-numeric
    tail, and the finals run ids end in `-mesh` / `-weighted` instead, so it
    hands them back whole and the legend prints the role twice.
    """
    stem = run_id
    for suffix in _ROLE_SUFFIXES:
        stem = stem.removesuffix(suffix)
    return f"{short_dataset(stem)} {role}"


def _weighted_label(run_id: str) -> str:
    """The red layer's legend text, warning when its weights are synthetic.

    A safeguard, not decoration. No weight-bearing export has been collected --
    weights are enterprise-sensitive -- so the only run that can populate this
    layer today is the `.randweight` fixture, and a legend reading
    "traffic-weighted" would be taken as evidence of real weights. Derived here
    rather than by the caller so it holds at every entry point;
    `figure_distance_rtt` carries the same warning off the overlay's filename,
    which is what names the dataset there.
    """
    label = _layer_label(run_id, "traffic-weighted")
    if _SYNTHETIC_MARKER in run_id:
        label += " (SYNTHETIC weights)"
    return label


def _series_label(label: str, stats: dict) -> str:
    """`<name>` over `(n=399 targets, 26 distinct)` — both counts, always.

    The second count is not a footnote. Without it the panel asserts a sample
    size it does not have, and the replica structure is invisible in exactly the
    figures it distorts most.

    Wrapped onto a second line rather than run on, because the legend sits
    OUTSIDE the axes and so is not part of what `tight_layout` sizes: a single
    long line is silently clipped at the figure edge instead of widening it.
    A full run id plus the synthetic-weights warning plus both counts exceeds
    5.8 in at 7.5 pt, and truncating a run id to fit would cost the panel its
    identity. Unconditional so the two entries stay the same shape.
    """
    return (
        f"{label}\n"
        f"(n={stats['n_targets']:,} targets, "
        f"{stats['n_distinct_points']:,} distinct)"
    )


def build(
    mesh: pd.DataFrame,
    tw: pd.DataFrame,
    *,
    mesh_label: str,
    tw_label: str,
    stats: dict,
    marker: str,
    point_size: float,
    alpha: float,
    x_max: float | None,
    y_max: float | None,
    log_axes: bool,
    title: str | None,
    out_png: Path,
) -> int:
    """Draw the overlay: mesh underneath in grey, weighted on top in red.

    Returns the number of marks clamped to the log floor (always 0 on linear
    axes), so the caller can record it rather than let a floor pile-up read as
    a cluster of genuinely co-located targets.

    **Linear is the default** because the claim is about a corner at the origin,
    and linear is the only scale that has an origin to put it at. `log_axes`
    exists because these legs span four orders of magnitude on real data — 0.6
    km to 3,100 km on as01 — so a linear panel spends most of its area empty and
    packs the co-located mass every claim is about into the bottom-left few
    percent. The two are honest views of the same points; the choice is the
    caller's and is recorded in the stats file rather than inferred from the
    data's range.
    """
    fig, ax = plt.subplots(figsize=(5.8, 5.4), dpi=200)
    n_clamped = 0

    # (frame, label, colour, key, zorder, area_scale)
    layers = [
        (mesh, mesh_label, _C_MESH, "mesh", 2, 1.0),
        (tw, tw_label, _C_TW, "traffic_weighted", 3, _OVERLAY_AREA_SCALE),
    ]

    for frame, label, colour, key, z, area_scale in layers:
        distinct = collapse(frame)
        if log_axes:
            below = (distinct["x_km"] < _MIN_LOG_KM) | (distinct["y_km"] < _MIN_LOG_KM)
            n_clamped += int(below.sum())
            distinct = distinct.assign(
                x_km=distinct["x_km"].clip(lower=_MIN_LOG_KM),
                y_km=distinct["y_km"].clip(lower=_MIN_LOG_KM),
            )
        ax.scatter(
            distinct["x_km"],
            distinct["y_km"],
            s=marker_sizes(distinct, point_size=point_size, marker=marker)
            * area_scale,
            alpha=alpha,
            color=colour,
            edgecolors="none",
            zorder=z,
            label=_series_label(label, stats[key]),
        )

    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        # The floor is pinned only when a mark actually sits on it. Forcing it
        # otherwise spends a whole empty decade on a region the data never
        # reaches, which on as01 (smallest leg 0.578 km) is a quarter of the
        # panel. `None` hands that end back to matplotlib's autoscale.
        floor = _MIN_LOG_KM if n_clamped else None
        ax.set_xlim(left=floor, right=x_max)
        ax.set_ylim(bottom=floor, top=y_max)
    else:
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
    ax.set_xlabel("d(shortest-ping VP, selected PNI)  [km]")
    ax.set_ylabel("d(selected PNI, target)  [km]")
    if title:
        ax.set_title(title, color=_C_INK)
    ax.grid(True, color=_C_GRID, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_C_AXIS)

    # Legend marks are sized and opaque independently of the data marks: under
    # --marker count a swatch drawn at the data's own area would be either
    # invisible or enormous, and identity must not rest on a mark that is either.
    # Above the axes rather than inside them: on log axes the cloud reaches
    # every corner, so any in-panel anchor collides with data, and the dataset
    # labels carry full run ids that no in-panel box has the width for.
    leg = ax.legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=7.5,
        frameon=False, scatterpoints=1, borderaxespad=0.0,
    )
    for handle in leg.legend_handles:
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([28.0])
        handle.set_alpha(1.0)
    if marker == MARKER_COUNT:
        ax.annotate(
            "mark area ∝ targets at that position",
            xy=(0.98, 0.02), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=6.5, color=_C_MUTED,
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return n_clamped


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("plot-pni-colocation")
    def plot_pni_colocation_cmd(
        run_id: str = typer.Option(
            ...,
            "--run-id",
            help="The TRAFFIC-WEIGHTED run, drawn in red on top. The figure is "
            "written into this run's pni-graph/, and `--config "
            "configs/<weighted-run>.yaml` supplies it automatically. One run is "
            "one peer ASN with its own site list, so there is no --all-runs and "
            "no pooled panel.",
        ),
        mesh_run_id: str = typer.Option(
            ...,
            "--mesh-run-id",
            help="The unweighted parent of --run-id, drawn grey underneath. "
            "Required: the figure IS the subset-against-parent comparison, so a "
            "lone population has no corner claim to support. Declare it once as "
            "analysis.plot-pni-colocation.mesh_run_id in the weighted config.",
        ),
        marker: str = typer.Option(
            MARKER_FIXED,
            "--marker",
            help=f"{MARKER_FIXED} (default) draws one mark per distinct position; "
            f"{MARKER_COUNT} scales its area by the targets stacked there. The "
            "operator datasets put ~20 replicas on one coordinate, so the two "
            "views differ a lot.",
        ),
        point_size: float = typer.Option(
            26.0, "--point-size", help=f"Mark area under --marker {MARKER_FIXED}."
        ),
        alpha: float = typer.Option(0.55, "--alpha", help="Opacity of both layers."),
        log_axes: bool = typer.Option(
            False,
            "--log-axes",
            help="Log both axes. The legs span four orders of magnitude on real "
            "data, so a linear panel packs the co-located mass into its bottom-"
            f"left corner. Marks below {_MIN_LOG_KM} km are clamped to it for "
            "drawing and counted; no statistic changes. Adds `.log` to the "
            "filename so both views can coexist.",
        ),
        x_max: float = typer.Option(
            None, "--x-max", help="Clip the x axis. Statistics stay over all targets."
        ),
        y_max: float = typer.Option(
            None, "--y-max", help="Clip the y axis. Statistics stay over all targets."
        ),
        title: str = typer.Option(None, "--title", help="Figure title. Default: none."),
        out_dir: Path = typer.Option(
            None,
            "--out-dir",
            help="Where to write. Default: the traffic-weighted run's pni-graph/.",
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
    ) -> None:
        """Scatter d(sping VP, selected PNI) against d(selected PNI, target).

        One point per target, the mesh parent grey underneath and the
        traffic-weighted arm red on top. `--run-id` is the WEIGHTED run and the
        output lands in its pni-graph/. Writes pni_colocation.png, _points.csv
        and _stats.json. Needs `build-pni-graph` on both runs named.
        """
        if marker not in MARKERS:
            raise typer.BadParameter(
                f"unknown --marker {marker!r}; pick from {list(MARKERS)}"
            )

        # A config typo naming the weighted run as its own parent would draw
        # two perfectly coincident layers and report a comparison that never
        # happened -- and with the overlay scaled down it would even look like
        # a legitimate rim-and-centre plot. Cheap to refuse, invisible to
        # diagnose afterwards.
        if mesh_run_id == run_id:
            raise typer.BadParameter(
                f"--mesh-run-id {mesh_run_id} is --run-id itself; the two layers "
                "are the weighted arm and the mesh it was filtered from, which "
                "cannot be one run"
            )

        mesh_run = resolve_run(mesh_run_id, outputs_root)
        mesh, mesh_diag = load_sping_legs(mesh_run.pni_graph_dir())
        mesh["series"] = "mesh"

        tw_run = resolve_run(run_id, outputs_root)
        tw, tw_diag = load_sping_legs(tw_run.pni_graph_dir())
        tw["series"] = "traffic_weighted"

        stats = {
            "mesh": series_stats(mesh),
            "traffic_weighted": series_stats(tw),
            "diagnostics": {"mesh": mesh_diag, "traffic_weighted": tw_diag},
            "runs": {"mesh": mesh_run_id, "traffic_weighted": run_id},
            "axes": {
                "x": f"{X_COLUMN} — great-circle km from the shortest-ping VP to "
                     "the site build-pni-graph assigned that pair",
                "y": f"{Y_COLUMN} — great-circle km from that site to the target",
                "scope": "one row per target, on the is_sping_vp edge only",
            },
            "scale": "log" if log_axes else "linear",
            "encoding": (
                "colour = dataset (mesh grey underneath, traffic-weighted red on "
                f"top); marker = {marker}"
            ),
        }

        dest = Path(out_dir) if out_dir else tw_run.pni_graph_dir()
        dest.mkdir(parents=True, exist_ok=True)
        mesh_label = _layer_label(mesh_run_id, "mesh")
        tw_label = _weighted_label(run_id)

        stem = PNG_NAME.removesuffix(".png") + (".log" if log_axes else "")
        out_png = dest / f"{stem}.png"
        n_clamped = build(
            mesh, tw,
            mesh_label=mesh_label, tw_label=tw_label, stats=stats, marker=marker,
            point_size=point_size, alpha=alpha, x_max=x_max, y_max=y_max,
            log_axes=log_axes, title=title, out_png=out_png,
        )
        stats["n_marks_clamped_to_log_floor"] = n_clamped
        points = pd.concat([mesh, tw], ignore_index=True)
        points.to_csv(dest / POINTS_NAME, index=False)
        (dest / STATS_NAME).write_text(json.dumps(stats, indent=2) + "\n")

        typer.echo(f"wrote {out_png}")
        typer.echo(f"wrote {dest / POINTS_NAME}")
        typer.echo(f"wrote {dest / STATS_NAME}")
        for key in ("mesh", "traffic_weighted"):
            block = stats.get(key)
            if not block:
                continue
            typer.echo(
                f"{key}: n={block['n_targets']} targets at "
                f"{block['n_distinct_points']} distinct positions; "
                f"co-location radius p50 {block['colocation_radius_km_p50']:,.1f} km, "
                f"p90 {block['colocation_radius_km_p90']:,.1f} km; "
                f"sel==tg-nearest {block.get('sel_pni_is_tg_nearest_share', float('nan')):.3f}"
            )
