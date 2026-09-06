"""Three figures for the bipartite graph: topology, flows, and the distance CDF.

Renders what `build-bipartite-graph` produced. The two maps are complements and
neither replaces the other, which is why both are drawn:

* **`bipartite_nodes_map.png` — topology.** Where the VPs and targets are,
  against the class regions they have to resolve. No edges, so nothing occludes
  the geometry. This is the figure that makes the occupied-cell counts visible:
  "134 VPs in 31 occupied cells" is the honest denominator for any claim resting
  on independent observations, and a scalar does not show that two dozen of them
  sit in one metro.
* **`bipartite_flows_map.png` — flows.** Every measured edge, so what the
  campaign actually covers reads directly. The topology map cannot show this and
  this map cannot show the topology, because the ink that carries the flows is
  the ink that hides the nodes.
* **`distance_cdf.png`** — the figure §7.3 asks for behind the diameter and p95
  scalars ("the full pairwise distance CDF is the figure behind those two
  scalars"), for the VP side, the target side, and the observed edges.

Both maps reuse `mapping.seed_voronoi`, so the class boundary they draw is
identical to `plot-answer-space`'s and is the classifier's own top-1 decision
boundary rather than an illustration of one.

Command: `plot-bipartite-graph`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from scripts.analysis.v3.modules.answer_space import AnswerSpace, load_answer_space
from scripts.analysis.v3.modules.bipartite import BipartiteGraph, load_bipartite
from scripts.analysis.v3.modules.grid import (
    DEFAULT_GRID,
    GRID_HELP,
    RESOLUTION_HELP,
    SWEEP_HELP,
    get_grid,
    resolve_cli_grid,
)
from scripts.analysis.v3.modules.mapping import (
    US_MAINLAND_EXTENT,
    _auto_extent,
    _seed_colors,
    draw_basemap,
    draw_cells,
    draw_voronoi,
    great_circle_segments,
    seed_voronoi,
)
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    discover_runs,
    resolve_run,
)

NODES_PNG = "bipartite_nodes_map.png"
FLOWS_PNG = "bipartite_flows_map.png"
CDF_PNG = "distance_cdf.png"

#: Three hues from the same categorical theme as
#: `diagram/common/palette.py::_VARIANT_HUES`, but *not* that tuple — colour there
#: is bound to a CBG variant's identity, and a VP-pairwise curve is not a
#: variant. These three are picked from the five members of the theme that clear
#: 3:1 on white, and validated as their own palette with the dataviz skill's
#: `validate_palette.js --mode light --pairs all`: all five checks pass, worst
#: all-pairs dE 13.0 (deutan) / 7.6 (tritan) and normal-vision 16.3. The tritan
#: figure sits in the 6-8 band that is legal only with secondary encoding, which
#: is satisfied twice — each curve is direct-labelled and carries its own dash.
#:
#: There are **four** curves on three hues, deliberately. The latent and observed
#: VP-to-target distributions are two samplings of one entity, not two entities,
#: so they share the violet slot and separate by dash; giving latent its own hue
#: would both assert a distinction the data does not have and force a 4-hue
#: palette whose best red/green pair only reaches dE 7.2 under protanopia.
_VP_PAIRWISE_HUE = "#2a78d6"
_TARGET_PAIRWISE_HUE = "#008300"
_CROSS_HUE = "#4a3aa7"

#: Quantile each distance curve is direct-labelled at. Spread rather than
#: shared, because the distributions nearly coincide on these datasets.
_CDF_LABEL_ANCHORS = (0.80, 0.60, 0.40, 0.20)

#: Efficiency above which the ratio panel switches to a log x axis. A ratio
#: is bounded below by 1 and unbounded above, so a linear axis is right for
#: a healthy campaign and useless for a badly allocated one.
_EFFICIENCY_LOG_ABOVE = 4.0

#: Ink for the VP marker and the flow lines. `#52514e` is the theme's secondary
#: ink: the flows are a density field rather than a category, so they carry no
#: hue, which also leaves the one hue on the figure (the red boundary) meaning
#: what it means on every other map in this package.
_VP_COLOR = "#2a78d6"
_FLOW_COLOR = "#52514e"

#: Flow-line alpha and width. A near-complete bipartite graph puts thousands of
#: lines on one frame, so both are deliberately at the low end: density has to
#: come from accumulation, not from each line being visible.
_FLOW_ALPHA = 0.18
_FLOW_LINEWIDTH = 0.25


def _grid_and_resolution(space: AnswerSpace):
    return (
        get_grid(str(space.seeds["grid_scheme"].iloc[0])),
        int(space.seeds["grid_resolution"].iloc[0]),
    )


def _new_axes(extent, *, figsize=(14.0, 8.0), basemap_kwargs=None):
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    draw_basemap(ax, **(basemap_kwargs or {}))
    return fig, ax


#: Text/grid tokens, matching `diagram/common/palette.py`. Text wears text
#: tokens and never a series colour; the coloured mark beside it carries identity.
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"


def _style_cdf_axes(ax) -> None:
    """Recessive grid and spines, y clamped to a share. One place, two panels."""
    ax.set_facecolor("#ffffff")
    ax.grid(True, color=_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(labelsize=8, colors=_INK_2)


def _caption(ax, text: str) -> None:
    ax.annotate(
        text,
        xy=(0.5, -0.03),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#444444",
    )


def _save(fig, out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _extent_for(space: AnswerSpace, graph: BipartiteGraph, extent):
    """The frame, padded around **both** node sets when auto-derived.

    `plot-answer-space` frames on targets alone, which is right there; here a VP
    outside the target hull would be silently cropped, and an off-frame VP is
    exactly the thing the topology figure exists to show.
    """
    if extent is not None:
        return extent
    lats = np.concatenate(
        [space.assignments["target_lat"].to_numpy(), graph.vp_nodes["vp_lat"].to_numpy()]
    )
    lons = np.concatenate(
        [space.assignments["target_lon"].to_numpy(), graph.vp_nodes["vp_lon"].to_numpy()]
    )
    return _auto_extent(lats, lons)


def plot_bipartite_nodes(
    space: AnswerSpace,
    graph: BipartiteGraph,
    out_path: Path,
    *,
    extent: tuple[float, float, float, float] | None = None,
    title: str | None = None,
    cell_step: int = 8,
    target_size: float = 7.0,
    vp_size: float = 13.0,
    voronoi: bool = True,
    vp_cells: bool = True,
) -> Path:
    """Nodes, cells and class regions — no edges."""
    import cartopy.crs as ccrs

    grid, resolution = _grid_and_resolution(space)
    seeds, assignments = space.seeds, space.assignments
    extent = _extent_for(space, graph, extent)

    fig, ax = _new_axes(extent)

    partition = seed_voronoi(seeds, extent) if voronoi else None
    if partition is not None:
        draw_voronoi(ax, partition)

    # Target cells, filled per seed — one occupied cell is exactly one class.
    draw_cells(
        ax, grid, seeds["cell_id"].to_numpy(), resolution, colors=_seed_colors(len(seeds)), step=cell_step
    )

    # VP cells, outlined only and above the fills so they stay readable where a
    # VP cell and a target cell coincide. zorder is fractional on purpose: this
    # layer belongs between the cells (2) and the boundary (3).
    n_vp_cells = 0
    if vp_cells:
        n_vp_cells = draw_cells(
            ax,
            grid,
            graph.vp_nodes["cell_id"].drop_duplicates().to_numpy(),
            resolution,
            facecolor="none",
            step=cell_step,
            edgecolor=_VP_COLOR,
            linewidth=0.7,
            linestyle=":",
            alpha=0.95,
            zorder=2.6,
        )

    ax.scatter(
        assignments["target_lon"].to_numpy(),
        assignments["target_lat"].to_numpy(),
        s=target_size,
        c="#111111",
        marker="o",
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
        transform=ccrs.PlateCarree(),
        zorder=4,
        label=f"Target ({len(assignments):,})",
    )
    ax.scatter(
        seeds["seed_lon"].to_numpy(),
        seeds["seed_lat"].to_numpy(),
        s=14,
        c="#d62728",
        marker="x",
        linewidths=0.9,
        transform=ccrs.PlateCarree(),
        zorder=5,
        label=f"Region ({len(seeds)})",
    )
    # VPs last and largest: they are what this figure adds over the answer-space
    # map, and a white ring keeps a cluster of co-located VPs countable.
    ax.scatter(
        graph.vp_nodes["vp_lon"].to_numpy(),
        graph.vp_nodes["vp_lat"].to_numpy(),
        s=vp_size,
        c=_VP_COLOR,
        marker="^",
        alpha=0.95,
        edgecolors="white",
        linewidths=0.35,
        transform=ccrs.PlateCarree(),
        zorder=6,
        label=f"VP ({len(graph.vp_nodes):,})",
    )
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    m = graph.meta
    n_tg_cells = m["nodes"]["targets"]["dispersion"]["effective_count"]
    vp_occ = m["nodes"]["vps"]["dispersion"]["occupancy_ratio"]
    e = m["edges"]
    ax.set_title(title or "Bipartite graph — topology", fontsize=12)
    _caption(
        ax,
        f"{grid.name} {grid.resolution_arg}={resolution} "
        f"(~{grid.nominal_cell_km(resolution):.0f} km cells) · "
        f"{len(graph.vp_nodes):,} VPs in {n_vp_cells or '-'} occupied cells "
        f"(occupancy {vp_occ:.2f}) · "
        f"{len(assignments):,} targets in {n_tg_cells} occupied cells = K={len(seeds)} classes\n"
        f"{e['n_edges']:,} measured edges · density {e['edge_density']:.3f} · "
        f"{e['connected_components']['n_components']} component(s) · "
        f"median nearest measured VP "
        f"{e['nearest_vp_km']['observed']['percentiles']['p50']:.0f} km · "
        f"median max angular gap "
        f"{m['angular']['max_angular_gap_deg']['percentiles']['p50']:.0f}°\n"
        f"Dotted outlines are VP-occupied cells — the denominator behind any "
        f"independent-observation claim, since VPs sharing a cell produce "
        f"near-identical constraints.",
    )
    return _save(fig, out_path)


def plot_bipartite_flows(
    space: AnswerSpace,
    graph: BipartiteGraph,
    out_path: Path,
    *,
    extent: tuple[float, float, float, float] | None = None,
    title: str | None = None,
    max_segments: int | None = None,
    voronoi: bool = True,
    seed: int = 20260903,
) -> Path:
    """Every measured edge as a great-circle line, over the class regions.

    Lines are the **distinct** `(VP coord, target coord)` pairs from
    `edge_segments.csv`, not one per edge, with width carrying multiplicity — see
    `bipartite.edge_segments` for why. The basemap is muted so that accumulated
    line ink, not the land fill, is what the eye reads as density.
    """
    import cartopy.crs as ccrs
    from matplotlib.collections import LineCollection

    seeds = space.seeds
    extent = _extent_for(space, graph, extent)

    fig, ax = _new_axes(
        extent,
        basemap_kwargs={
            "ocean": "#f4f8fb",
            "land": "#fbfaf7",
            "coastline": "#c4c4c4",
            "borders": "#dedede",
        },
    )

    seg = graph.edge_segments
    n_all = len(seg)
    sampled = False
    if max_segments is not None and n_all > int(max_segments):
        seg = seg.sample(n=int(max_segments), random_state=seed).sort_index()
        sampled = True

    if len(seg):
        paths = great_circle_segments(
            seg["vp_lat"].to_numpy(dtype=float),
            seg["vp_lon"].to_numpy(dtype=float),
            seg["target_lat"].to_numpy(dtype=float),
            seg["target_lon"].to_numpy(dtype=float),
        )
        mult = seg["n_edges"].to_numpy(dtype=float)
        widths = _FLOW_LINEWIDTH * (1.0 + np.log10(np.maximum(mult, 1.0)))
        ax.add_collection(
            LineCollection(
                list(paths),
                colors=_FLOW_COLOR,
                linewidths=widths,
                alpha=_FLOW_ALPHA,
                transform=ccrs.PlateCarree(),
                zorder=1.5,
            )
        )

    partition = seed_voronoi(seeds, extent) if voronoi else None
    if partition is not None:
        draw_voronoi(ax, partition, linewidth=0.6)

    ax.scatter(
        seeds["seed_lon"].to_numpy(),
        seeds["seed_lat"].to_numpy(),
        s=18,
        c="#d62728",
        marker="x",
        linewidths=1.0,
        transform=ccrs.PlateCarree(),
        zorder=5,
        label=f"Region ({len(seeds)})",
    )
    ax.scatter(
        graph.vp_nodes["vp_lon"].to_numpy(),
        graph.vp_nodes["vp_lat"].to_numpy(),
        s=26,
        c=_VP_COLOR,
        marker="^",
        alpha=0.9,
        edgecolors="white",
        linewidths=0.5,
        transform=ccrs.PlateCarree(),
        zorder=6,
        label=f"VP ({len(graph.vp_nodes):,})",
    )
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    e = graph.meta["edges"]
    length = e["length_km"]["observed"]["percentiles"]
    sample_note = (
        f" Sampled to {len(seg):,} of them (seed {seed})." if sampled else ""
    )
    ax.set_title(title or "Bipartite graph — measured flows", fontsize=12)
    _caption(
        ax,
        f"{e['n_edges']:,} measured (VP, target) edges, drawn as "
        f"{n_all:,} geometrically distinct great-circle lines; width carries how "
        f"many edges each stands for.{sample_note}\n"
        f"Edge length p50 {length['p50']:,.0f} km, p90 {length['p90']:,.0f} km · "
        f"density {e['edge_density']:.3f} · "
        f"{e['connected_components']['n_components']} component(s)\n"
        f"Duplicate IPs at one facility collapse into one line on purpose: one "
        f"line per row would report them as heavier traffic.",
    )
    return _save(fig, out_path)


def plot_distance_cdf(
    graph: BipartiteGraph,
    out_path: Path,
    *,
    title: str | None = None,
) -> Path:
    """Distance CDFs and the measurement-efficiency CDF, as two panels.

    **Left: distances, one axes and one x scale**, because all four series are
    great-circle km. The VP-to-target distribution appears twice — over measured
    edges (solid) and over all pairs (dotted) — which is §7.3's latent/observed
    pair drawn as a pair, so the gap between the two *is* the campaign's sampling
    bias and cannot be overlooked by reading one curve.

    **Right: measurement efficiency**, on its own panel because it is a ratio and
    not a distance. Sharing the left panel's x axis would be the two-scales
    error; a second y axis would be worse. The y axis is shared instead, since
    both panels are cumulative shares — of pairs on the left, of targets on the
    right.

    Each curve is direct-labelled as well as legended, so identity never rests
    on colour alone.
    """
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    pw = graph.pairwise_distance_cdf
    ec = graph.edge_length_cdf
    q_pw = pw["quantile"].to_numpy()
    q_ec = ec["quantile"].to_numpy()
    series = [
        ("VP pairwise", q_pw, pw["vp_pairwise_km"].to_numpy(), _VP_PAIRWISE_HUE, ()),
        ("Target pairwise", q_pw, pw["target_pairwise_km"].to_numpy(), _TARGET_PAIRWISE_HUE, (5.0, 2.0)),
        ("VP→target, measured", q_ec, ec["observed_edge_km"].to_numpy(), _CROSS_HUE, ()),
        ("VP→target, all pairs", q_ec, ec["latent_pair_km"].to_numpy(), _CROSS_HUE, (1.5, 1.5)),
    ]

    fig, (ax, ax_eff) = plt.subplots(
        1, 2, figsize=(11.4, 4.4), sharey=True, gridspec_kw={"width_ratios": [1.75, 1.0]}
    )

    medians: list[str] = []
    for (name, q, v, hue, dash), anchor in zip(series, _CDF_LABEL_ANCHORS):
        finite = np.isfinite(v)
        if not finite.any():
            continue
        line, = ax.plot(v[finite], q[finite], color=hue, linewidth=2.0, label=name, zorder=3)
        if dash:
            line.set_dashes(list(dash))
        medians.append(f"{name} p50 {float(np.interp(0.5, q[finite], v[finite])):,.0f} km")
        # Anchored at its own quantile rather than all four at p50: on these
        # datasets the medians sit within ~200 km of each other, so a shared
        # anchor overprints every label into one illegible line.
        # A white stroke behind the text, because a direct label sits on the
        # curve it names and four curves that nearly coincide leave nowhere to
        # put it that is clear of all of them.
        ax.annotate(
            name,
            xy=(float(np.interp(anchor, q[finite], v[finite])), anchor),
            xytext=(9, 7),
            textcoords="offset points",
            fontsize=7.5,
            color=hue,
            zorder=5,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )
    _style_cdf_axes(ax)
    ax.set_xlabel("Great-circle distance (km)", fontsize=9, color=_INK_2)
    ax.set_ylabel("Cumulative share", fontsize=9, color=_INK_2)
    ax.set_xlim(left=0.0)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)

    # --- right panel: the ratio -------------------------------------------
    eff = ec["measured_nearest_vp_ratio"].to_numpy()
    finite = np.isfinite(eff)
    e_meta = graph.meta["edges"]
    # Read off the per-target column rather than a meta scalar: the count is
    # exactly "efficiency == 1.0", which the metric has already snapped.
    n_got = int(graph.target_nodes["nearest_vp_is_measured"].sum())
    n_targets = graph.meta["nodes"]["targets"]["count"]
    degenerate = finite.any() and float(np.nanmax(eff[finite])) <= 1.0

    if degenerate:
        # Every target measured its nearest VP, so the CDF is a vertical line at
        # 1.0 carrying no information. Drawing it anyway would imply a
        # distribution the data does not have; the sentence is the whole result,
        # and it is a *useful* one — campaign allocation cannot be the
        # explanation for anything §8 finds on this dataset.
        ax_eff.axvline(1.0, color=_CROSS_HUE, linewidth=2.0, zorder=3)
        ax_eff.text(
            0.5,
            0.62,
            f"All {n_targets:,} targets measured\ntheir nearest VP.\n"
            f"The ratio is exactly 1.00,\nso there is no distribution\nto draw.",
            transform=ax_eff.transAxes,
            ha="center",
            va="center",
            fontsize=8.5,
            color=_INK_2,
            zorder=4,
        )
        ax_eff.set_xlim(0.9, 1.6)
    elif finite.any():
        ax_eff.plot(eff[finite], q_ec[finite], color=_CROSS_HUE, linewidth=2.0, zorder=3)
        ax_eff.axvline(1.0, color=_GRID, linewidth=1.0, zorder=1)
        hi = float(np.nanmax(eff[finite]))
        # A badly allocated campaign can put the ratio three orders of magnitude
        # above 1 (1,112x on a synthetic check), which on a linear axis squashes
        # everything interesting into the leftmost pixel. Log below the tick
        # where that starts to bite; the axis is bounded at 1 either way, since
        # the ratio cannot go under it.
        if hi > _EFFICIENCY_LOG_ABOVE:
            ax_eff.set_xscale("log")
        ax_eff.set_xlim(1.0 / 1.02, max(hi * 1.05, 1.1))
        # The share sitting at exactly 1.0 is the number to read, and a step's
        # height is hard to eyeball against a gridline. Placed in axes fractions
        # so it cannot land on the x label the way a data-anchored offset did.
        share = n_got / n_targets if n_targets else 0.0
        ax_eff.text(
            0.97,
            0.06,
            f"{n_got:,}/{n_targets:,} targets ({share:.1%})\nmeasured their nearest VP",
            transform=ax_eff.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=_INK_2,
            zorder=4,
        )
    _style_cdf_axes(ax_eff)
    ax_eff.set_xlabel(
        "Measured nearest-VP ratio\n(nearest measured VP ÷ nearest VP)",
        fontsize=9,
        color=_INK_2,
    )
    ratio = e_meta["measured_nearest_vp_ratio_per_target"]
    ax_eff.set_title(
        "no allocation bias"
        if degenerate
        else f"p50 {ratio['percentiles']['p50']:.3f} · "
             f"p90 {ratio['percentiles']['p90']:.3f} · max {ratio['max']:.3f}",
        fontsize=9,
        color=_INK_2,
    )

    nodes = graph.meta["nodes"]
    fig.suptitle(title or "Node and edge distance CDFs", fontsize=11, color=_INK)
    undefined = n_targets - int(
        e_meta["measured_nearest_vp_ratio_per_target"].get("n", 0)
    )
    undefined_note = (
        f" {undefined} target(s) have a co-located but unmeasured VP, so their "
        f"ratio is infinite and is counted rather than plotted."
        if undefined
        else ""
    )
    fig.text(
        0.008,
        -0.06,
        f"{' · '.join(medians)}\n"
        f"VP diameter {nodes['vps']['geographic_diameter_km']:,.0f} km "
        f"(p95 {nodes['vps']['pairwise_p95_km']:,.0f}) · "
        f"target diameter {nodes['targets']['geographic_diameter_km']:,.0f} km "
        f"(p95 {nodes['targets']['pairwise_p95_km']:,.0f}). A diameter is a "
        f"maximum one node can set alone; the curve is what it summarizes.\n"
        f"The dotted VP→target curve is over all "
        f"{e_meta['length_km']['n_latent_pairs']:,} pairs and the solid one over "
        f"the {e_meta['n_edges']:,} measured; the gap between them is the "
        f"campaign's sampling bias.{undefined_note}",
        ha="left",
        va="top",
        fontsize=7.5,
        color=_MUTED,
    )
    fig.tight_layout()
    return _save(fig, out_path)


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("plot-bipartite-graph")
    def plot_bipartite_cmd(
        run_id: str = typer.Option(
            None, help="Run to plot. Omit with --all-runs to do every run."
        ),
        all_runs: bool = typer.Option(
            False, "--all-runs", help="Plot every run under --outputs-root."
        ),
        bipartite_dir: Path = typer.Option(
            None,
            help="Bipartite-graph dir (from build-bipartite-graph). Defaults to "
                 "this run's bipartite-graph/<grid>-<resolution>/ under "
                 "--analysis-root.",
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option(
            [], "--resolution", "-r", help=RESOLUTION_HELP
        ),
        sweep: bool = typer.Option(False, "--sweep", help=SWEEP_HELP),
        us_only: bool = typer.Option(
            False, "--us-only", help="Clamp the view to the continental US."
        ),
        extent: tuple[float, float, float, float] = typer.Option(
            (None, None, None, None),
            "--extent",
            help="LON_MIN LON_MAX LAT_MIN LAT_MAX. Overrides --us-only.",
        ),
        target_size: float = typer.Option(7.0, help="Target dot size."),
        vp_size: float = typer.Option(13.0, help="VP marker size."),
        no_voronoi: bool = typer.Option(
            False,
            "--no-voronoi",
            help="Skip the nearest-seed overlay. On by default: it is the "
                 "classifier's own top-1 decision boundary.",
        ),
        no_vp_cells: bool = typer.Option(
            False,
            "--no-vp-cells",
            help="Skip the VP-occupied-cell outlines on the topology map.",
        ),
        max_segments: int = typer.Option(
            None,
            help="Cap the flow map's distinct lines, sampled deterministically. "
                 "Unset draws them all.",
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """Topology map, flow map and distance CDF for the bipartite graph.

        Writes bipartite_nodes_map.png, bipartite_flows_map.png and
        distance_cdf.png into the bipartite-graph dir.
        """
        if all_runs == (run_id is not None):
            raise typer.BadParameter("pass exactly one of --run-id or --all-runs")
        if all_runs and bipartite_dir is not None:
            raise typer.BadParameter("--bipartite-dir cannot be combined with --all-runs")

        chosen_extent: tuple[float, float, float, float] | None = None
        if extent and all(v is not None for v in extent):
            chosen_extent = tuple(float(v) for v in extent)  # type: ignore[assignment]
        elif us_only:
            chosen_extent = US_MAINLAND_EXTENT

        g, resolutions = resolve_cli_grid(grid, resolution, sweep=sweep)
        runs = discover_runs(outputs_root) if all_runs else [resolve_run(run_id, outputs_root)]
        jobs = (
            [(r, None) for r in runs]
            if bipartite_dir is not None
            else [(r, x) for r in runs for x in resolutions]
        )
        for run, want_res in jobs:
            out_dir = bipartite_dir or run.bipartite_dir(
                root=analysis_root, grid=g.name, resolution=want_res
            )
            graph = load_bipartite(out_dir)
            space_grid = str(graph.meta["grid"]["scheme"])
            space_res = int(graph.meta["grid"]["resolution"])
            space = load_answer_space(
                run.answer_space_dir(
                    root=analysis_root, grid=space_grid, resolution=space_res
                )
            )
            label = f"{space_grid} {get_grid(space_grid).resolution_arg}={space_res}"
            nodes = plot_bipartite_nodes(
                space,
                graph,
                out_dir / NODES_PNG,
                extent=chosen_extent,
                title=f"{run.run_id} — bipartite graph, topology ({label})",
                target_size=target_size,
                vp_size=vp_size,
                voronoi=not no_voronoi,
                vp_cells=not no_vp_cells,
            )
            flows = plot_bipartite_flows(
                space,
                graph,
                out_dir / FLOWS_PNG,
                extent=chosen_extent,
                title=f"{run.run_id} — bipartite graph, measured flows ({label})",
                max_segments=max_segments,
                voronoi=not no_voronoi,
            )
            cdf = plot_distance_cdf(
                graph,
                out_dir / CDF_PNG,
                title=f"{run.run_id} — node and edge distance CDFs ({label})",
            )
            typer.echo(
                f"{run.run_id}: {label} · {graph.n_vps} VPs, {graph.n_targets} targets, "
                f"{graph.meta['edges']['n_edges']:,} edges "
                f"({graph.meta['edges']['n_distinct_geometry_edge']:,} distinct lines) -> "
                f"{nodes.name}, {flows.name}, {cdf.name}"
            )
