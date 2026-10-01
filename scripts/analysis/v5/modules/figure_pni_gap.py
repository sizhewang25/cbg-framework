"""`d_pni` against the S-P gap, one marker per scatter point, coloured by cluster.

The data and the clustering are `pni_gap`'s. This module draws them, and it is
where `plot-pni-gap` writes the three artifacts together: the clusters the RTT
figure reads, the points, and this PNG. One command writes all three, so the
picture cannot show a partition the CSV does not hold.

## Axes

Both axes are symlog with `pni_gap`'s parameters: linear 0-100 km with ticks
every 25, log from 100 to 4,000 km, and equal aspect, since both axes are
distances on the same transform. A dotted line marks the 100 km switch on
each axis, because the change of scale is invisible otherwise and a reader
would compare a 0-100 km step with a 100-1,000 km one.

A gap of exactly zero sits on the x axis; symlog draws zero, so there is no
fold-onto-the-floor trick here as in `figure_vp_distance_cdf`. Markers are
drawn unclipped so a point on an axis is a whole marker, not half of one.

## Encoding

Marker area grows with `n_tgs`, because a point holds up to ~20 replicas
and a split site's 5-replica point is a real minority. Cluster is carried by
**hue and shape together** plus a direct label, because three of the six
palette hues are under 3:1 against the surface (validated:
`validate_palette.js --mode light`, CVD worst adjacent dE 9.1). The hues are
the reference categorical palette in fixed order, so cluster 1 is always the
same blue whatever k is.

## Pooled

`--layout pooled` draws every run's points on one panel, clustered once. The
marker encodes cluster only, not dataset: the question is whether the regimes
recur across operators, and `clusters_by_run` in the manifest says which runs
each cluster is made of.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.analysis.v5.modules import pni_gap as P  # noqa: E402
from scripts.analysis.v5.modules.mapping import INK_2, MUTED  # noqa: E402
from scripts.analysis.v5.modules.paths import RunPaths  # noqa: E402

PNG_NAME = "pni_gap_scatter.png"

#: Reference categorical palette, fixed order (dataviz `palette.md`, light).
CLUSTER_HUES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
CLUSTER_MARKERS = ("o", "s", "^", "D", "v", "P")

MAJOR_TICKS_KM = (0, 25, 50, 75, 100, 1000, 4000)
MINOR_TICKS_KM = (200, 300, 400, 500, 600, 700, 800, 900, 2000, 3000)


def cluster_style(cluster: int) -> tuple[str, str]:
    """`(hue, marker)` for cluster `1..k`. Refuses k beyond the palette."""
    i = int(cluster) - 1
    if not 0 <= i < len(CLUSTER_HUES):
        raise ValueError(
            f"cluster {cluster} has no colour: the palette holds {len(CLUSTER_HUES)} "
            f"(generating more hues breaks the CVD validation). Use a smaller --k."
        )
    return CLUSTER_HUES[i], CLUSTER_MARKERS[i]


def cluster_label(cluster: int) -> str:
    return f"C{int(cluster)}"


def marker_area(n_tgs) -> np.ndarray:
    return 10.0 + 2.5 * np.asarray(n_tgs, dtype=float)


def style_symlog_axis(axis) -> None:
    axis.set_major_locator(ticker.FixedLocator(MAJOR_TICKS_KM))
    axis.set_minor_locator(ticker.FixedLocator(MINOR_TICKS_KM))
    axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def plot(pts: pd.DataFrame, summary: pd.DataFrame, *, meta: dict, out_png: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.3, 3.8))  # square panel + legend column on the right
    for c, block in pts.groupby(P.CLUSTER_COL):
        hue, marker = cluster_style(c)
        row = summary.set_index(P.CLUSTER_COL).loc[c]
        ax.scatter(
            block[P.D_PNI], block[P.GAP], s=marker_area(block.n_tgs), c=hue, marker=marker,
            alpha=0.75, edgecolor="white", linewidth=0.6, clip_on=False, zorder=3,
            label=f"{cluster_label(c)}  {int(row.n_tgs)} TGs, {int(row.n_sites)} sites",
        )
        # Direct label at the cluster's top-right point, nudged off the marker.
        top = block.loc[block[P.GAP].idxmax()]
        ax.annotate(
            cluster_label(c), (top[P.D_PNI], top[P.GAP]), xytext=(6, 4),
            textcoords="offset points", fontsize=7.5, color=INK_2, zorder=4,
        )

    ax.set_xscale("symlog", linthresh=P.LINTHRESH_KM, linscale=P.LINSCALE)
    ax.set_yscale("symlog", linthresh=P.LINTHRESH_KM, linscale=P.LINSCALE)
    ax.set_xlim(0, P.AXIS_MAX_KM)
    ax.set_ylim(0, P.AXIS_MAX_KM)
    ax.set_aspect("equal")
    for axis in (ax.xaxis, ax.yaxis):
        style_symlog_axis(axis)
    ax.axvline(P.LINTHRESH_KM, color=MUTED, lw=0.5, ls=":", zorder=1)
    ax.axhline(P.LINTHRESH_KM, color=MUTED, lw=0.5, ls=":", zorder=1)

    beyond = meta["axes"]["n_points_beyond_max"]
    if beyond:
        ax.text(0.98, 0.02, f"{beyond} points beyond {P.AXIS_MAX_KM:g} km not drawn",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=INK_2)

    ax.set_xlabel(r"$d(\mathrm{TG},\ \mathrm{nearest\ PNI})$ (km)", fontsize=8)
    ax.set_ylabel(r"$d_\mathrm{sp}-d_\mathrm{geo}$ (km)", fontsize=8)
    ax.tick_params(labelsize=7.5, length=3)
    ax.tick_params(which="minor", length=1.8)
    ax.grid(alpha=0.25, lw=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # Outside the axes: every corner of the panel holds data on the pooled meshes.
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6.5, frameon=False,
                    borderaxespad=0.0,
                    handletextpad=0.3, labelspacing=0.5)
    for text in leg.get_texts():
        text.set_color(INK_2)
    for handle in leg.legend_handles:
        handle.set_sizes([28.0])  # the key names a cluster, not a TG count

    fig.tight_layout(pad=0.4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _checked_k(k: int | None) -> None:
    if k is not None:
        cluster_style(k)  # refuse an unpaintable --k before fitting anything


def _draw(tgs: pd.DataFrame, pts: pd.DataFrame, meta: dict, out_dir: Path) -> Path:
    for c in pts[P.CLUSTER_COL].unique():
        cluster_style(c)  # and a chosen k, before anything is written
    P.write(tgs, pts, meta, out_dir)
    return plot(pts, P.cluster_summary(tgs, pts), meta=meta, out_png=out_dir / PNG_NAME)


def build_for_runs(
    runs: list[RunPaths],
    pni_csvs: dict[str, Path],
    *,
    layouts: tuple[str, ...] = (P.PER_RUN,),
    k: int | None = None,
    method: str = P.DEFAULT_METHOD,
    analysis_root: Path | None = None,
    source_csvs: dict[str, Path] | None = None,
) -> list[Path]:
    """Every requested layout. `per-run` writes one figure per run into that
    run's tree; `pooled` writes one figure over all of them into `_cross/`.
    Each run is measured against its own PNI list either way."""
    _checked_k(k)
    bad = [lay for lay in layouts if lay not in P.LAYOUTS]
    if bad:
        raise ValueError(f"unknown layout {bad}; expected {list(P.LAYOUTS)}")
    pngs = []
    if P.PER_RUN in layouts:
        for run in runs:
            one = {run.run_id: (source_csvs or {}).get(run.run_id)} if source_csvs else None
            tgs, pts, meta = P.compute_runs([run], pni_csvs, layout=P.PER_RUN, k=k, method=method,
                                             source_csvs=one)
            out_dir = P.output_dir(run.run_id, pni_csvs[run.run_id], analysis_root=analysis_root)
            pngs.append(_draw(tgs, pts, meta, out_dir))
    if P.POOLED in layouts:
        tgs, pts, meta = P.compute_runs(runs, pni_csvs, layout=P.POOLED, k=k, method=method,
                                         source_csvs=source_csvs)
        out_dir = P.pooled_output_dir([r.run_id for r in runs], analysis_root=analysis_root)
        pngs.append(_draw(tgs, pts, meta, out_dir))
    return pngs


def build_for_run(
    run: RunPaths,
    pni_csv: Path,
    *,
    k: int | None = None,
    method: str = P.DEFAULT_METHOD,
    analysis_root: Path | None = None,
    source_csv: Path | None = None,
) -> list[Path]:
    """Clusters CSV, points CSV, manifest and the scatter for one run, per-run layout."""
    return build_for_runs(
        [run], {run.run_id: pni_csv}, k=k, method=method, analysis_root=analysis_root,
        source_csvs={run.run_id: source_csv} if source_csv is not None else None,
    )
