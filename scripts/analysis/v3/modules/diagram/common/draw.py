"""The matplotlib primitives both layouts are assembled from.

Also the one place the Agg backend is selected. Every plotting module in this
package imports `plt` from here rather than importing `matplotlib.pyplot`
itself, so the backend is guaranteed to be set before pyplot is first imported
— an ordering that is otherwise only true by accident of import order.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.v3.modules.diagram.common.palette import (  # noqa: E402
    _C_INK,
    _C_INK_2,
    _C_MUTED,
)

__all__ = ["plt", "_label_anchor", "_draw_circles", "_annotate_figure"]


def _label_anchor(patch):
    """A point comfortably inside `patch`, even when it is a curved sliver.

    `representative_point` only promises to be inside, which for a crescent puts
    the label hard against an arc. `polylabel` returns the pole of
    inaccessibility — the point furthest from any edge — which is what keeps a
    5-set sliver's label off its own boundary.
    """
    from shapely.ops import polylabel

    geom = patch
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    try:
        return polylabel(geom, tolerance=1e-3)
    except Exception:
        return geom.representative_point()


def _draw_circles(ax, order: list[str], colors: dict[str, str], centres, radii) -> None:
    """Fill every circle, then outline every circle. Shared by both layouts.

    Two passes rather than one, so no circle's fill lands on top of another's
    edge. Translucent fills so overlaps blend: no point is covered by more than
    one fill per circle, so the stack tops out at `n` rather than compounding an
    arbitrary number of times over the same pixel.

    **Largest first**, so the biggest set sits at the bottom of the stack and the
    smallest on top. Same-`zorder` artists draw in insertion order, so this is
    the whole mechanism. It matters on the Euler layout, where the radii differ:
    drawing in method order lets a 60%-of-the-population circle be laid over a
    34% one, burying the smaller set's outline in the larger's fill. On the ring
    every radius is equal, so the sort is stable and the ring order survives.
    """
    from matplotlib.patches import Circle

    stacked = sorted(range(len(order)), key=lambda i: -float(radii[i]))
    for i in stacked:
        ax.add_patch(
            Circle(centres[i], radii[i], facecolor=colors[order[i]],
                   edgecolor="none", alpha=0.22, zorder=1)
        )
    for i in stacked:
        ax.add_patch(
            Circle(centres[i], radii[i], facecolor="none",
                   edgecolor=colors[order[i]], linewidth=1.8, zorder=2)
        )


def _annotate_figure(ax, *, title: str, subtitle: str, note: str) -> None:
    """Title above, subtitle under it, footnote below the axes."""
    ax.set_title(title, fontsize=13, fontweight="bold", color=_C_INK, pad=16)
    if subtitle:
        ax.annotate(
            subtitle, xy=(0.5, 1.005), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=9.5, color=_C_INK_2,
        )
    ax.annotate(
        note, xy=(0.5, -0.015), xycoords="axes fraction",
        ha="center", va="top", fontsize=8.5, color=_C_MUTED,
    )
