"""Drawing a fitted layout: names on the circles, numbers almost nowhere.

No region carries a label — at six sets there are thirty-one of them, and the
combination a region stands for is legible from the arcs bounding it. What is
printed is each set's own share (which by `circle_radii` *is* its drawn area, so
the number and the ink cannot drift), the share no method got right, and the
figure's own fit error.

Label placement is the fiddly part and `_euler_label_points` is where it lives:
every name sits on the circle it names, which is what lets the figure drop
leader lines entirely.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from scripts.analysis.v3.modules.diagram.common.draw import (
    _annotate_figure,
    _draw_circles,
    _label_anchor,
    plt,
)
from scripts.analysis.v3.modules.diagram.common.labels import label_for
from scripts.analysis.v3.modules.diagram.common.palette import (
    _C_MUTED,
    _SURFACE,
    method_colors,
)
from scripts.analysis.v3.modules.diagram.euler.layout import EulerLayout


#: Printed on every Euler figure. The counterpart to `RING_CAVEAT`, and its
#: mirror image: there, nothing but the labels carried data; here, nothing *but*
#: the geometry does.
EULER_CAPTION = (
    "Circle area = share of targets correct; shared area = share both get right."
)


#: Blank layout units left around the circles. Labels sit on their own circles
#: now, so this only has to clear the stroke and the type's own height.
EULER_MARGIN = 0.22

#: How far in from its own boundary a boundary-anchored label sits, as a share
#: of that circle's radius. Far enough that the type clears the stroke, near
#: enough that it still reads as belonging to the arc rather than the interior.
BOUNDARY_LABEL_INSET = 0.16

#: Minimum distance between two labels, in layout units — roughly the height of
#: a name at the size they are drawn.
LABEL_MIN_GAP = 0.26

#: A set is named inside its own exclusive lobe when that lobe is at least this
#: share of its circle. Below it the lobe is a crescent, and a name centred in a
#: crescent reads as belonging to whatever fills the rest of the circle.
LOBE_LABEL_SHARE = 0.12


def _spread_angles(angles: list[float], min_gap: float) -> list[float]:
    """Nudge angles apart until adjacent ones clear `min_gap`, keeping their order.

    Order is preserved so the labels stay in the same rotational sequence as the
    circles they name, which is what keeps the leader lines from crossing. A few
    relaxation passes are enough for the handful of labels a legible Euler
    diagram can carry; if the ring is too crowded to satisfy the gap, the passes
    simply end with the best spacing they reached.
    """
    if len(angles) < 2:
        return list(angles)
    out = list(angles)
    for _ in range(64):
        moved = False
        for i in range(len(out)):
            j = (i + 1) % len(out)
            gap = (out[j] - out[i]) % (2 * math.pi)
            if gap < min_gap:
                push = (min_gap - gap) / 2
                out[i] -= push
                out[j] += push
                moved = True
        if not moved:
            break
    return out


def _euler_label_points(layout: EulerLayout) -> list[tuple[float, float]]:
    """Where to write each circle's name. Always a point **on that circle**.

    Two placements, in order of preference:

    *Inside its own exclusive lobe*, when the lobe is at least
    `LOBE_LABEL_SHARE` of the circle. That is the placement that reads as "this
    circle is Vanilla CBG", and it is what a well-separated set gets.

    *Just inside its own boundary, on the bearing away from the layout's
    centroid*, otherwise. When a set is nearly contained in another
    (Shortest-Ping sits inside SoI CBG on the operator runs, one target short of
    total) it has no lobe to speak of, and a name centred in a crescent reads as
    belonging to whatever fills the rest of the circle. The outward arc is the
    part of that circle least covered by the others, so a name there is
    attributable to the arc it sits on.

    Every point is within its own circle, which is what lets the figure drop
    leader lines: a label never has to be connected to the thing it names,
    because it is already on it.
    """
    from shapely.geometry import Point
    from shapely.ops import unary_union

    disks = [
        Point(c).buffer(r, quad_segs=128)
        for c, r in zip(layout.centres, layout.radii)
    ]
    hub = layout.centres.mean(axis=0)
    points: dict[int, tuple[float, float]] = {}
    on_boundary: list[int] = []
    for i, disk in enumerate(disks):
        others = [d for k, d in enumerate(disks) if k != i]
        lobe = disk.difference(unary_union(others)) if others else disk
        if not lobe.is_empty and lobe.area >= LOBE_LABEL_SHARE * disk.area:
            point = _label_anchor(lobe)
            points[i] = (point.x, point.y)
        else:
            on_boundary.append(i)

    # Near-coincident circles (Shortest-Ping and SoI CBG differ by one target,
    # so the fit puts them almost on top of each other) leave the hub on almost
    # the same bearing and would stack their names. Spreading the *angles*
    # rather than walking outwards keeps each label on its own circle, which is
    # the property the whole placement rests on.
    bearings = {i: math.atan2(*(layout.centres[i] - hub)[::-1]) for i in on_boundary}
    # Ties would leave the spreading with no order to preserve; index breaks them.
    ranked = sorted(on_boundary, key=lambda i: (bearings[i], i))
    smallest = min((layout.radii[i] for i in on_boundary), default=1.0)
    spread = _spread_angles(
        [bearings[i] for i in ranked], LABEL_MIN_GAP / max(float(smallest), 1e-6)
    )
    for i, angle in zip(ranked, spread):
        reach = layout.radii[i] * (1.0 - BOUNDARY_LABEL_INSET)
        points[i] = tuple(
            layout.centres[i] + reach * np.array([math.cos(angle), math.sin(angle)])
        )

    return [points[i] for i in range(len(disks))]


#: What the space outside every circle is called. It is a real region of the
#: population — the targets no method placed correctly — and leaving it blank
#: invites reading the union of the circles as the whole population when on the
#: pooled operator runs it is 85% of it.
#:
#: Its percentage is `observed[0]` and is exact, but — unlike every circle — its
#: **area is not to scale**: the space outside the union is whatever frame is
#: left over after the layout, not a fitted share. It is the one label here whose
#: number and ink are unrelated, which is why it is drawn muted and outside.
OUTSIDE_LABEL = "None"


def plot_euler(
    layout: EulerLayout,
    out_path: Path,
    *,
    title: str,
    subtitle: str = "",
    fit_ref: str = "overlap_euler_fit.csv",
    caption: bool = True,
) -> Path:
    """Draw the fitted layout: names on the circles, numbers nowhere.

    No region carries a label. At six sets there are thirty-one of them and the
    combination a region stands for is legible from the arcs bounding it, so
    printing every one buries the relationship the layout exists to show.

    Names only — no letter ids. The letters still key `euler_fit_table`'s
    `region` column, but that table also names the methods in full, and the
    ring's own region key is where a letter legend belongs. On this figure they
    were a second line of type per circle buying nothing the colour did not
    already say.

    `caption=False` drops the note under the figure for slide use. It defaults
    on because the layout misplaces a real fraction of the targets — 16% at six
    sets — and a figure that says nothing about that asserts an exactness it
    does not have.
    """
    from matplotlib.patheffects import withStroke

    order = layout.order
    colors = method_colors(order)

    # Window the union, not the origin: the fit centres the circles on their own
    # mean, which is not the middle of what they cover. The canvas then takes the
    # window's aspect, because a fitted layout is rarely square and a square
    # figure would pad the short axis with blank canvas.
    lo = (layout.centres - layout.radii[:, None]).min(axis=0)
    hi = (layout.centres + layout.radii[:, None]).max(axis=0)
    mid = (lo + hi) / 2
    half = (hi - lo) / 2 + EULER_MARGIN
    fig, ax = plt.subplots(
        figsize=(9.0, float(np.clip(9.0 * half[1] / half[0], 5.0, 11.0)))
    )
    ax.set_xlim(mid[0] - half[0], mid[0] + half[0])
    ax.set_ylim(mid[1] - half[1], mid[1] + half[1])
    ax.set_aspect("equal")
    ax.axis("off")
    _draw_circles(ax, order, colors, layout.centres, layout.radii)

    halo = [withStroke(linewidth=2.6, foreground=_SURFACE)]
    # Centred just under the lowest circle rather than at the foot of the frame:
    # in data units, so it tracks the layout instead of drifting with the margin.
    # `EULER_MARGIN` guarantees the band below `lo[1]` is clear of every circle
    # whatever the fit produced.
    ax.annotate(
        f"{OUTSIDE_LABEL}\n{100 * float(layout.observed[0]):.1f}%",
        xy=(mid[0], lo[1] - 0.03),
        ha="center", va="top", zorder=6,
        fontsize=10.5, fontweight="bold", color=_C_MUTED,
        linespacing=1.35, path_effects=halo,
    )

    for i, (lx, ly) in enumerate(_euler_label_points(layout)):
        method = order[i]
        # `pi * r**2` is not a re-derivation of the share — by `circle_radii`
        # it *is* the circle's drawn area, so the printed number and the ink
        # cannot drift apart. Sets only: an intersection's drawn area is fitted
        # and would disagree with its observed share by up to the figure's
        # error, which is what the caption reports instead.
        ax.annotate(
            f"{label_for(method)}\n{100 * math.pi * layout.radii[i] ** 2:.1f}%",
            xy=(lx, ly),
            ha="center", va="center", zorder=6,
            fontsize=10.5, fontweight="bold", color=colors[method],
            linespacing=1.35, path_effects=halo,
        )

    # One line, not five. The figure's own accuracy is the only thing a reader
    # cannot get from the geometry, so it stays; the rest of what the old
    # caption spelled out is what the picture is already showing them.
    # Two lines, not one: `bbox_inches="tight"` grows the canvas to fit the
    # widest artist, so a single ~200-character note pads the figure out
    # sideways and shrinks the circles it is describing.
    note = (
        f"{EULER_CAPTION}\n"
        f"Fitted layout: {100 * layout.placed:.1f}% of targets land in the "
        f"region drawn (largest pair error {100 * layout.pair_error():.1f}%) "
        f"— see {fit_ref}."
    ) if caption else ""
    _annotate_figure(ax, title=title, subtitle=subtitle, note=note)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return out_path
