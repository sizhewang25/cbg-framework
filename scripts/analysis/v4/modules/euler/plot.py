"""Drawing a fitted layout: names on the circles, numbers almost nowhere.

Circle area *is* the set size, the area two circles share *is* the size of
their intersection, methods that never agree are drawn apart and a method
contained in another is drawn inside it. No region carries a label — at six
sets there are thirty-one of them, and the combination a region stands for is
legible from the arcs bounding it.

What is printed is each set's own share (which by `circle_radii` is its drawn
area, so the number and the ink cannot drift), the share no method got right,
and the figure's own fit error.

Ported from v3's `diagram/euler/plot.py`; the differences are all v4's:
labels come from `modules.methods`, the caption names the **ring tolerance**
the figure was drawn at rather than a seed rank, and a footnote names any set
that was empty at that tolerance and therefore has no circle.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.v4.modules.euler.layout import (  # noqa: E402
    DISJOINT_MARGIN,
    EulerLayout,
    circle_radii,
)
from scripts.analysis.v4.modules.methods import (  # noqa: E402
    method_colors,
    method_label,
)

_SURFACE = "#ffffff"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"

#: Printed on every Euler figure. The figure's whole claim in two lines: unlike
#: the outcome bars, where every quantity is a printed label, here nothing but
#: the geometry carries data.
EULER_CAPTION = (
    "Circle area = share of targets placed within the tolerance; shared area = "
    "share both place.\n"
    "Dashed circle: the share no method placed — same area scale, set apart "
    "because it meets none of them."
)

#: Blank layout units left around the circles. Labels sit on their own circles,
#: so this only has to clear the stroke and the type's own height.
EULER_MARGIN = 0.22

#: How far in from its own boundary a boundary-anchored label sits, as a share
#: of that circle's radius. Far enough that the type clears the stroke, near
#: enough that it still reads as belonging to the arc rather than the interior.
BOUNDARY_LABEL_INSET = 0.16

#: Minimum distance between two labels, in layout units. A label is two lines
#: of bold 10.5 pt — a name over a percentage — so it is about as wide as it is
#: tall at this scale, and the gap is one such block. Separation past this buys
#: nothing, so the placement search caps its reward there and spends the
#: remaining freedom on keeping each name in its own exclusive lobe.
LABEL_MIN_GAP = 0.30

#: A set is named inside its own exclusive lobe when that lobe is at least this
#: share of its circle. Below it the lobe is a crescent, and a name centred in
#: a crescent reads as belonging to whatever fills the rest of the circle.
LOBE_LABEL_SHARE = 0.12

#: Candidate positions tried around each circle's own boundary, in addition to
#: its lobe anchor. 36 is one every 10 degrees: finer than the ~0.3-unit gap
#: the search is resolving on any circle big enough to carry a label, so the
#: grid is never what limits the answer.
BOUNDARY_CANDIDATES = 36

#: Reward, as a multiple of `LABEL_MIN_GAP`, for a position that sits in the
#: circle's **exclusive** lobe — inside its own circle and no other. That is
#: the placement that reads as "this circle is Vanilla"; anywhere else the name
#: sits in an overlap and could be read as belonging to either set.
#:
#: Below 1.0 deliberately: a clear, correctly attributable label beats a
#: perfectly attributable one that collides with its neighbour, so separation
#: always outranks exclusivity when the two conflict.
EXCLUSIVE_BONUS = 0.55

#: A smaller reward for the lobe's pole of inaccessibility specifically — the
#: point furthest from any edge. Every exclusive position is attributable; this
#: one is also comfortable, so it breaks ties among them without ever
#: outweighing the separation term.
LOBE_ANCHOR_BONUS = 0.12

#: Clear space between the `None` circle and the nearest set circle, in layout
#: units. `DISJOINT_MARGIN` alone would do for the geometry, but the set
#: nearest the `None` circle is usually labelled on its *outward* arc and that
#: name overhangs its own boundary by roughly half its width.
OUTSIDE_STANDOFF = 0.24

#: What the space outside every circle is called. It is a real region of the
#: population — the targets no method placed within the tolerance — and leaving
#: it blank invites reading the union of the circles as the whole population,
#: when at nside 128 top-1 it is two thirds of it.
OUTSIDE_LABEL = "None"


def _label_anchor(patch):
    """A point comfortably inside `patch`, even when it is a curved sliver.

    `representative_point` only promises to be inside, which for a crescent
    puts the label hard against an arc. `polylabel` returns the pole of
    inaccessibility — the point furthest from any edge.
    """
    from shapely.ops import polylabel

    geom = patch
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    try:
        return polylabel(geom, tolerance=1e-3)
    except Exception:
        return geom.representative_point()


def _candidate_points(layout: EulerLayout, i: int, disks, lobes):
    """Where circle `i`'s name could go: its lobe anchor, then its own boundary.

    Every candidate is a point **inside circle `i`**, which is what lets the
    figure drop leader lines — a label never has to be connected to the thing
    it names, because it is already on it.
    """
    centre, radius = layout.centres[i], float(layout.radii[i])
    out = []
    lobe = lobes[i]
    if not lobe.is_empty and lobe.area >= LOBE_LABEL_SHARE * disks[i].area:
        point = _label_anchor(lobe)
        out.append((np.array([point.x, point.y]), True))
    reach = radius * (1.0 - BOUNDARY_LABEL_INSET)
    for k in range(BOUNDARY_CANDIDATES):
        angle = 2 * math.pi * k / BOUNDARY_CANDIDATES
        out.append(
            (centre + reach * np.array([math.cos(angle), math.sin(angle)]), False)
        )
    return out


def _euler_label_points(layout: EulerLayout) -> list[tuple[float, float]]:
    """Where to write each circle's name. Always a point **on that circle**.

    Chosen globally rather than one circle at a time, which is the difference
    from v3's placement and the reason this was rewritten. There, each name
    went to its own exclusive lobe when it had one and to its outward arc when
    it did not, and only the *outward-arc* names were then spread apart. Two
    names picked by different branches could therefore land on top of each
    other — on the pooled meshes at one ring out, "SoI 49.1%" was drawn over
    "Octant-Spline 40.4%", and "Vanilla" over "Shortest-Ping".

    Instead every circle offers the same menu — its lobe anchor plus
    `BOUNDARY_CANDIDATES` points around its own boundary — and positions are
    taken in **ascending radius order**, smallest circle first. Most-constrained
    first: a small circle has a short boundary and few distinguishable places to
    put a name, while a large one can almost always find another arc.

    Each circle takes the candidate maximizing

        clear = min(gap / LABEL_MIN_GAP, 1)   # gap = distance to nearest name

        clear * LABEL_MIN_GAP
        + EXCLUSIVE_BONUS   * LABEL_MIN_GAP * clear   if in this circle alone
        + LOBE_ANCHOR_BONUS * LABEL_MIN_GAP * clear   if at the lobe's pole

    Capping the distance term is what keeps the two goals from fighting: once
    a name is a full gap clear of its neighbours, more distance is worth
    nothing and the search spends what is left on attributability. Scaling the
    bonuses by the same `clear` is what stops them buying a collision —
    unscaled, an exclusive crescent 0.12 units from its neighbour outscored a
    clear arc, which is how "SoI 20.4%" and "Octant-Hull 26.8%" ended up drawn
    over each other at top-1.

    Deterministic — no randomness, and ties break on candidate order — so the
    same layout always yields the same figure.
    """
    from shapely.geometry import Point
    from shapely.ops import unary_union

    disks = [
        Point(c).buffer(r, quad_segs=128)
        for c, r in zip(layout.centres, layout.radii)
    ]
    lobes = [
        d.difference(unary_union([o for k, o in enumerate(disks) if k != i]))
        if len(disks) > 1 else d
        for i, d in enumerate(disks)
    ]

    placed: dict[int, np.ndarray] = {}
    for i in sorted(range(len(disks)), key=lambda k: (float(layout.radii[k]), k)):
        best, best_score = None, -math.inf
        for point, is_anchor in _candidate_points(layout, i, disks, lobes):
            gap = min(
                (float(np.hypot(*(point - q))) for q in placed.values()),
                default=LABEL_MIN_GAP,
            )
            score = min(gap, LABEL_MIN_GAP)
            # Both bonuses are **scaled by how clear the position already is**,
            # so a colliding candidate earns none of them. Adding them flat let
            # exclusivity buy a collision: at top-1 the Octant-Hull name took an
            # exclusive crescent 0.12 units from "SoI 20.4%" over a clear arc,
            # because 0.12 + 0.55*0.30 beat 0.30. Scaling makes separation
            # strictly dominant and leaves exclusivity to break ties among
            # positions that are already clear.
            clear = min(gap / LABEL_MIN_GAP, 1.0)
            if lobes[i].contains(Point(point)):
                score += EXCLUSIVE_BONUS * LABEL_MIN_GAP * clear
            if is_anchor:
                score += LOBE_ANCHOR_BONUS * LABEL_MIN_GAP * clear
            if score > best_score:
                best, best_score = point, score
        placed[i] = best

    return [tuple(placed[i]) for i in range(len(disks))]


def outside_circle(layout: EulerLayout) -> tuple[np.ndarray, float]:
    """Centre and radius of the `None` circle: area to scale, position not.

    The region outside every set is as real as any of them — at nside 128 top-1
    it is the majority of the population — and drawing it as leftover frame
    would make it the one quantity on the figure whose number and ink are
    unrelated. Its radius comes from the same `circle_radii` as every set, so
    its area *is* its share and can be compared against theirs directly.

    What its position cannot carry is a relationship, because it has none: the
    never-placed targets are by construction in no set, so the circle meets
    nothing. It is placed on the layout's horizontal midline, as far left as it
    can sit while clearing every circle by `DISJOINT_MARGIN` — the standoff the
    fit puts between two sets that never co-occur, so "these do not meet" looks
    the same wherever it appears — or by `OUTSIDE_STANDOFF` where that is
    wider. `plot_euler` draws it dashed and muted to say the placement itself
    is not fitted.
    """
    radius = float(circle_radii(np.array([float(layout.observed[0])]))[0])
    lo = (layout.centres - layout.radii[:, None]).min(axis=0)
    hi = (layout.centres + layout.radii[:, None]).max(axis=0)
    y = float((lo[1] + hi[1]) / 2)
    # Clear of the layout's own bounding box to begin with, then pushed right
    # by whichever circle the midline actually runs closest to.
    x = float(hi[0]) + radius + OUTSIDE_STANDOFF
    for centre, r in zip(layout.centres, layout.radii):
        gap = max(DISJOINT_MARGIN * min(float(r), radius), OUTSIDE_STANDOFF)
        reach = float(r) + radius + gap
        dy = y - float(centre[1])
        if abs(dy) < reach:
            x = max(x, float(centre[0]) + math.sqrt(reach * reach - dy * dy))
    return np.array([x, y]), radius


def _draw_circles(ax, order, colors, centres, radii) -> None:
    """Fill every circle, then outline every circle.

    Two passes rather than one, so no circle's fill lands on top of another's
    edge. Translucent fills so overlaps blend.

    **Largest first**, so the biggest set sits at the bottom of the stack and
    the smallest on top. Same-`zorder` artists draw in insertion order, so this
    is the whole mechanism, and it matters here because the radii differ:
    drawing in method order lets a 60%-of-the-population circle be laid over a
    34% one, burying the smaller set's outline in the larger's fill.
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


def plot_euler(
    layout: EulerLayout,
    out_path: Path,
    *,
    title: str,
    subtitle: str = "",
    fit_ref: str = "euler_fit.csv",
    empty_sets: list[str] | None = None,
    caption: bool = True,
) -> Path:
    """Draw the fitted layout: names on the circles, numbers nowhere else.

    `empty_sets` names methods that placed **no** target within this figure's
    tolerance. They have no circle — a zero-radius one is a dot a reader takes
    for "very small" rather than "never" — and removing a set nothing belongs
    to changes no other region, so the drawing is unaffected. Naming them in
    the footnote is what stops their absence reading as an arbitrary exclusion.

    `caption=False` drops the note under the figure for slide use. It defaults
    on because the layout misplaces a real fraction of the targets and a figure
    that says nothing about that asserts an exactness it does not have.
    """
    from matplotlib.patches import Circle
    from matplotlib.patheffects import withStroke

    order = layout.order
    colors = method_colors(order)
    out_centre, out_radius = outside_circle(layout)

    # Window the union, not the origin: the fit centres the circles on their
    # own mean, which is not the middle of what they cover. The `None` circle
    # is part of what has to fit — it always extends the window rightwards, and
    # dominates it when few targets are placed. The canvas takes the window's
    # aspect, because a fitted layout is rarely square.
    lo = np.minimum(
        (layout.centres - layout.radii[:, None]).min(axis=0), out_centre - out_radius
    )
    hi = np.maximum(
        (layout.centres + layout.radii[:, None]).max(axis=0), out_centre + out_radius
    )
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
    # The never-placed region, on the same area scale as every set but drawn
    # dashed: the area is to scale, the position is not, and the dash says so.
    # Skipped entirely at a zero share — there is no circle to draw, and a dot
    # would read as a very small one.
    if out_radius > 0:
        ax.add_patch(
            Circle(out_centre, out_radius, facecolor=_MUTED,
                   edgecolor="none", alpha=0.10, zorder=1)
        )
        ax.add_patch(
            Circle(out_centre, out_radius, facecolor="none", edgecolor=_MUTED,
                   linewidth=1.5, linestyle=(0, (6, 4)), zorder=2)
        )
    ax.annotate(
        f"{OUTSIDE_LABEL}\n{100 * float(layout.observed[0]):.1f}%",
        xy=tuple(out_centre),
        ha="center", va="center", zorder=6,
        fontsize=10.5, fontweight="bold", color=_MUTED,
        linespacing=1.35, path_effects=halo,
    )

    for i, (lx, ly) in enumerate(_euler_label_points(layout)):
        # `pi * r**2` is not a re-derivation of the share — by `circle_radii`
        # it *is* the circle's drawn area, so the printed number and the ink
        # cannot drift apart. Sets only: an intersection's drawn area is fitted
        # and would disagree with its observed share by up to the figure's
        # error, which is what the caption reports instead.
        ax.annotate(
            f"{method_label(order[i])}\n"
            f"{100 * math.pi * layout.radii[i] ** 2:.1f}%",
            xy=(lx, ly),
            ha="center", va="center", zorder=6,
            fontsize=10.5, fontweight="bold", color=colors[order[i]],
            linespacing=1.35, path_effects=halo,
        )

    lines = []
    if caption:
        lines.append(EULER_CAPTION)
        lines.append(
            f"Fitted layout: {100 * layout.placed:.1f}% of targets land in the "
            f"region drawn (largest pair error {100 * layout.pair_error():.1f}%) "
            f"— see {fit_ref}."
        )
    if empty_sets:
        names = ", ".join(method_label(m) for m in empty_sets)
        lines.append(
            f"No circle for {names}: nothing placed within this tolerance. An "
            f"empty set belongs to no region, so every other region is unchanged."
        )

    ax.set_title(title, fontsize=13, fontweight="bold", color=_INK, pad=16)
    if subtitle:
        ax.annotate(
            subtitle, xy=(0.5, 1.005), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=9.5, color=_INK_2,
        )
    if lines:
        ax.annotate(
            "\n".join(lines), xy=(0.5, -0.015), xycoords="axes fraction",
            ha="center", va="top", fontsize=8.5, color=_MUTED,
        )

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return out_path
