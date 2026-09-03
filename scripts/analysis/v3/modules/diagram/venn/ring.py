"""The `n`-way ring Venn: a fixed template with the numbers printed on it.

`n` equal circles on a ring is the conventional presentation "6-way Venn". It is
not a mathematical 6-set Venn — that needs 63 regions and cannot be drawn with
circles at all — so a fraction of the observed intersections have no region
here. `ring_coverage_table` and the figure's own footnote name which, by count
and by method, which is what makes the omission a disclosure rather than a
silent loss.

The geometry is a pure function of `n` and never of the data: same centres, same
radii, every run. That is the whole premise — a region's size and position carry
nothing, and every quantity on the figure is a label. `euler/` is the other half
of the pair, and neither is sufficient alone.
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import pandas as pd

from scripts.analysis.v3.modules.diagram.common.draw import (
    _annotate_figure,
    _draw_circles,
    _label_anchor,
    plt,
)
from scripts.analysis.v3.modules.diagram.common.labels import label_for, region_key
from scripts.analysis.v3.modules.diagram.common.palette import (
    _C_INK,
    _C_MUTED,
    _SURFACE,
    method_colors,
)
from scripts.analysis.v3.modules.diagram.common.tables import exact_combination_counts


#: Circle radius divided by the radius of the ring the centres sit on. Fixed at
#: the value that reproduces the reference figure: every circle passes close to
#: the ring's centre, so all `n` overlap in the middle and the region count is
#: maximal. The layout is stable for anything in 1.05–1.55 (the region set is
#: identical throughout); 1.2 is what the reference looks like.
RING_RATIO = 1.2

#: Circle radius in figure units. Arbitrary — the axes are scaled to the union —
#: but fixing it keeps `ring_centres` returning literal coordinates a test can
#: compare.
RING_CIRCLE_RADIUS = 1.0

#: Ring layouts are only legible in this range. Below 3 there is no ring (use
#: `plot_venn`); above 8 the smallest region falls under ~30 px at 200 dpi and
#: its label no longer fits inside it.
RING_MIN_SETS, RING_MAX_SETS = 3, 8


def ring_centres(n: int) -> list[tuple[float, float]]:
    """Circle centres for an `n`-set ring, first at 12 o'clock then clockwise.

    Pure function of `n` — **never of the data**. That is the whole premise of
    this figure: it is a template, so a region's position and size say nothing
    about its count and only the printed label does. Anything that made the
    geometry depend on the membership matrix would reintroduce the
    area-encoding this layout was chosen to avoid.
    """
    d = RING_CIRCLE_RADIUS / RING_RATIO
    out = []
    for i in range(n):
        angle = math.pi / 2 - 2 * math.pi * i / n
        out.append((d * math.cos(angle), d * math.sin(angle)))
    return out


@lru_cache(maxsize=8)
def ring_regions(n: int) -> dict[frozenset[int], object]:
    """Every non-empty region of the `n`-circle ring, keyed by its member set.

    A region is the intersection of its members minus the union of everything
    else, so the regions partition the union and each one means "exactly these
    methods, no others" — the same reading as `intersection_table`.

    There are exactly `n*(n-1)+1` of them, and they are precisely the subsets
    that are **contiguous around the ring** (plus the all-`n` centre): six
    circles yield 31, not the 63 a mathematical 6-set Venn needs. That gap is
    not a defect of this implementation — no six circles can realize 63 regions
    — and it is why `ring_coverage_table` exists.
    """
    from shapely.geometry import Point
    from shapely.ops import unary_union

    disks = [
        Point(c).buffer(RING_CIRCLE_RADIUS, quad_segs=256) for c in ring_centres(n)
    ]
    regions: dict[frozenset[int], object] = {}
    for k in range(1, n + 1):
        for combo in combinations(range(n), k):
            patch = disks[combo[0]]
            for i in combo[1:]:
                patch = patch.intersection(disks[i])
                if patch.is_empty:
                    break
            if patch.is_empty:
                continue
            others = [disks[i] for i in range(n) if i not in combo]
            if others:
                patch = patch.difference(unary_union(others))
            # Boolean ops on 256-segment circles leave slivers of ~1e-12 where
            # two arcs graze; the true regions are all >1e-3 of the union, so
            # this threshold separates them by nine orders of magnitude.
            if patch.is_empty or patch.area < 1e-9:
                continue
            regions[frozenset(combo)] = patch
    return regions


def ring_order_for(membership: pd.DataFrame, ring_order=None) -> list[str]:
    """Validate an explicit ring order, or take the frame's column order.

    An explicit order must be a **permutation** of the columns, not a subset: a
    method left off the ring would still be in the membership matrix, so its
    targets would fall into intersections the ring has no region for and vanish
    from the figure without appearing in the undrawn count either.
    """
    columns = list(membership.columns)
    if ring_order is None:
        order = columns
    else:
        order = list(ring_order)
        if sorted(order) != sorted(columns):
            raise ValueError(
                f"ring order must be a permutation of the scored methods; got "
                f"{order} against {columns}. Drop methods with --method, not by "
                f"omitting them from --ring-order."
            )
    if not RING_MIN_SETS <= len(order) <= RING_MAX_SETS:
        raise ValueError(
            f"the ring layout takes {RING_MIN_SETS}-{RING_MAX_SETS} methods, got "
            f"{len(order)}"
        )
    return order


def ring_coverage_table(
    membership: pd.DataFrame, ring_order=None
) -> pd.DataFrame:
    """Every observed intersection, with whether the ring can draw it.

    This is the audit trail for the figure's central omission. The ring realizes
    `n*(n-1)+1` of the `2**n - 1` possible combinations, so on real data some
    non-empty intersections have nowhere to go — around 11-18% of solved targets
    on as01+as02+as03, depending on the ring order. Without this table a reader
    would take the figure's labels for the whole population and be wrong.

    Targets no method got right are **not** listed: they sit outside the union
    rather than in an undrawable region, so counting them as undrawn would
    conflate "the layout cannot show this" with "there is nothing to show".
    """
    order = ring_order_for(membership, ring_order)
    drawable = {frozenset(order[i] for i in combo) for combo in ring_regions(len(order))}
    rows = []
    for combo, n in exact_combination_counts(membership, order).items():
        if not combo:
            continue
        members = [m for m in order if m in combo]
        rows.append(
            {
                "methods": "|".join(label_for(m) for m in members),
                # The same letters `euler_fit_table` keys its own `region`
                # column on, so a row can be traced across the two tables.
                "region": region_key(order, combo),
                "n_methods": len(combo),
                "n_targets": n,
                "share": round(float(n) / len(membership), 4),
                "drawn": combo in drawable,
            }
        )
    return (
        pd.DataFrame(
            rows,
            columns=["methods", "region", "n_methods", "n_targets", "share", "drawn"],
        )
        .sort_values(["drawn", "n_targets"], ascending=[False, False])
        .reset_index(drop=True)
    )


#: Printed on every ring figure. The reference layout looks like an
#: area-proportional diagram and is not one: the circles are identical in every
#: run, so a region being large means the template put it there, not that many
#: targets landed in it. Every quantity on this figure is a printed label.
RING_CAVEAT = (
    "Circle positions and sizes are a fixed template and carry no data — "
    "only the printed percentages do.\n"
    "Each number is the share of the target population solved by exactly that "
    "combination of methods."
)


def _region_font_size(area_share: float) -> float:
    """Font tier from the region's share of the union's *area*, not its count.

    Tiered rather than continuous so equally-sized regions get equal type: a
    smooth scale would make the six singles differ by a fraction of a point for
    no reason. The floor is 6.5 pt because below that the digits stop resolving
    at 200 dpi.
    """
    for threshold, size in ((0.03, 12.0), (0.01, 10.0), (0.003, 8.5)):
        if area_share >= threshold:
            return size
    return 6.5


def _draw_ring_outer_labels(
    ax, order: list[str], colors: dict[str, str], texts: dict[str, str]
) -> None:
    """Name each circle just outside its own arc, radially aligned.

    Both ring figures label the same circles in the same places; only the text
    differs (a success rate on the data figure, a letter on the key), so the
    caller supplies it per method.
    """
    n = len(order)
    for i, method in enumerate(order):
        angle = math.pi / 2 - 2 * math.pi * i / n
        reach = RING_CIRCLE_RADIUS / RING_RATIO + RING_CIRCLE_RADIUS + 0.06
        x, y = reach * math.cos(angle), reach * math.sin(angle)
        ax.annotate(
            texts[method],
            xy=(x, y),
            ha="center" if abs(x) < 0.2 else ("left" if x > 0 else "right"),
            va="center" if abs(y) < 0.2 else ("bottom" if y > 0 else "top"),
            fontsize=10, fontweight="bold", color=colors[method], zorder=5,
            linespacing=1.35,
        )


def _finish_ring_axes(ax, *, title: str, subtitle: str, note: str) -> None:
    """Square, unframed axes scaled to the ring template."""
    span = RING_CIRCLE_RADIUS / RING_RATIO + RING_CIRCLE_RADIUS
    ax.set_xlim(-span - 0.55, span + 0.55)
    ax.set_ylim(-span - 0.5, span + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    _annotate_figure(ax, title=title, subtitle=subtitle, note=note)


def plot_ring_venn(
    membership: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    subtitle: str = "",
    ring_order=None,
    coverage_ref: str = "overlap_ring_coverage.csv",
) -> Path:
    """The `n`-way ring Venn: fixed circles, percentages written into the regions.

    Six equal circles on a ring is the conventional presentation "6-way Venn".
    It is not a mathematical 6-set Venn — that needs 63 regions and cannot be
    drawn with circles at all — so a fraction of the observed intersections have
    no region here. Both the footnote and `ring_coverage_table` say which, by
    name and by count; that disclosure is what makes the figure honest rather
    than optional polish.

    The geometry never changes: same centres, same radii, every run. Only the
    labels move. That is deliberate — three earlier forms of this figure tried
    to encode counts in area and each one either lost the higher-order regions
    or implied a containment the sets do not have.

    At six methods the middle of the ring is thirteen regions inside a circle's
    radius, so a percentage there cannot also carry the names of the methods it
    belongs to. `plot_euler` is the other half: the same six sets with the
    geometry fitted to the intersections and no number printed anywhere.
    """
    from matplotlib.patheffects import withStroke
    from shapely.ops import unary_union

    order = ring_order_for(membership, ring_order)
    n = len(order)
    total = len(membership)
    if total == 0:
        raise ValueError("no targets to draw")

    regions = ring_regions(n)
    union_area = unary_union(list(regions.values())).area
    counts = exact_combination_counts(membership, order)
    colors = method_colors(order)

    drawable = {frozenset(order[i] for i in combo) for combo in regions}
    n_solved = total - counts.get(frozenset(), 0)
    undrawn = {
        combo: c for combo, c in counts.items() if combo and combo not in drawable
    }
    n_undrawn = sum(undrawn.values())

    fig, ax = plt.subplots(figsize=(9.0, 9.0))
    _draw_circles(
        ax, order, colors, ring_centres(n), [RING_CIRCLE_RADIUS] * n
    )

    halo = [withStroke(linewidth=2.4, foreground=_SURFACE)]
    for combo, patch in regions.items():
        key = frozenset(order[i] for i in combo)
        n_here = counts.get(key, 0)
        pct_here = 100.0 * n_here / total if total else 0.0
        point = _label_anchor(patch)
        ax.annotate(
            f"{pct_here:.1f}%",
            xy=(point.x, point.y),
            ha="center", va="center", zorder=4,
            fontsize=_region_font_size(patch.area / union_area),
            # An empty region is drawn, not omitted: "0.0%" is a finding, and a
            # blank space would read as a region the figure forgot.
            color=_C_INK if n_here else _C_MUTED,
            fontweight="bold" if n_here else "normal",
            path_effects=halo,
        )

    _draw_ring_outer_labels(
        ax,
        order,
        colors,
        {
            m: f"{label_for(m)}\n"
               f"{100.0 * int(membership[m].sum()) / total if total else 0.0:.1f}%"
            for m in order
        },
    )

    note = RING_CAVEAT
    if undrawn:
        pct = 100.0 * n_undrawn / n_solved if n_solved else 0.0
        note += (
            f"\n{len(undrawn)} observed combination(s) — {n_undrawn} targets, "
            f"{pct:.1f}% of the {n_solved} solved — have no region in this "
            f"layout; see {coverage_ref}."
        )
    _finish_ring_axes(ax, title=title, subtitle=subtitle, note=note)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return out_path
