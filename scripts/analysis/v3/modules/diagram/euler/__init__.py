"""Area-proportional Euler layout: the geometry carries the numbers.

The opposite bargain to `venn/ring.py`. Circle areas *are* the set sizes,
centres are fitted to the observed intersections, and nothing is printed inside
a region — a pair that never agrees is drawn apart, a set contained in another
is drawn inside it, and the reader gets the relationships by looking.

What no fit can buy is exactness: `n` circles have 2n degrees of freedom against
`2**n - 1` regions, so past two sets the system is overdetermined and some
combination is always misdrawn. `layout` measures exactly how much and `plot`
prints it, so the figure states its own error rather than implying it has none.
"""

from scripts.analysis.v3.modules.diagram.euler.layout import (
    DISJOINT_MARGIN,
    EULER_RESTARTS,
    EulerLayout,
    circle_radii,
    combination_shares,
    euler_fit_table,
    fit_euler_layout,
    lens_area,
    separation_for_overlap,
    set_shares,
)
from scripts.analysis.v3.modules.diagram.euler.plot import (
    EULER_CAPTION,
    LABEL_MIN_GAP,
    OUTSIDE_LABEL,
    plot_euler,
)

__all__ = [
    "DISJOINT_MARGIN",
    "EULER_CAPTION",
    "EULER_RESTARTS",
    "EulerLayout",
    "LABEL_MIN_GAP",
    "OUTSIDE_LABEL",
    "circle_radii",
    "combination_shares",
    "euler_fit_table",
    "fit_euler_layout",
    "lens_area",
    "plot_euler",
    "separation_for_overlap",
    "set_shares",
]
