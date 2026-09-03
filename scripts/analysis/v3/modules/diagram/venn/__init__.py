"""Fixed-geometry overlap figures: the number is always a printed label.

Three layouts, in ascending arity. `classic` draws the 2- or 3-set Venn
matplotlib-venn can lay out exactly. `ring` draws the conventional presentation
"`n`-way Venn" — equal circles on a ring, identical in every run — for 3 to 8
sets, and reports which observed intersections it has no region for. `upset` is
the readable form once even that stops working.

What they share is the bargain: the geometry is a template, so a region being
large means the template put it there. Every quantity is a label. `euler/` is
the opposite trade.
"""

from scripts.analysis.v3.modules.diagram.venn.classic import (
    CBG_ANY_LABEL,
    SP_VS_CBG_LABEL,
    collapse_to_sp_vs_cbg,
    plot_sp_vs_cbg_venn,
    plot_venn,
)
from scripts.analysis.v3.modules.diagram.venn.ring import (
    RING_CAVEAT,
    RING_CIRCLE_RADIUS,
    RING_MAX_SETS,
    RING_MIN_SETS,
    RING_RATIO,
    plot_ring_venn,
    ring_centres,
    ring_coverage_table,
    ring_order_for,
    ring_regions,
)
from scripts.analysis.v3.modules.diagram.venn.upset import (
    COLUMN_METRIC_LABEL,
    ROW_METRIC_LABEL,
    plot_upset,
)

__all__ = [
    "CBG_ANY_LABEL",
    "COLUMN_METRIC_LABEL",
    "RING_CAVEAT",
    "RING_CIRCLE_RADIUS",
    "RING_MAX_SETS",
    "RING_MIN_SETS",
    "RING_RATIO",
    "ROW_METRIC_LABEL",
    "SP_VS_CBG_LABEL",
    "collapse_to_sp_vs_cbg",
    "plot_ring_venn",
    "plot_sp_vs_cbg_venn",
    "plot_upset",
    "plot_venn",
    "ring_centres",
    "ring_coverage_table",
    "ring_order_for",
    "ring_regions",
]
