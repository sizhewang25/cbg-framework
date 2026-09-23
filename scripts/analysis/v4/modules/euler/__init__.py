"""Area-proportional Euler layout over v4's ring-graded classifications.

Three layers, each a pure function of the one below:

* `membership` — the boolean matrix. This is where `top_n` means a **ring**
  (`top_n=1` is `ring == 0`, the truth's own cell) rather than v3's seed rank.
* `layout` — the geometry fit. Knows nothing about HEALPix; it sees a boolean
  matrix and places circles whose areas are the set sizes.
* `plot` — the drawing. Names on the circles, numbers nowhere else.

`tables` holds the exact counts the figure deliberately does not print.
`figure_euler` (one level up) assembles them into the `_cross` artifact set.
"""

from scripts.analysis.v4.modules.euler.layout import (
    EulerLayout,
    circle_radii,
    combination_shares,
    euler_fit_table,
    fit_euler_layout,
    set_shares,
)
from scripts.analysis.v4.modules.euler.membership import (
    MAX_TOP_N,
    TOLERANCE_LABELS,
    available_methods,
    build_membership,
    correct_at,
    pooled_membership,
    validate_top_n,
)
from scripts.analysis.v4.modules.euler.plot import plot_euler
from scripts.analysis.v4.modules.euler.tables import (
    intersection_table,
    letter_map,
    pairwise_table,
    region_key,
)

__all__ = [
    "MAX_TOP_N",
    "TOLERANCE_LABELS",
    "EulerLayout",
    "available_methods",
    "build_membership",
    "circle_radii",
    "combination_shares",
    "correct_at",
    "euler_fit_table",
    "fit_euler_layout",
    "intersection_table",
    "letter_map",
    "pairwise_table",
    "plot_euler",
    "pooled_membership",
    "region_key",
    "set_shares",
    "validate_top_n",
]
