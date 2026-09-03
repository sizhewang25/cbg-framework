"""Set-overlap figures, grouped by the bargain each one makes.

`venn/` fixes the geometry and prints every number on it — the classic 2/3-set
Venn, the UpSet, and the `n`-way ring template. `euler/` makes the opposite
trade: it fits the geometry *to* the numbers and prints almost none of them.
Neither subsumes the other, which is why both ship.

`common/` is what they both need — the labels and palette that keep a variant
the same name and colour in every figure, the membership matrix every figure is
computed from, the exact-intersection tables they are all checked against, and
the handful of matplotlib primitives they share.

`modules/venn.py` is the `plot-venn` command that assembles these into an
artifact set, and is the one import surface the rest of the package uses.
"""
