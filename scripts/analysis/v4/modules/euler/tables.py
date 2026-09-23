"""Exact counts behind the figure — the numbers the circles cannot print.

An area-proportional Euler diagram states relationships and quantifies almost
nothing: circle area is a set's size, but no region carries a label, so the
only way to check what the picture claims is to read the counts beside it.
These are those counts.

`intersection_table` is the human-readable artifact and keys on display labels.
`region_key` gives every combination the same letter key the fit table uses, so
a row in one table can be matched to a row in the other.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from scripts.analysis.v4.modules.methods import method_label

#: A letter per set, assigned by position in the drawing order, so an
#: intersection can be named in the width a table column allows. Eight is the
#: cap: past that the layout has more regions than a reader can trace.
SET_LETTERS = "ABCDEFGH"


def letter_map(order: list[str]) -> dict[str, str]:
    """Method id -> its single letter, assigned by **position** in `order`."""
    if len(order) > len(SET_LETTERS):
        raise ValueError(
            f"an Euler layout takes at most {len(SET_LETTERS)} sets, got "
            f"{len(order)}"
        )
    return {method: SET_LETTERS[i] for i, method in enumerate(order)}


def region_key(order: list[str], members) -> str:
    """`{vanilla_cbg, shortest_ping}` -> `"AC"` — the members' letters, in order.

    Concatenated rather than joined with a separator: these keys appear in
    narrow table columns and the convention is stated once in the manifest
    rather than paid for on every row.
    """
    letters = letter_map(order)
    return "".join(letters[m] for m in order if m in members)


def intersection_table(membership: pd.DataFrame) -> pd.DataFrame:
    """Every non-empty **exact** intersection, largest first.

    "Exact" means the set of methods correct on that target is precisely this
    combination, so the rows partition the population and sum to `n_targets` —
    including the `(none)` row, which is the share outside every circle.
    """
    methods = list(membership.columns)
    keys = membership.apply(lambda r: tuple(m for m in methods if r[m]), axis=1)
    rows = [
        {
            "methods": "|".join(method_label(m) for m in combo) if combo else "(none)",
            "region": region_key(methods, set(combo)) if combo else "(none)",
            "n_methods": len(combo),
            "n_targets": int(n),
            "share": round(float(n) / len(membership), 4),
        }
        for combo, n in keys.value_counts().items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values(["n_targets", "n_methods"], ascending=[False, True])
        .reset_index(drop=True)
    )


def pairwise_table(membership: pd.DataFrame) -> pd.DataFrame:
    """Per method pair: both, one only, the other only, neither.

    `a_only` is the **regression** column when `a` is the baseline — targets
    Shortest-Ping places within tolerance and the variant does not. That trade
    is the thing an aggregate accuracy number cannot show and the reason this
    figure exists.
    """
    rows = []
    for a, b in combinations(membership.columns, 2):
        sa, sb = membership[a], membership[b]
        rows.append(
            {
                "method_a": method_label(a),
                "method_b": method_label(b),
                "both": int((sa & sb).sum()),
                "a_only": int((sa & ~sb).sum()),
                "b_only": int((~sa & sb).sum()),
                "neither": int((~sa & ~sb).sum()),
                "jaccard": round(float((sa & sb).sum() / max((sa | sb).sum(), 1)), 4),
            }
        )
    return pd.DataFrame(rows)
