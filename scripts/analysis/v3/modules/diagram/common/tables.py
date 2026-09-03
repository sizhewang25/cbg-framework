"""Exact intersection counts — the numbers behind every figure.

Written alongside the figures so the counts are checkable without reading one,
and used by the figures themselves: `intersection_table` is the human-readable
artifact, `exact_combination_counts` its method-id keyed twin that the ring
indexes regions by. Both use the same disjoint reading, so they cannot drift.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from scripts.analysis.v3.modules.diagram.common.labels import label_for


def intersection_table(membership: pd.DataFrame) -> pd.DataFrame:
    """Every non-empty exact intersection, largest first.

    "Exact" means the set of methods that got the target right is precisely this
    combination — so rows partition the target population and sum to n_targets.
    """
    methods = list(membership.columns)
    keys = membership.apply(
        lambda r: tuple(m for m in methods if r[m]), axis=1
    )
    grouped = keys.value_counts()
    rows = [
        {
            "methods": "|".join(label_for(m) for m in combo) if combo else "(none)",
            "n_methods": len(combo),
            "n_targets": int(n),
            "share": round(float(n) / len(membership), 4),
        }
        for combo, n in grouped.items()
    ]
    return pd.DataFrame(rows).sort_values(
        ["n_targets", "n_methods"], ascending=[False, True]
    ).reset_index(drop=True)


def exact_combination_counts(
    membership: pd.DataFrame, order: list[str]
) -> dict[frozenset[str], int]:
    """Method-id keyed twin of `intersection_table`, for indexing regions by.

    `intersection_table` keys on display *labels* and is the human-readable
    artifact; the figure needs to look regions up by method id, and the two
    must agree exactly — a test asserts every drawn label equals that table's
    count for the same combination.
    """
    keys = membership.apply(lambda r: frozenset(m for m in order if r[m]), axis=1)
    if keys.empty:
        return {}
    return {k: int(v) for k, v in keys.value_counts().items()}


def pairwise_table(membership: pd.DataFrame) -> pd.DataFrame:
    """Per method pair: won-only, lost-only, both, neither.

    `a_only` is the regression column when `a` is the baseline — targets
    Shortest-Ping solves and the variant does not.
    """
    rows = []
    for a, b in combinations(membership.columns, 2):
        sa, sb = membership[a], membership[b]
        rows.append(
            {
                "method_a": label_for(a),
                "method_b": label_for(b),
                "both": int((sa & sb).sum()),
                "a_only": int((sa & ~sb).sum()),
                "b_only": int((~sa & sb).sum()),
                "neither": int((~sa & ~sb).sum()),
                "jaccard": round(
                    float((sa & sb).sum() / max((sa | sb).sum(), 1)), 4
                ),
            }
        )
    return pd.DataFrame(rows)


#: Letters the generic Venn tools key their inputs on — `A ^ B`, `A ^ B ^ C`.
#: Method labels are far too long for that role ("Octant-Spline CBG"), so the
#: spec carries both: a letter as the key and the real label alongside it.
SET_IDS = "ABCDEFGH"


def venn_spec(membership: pd.DataFrame, *, exclusive: bool = False) -> dict:
    """The membership matrix as a generic Venn-tool input document.

    Mirrors the shape those generators ask for: a set count, one entry per set
    with a name and a size, and one entry per *combination* of two or more sets.
    Written as JSON so the numbers can be pasted into a drawing tool without
    re-deriving them from `overlap_membership.csv`.

    `relations` are **cumulative** by default — `A^B` is the full `|A ∩ B|`,
    which counts targets that are also in `C`. That is the set-theoretic
    reading, and the one those forms assume: it is what makes `size` and the
    relations satisfy inclusion-exclusion, so a tool can solve for the regions
    itself. `exclusive=True` switches to the disjoint reading instead — `A^B`
    becomes targets in `A` and `B` and **nothing else** — which is what
    `intersection_table` and the ring figure report. The two differ wherever a
    higher-order region is non-empty, so the convention is recorded in the
    document rather than left for a reader to infer.

    Combinations with no targets are included with a zero. A Venn tool needs a
    value for every relation it will draw, and omitting the empties would make
    a reader guess whether the region is empty or the number is missing.
    """
    methods = list(membership.columns)
    if len(methods) > len(SET_IDS):
        raise ValueError(
            f"venn spec covers at most {len(SET_IDS)} sets, got {len(methods)}"
        )
    ids = {m: SET_IDS[i] for i, m in enumerate(methods)}
    columns = {m: membership[m].to_numpy() for m in methods}

    relations = {}
    for k in range(2, len(methods) + 1):
        for combo in combinations(methods, k):
            mask = columns[combo[0]].copy()
            for m in combo[1:]:
                mask = mask & columns[m]
            if exclusive:
                for m in methods:
                    if m not in combo:
                        mask = mask & ~columns[m]
            relations[" ^ ".join(ids[m] for m in combo)] = int(mask.sum())

    return {
        "number_of_sets": len(methods),
        "n_targets": int(len(membership)),
        # Targets no method got right sit outside every circle, so no Venn
        # region holds them and the count would otherwise vanish.
        "n_none": int((~membership.any(axis=1)).sum()),
        "relation_convention": "exclusive" if exclusive else "cumulative",
        "sets": [
            {
                "id": ids[m],
                "name": label_for(m),
                "method": m,
                "size": int(membership[m].sum()),
            }
            for m in methods
        ],
        "relations": relations,
    }
