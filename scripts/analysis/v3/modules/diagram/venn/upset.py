"""The UpSet plot — the readable form once arity exceeds three.

Both of UpSet's bar charts are suppressed and their magnitudes written as text,
because the two number sets are different kinds of quantity and a shared axis
would imply otherwise. `plot_upset` documents which is which.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analysis.v3.modules.diagram.common.draw import plt
from scripts.analysis.v3.modules.diagram.common.labels import label_for


#: Header for the per-column numbers (exact disjoint intersection shares).
COLUMN_METRIC_LABEL = "Intersections (%)"

#: Header for the per-row numbers (per-method success rate). Deliberately a
#: different word from the column header: these do not sum to 100%.
ROW_METRIC_LABEL = "True (%)"


def plot_upset(
    membership: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    min_subset_size: int = 1,
) -> Path:
    """UpSet plot — the readable form once arity exceeds three.

    Both bar charts are suppressed and their magnitudes written as text
    instead: shares above each column, per-method success rate to the right of
    each row. The two number sets are *not* the same kind of quantity, which is
    why each carries its own header —

    * columns are **exact, disjoint** intersections ("these methods correct,
      all others wrong"), so they partition the targets and sum to 100%;
    * rows are set totals, which overlap and therefore do not.

    Category order is pinned to the caller's column order
    (`sort_categories_by=None`) so the dot pattern means the same thing in
    every run's figure. Columns are ranked by intersection size, largest first,
    so the numbers read monotonically left to right. That ordering is
    data-dependent — the same dot combination sits at a different x in another
    run — so compare columns by their dots, never by position.
    """
    from upsetplot import UpSet, from_indicators

    renamed = membership.rename(columns={m: label_for(m) for m in membership.columns})
    total = len(renamed)
    # upsetplot draws the first category at the *bottom*, so hand it the
    # reversed order to get the caller's order reading top-to-bottom.
    data = from_indicators(list(renamed.columns)[::-1], renamed)

    upset = UpSet(
        data,
        subset_size="count",
        show_counts=False,
        sort_by="cardinality",
        sort_categories_by=None,
        min_subset_size=min_subset_size,
        # Both bars off; the numbers are annotated onto the matrix below.
        intersection_plot_elements=0,
        totals_plot_elements=0,
    )

    n_cols = len(upset.intersections)
    fig = plt.figure(figsize=(max(7.0, 0.62 * n_cols + 4.2), 0.46 * len(renamed.columns) + 2.6))
    axes = upset.plot(fig=fig)
    ax = axes["matrix"]

    # `intersections` is in plotted order, so position i sits at x == i.
    shares = 100.0 * upset.intersections.to_numpy() / total if total else []
    y_top = len(renamed.columns) - 0.5
    for i, pct in enumerate(shares):
        ax.text(
            i, y_top + 0.30, f"{pct:.1f}", ha="center", va="bottom",
            fontsize=7.5, rotation=90, clip_on=False,
        )
    ax.text(
        -0.9, y_top + 0.30, COLUMN_METRIC_LABEL, ha="right", va="bottom",
        fontsize=8, fontweight="bold", clip_on=False,
    )

    # `totals` is indexed by category in the same order as the y ticks.
    x_right = n_cols - 0.4
    ticks = ax.get_yticks()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    for y, lab in zip(ticks, labels):
        # A method with zero correct targets is dropped from `totals`, so read
        # it defensively rather than KeyError-ing on the worst-performing arm.
        pct = 100.0 * float(upset.totals.get(lab, 0)) / total if total else 0.0
        ax.text(
            x_right, y, f"{pct:.1f}", ha="left", va="center",
            fontsize=8, clip_on=False,
        )
    ax.text(
        x_right, y_top + 0.30, ROW_METRIC_LABEL, ha="left", va="bottom",
        fontsize=8, fontweight="bold", clip_on=False,
    )

    ax.set_ylim(-0.6, y_top + 0.2)
    # Title on the matrix axis, not the figure: the rotated column numbers sit
    # above the axis, so a figure-level suptitle lands on top of them. `pad` is
    # in points and must clear the tallest number plus its header.
    ax.set_title(title, fontsize=11, pad=44)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
