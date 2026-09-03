"""The 2- and 3-set Venn, and the Shortest-Ping vs ">=1 CBG" collapse.

The only layouts here that matplotlib-venn can draw exactly. Past three sets the
region count outruns what circles can realize, which is where `ring` and `upset`
take over.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analysis.v3.modules.classify import SHORTEST_PING
from scripts.analysis.v3.modules.diagram.common.draw import plt


#: Display label for the collapsed "at least one CBG variant is correct" set.
CBG_ANY_LABEL = "≥1 CBG"

#: Display label for the Shortest-Ping side of that same collapse. Deliberately
#: not `label_for(SHORTEST_PING)` ("Shortest-Ping"): this pairing is a figure of
#: its own, and reads better without the hyphen alongside the terse "≥1 CBG".
SP_VS_CBG_LABEL = "Shortest Ping"


def _region_labeller(total: int):
    """Format each Venn region as `n` over `(pct%)` of the target population."""

    def fmt(value) -> str:
        n = int(value if not hasattr(value, "__len__") else len(value))
        pct = 100.0 * n / total if total else 0.0
        return f"{n}\n({pct:.1f}%)"

    return fmt


def plot_venn(
    membership: pd.DataFrame, out_path: Path, *, title: str, weighted: bool = False
) -> Path:
    """Classic 2- or 3-set Venn. Requires 2 or 3 methods.

    Column names are taken as the display labels verbatim — the caller must
    already have named them (`label_for`, or a hand-picked string like the
    SP-vs-CBG collapse's "Shortest Ping"). Applying `label_for` in here too
    would double up on any column that is already a display name rather than
    a raw method id, which is exactly how this used to render "≥1 CBG CBG".

    Unweighted (fixed-size, evenly positioned circles) by default rather than
    area-proportional, because the correctness sets are routinely nested —
    `all-CBG ⊆ Shortest-Ping ⊆ ≥1-CBG` holds on every run we have — and a
    3-circle area-proportional layout cannot render a strict 3-way chain: at
    least one region's exact-zero area is unsatisfiable by any real circle
    configuration, so the solver distorts the whole figure to approximate it.
    The counts carry the information instead of the areas.

    `weighted=True` switches to the real area-proportional layout — circle and
    overlap size driven by the actual subset sizes — so a near-zero region
    reads as near-total inclusion instead of an evenly-sized sliver. Only
    supported at 2 sets: the chain-nesting failure above is specifically a
    3-circle problem, and every current caller of `weighted=True` is the 2-set
    Shortest-Ping-vs-CBG collapse.

    Set labels are drawn as a legend rather than matplotlib-venn's default
    text beside each circle: at near-total overlap (the `weighted=True` case
    this exists for) the smaller circle's own label sits inside the bigger
    circle's territory, which reads as if it belongs to the wrong set.
    """
    # `venn2_unweighted` / `venn3_unweighted` are deprecated in matplotlib-venn
    # 1.1.2 *and* broken — they forward `normalize_to` into a custom layout that
    # rejects it. Drive the layout algorithm directly instead.
    from matplotlib.patches import Patch
    from matplotlib_venn import venn2, venn3
    from matplotlib_venn.layout.venn2 import DefaultLayoutAlgorithm as Venn2Layout
    from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm as Venn3Layout

    methods = list(membership.columns)
    if len(methods) not in (2, 3):
        raise ValueError(f"Venn needs 2 or 3 methods, got {len(methods)}")
    if weighted and len(methods) != 2:
        raise ValueError(
            f"weighted=True is only supported at 2 sets (got {len(methods)}); "
            f"a 3-circle area-proportional layout cannot render the chain "
            f"nesting these sets routinely have"
        )
    sets = [set(membership.index[membership[m]]) for m in methods]
    labels = tuple(str(m) for m in methods)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    if len(methods) == 2:
        draw = venn2
        layout = Venn2Layout() if weighted else Venn2Layout(fixed_subset_sizes=(1, 1, 1))
        region_ids = ("10", "01")
    else:
        draw, layout = venn3, Venn3Layout(fixed_subset_sizes=(1,) * 7)
        region_ids = ("100", "010", "001")
    v = draw(
        sets,
        # No text beside the circles — the legend below carries the names.
        set_labels=None,
        ax=ax,
        layout_algorithm=layout,
        subset_label_formatter=_region_labeller(len(membership)),
    )
    # Swatch colour comes from the rendered "this set only" patch rather than
    # a second, independent colour choice, so the legend can never disagree
    # with what is actually on the canvas. That region is occasionally empty
    # (no patch drawn), hence the grey fallback.
    handles = [
        Patch(
            facecolor=(
                v.get_patch_by_id(rid).get_facecolor()
                if v.get_patch_by_id(rid) is not None
                else "#999999"
            ),
            edgecolor="none",
            label=lab,
        )
        for rid, lab in zip(region_ids, labels)
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=len(labels),
        frameon=False,
        fontsize=10,
    )

    # The "no method correct" region falls outside every circle, so a Venn
    # cannot show it. It is part of the denominator and often large, so state
    # it rather than leaving the reader to subtract.
    n_total = len(membership)
    n_none = int((~membership.any(axis=1)).sum())
    pct_none = 100.0 * n_none / n_total if n_total else 0.0
    ax.annotate(
        f"All Failed: {n_none} ({pct_none:.1f}%)",
        xy=(0.5, -0.14),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
    )

    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def collapse_to_sp_vs_cbg(membership: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Collapse per-method membership to `Shortest-Ping` vs `≥1 CBG`.

    Returns the 2-column frame and the size of the CBG pool that was collapsed.
    The pool size matters for interpretation and must reach the figure: on
    `as7018_us_test01` the default method set is 16 combos including 11 ablation
    arms, and the CBG-only region reads 40 over all of them versus 31 over the
    five published variants.
    """
    if SHORTEST_PING not in membership.columns:
        raise ValueError(
            f"membership has no {SHORTEST_PING!r} column; the Shortest-Ping "
            f"baseline is required for the SP-vs-CBG view. Got: "
            f"{list(membership.columns)}"
        )
    cbg = membership.drop(columns=[SHORTEST_PING])
    if cbg.shape[1] == 0:
        raise ValueError("no CBG methods to collapse; need at least one besides the baseline")
    collapsed = pd.DataFrame(
        {
            SP_VS_CBG_LABEL: membership[SHORTEST_PING],
            CBG_ANY_LABEL: cbg.any(axis=1),
        },
        index=membership.index,
    )
    return collapsed, cbg.shape[1]


def plot_sp_vs_cbg_venn(
    membership: pd.DataFrame, out_path: Path, *, title: str
) -> Path:
    """Area-proportional 2-set Venn: Shortest-Ping vs "at least one CBG works".

    This is the rescue-vs-regression trade an aggregate accuracy number hides.
    The CBG-only region counts targets CBG rescues; the Shortest-Ping-only
    region counts the ones it regresses on. Weighted rather than fixed-size —
    unlike the 3-way chain `plot_venn`'s docstring warns about, this is exactly
    2 sets, so a near-zero regression region can render as near-total
    inclusion instead of an evenly-sized sliver that overstates it.
    """
    collapsed, n_cbg = collapse_to_sp_vs_cbg(membership)
    subtitle = f"≥1 of {n_cbg} CBG variant{'s' if n_cbg != 1 else ''}"
    return plot_venn(collapsed, out_path, title=f"{title}\n{subtitle}", weighted=True)
