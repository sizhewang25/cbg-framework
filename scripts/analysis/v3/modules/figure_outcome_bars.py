"""§8.1's outcome bars: what each method did with every target, by dataset type.

The headline table states two numbers per cell — top-N accuracy and, where it is
non-zero, the fallback rate. This figure draws the same two, plus the third that
completes them: each bar is one method's entire target set partitioned into
**correct · wrong · fallback · error**, so the accuracy is the bottom segment,
the failure rate is the top one, and their complement is visible rather than
implied.

Four segments rather than three. `n_error` is zero on every run today, but
`n_targets == n_solved + n_fallback + n_error` is the identity that makes the
stack a partition, and folding a run failure into "wrong" would read as a
scoring miss instead of a crash. The identity is asserted per bar.

## Colour is the outcome here, not the method

The one place in this paper where hue does **not** mean the variant. Green
right, red wrong, grey never answered; the method is the x position and its
label. That is a deliberate departure, made because the question this figure
answers is "what happened to the targets", and a reader who has to hold
"green = Octant-Hull" while reading a green segment that means "correct" is
doing the figure's work for it. Nothing in the bars is variant-hued, so inside
this figure there is no second meaning to collide with, and the six variant
hues stay untouched everywhere else.

Hatch is the **dataset type** — mesh solid, traffic-weighted hatched — the
comparison §8.1 hands off to. That leaves the constraint that **no segment may
carry a texture of its own**, so the outcome scheme has to do its separating in
colour and lightness alone, which is what its three checks are for (see
`SEGMENT_INK`).

## The weighted half is drawn empty on purpose

No run carries traffic weights (`has_weight: false` on all four), so every
hatched slot is a dashed outline with no fill. Leaving it off the page entirely
would make six solid bars look like the whole comparison; drawing it as a zero
would say the weighted campaign scored nothing. A ghost says what it is — the
same stance as the table's `—` rows.

## Arithmetic comes from the table, not from beside it

Counts, pooling and the micro-average are `headline_table`'s, imported rather
than reimplemented, so the figure's CSV twin and the table's long CSV are the
same numbers by construction. A second implementation would be free to drift by
a rounding rule and nobody would notice until the paper printed both.

Command: `plot-outcome-bars`. Writes to
`outputs/analysis/v3/_cross/accuracy-table/<dataset-set>/`, beside the table it
draws.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules import headline_table as H
from scripts.analysis.v3.modules.accuracy_table import accuracy_rows
from scripts.analysis.v3.modules.cross import cross_dir
from scripts.analysis.v3.modules.diagram.common.draw import plt
from scripts.analysis.v3.modules.diagram.common.labels import (
    PUBLISHED_METHODS,
    short_label,
)
from scripts.analysis.v3.modules.diagram.common.palette import (
    _C_AXIS,
    _C_GRID,
    _C_INK,
    _C_INK_2,
    _C_MUTED,
    _SURFACE,
)
from scripts.analysis.v3.modules.grid import (
    DEFAULT_GRID,
    GRID_HELP,
    RESOLUTION_HELP,
    resolve_cli_grid,
)
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    RunPaths,
    grid_slug,
    resolve_run,
)

POOLED = "pooled"
COMPARE = "compare"
LAYOUTS: tuple[str, ...] = (POOLED, COMPARE)

#: The counts carried in the CSV. `n_fallback` and `n_error` are kept apart here
#: even though the figure draws them as one bar segment, so the split stays
#: recoverable from the artifact.
COUNTS: tuple[str, ...] = ("n_correct", "n_wrong", "n_fallback", "n_error")

#: Bottom to top, and the reading order of the claim: how much was right, how
#: much was wrong, how much never produced an answer. `n_failed` is
#: `n_fallback + n_error` — a give-up and a crash are both failures to answer,
#: which is why they merge here and must never merge into `n_wrong`, where they
#: would read as a scoring miss instead.
SEGMENTS: tuple[str, ...] = ("n_correct", "n_wrong", "n_failed")
SEGMENT_LABELS = {"n_correct": "correct", "n_wrong": "wrong", "n_failed": "failed"}

#: Outcome, not identity: green right, red wrong, grey never answered.
#:
#: This spends the colour channel on the *result* rather than on the method, so
#: the six variant hues do not appear in the bars at all — the method is the x
#: position, and its tick label keeps its hue so the mapping is still on the
#: page. Green and red are Octant-Hull and Spotter elsewhere in §8.1; inside a
#: figure that hue-codes no method there is nothing for them to be confused
#: with, and the legend and the per-segment labels name every one.
#:
#: Chosen against three checks on `_SURFACE` white rather than by eye.
#: Lightness is **monotone** up the stack (L* 34 / 54 / 73, min step 19), so the
#: order survives greyscale and is the separator colour vision cannot take away.
#: Worst pairwise dE under simulated deuteranopia and protanopia is 16.8, above
#: the 8 that would oblige a secondary encoding — the red/green pair is legible
#: here on its own, and the labels are belt and braces. And the nearest of the
#: six variant hues is dE 11.8 away, so none of these reads as a variant.
SEGMENT_INK = {
    "n_correct": "#0d5c33",
    "n_wrong": "#d65a4e",
    "n_failed": "#b5b4ad",
}

#: Mesh is solid; the weighted campaign is hatched. A diagonal rather than a
#: cross-hatch: at these bar widths a cross reads as a fill and loses the
#: distinction it exists to make.
#:
#: Sparse (`//`, not `///`) and drawn in a half-transparent white at 0.6 pt, so
#: the hatch is a veil over the fill rather than a second pattern competing with
#: it. Dense opaque hatching lightened the whole bar enough that a hatched green
#: read as a *different colour* from a solid green — which is the one thing this
#: encoding may not do, since colour is the outcome.
KIND_HATCH = {H.MESH: "", H.WEIGHTED: "//"}
HATCH_INK = (1.0, 1.0, 1.0, 0.5)
HATCH_LINEWIDTH = 0.6

# Hatch line weight is an rcParam, not a Patch property — matplotlib exposes no
# per-artist `hatch_linewidth` — so it is set here, beside the pattern it
# belongs to, rather than left at the 1.0 default that made `//` read as `///`.
plt.rcParams["hatch.linewidth"] = HATCH_LINEWIDTH

#: A segment thinner than this cannot hold `100.0%` at 6.5 pt, so its label is
#: dropped rather than drawn over its neighbours. Nothing on as01/02/03 comes
#: close — the smallest drawn segment is 18.2% — so this is a guard, not a
#: routine case, and the share is in the CSV either way.
MIN_LABEL_SHARE = 0.045

#: Figure fraction the legend line sits on, and the room reserved above the
#: axes for it. Both legends sit between the title and the plotting area, so the
#: axes have to be pushed down by hand — `tight_layout` lays out axes and knows
#: nothing about a figure-level legend.
LEGEND_Y = 0.925
AXES_TOP = 0.885


#: The comparison grid needs a wider band: its panels carry their own titles
#: (`AS01 · n=399`), which sit above the axes and would otherwise share a line
#: with the legend.
COMPARE_LEGEND_Y = 0.945
COMPARE_AXES_TOP = 0.845

#: Gaps in figure fractions: between a title and its own entries, and between
#: the two groups. The second is the larger of the two, which is what makes the
#: line read as two keys rather than as five entries with two stray words in it.
TITLE_PAD = 0.008
GROUP_GAP = 0.045

DPI = 200
BAR_WIDTH = 0.40
PANEL_WIDTH_INCHES = 9.2
PANEL_HEIGHT_INCHES = 5.0


def method_order(table: pd.DataFrame, methods: list[str]) -> list[str]:
    """Methods by pooled correct rate, best first.

    Ranked on the **mesh** rows pooled over every dataset in the table, so the
    order is one property of the figure rather than of whichever panel is being
    drawn. In `compare` that is what keeps an x position meaning the same method
    in all three panels — ranking each panel on its own numbers would put
    Octant-Hull in a different column per dataset and turn a straight-line
    comparison into a search.

    Ties fall back to `PUBLISHED_METHODS` order, so the layout is deterministic
    when two methods are genuinely level.
    """
    mesh = table[(table["kind"] == H.MESH) & (~table["pending"])]
    fallback_rank = {m: i for i, m in enumerate(methods)}
    rates: dict[str, float] = {}
    for method in methods:
        rows = mesh[mesh["method"] == method]
        n = float(rows["n_targets"].sum())
        rates[method] = float(rows["n_correct"].sum()) / n if n else -1.0
    return sorted(methods, key=lambda m: (-rates[m], fallback_rank[m]))


def label_ink(fill: str) -> str:
    """Whichever of white or ink reads better on this fill.

    The stack runs from a dark green to a light grey, so one fixed label colour
    fails at one end whichever end it is chosen for. Picking per fill by
    relative luminance is what lets the fills be chosen for their own separation
    instead of for their ability to host white text.
    """
    rgb = [int(fill.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    luminance = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    on_white = 1.05 / (luminance + 0.05)
    on_ink = (luminance + 0.05) / 0.05
    return _SURFACE if on_white >= on_ink else _C_INK


def outcome_table(
    accuracy: pd.DataFrame, plan: list[dict], *, top_n: int, methods: list[str]
) -> pd.DataFrame:
    """One row per (kind, method): the four counts and their shares.

    Built from `headline_table`'s own row plan and pooling so the bars cannot
    disagree with the table beside them, and restricted to the aggregate rows —
    a bar per dataset type is what the hatch encodes, and the per-dataset view
    is `--layout compare`.
    """
    by_run = {run_id: g.set_index("method") for run_id, g in accuracy.groupby("run_id")}
    rows: list[dict] = []
    for entry in plan:
        if entry["scope"] != H.AGGREGATE:
            continue
        sources = [by_run.get(r) for r in entry["run_ids"]]
        placeholder = H.provisional_for(
            entry["kind"], entry["scope"], top_n, len(entry["run_ids"])
        )
        for method in methods:
            pooled = H.pool_method_counts(
                [H.method_counts(src, method, top_n=top_n) for src in sources]
            )
            counts = pooled or {}
            n = int(counts.get("n_targets", 0))
            row = {
                "kind": entry["kind"],
                "kind_label": H.KIND_LABELS[entry["kind"]],
                "row_label": entry["label"],
                "datasets": ";".join(entry["datasets"]),
                "n_runs": len(entry["run_ids"]),
                "method": method,
                "method_label": short_label(method),
                "top_n": top_n,
                "n_targets": n,
                "provisional": placeholder is not None and method in placeholder,
                "pending": (pooled is None or n == 0)
                and not (placeholder is not None and method in placeholder),
            }
            rows.append(
                _with_shares(
                    row, counts, n, provisional=(placeholder or {}).get(method)
                )
            )
    table = pd.DataFrame(rows)
    guard_partition(table)
    return table


def _with_shares(row: dict, counts: dict, n: int, *, provisional: dict | None = None) -> dict:
    """Attach the four counts, the three drawn segments, and their shares.

    A provisional row carries **shares only**: the placeholder in
    `headline_table` is rates from an earlier run with no denominator, so its
    counts stay NaN rather than being back-computed against an `n` nobody
    supplied. The bars draw from shares either way, so the figure needs nothing
    that does not exist.
    """
    if provisional is not None:
        for key in COUNTS + ("n_failed",):
            row[key] = np.nan
        correct = float(provisional["accuracy"])
        failed = float(provisional.get("fallback_rate", 0.0))
        row["share_correct"] = correct
        row["share_failed"] = failed
        row["share_wrong"] = 1.0 - correct - failed
        return row
    for key in COUNTS:
        row[key] = int(counts.get(key, 0)) if counts else 0
    row["n_failed"] = row["n_fallback"] + row["n_error"]
    for key in SEGMENTS:
        row[f"share_{SEGMENT_LABELS[key]}"] = row[key] / n if n else np.nan
    return row


def guard_partition(table: pd.DataFrame) -> None:
    """The four segments must account for every target, exactly once.

    A stacked bar whose segments do not sum to the whole is a chart that looks
    like a composition and is not one, and the failure is silent — the bar just
    ends early and reads as a shorter total. `topn_accuracy.csv` guarantees
    `n_targets == n_solved + n_fallback + n_error` and the correct/wrong split
    partitions `n_solved`, so this holds by construction; it is asserted because
    "by construction" is what stops being true when someone adds a fifth status.
    """
    drawn = table[~table["pending"]]
    if drawn.empty:
        return

    counted = drawn[~drawn["provisional"]]
    if len(counted):
        totals = counted[list(SEGMENTS)].sum(axis=1)
        bad = counted[totals != counted["n_targets"]]
        if len(bad):
            first = bad.iloc[0]
            raise ValueError(
                f"outcome segments do not partition the target set: "
                f"{first['method']} on {first['row_label']} sums to "
                f"{int(totals.loc[bad.index[0]])} against "
                f"n_targets={int(first['n_targets'])}. The bar would render as a "
                "composition it is not."
            )

    # A provisional bar has no counts to check, so the same promise is made
    # against its shares — otherwise a placeholder could be typed in wrong and
    # render as a bar that stops short of the top with nothing to say so.
    share_columns = [f"share_{SEGMENT_LABELS[k]}" for k in SEGMENTS]
    shares = drawn[share_columns].sum(axis=1)
    off = drawn[~np.isclose(shares, 1.0)]
    if len(off):
        first = off.iloc[0]
        raise ValueError(
            f"outcome shares do not sum to 1 for {first['method']} on "
            f"{first['row_label']}: {float(shares.loc[off.index[0]]):.4f}. The bar "
            "would render as a composition it is not."
        )


def _draw_panel(
    ax,
    table: pd.DataFrame,
    methods: list[str],
    *,
    kinds: list[str],
    label_rotation: int = 0,
) -> None:
    """One panel: methods across x, a bar per kind, segments stacked in y.

    `label_rotation` turns the in-bar accuracy readout upright. Six methods in a
    full-width panel leave room for `0.622` across the bar; three such panels
    side by side do not, and a horizontal label there spills onto its
    neighbour's ghost.
    """
    offsets = np.linspace(
        -BAR_WIDTH / 2 * (len(kinds) - 1), BAR_WIDTH / 2 * (len(kinds) - 1), len(kinds)
    )
    indexed = table.set_index(["kind", "method"])
    for slot, kind in enumerate(kinds):
        for x, method in enumerate(methods):
            key = (kind, method)
            centre = x + offsets[slot]
            if key not in indexed.index or bool(indexed.loc[key, "pending"]):
                # A ghost, not a zero: the campaign is uncollected, and a bar of
                # height 0 would say it scored nothing.
                ax.bar(
                    centre, 1.0, BAR_WIDTH, facecolor="none", edgecolor=_C_MUTED,
                    linewidth=0.9, linestyle=(0, (3, 2)), zorder=2,
                )
                continue
            row = indexed.loc[key]
            bottom = 0.0
            for segment in SEGMENTS:
                # Shares, not counts over `n_targets`: a provisional bar is
                # rates from an earlier run with no denominator, and every bar
                # is a composition of the whole anyway.
                height = float(row[f"share_{SEGMENT_LABELS[segment]}"])
                if not np.isfinite(height) or height <= 0:
                    continue
                fill = SEGMENT_INK[segment]
                hatch = KIND_HATCH[kind] or None
                ax.bar(
                    centre, height, BAR_WIDTH, bottom=bottom,
                    facecolor=fill,
                    # One property draws both the hatch and the segment border,
                    # so a hatched bar's separators soften with its hatch.
                    edgecolor=HATCH_INK if hatch else _SURFACE,
                    linewidth=HATCH_LINEWIDTH if hatch else 0.6,
                    hatch=hatch, zorder=3,
                )
                # Every segment carries its own share, inside itself: a stack
                # labelled only at the bottom asks the reader to subtract, and
                # the failure rate is one of the two numbers this figure is for.
                if height >= MIN_LABEL_SHARE:
                    ax.annotate(
                        f"{height * 100:.1f}%",
                        xy=(centre, bottom + height / 2),
                        ha="center", va="center", fontsize=6.5,
                        color=label_ink(fill), fontweight="bold",
                        zorder=4, rotation=label_rotation,
                        # On a hatched bar the hatch is drawn in the surface
                        # colour, so it strikes straight through a white label.
                        # A patch of the segment's own fill behind the text
                        # restores the contrast `label_ink` computed for.
                        bbox={
                            "facecolor": fill,
                            "edgecolor": "none",
                            "pad": 0.8,
                        },
                    )
                bottom += height

    ax.set_xticks(range(len(methods)))
    # Deliberately neutral. Tinting these with the variant hues would put
    # "Octant-Hull" in green and "Spotter" in red directly beneath green and red
    # segments, and colour would mean outcome inside the bars and identity an
    # inch below them. The method is the x position; nothing else needs to say so.
    ax.set_xticklabels([short_label(m) for m in methods], fontsize=8.5, color=_C_INK_2)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    # Percent, because every in-bar label is a percent — an axis in fractions
    # beside labels in percent makes the reader convert to check one against
    # the other.
    ax.set_yticklabels([f"{v:.0f}%" for v in np.arange(0, 101, 20)])
    ax.set_ylabel("Share of targets", fontsize=9.5, color=_C_INK_2)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=_C_GRID, linewidth=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_C_AXIS)


def outcome_handles():
    """Legend proxies for the three outcomes, in stack order.

    All three are always listed even where one is empty: this is the key to a
    fixed scheme, and an entry that comes and goes with the data makes the
    scheme itself look data-dependent.
    """
    from matplotlib.patches import Patch

    return [
        Patch(facecolor=SEGMENT_INK[key], edgecolor=_SURFACE, label=SEGMENT_LABELS[key])
        for key in SEGMENTS
    ]


def kind_handles(kinds: list[str], *, pending: set[str]):
    """Legend proxies for the dataset types, carrying no fill.

    Every neutral that would fill them is within reach of the `failed` grey, and
    a filled `mesh` swatch beside it is the one confusion this figure cannot
    afford. Outline plus hatch says "this channel is texture, not colour".

    A kind with no data anywhere keeps its plain name and takes the dashed
    outline its bars have, so the swatch is the mark the reader will meet rather
    than a description of it. That the campaign is uncollected is a fact about
    the data, not about the encoding, so it belongs in the caption and the
    manifest and not in the key.
    """
    from matplotlib.patches import Patch

    return [
        Patch(
            facecolor="none",
            edgecolor=_C_MUTED,
            hatch=KIND_HATCH[k] or None,
            linestyle="solid" if k not in pending else (0, (3, 2)),
            linewidth=1.0,
            label=H.KIND_LABELS[k].lower(),
        )
        for k in kinds
    ]


def pending_kinds(table: pd.DataFrame) -> set[str]:
    """The dataset types with no data at all, drawn as ghosts."""
    return {
        str(kind)
        for kind, group in table.groupby("kind", sort=False)
        if bool(group["pending"].all())
    }


def provisional_kinds(table: pd.DataFrame) -> set[str]:
    """The dataset types drawn from hard-coded placeholder rates.

    **The figure does not mark these.** A bar drawn from a constant in a source
    file and a bar drawn from the pipeline are identical once rendered, and only
    one of them can be cited — so the caveat lives in the table's footnote, the
    `provisional` column of this figure's CSV twin, and the manifest, and any
    caption using this PNG has to carry it. This function is what those readers
    query; it exists so the fact is one predicate rather than three.
    """
    if "provisional" not in table.columns:
        return set()
    return {
        str(kind)
        for kind, group in table.groupby("kind", sort=False)
        if bool(group["provisional"].any())
    }


def _place_legend(fig, renderer, handles, y: float, ncol: int):
    """One legend in a single row, placed at x = 0 and shifted by the caller.

    No title: the two groups are told apart by the marks themselves — filled
    swatches are outcomes, outlines are dataset types — and by the gap between
    them, so a pair of words above the row only bought vertical space.

    Placed provisionally because the caller centres both groups together and
    cannot do that until it knows how wide each rendered, which depends on the
    figure size, and this module draws at two of them.
    """
    legend = fig.legend(
        handles=handles, ncol=ncol, loc="center left", bbox_to_anchor=(0.0, y),
        frameon=False, fontsize=8.5, labelcolor=_C_INK_2, handlelength=1.5,
        columnspacing=1.2, handletextpad=0.45,
        # A frameless legend still reserves its border padding, which is what
        # opened a gap wide enough to read as a gap between the two groups.
        borderpad=0.0, borderaxespad=0.0,
    )
    return legend, legend.get_window_extent(renderer).width / fig.bbox.width


def _axes_top(base: float, table: pd.DataFrame) -> float:
    """The axes box top. One value now, kept as a seam for per-layout bands."""
    del table
    return base


def _add_legends(
    fig, table: pd.DataFrame, kinds: list[str], *, y: float = LEGEND_Y
) -> None:
    """Both legends as a single centred row under the title.

    Two legends rather than one because there are two independent encodings: a
    combined key would read as five alternatives on one scale, when every bar is
    in fact one outcome stack *and* one dataset type. The gap between them is
    wider than the gap inside either, which is what keeps them two keys.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    groups = [
        _place_legend(fig, renderer, outcome_handles(), y, len(SEGMENTS)),
        _place_legend(
            fig, renderer,
            kind_handles(kinds, pending=pending_kinds(table)),
            y, len(kinds),
        ),
    ]
    total = sum(width for _, width in groups) + GROUP_GAP
    x = 0.5 - total / 2
    for legend, width in groups:
        legend.set_bbox_to_anchor((x, y))
        x += width + GROUP_GAP


def plot_bars(
    table: pd.DataFrame, out_path: Path, *, methods: list[str], title: str
) -> Path:
    kinds = [k for k in H.KINDS if (table["kind"] == k).any()]
    ordered = method_order(table, methods)
    fig, ax = plt.subplots(
        figsize=(PANEL_WIDTH_INCHES, PANEL_HEIGHT_INCHES), dpi=DPI, facecolor=_SURFACE
    )
    ax.set_facecolor(_SURFACE)
    _draw_panel(ax, table, ordered, kinds=kinds)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=_C_INK, y=0.99)
    fig.tight_layout()
    # `rect=` reserves a band for the *axes and its decorations*, which leaves
    # the axes box itself well below the band's top. Setting the box directly is
    # what puts the bars where the legend line ends.
    fig.subplots_adjust(top=_axes_top(AXES_TOP, table))
    _add_legends(fig, table, kinds)
    fig.savefig(out_path, dpi=DPI, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def compare_table(
    accuracy: pd.DataFrame, plan: list[dict], *, top_n: int, methods: list[str]
) -> pd.DataFrame:
    """The same four counts, one row per (dataset, kind, method).

    The breakdown the pooled panel averages over. Each dataset keeps its own
    denominator — as01/02/03 carry 399 / 412 / 458 targets, so one pooled share
    weights as03 heaviest and describes no dataset on its own.
    """
    by_run = {run_id: g.set_index("method") for run_id, g in accuracy.groupby("run_id")}
    rows: list[dict] = []
    for entry in plan:
        if entry["scope"] != H.DATASET:
            continue
        source = by_run.get(entry["run_id"]) if entry["run_id"] else None
        for method in methods:
            counts = H.method_counts(source, method, top_n=top_n)
            n = int(counts["n_targets"]) if counts else 0
            row = {
                "dataset": entry["dataset"],
                "kind": entry["kind"],
                "kind_label": H.KIND_LABELS[entry["kind"]],
                "row_label": entry["label"],
                "run_id": entry["run_id"],
                "method": method,
                "method_label": short_label(method),
                "top_n": top_n,
                "n_targets": n,
                "provisional": False,
                "pending": counts is None or n == 0,
            }
            rows.append(_with_shares(row, counts or {}, n))
    table = pd.DataFrame(rows)
    guard_partition(table)
    return table


def plot_compare(
    table: pd.DataFrame, out_path: Path, *, methods: list[str], title: str
) -> Path:
    """One panel per dataset, sharing the y axis so bar heights compare directly."""
    datasets = list(dict.fromkeys(table["dataset"]))
    kinds = [k for k in H.KINDS if (table["kind"] == k).any()]
    ordered = method_order(table, methods)
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(PANEL_WIDTH_INCHES * len(datasets) * 0.50, PANEL_HEIGHT_INCHES),
        dpi=DPI,
        facecolor=_SURFACE,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets):
        ax.set_facecolor(_SURFACE)
        _draw_panel(
            ax,
            table[table["dataset"] == dataset],
            ordered,
            kinds=kinds,
            label_rotation=90,
        )
        n = int(table.loc[table["dataset"] == dataset, "n_targets"].max())
        ax.set_title(
            f"{str(dataset).upper()}  ·  n={n:,}", fontsize=10, color=_C_INK, pad=8
        )
        ax.tick_params(axis="x", labelrotation=90)
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.suptitle(title, fontsize=13, fontweight="bold", color=_C_INK, y=1.0)
    fig.tight_layout()
    fig.subplots_adjust(top=_axes_top(COMPARE_AXES_TOP, table))
    _add_legends(fig, table, kinds, y=COMPARE_LEGEND_Y)
    fig.savefig(out_path, dpi=DPI, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build(
    mesh_runs: dict[str, RunPaths],
    weighted_runs: dict[str, RunPaths],
    *,
    analysis_root: Path | None = None,
    grid: str = DEFAULT_GRID,
    resolution: int,
    methods: list[str] | None = None,
    top_n: int = H.BODY_TOP_N,
    layouts: tuple[str, ...] = (POOLED,),
) -> tuple[dict[str, pd.DataFrame], dict]:
    """The outcome tables per layout, plus the manifest, before rendering."""
    wanted = list(methods) if methods else list(PUBLISHED_METHODS)
    plan = H.row_plan(mesh_runs, weighted_runs)
    all_runs = {**mesh_runs, **{r.run_id: r for r in weighted_runs.values()}}
    accuracy = accuracy_rows(
        all_runs,
        analysis_root=analysis_root,
        grid=grid,
        resolution=resolution,
        methods=wanted,
        include_counts=True,
    )
    present = set(accuracy["method"])
    printed = [m for m in wanted if m in present]
    if not printed:
        raise ValueError(
            f"none of {wanted} appear in the selected runs' topn_accuracy.csv"
        )

    tables: dict[str, pd.DataFrame] = {}
    if POOLED in layouts:
        tables[POOLED] = outcome_table(accuracy, plan, top_n=top_n, methods=printed)
    if COMPARE in layouts:
        tables[COMPARE] = compare_table(accuracy, plan, top_n=top_n, methods=printed)

    manifest = {
        "mesh_runs": sorted(mesh_runs),
        "weighted_runs": {d: r.run_id for d, r in sorted(weighted_runs.items())},
        "grid": {"scheme": grid, "resolution": resolution},
        "methods": printed,
        "methods_requested_but_absent": [m for m in wanted if m not in present],
        "top_n": top_n,
        "layouts": list(layouts),
        "segments": list(SEGMENTS),
        "encoding": (
            "fill = method (the paper's fixed variant hues); hatch = dataset type "
            "(mesh solid, traffic-weighted hatched); stack = outcome. No segment "
            "carries a texture of its own, because hatch is spoken for."
        ),
        "partition": (
            "n_targets == n_solved + n_fallback + n_error, and correct/wrong splits "
            "n_solved, so the four segments partition the target set. Asserted per "
            "bar by guard_partition."
        ),
        "arithmetic": (
            "headline_table.method_counts / pool_method_counts, imported rather than "
            "reimplemented, so these bars and headline_top{n}.csv are the same "
            "numbers by construction"
        ),
        "method_order": (
            "bars run left to right by pooled correct rate, best first, ranked on "
            "the mesh rows over every dataset in the figure so an x position means "
            "the same method in every panel"
        ),
        "pending_note": (
            "traffic-weighted bars are drawn as dashed outlines: has_weight is false "
            "on every run, so the campaign is uncollected rather than scored zero"
        ),
    }
    return tables, manifest


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("plot-outcome-bars")
    def plot_outcome_bars_cmd(
        run_id: list[str] = typer.Option(
            None, "--run-id", help="Mesh runs, one per dataset (repeatable)."
        ),
        weighted_run_id: list[str] = typer.Option(
            None,
            "--weighted-run-id",
            help="Traffic-weighted twin, as <dataset>=<run_id> (repeatable). None "
            "exist yet, so those bars draw as empty outlines.",
        ),
        layout: list[str] = typer.Option(
            [],
            "--layout",
            help=f"{POOLED} (dataset types pooled) or {COMPARE} (one panel per "
            f"dataset). Repeatable; default {POOLED}.",
        ),
        top_n: int = typer.Option(
            H.BODY_TOP_N, "--top-n", help="Which accuracy the correct segment is."
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option(
            [], "--resolution", "-r", help=RESOLUTION_HELP
        ),
        method: list[str] = typer.Option(
            None, "--method", help="Override the six published variants (repeatable)."
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """§8.1's outcome bars: correct / wrong / fallback per method and dataset type.

        Writes outcome_bars[_by_dataset].<grid>.top{n}.{png,csv} + a manifest into
        _cross/accuracy-table/<dataset-set>/. Needs `classify` on every run.
        """
        if not run_id:
            raise typer.BadParameter(
                "pass at least one --run-id. Like `table-headline`, this command has "
                "no --all-runs: which datasets belong in one figure is the caller's "
                "call, and §7.3 declines the operator/public head-to-head."
            )
        unknown = [x for x in layout if x not in LAYOUTS]
        if unknown:
            raise typer.BadParameter(f"unknown --layout {unknown}; pick from {list(LAYOUTS)}")
        layouts = tuple(dict.fromkeys(layout)) or (POOLED,)

        g, resolutions = resolve_cli_grid(grid, resolution, sweep=False)
        mesh_runs = {rid: resolve_run(rid, outputs_root) for rid in run_id}
        weighted_runs = {
            dataset: resolve_run(rid, outputs_root)
            for dataset, rid in H.parse_weighted_pairs(weighted_run_id or []).items()
        }

        for res in resolutions:
            tables, manifest = build(
                mesh_runs,
                weighted_runs,
                analysis_root=analysis_root,
                grid=g.name,
                resolution=res,
                methods=list(method) if method else None,
                top_n=top_n,
                layouts=layouts,
            )
            out_dir = cross_dir(analysis_root, sorted(mesh_runs), kind=H.CROSS_KIND)
            slug = grid_slug(g.name, res)
            methods_printed = manifest["methods"]
            written: list[Path] = []
            for name, table in tables.items():
                stem = "outcome_bars" if name == POOLED else "outcome_bars_by_dataset"
                table.to_csv(out_dir / f"{stem}.{slug}.top{top_n}.csv", index=False)
                png = out_dir / f"{stem}.{slug}.top{top_n}.png"
                title = f"Top-{top_n} outcome composition by method"
                renderer = plot_bars if name == POOLED else plot_compare
                written.append(
                    renderer(table, png, methods=methods_printed, title=title)
                )
            (out_dir / f"outcome_bars.{slug}.top{top_n}.manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            typer.echo(
                f"{len(methods_printed)} methods x {len(H.KINDS)} dataset types at "
                f"{slug} top-{top_n} · layouts {', '.join(tables)} -> "
                f"{', '.join(p.name for p in written)}"
            )
