"""Outcome composition bars: cell label first, then ring tier.

Each bar is one method's **entire** TG set, stacked bottom-up by cell label

    true | wrong | outland | no answer

and each cell-label group broken down by ring tier (ring0, ring1, ring2,
beyond). The bar's primary split answers "did the prediction land in the TG's
serving region?"; the breakdown answers "and how far off?". It is the same
cross-tab classify writes (`n_<tier>_cell_<label>`); grouping by cell label
first puts the operator's first question on the bar's main division.

## Encoding

* **Colour = ring tier**: green (in the TG grid), light blue (1 ring out),
  light purple (2 rings out), light grey (further out -- ungraded, so no
  hue), dark grey (no answer). See `TIER_INK` for the reasoning and the
  validation.
* **Stripe = cell label.** Plain for `true`, `//` for `wrong`, `\\\\` for
  `outland`. Two stripe directions rather than two densities: direction
  survives print and colour-vision deficiency, and the empty pattern is the
  good outcome so the eye reads stripes as "something went wrong". Stripes are
  white on dark fills and ink on light ones, by the same rule as the labels.
* **A thicker border around each cell-label group**, "no answer" included,
  so a group's ring breakdown reads as one unit. Every group takes the same
  border, so no segment gains apparent width -- in a stacked share chart width
  is quantitative. Sub-segments have no stroke of their own.
* Inside a group, a **white line** separates the ring tiers -- interior
  boundaries only, drawn as lines across the bar.
* Every sub-segment above `_LABEL_FLOOR` prints **its own share**, on a solid
  chip so the stripe never runs through the digits; thinner slivers stay in
  the CSV twin.
* A **rail** right of each bar carries one segment per cell-label group,
  white with that group's stripe and a border, and the group's **total**
  beside it, set vertically in bold ink. Labels of thin adjacent groups are
  nudged apart, never reordered. "No answer" has no rail: it has no
  breakdown, so its in-bar label is its total.

## Order

Each panel ranks itself, descending, on the cumulative key

    (true, true & ring0, true & <=ring1, true & <=ring2)

at the reported precision, so the displayed order follows from the displayed
values: most TGs in their serving region first, and among equals, the one
whose correct-region predictions are tighter. The pooled cross-dataset order
breaks remaining ties.

## Layouts and pooling

As v4. `compare` is one panel per dataset; `pooled` is one panel over every
input run's TGs, a micro-average recounted from the per-TG parquets by
`classify.summarize`. Coverage is strict and TG ids must be disjoint across
runs.

Command: `plot-outcome-bars`. Needs `classify` on every run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules import methods
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths, grid_slug

#: Ring tiers bottom-up, then the unanswered slot.
TIERS: tuple[str, ...] = C.RING_TIERS
FAILED = "failed"
STACK: tuple[str, ...] = (*TIERS, FAILED)

TIER_LABELS = {
    "ring0": "in the TG grid",
    "ring1": "1 ring out",
    "ring2": "2 rings out",
    "beyond": "further out",
    FAILED: "no answer",
}

#: Ring tier -> fill. Hue for the three graded tiers -- green for the TG's own
#: grid, light blue one ring out, light purple two rings out -- so adjacent
#: tiers differ in hue, not only in lightness. "Further out" is a light grey:
#: the ring metric has stopped grading there, so it takes no hue. "No answer"
#: is a dark grey, the heaviest ink on the bar, so a method that declined to
#: answer cannot be mistaken for one that answered far away.
#:
#: The dark grey `#6b6a65` is kept lighter than `GROUP_EDGE` (`#52514e`) so the
#: tier border still shows around it.
#:
#: Validated with the dataviz `validate_palette.js --mode light` (adjacent
#: pairs, the stack's pairlist): worst adjacent CVD dE 13.4 (deutan, blue vs
#: green), tritan 11.4; normal-vision floor 15.7 -- both hard gates pass.
#: Rejected on the way: a cyan-leaning blue (#8ec5ec) against a lavender
#: (#b8a4e3) collapsed to dE 4.0 under deuteranopia and 9.4 in normal vision,
#: and every blue darker than #b3cdf6 sat under the 15 floor against the green.
#: The lightness-band and chroma checks FAIL by construction (light tints and
#: two greys are outside the mid-tone categorical band), and the contrast WARN
#: is relieved by the tier borders, the in-place labels and the CSV twin.
TIER_INK = {
    "ring0": "#79bf9b",
    "ring1": "#b8d2f8",
    "ring2": "#a58bd6",
    "beyond": "#d8d7cf",
    FAILED: "#6b6a65",
}

#: Cell label -> hatch. Plain is the good outcome.
CELL_HATCH = {"true": "", "wrong": "//", "outland": "\\\\"}
CELL_LEGEND = {
    "true": "true cell",
    "wrong": "wrong cell (//)",
    "outland": "outland (\\\\)",
}

#: Cell-label groups bottom-up, then the unanswered slot.
GROUPS: tuple[str, ...] = (*C.CELL_LABELS, FAILED)

METHOD_LABELS = methods.METHOD_LABELS
method_label = methods.method_label

_SURFACE = "#ffffff"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"

#: Bar thickness as a share of the slot. Narrower than v4's 0.62 to leave room
#: on the right of each bar for its rail and the group totals beside it.
_BAR_FRAC = 0.50
_PANEL_W = 5.0

#: The rail: a narrow strip to the right of each bar, one segment per
#: cell-label group spanning that group's height, with the group's total
#: printed beside it. The bar breaks each group down by ring tier; the rail
#: gives the group back as one number.
_RAIL_GAP = 0.08
_RAIL_W = 0.07
_RAIL_TEXT_PAD = 0.015

#: Vertical gap between consecutive rail segments, in share units, so each
#: group reads as its own mark (as the bar's group borders do).
_RAIL_SEG_GAP = 0.006

#: Minimum vertical distance between two rail labels, in share units. The
#: labels run vertically, so this is one rotated "100%" plus air; thin
#: adjacent groups are nudged apart (never reordered) until they clear it.
_RAIL_LABEL_GAP = 0.085

#: Figure-width floor, for the figure-level title and legends on a single
#: panel. Measured against the subjects in `_SUBJECTS`; lengthening one can
#: clip the pooled title at both ends.
_MIN_FIG_W = 8.0

#: Figure width, in inches, below which the method-term caption wraps onto
#: two lines.
_TERMS_ONE_LINE_W = 11.0

#: The rail's stripe ink on its white fill.
_RAIL_HATCH_INK = _INK_2
#: Below this share a sub-segment is too thin to hold its label; the value
#: stays in the CSV twin.
_LABEL_FLOOR = 0.035

#: The group border. Heavier than the white separators so it reads as the
#: grouping, and ink so it survives a stripe crossing it.
GROUP_EDGE = _INK_2
GROUP_EDGE_PT = 1.3

#: White separator between ring-tier sub-segments inside one cell-label
#: group: lighter than the group border, so the group still reads as the unit.
SEPARATOR_PT = 1.0

#: Stripe stroke. Thinner than matplotlib's 1.0 default: at a 24 px bar a
#: full-weight stripe outweighs the fill it sits on.
_HATCH_PT = 0.6

#: Reported and ranked at whole percent -- see v4's module for the case
#: (as01, nside 16) where ranking on the exact share contradicted the labels.
ACCURACY_DECIMALS = 2

COMPARE = "compare"
POOLED = "pooled"
LAYOUTS: tuple[str, ...] = (COMPARE, POOLED)

NAMES: dict[str, tuple[str, str, str]] = {
    COMPARE: ("outcome_bars.{slug}.png", "outcome_bars.{slug}.csv", "outcome_bars.{slug}.manifest.json"),
    POOLED: (
        "outcome_bars.pooled.{slug}.png",
        "outcome_bars.pooled.{slug}.csv",
        "outcome_bars.pooled.{slug}.manifest.json",
    ),
}

_SUBJECTS = {
    COMPARE: "Serving region, then distance",
    POOLED: "Serving region, then distance, datasets pooled",
}


def count_col(tier: str, label: str | None = None) -> str:
    """The `accuracy.csv` count column for a tier, or a tier x cell label."""
    if tier == FAILED:
        return "n_failed"
    return f"n_{tier}" if label is None else f"n_{tier}_cell_{label}"


def group_col(group: str) -> str:
    """The `accuracy.csv` count column for a whole cell-label group."""
    return "n_failed" if group == FAILED else f"n_cell_{group}"


def segments() -> list[tuple[str, str | None]]:
    """Every drawn segment bottom-up, as `(tier, cell label)`: cell label
    outer, ring tier inner; `(failed, None)` last."""
    return [(t, lab) for lab in C.CELL_LABELS for t in TIERS] + [(FAILED, None)]


def _add_shares(table: pd.DataFrame) -> pd.DataFrame:
    """Exact shares -- the drawn geometry. Rounding lives in labels and sort keys."""
    for tier, lab in segments():
        col = count_col(tier, lab)
        table[f"share_{col}"] = table[col] / table["n_tgs"]
    for group in GROUPS:
        table[f"share_{group_col(group)}"] = table[group_col(group)] / table["n_tgs"]
    return table


def load_rung(run: RunPaths, nside: int, *, analysis_root: Path | None = None) -> pd.DataFrame:
    """One run's `accuracy.csv` at one rung."""
    path = run.classify_dir(nside, root=analysis_root) / C.ACCURACY_CSV
    if not path.exists():
        raise MissingArtifactError(f"{path} missing; run `classify --run-id {run.run_id}` first")
    df = pd.read_csv(path)
    df.insert(0, "run_id", run.run_id)
    df.insert(1, "dataset", cross.short_dataset(run.run_id))
    return df


def build_table(
    runs: list[RunPaths],
    nside: int,
    *,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
) -> pd.DataFrame:
    """Long frame: one row per (dataset, method), shares summing to 1."""
    table = pd.concat(
        [load_rung(r, nside, analysis_root=analysis_root) for r in runs], ignore_index=True
    )
    if methods:
        table = table[table["method"].isin(methods)]
        if table.empty:
            raise ValueError(f"none of {methods} are scored at nside={nside}")
    C.guard_partition(table)
    C.guard_cross_tab(table)
    return _add_shares(table.reset_index(drop=True))


def _load_tgs(run: RunPaths, method: str, nside: int, *, analysis_root: Path | None = None):
    path = run.classify_dir(nside, root=analysis_root) / C.TGS_PARQUET.format(method=method)
    if not path.exists():
        raise MissingArtifactError(f"{path} missing; run `classify --run-id {run.run_id}` first")
    return pd.read_parquet(path)


def pooled_table(
    runs: list[RunPaths],
    nside: int,
    *,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
) -> pd.DataFrame:
    """Every run's TGs as one population, recounted: one row per method.

    Concatenating per-TG rows is sound because each label is decided within
    its own row against its own run's answer space (`pred_seed_id` against
    `tg_seed_id`), so the runs' seed namespaces never meet.
    """
    scored = {
        r.run_id: set(load_rung(r, nside, analysis_root=analysis_root)["method"]) for r in runs
    }
    if methods:
        wanted = set(methods)
        scored = {r: ms & wanted for r, ms in scored.items()}
        if not any(scored.values()):
            raise ValueError(f"none of {methods} are scored at nside={nside}")
    common = cross.guard_common_methods(scored)

    frames: dict[str, pd.DataFrame] = {}
    for i, method in enumerate(common):
        per_run = {r.run_id: _load_tgs(r, method, nside, analysis_root=analysis_root) for r in runs}
        if i == 0:
            cross.guard_disjoint_tgs({rid: set(df["tg_id"]) for rid, df in per_run.items()})
        frames[method] = pd.concat(per_run.values(), ignore_index=True)

    table = C.summarize(frames, nside)
    sizes = set(table["n_tgs"].astype(int))
    if len(sizes) > 1:
        raise ValueError(
            f"cannot pool: methods were scored on different TG counts "
            f"({dict(zip(table['method'], table['n_tgs'].astype(int)))}). "
            f"Pass --method to pick methods that share a population, or --layout compare."
        )
    run_ids = [r.run_id for r in runs]
    table.insert(0, "run_id", "+".join(sorted(run_ids)))
    table.insert(1, "dataset", cross.dataset_slug(run_ids))
    return _add_shares(table.reset_index(drop=True))


#: The sort key, all descending: correct-region share, then how tight the
#: correct-region predictions are.
_RANK_KEYS: tuple[str, ...] = (
    "cell_true", "true_within_ring0", "true_within_ring1", "true_within_ring2",
)


def _with_rank_keys(table: pd.DataFrame) -> pd.DataFrame:
    """Cumulative counts, divided and rounded **once** -- summing rounded
    shares compounds the error."""
    out = table.copy()
    out["cell_true"] = (out["n_cell_true"] / out["n_tgs"]).round(ACCURACY_DECIMALS)
    running = 0
    for key, tier in zip(_RANK_KEYS[1:], TIERS):
        running = running + out[count_col(tier, "true")]
        out[key] = (running / out["n_tgs"]).round(ACCURACY_DECIMALS)
    return out


def _sorted_methods(frame: pd.DataFrame, tiebreak: dict[str, int]) -> list[str]:
    f = frame.assign(_tie=frame["method"].map(tiebreak).fillna(len(tiebreak)))
    return f.sort_values([*_RANK_KEYS, "_tie"], ascending=[False] * len(_RANK_KEYS) + [True])[
        "method"
    ].tolist()


def method_order(table: pd.DataFrame) -> list[str]:
    """The pooled order across datasets: manifest and last tiebreak."""
    means = _with_rank_keys(table).groupby("method", as_index=False)[list(_RANK_KEYS)].mean()
    return _sorted_methods(means, {m: i for i, m in enumerate(sorted(means["method"]))})


def panel_order(table: pd.DataFrame, dataset: str) -> list[str]:
    """One dataset's methods by correct-region share. Each panel ranks itself."""
    pooled = {m: i for i, m in enumerate(method_order(table))}
    keyed = _with_rank_keys(table)
    return _sorted_methods(keyed[keyed["dataset"] == dataset], pooled)


def _contrast_ink(face: str) -> str:
    """White or ink by the fill's luminance -- for labels and stripes alike."""
    r, g, b = (int(face[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.55 else _INK


def _draw_bar(ax, x: float, row: pd.Series, nd: int) -> None:
    """One method's stack: per cell-label group, its ring-tier sub-segments,
    white separators between them, the group border; every sub-segment
    labelled. Then the rail to its right."""
    half = _BAR_FRAC / 2
    groups: list[tuple[str, float, float]] = []
    bottom = 0.0
    for group in GROUPS:
        group_bottom = bottom
        parts = [(FAILED, None)] if group == FAILED else [(t, group) for t in TIERS]
        drawn: list[tuple[float, float, str]] = []
        for tier, lab in parts:
            v = float(row[f"share_{count_col(tier, lab)}"])
            face = TIER_INK[tier]
            if v > 0:
                ax.bar(
                    x, v, bottom=bottom, width=_BAR_FRAC,
                    facecolor=face,
                    hatch=CELL_HATCH[lab] if lab else "",
                    # The hatch takes the edge colour; linewidth 0 keeps the
                    # sub-segment unstroked -- separators and border are drawn
                    # as their own marks, so no segment gains apparent width.
                    edgecolor=_contrast_ink(face), linewidth=0, zorder=3,
                )
                drawn.append((bottom, v, face))
            bottom += v
        # White separators between ring tiers: interior boundaries only, as
        # lines across the bar rather than strokes around each sub-segment.
        for seg_bottom, _, _ in drawn[1:]:
            ax.hlines(
                seg_bottom, x - half, x + half, colors=_SURFACE,
                linewidth=SEPARATOR_PT, zorder=3.5,
            )
        height = bottom - group_bottom
        if height > 0:
            ax.bar(
                x, height, bottom=group_bottom, width=_BAR_FRAC,
                facecolor="none", edgecolor=GROUP_EDGE, linewidth=GROUP_EDGE_PT, zorder=4,
            )
        for seg_bottom, v, face in drawn:
            if v >= _LABEL_FLOOR:
                ax.text(
                    x, seg_bottom + v / 2, f"{round(v, nd) * 100:.0f}%",
                    ha="center", va="center", fontsize=7.5, color=_contrast_ink(face), zorder=5,
                    bbox=dict(boxstyle="square,pad=0.12", facecolor=face, edgecolor="none"),
                )
        if group != FAILED:
            groups.append((group, group_bottom, height))
    _draw_rail(ax, x, groups, nd)


def _spread(ys: list[float], gap: float) -> list[float]:
    """Nudge sorted label positions apart to at least `gap`, keeping each
    label's half-height inside [0, 1] so none crosses the baseline or the top.

    One pass up from the baseline, then one down from the top: only labels in
    a crowded run move, and a push down from the top stops at the first label
    it no longer collides with. (Shifting every label by the overflow instead
    pushed an unrelated bottom label below 0%.)
    """
    lo, hi = gap / 2, 1 - gap / 2
    out = [max(y, lo) for y in ys]
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + gap)
    if out:
        out[-1] = min(out[-1], hi)
    for i in range(len(out) - 2, -1, -1):
        out[i] = min(out[i], out[i + 1] - gap)
    return out


def _draw_rail(ax, x: float, groups: list[tuple[str, float, float]], nd: int) -> None:
    """One rail segment per cell-label group: white, the group's stripe, a
    border; the group's total beside it.

    "No answer" has no rail: it has no ring breakdown, so its in-bar label
    already is its total.
    """
    rail_x = x + _BAR_FRAC / 2 + _RAIL_GAP + _RAIL_W / 2
    shown = [(g, b, h) for g, b, h in groups if h > 0]
    for group, bottom, height in shown:
        trim = min(_RAIL_SEG_GAP / 2, height / 4)
        ax.bar(
            rail_x, height - 2 * trim, bottom=bottom + trim, width=_RAIL_W,
            facecolor=_SURFACE, hatch=CELL_HATCH[group],
            edgecolor=_RAIL_HATCH_INK, linewidth=0.8, zorder=3,
        )
    rail_labels(ax, rail_x, [(b, h) for _, b, h in shown], nd)


def rail_labels(ax, rail_x: float, spans: list[tuple[float, float]], nd: int) -> None:
    """Each span's total beside the rail, set **vertically** to save width.

    `spans` are `(bottom, height)` in stack order. Rotated 90 degrees, a label
    is taller than it is wide, so thin neighbours are nudged apart by
    `_RAIL_LABEL_GAP` -- which is sized for the rotated text -- never reordered.
    """
    labelled = [(b, h) for b, h in spans if round(h, nd) > 0]
    ys = _spread([b + h / 2 for b, h in labelled], _RAIL_LABEL_GAP)
    for (_, height), y in zip(labelled, ys):
        ax.text(
            rail_x + _RAIL_W / 2 + _RAIL_TEXT_PAD, y, f"{round(height, nd) * 100:.0f}%",
            ha="center", va="top", rotation=90, rotation_mode="anchor",
            fontsize=7.5, fontweight="bold", color=_INK, zorder=5,
        )


def _legend_handles():
    from matplotlib.patches import Patch

    tiers = [
        Patch(facecolor=TIER_INK[t], edgecolor=GROUP_EDGE, linewidth=0.8, label=TIER_LABELS[t])
        for t in STACK
    ]
    cells = [
        Patch(
            facecolor=_SURFACE, edgecolor=_INK_2, hatch=CELL_HATCH[lab], linewidth=0.8,
            label=CELL_LEGEND[lab],
        )
        for lab in C.CELL_LABELS
    ]
    return tiers, cells


def render(
    table: pd.DataFrame,
    nside: int,
    out_dir: Path,
    *,
    order: list[str] | None = None,
    decimals: int | None = None,
    subject: str = _SUBJECTS[COMPARE],
    png_name: str | None = None,
    dpi: int = 150,
) -> Path:
    """One panel per dataset, one bar per method, each panel ranked by itself."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted(table["dataset"].unique())
    nd = ACCURACY_DECIMALS if decimals is None else decimals
    panels_w = _PANEL_W * len(datasets)
    fig_w = max(panels_w, _MIN_FIG_W)

    with plt.rc_context({"hatch.linewidth": _HATCH_PT}):
        fig, axes = plt.subplots(1, len(datasets), figsize=(fig_w, 5.8), sharey=True, squeeze=False)
        axes = axes[0]
        fig.patch.set_facecolor(_SURFACE)
        for ax, ds in zip(axes, datasets):
            ranked = list(order) if order else panel_order(table, ds)
            sub = table[table["dataset"] == ds].set_index("method")
            ranked = [m for m in ranked if m in sub.index]
            ax.set_facecolor(_SURFACE)
            xs = np.arange(len(ranked))
            for x, m in zip(xs, ranked):
                _draw_bar(ax, x, sub.loc[m], nd)

            n = int(sub["n_tgs"].iloc[0]) if len(sub) else 0
            ax.set_title(f"{ds.upper()}  ·  n={n}", fontsize=11, color=_INK, pad=10)
            ax.set_xticks(xs)
            # Short terms (see `methods.METHOD_TERMS`), so no rotation.
            ax.set_xticklabels([method_label(m) for m in ranked], fontsize=9, color=_INK_2)
            # Room on the right of the last bar for its rail and labels.
            ax.set_xlim(-0.5, len(ranked) - 0.5 + 0.25)
            ax.set_ylim(0, 1)
            ticks = np.arange(0, 1.01, 0.2)
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{int(v * 100)}%" for v in ticks])
            ax.grid(True, axis="y", color=_GRID, linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.spines["left"].set_visible(ax is axes[0])
            for side in ("left", "bottom"):
                ax.spines[side].set_color(_AXIS)
                ax.spines[side].set_linewidth(0.8)
            ax.tick_params(colors=_MUTED, labelsize=9)
            if ax is not axes[0]:
                ax.tick_params(axis="y", length=0)
        axes[0].set_ylabel("Share of TGs", fontsize=11, color=_INK_2)

        tiers, cells = _legend_handles()
        fig.legend(
            handles=tiers, loc="upper center", ncol=len(tiers), frameon=False, fontsize=9,
            bbox_to_anchor=(0.5, 0.955), labelcolor=_INK_2,
        )
        fig.legend(
            handles=cells, loc="upper center", ncol=len(cells), frameon=False, fontsize=9,
            bbox_to_anchor=(0.5, 0.915), labelcolor=_INK_2, handleheight=1.2,
        )
        fig.suptitle(
            f"{subject}  ·  HEALPix nside={nside} (grid_km {G.grid_km(nside):.0f})",
            fontsize=13, color=_INK, y=0.995,
        )
        # The term lookup under the panels, so the figure reads without the
        # paper beside it. Only the terms this figure draws.
        # One line needs ~11 in (measured on the six published terms); a
        # narrower figure -- the single pooled panel -- wraps it in two.
        entries = [f"{t}: {name}" for t, name in methods.method_term_table(
            table["method"].unique()).items()]
        rows = [entries] if fig_w >= _TERMS_ONE_LINE_W else [
            entries[: (len(entries) + 1) // 2], entries[(len(entries) + 1) // 2:]
        ]
        fig.text(
            0.5, 0.012, "\n".join("   ·   ".join(r) for r in rows),
            ha="center", va="bottom", fontsize=8, color=_MUTED, linespacing=1.5,
        )
        inset = (fig_w - panels_w) / 2 / fig_w
        fig.tight_layout(rect=(inset, 0.04 * len(rows), 1 - inset, 0.885))

        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / (png_name or NAMES[COMPARE][0].format(slug=grid_slug(nside)))
        fig.savefig(png, dpi=dpi, facecolor=_SURFACE)
        plt.close(fig)
    return png


def csv_columns() -> list[str]:
    """The CSV twin's columns, in order."""
    counts = [count_col(t, lab) for t, lab in segments() if t != FAILED]
    groups = [group_col(g) for g in GROUPS]
    return [
        "run_id", "dataset", "method", "n_tgs", "n_solved", *groups, *counts,
        *[f"share_{c}" for c in (*groups, *counts)],
        "accuracy_ring0", "accuracy_ring1", "accuracy_ring2", "accuracy_cell_true",
        "pred_dist_to_tg_km_p50", "pred_dist_to_tg_km_p90",
        "pred_dist_to_seed_km_p50", "pred_dist_to_seed_km_p90",
    ]


def _tgs_per_run(runs, nside, *, analysis_root=None) -> dict[str, int]:
    out: dict[str, int] = {}
    for run in runs:
        sizes = set(load_rung(run, nside, analysis_root=analysis_root)["n_tgs"].astype(int))
        if len(sizes) > 1:
            raise ValueError(f"{run.run_id} scored its methods on different TG counts {sorted(sizes)}")
        out[run.run_id] = sizes.pop() if sizes else 0
    return out


def _manifest(layout, table, nside, png_name, csv_name, *, run_ids, per_run=None) -> str:
    body = {
        "figure": png_name,
        "csv": csv_name,
        "layout": layout,
        "grid": G.describe(nside),
        "runs": run_ids,
        "arm": cross.arm(run_ids),
        "methods": method_order(table),
        "panel_order": {ds: panel_order(table, ds) for ds in sorted(table["dataset"].unique())},
        "method_terms": {
            m: {"term": method_label(m), "name": methods.METHOD_TERMS.get(method_label(m))}
            for m in method_order(table)
        },
        "stack": [count_col(t, lab) for t, lab in segments()],
        "encoding": {
            "colour": {"ring tier": TIER_INK, "labels": TIER_LABELS},
            "stripe": {"cell label": CELL_HATCH, "labels": CELL_LEGEND},
            "group_border": (
                f"{GROUP_EDGE} {GROUP_EDGE_PT} pt around every cell-label group, 'no answer' included"
            ),
            "labels": "each cell-label x tier share at whole percent, above the label floor",
            "separator": f"white {SEPARATOR_PT} pt between ring tiers inside a group",
            "rail": (
                "per-group strip right of each bar, white with the group's stripe; "
                "group total beside it, vertical, in ink"
            ),
        },
        "grouping": "cell label outer (true, wrong, outland, no answer), ring tier inner",
        "order_note": (
            "each panel ranked by its own cumulative (true, true & ring0, true & <=ring1, "
            "true & <=ring2) shares at whole percent, descending; `methods` is the pooled order"
        ),
    }
    if layout == POOLED:
        total = int(table["n_tgs"].iloc[0]) if len(table) else 0
        body["pooling"] = {
            "rule": "micro-average: per-TG rows concatenated across runs and re-scored",
            "runs": per_run or {},
            "n_tgs": total,
            "largest_share": round(max(per_run.values()) / total, 4) if per_run and total else None,
            "coverage": "strict: a method absent from any input run is refused",
        }
    return json.dumps(body, indent=2) + "\n"


def build_for_runs(
    runs: list[RunPaths],
    *,
    nsides: tuple[int, ...] = G.NSIDE_LADDER,
    methods: list[str] | None = None,
    layouts: tuple[str, ...] = LAYOUTS,
    analysis_root: Path | None = None,
) -> list[Path]:
    """One figure, CSV twin and manifest per layout per rung. Returns the PNGs."""
    unknown = [x for x in layouts if x not in LAYOUTS]
    if unknown:
        raise ValueError(f"unknown layout {unknown}; pick from {list(LAYOUTS)}")
    wanted = tuple(dict.fromkeys(layouts)) or LAYOUTS
    run_ids = [r.run_id for r in runs]
    out_dir = cross.cross_dir(run_ids, analysis_root=analysis_root)
    builders = {COMPARE: build_table, POOLED: pooled_table}
    written: list[Path] = []
    for layout in wanted:
        png_t, csv_t, man_t = NAMES[layout]
        for nside in sorted({G.validate_nside(n) for n in nsides}, reverse=True):
            slug = grid_slug(nside)
            table = builders[layout](runs, nside, methods=methods, analysis_root=analysis_root)
            table[[c for c in csv_columns() if c in table.columns]].to_csv(
                out_dir / csv_t.format(slug=slug), index=False
            )
            png = render(
                table, nside, out_dir, subject=_SUBJECTS[layout], png_name=png_t.format(slug=slug)
            )
            per_run = _tgs_per_run(runs, nside, analysis_root=analysis_root) if layout == POOLED else None
            (out_dir / man_t.format(slug=slug)).write_text(
                _manifest(
                    layout, table, nside, png.name, csv_t.format(slug=slug),
                    run_ids=run_ids, per_run=per_run,
                )
            )
            written.append(png)
    return written
