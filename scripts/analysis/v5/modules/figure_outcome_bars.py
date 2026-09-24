"""Outcome composition bars: ring tier by colour, cell label by stripe.

Each bar is one method's **entire** TG set, stacked bottom-up:

    ring0 | ring1 | ring2 | beyond | no answer

as in v4, and inside each ring tier the cross-tab with the cell partition:

    true (plain) | wrong (right-leaning stripe //) | outland (left-leaning stripe \\\\)

so a reader sees, within "one ring out", how much still landed in the TG's
serving region, how much in a neighbour's, and how much off the landmass.

## Encoding

* **Colour = ring tier**: green (in the TG grid), light blue (1 ring out),
  light purple (2 rings out), light grey (further out -- ungraded, so no
  hue), dark grey (no answer). See `TIER_INK` for the reasoning and the
  validation.
* **Stripe = cell label.** Plain for `true`, `//` for `wrong`, `\\\\` for
  `outland`. Two stripe directions rather than two densities: direction
  survives print and colour-vision deficiency, and the empty pattern is the
  good outcome so the eye reads stripes as "something went wrong".
  v4 reserved the hatch channel for a traffic-weighted arm that was never
  drawn (and is drawn as its own figure when it exists); v5 spends it here.
* **A thicker border around each tier**, so the three sub-segments of one ring
  read as one unit and the tier boundaries stay visible when a stripe runs
  across them. Every tier, "no answer" included, takes the same border, so no
  segment gains apparent width over another -- in a stacked share chart width
  is quantitative. Sub-segments inside a tier have no stroke of their own.
* Stripes are drawn in white on the dark tiers and ink on the light ones, by
  the same luminance rule as the labels.
* Inside a tier, a **white line** separates the cell-label sub-segments --
  interior boundaries only, drawn as lines across the bar rather than strokes
  around each sub-segment, so widths stay equal.
* A **rail** to the right of each bar repeats the ring tiers as narrow
  segments in the same fills and borders, with each tier's **total** printed
  beside it in bold ink -- the bar breaks a tier down by cell label, the rail
  gives it back as one number. Labels of thin adjacent tiers are nudged apart,
  never reordered. "No answer" has no rail: its in-bar label is its total.
* Every sub-segment above `_LABEL_FLOOR` prints **its own share**, on a solid
  chip so the stripe never runs through the digits. A tier's total is the sum
  of its labels; thinner slivers stay in the CSV twin.

## Order, layouts, pooling

Unchanged from v4. Each panel ranks itself down the tolerance ladder
`(ring0, <=ring1, <=ring2, answered)`, cumulative and rounded to the reported
precision so the displayed order follows from the displayed values; the pooled
cross-dataset order breaks remaining ties. `compare` is one panel per dataset;
`pooled` is one panel over every input run's TGs, a micro-average recounted
from the per-TG parquets by `classify.summarize`. Coverage is strict and TG ids
must be disjoint across runs.

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
#: The dark grey `#6b6a65` is kept lighter than `TIER_EDGE` (`#52514e`) so the
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

#: Cell label -> hatch, bottom-up within a tier. Plain is the good outcome.
CELL_HATCH = {"true": "", "wrong": "//", "outland": "\\\\"}
CELL_LEGEND = {
    "true": "true cell",
    "wrong": "wrong cell (//)",
    "outland": "outland (\\\\)",
}

METHOD_LABELS = methods.METHOD_LABELS
method_label = methods.method_label

_SURFACE = "#ffffff"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"

#: Bar thickness as a share of the slot. Narrower than v4's 0.62, and the
#: panel wider, to leave room on the right of each bar for its rail and the
#: tier totals printed beside it.
_BAR_FRAC = 0.50
_PANEL_W = 5.6

#: The tier rail: a narrow strip to the right of each bar, one segment per
#: ring tier spanning that tier's height, in the tier's fill and border, with
#: the tier's total share printed beside it. The bar breaks each tier down by
#: cell label; the rail gives the tier back as one number.
_RAIL_GAP = 0.08
_RAIL_W = 0.07
_RAIL_TEXT_PAD = 0.03

#: Vertical gap between consecutive rail segments, in share units, so each
#: tier reads as its own mark (as the bar's tier borders do).
_RAIL_SEG_GAP = 0.006

#: Minimum vertical distance between two rail labels, in share units. Thin
#: adjacent tiers would otherwise print their totals on top of each other, so
#: labels are nudged apart (never reordered) until they clear this.
_RAIL_LABEL_GAP = 0.045
_MIN_FIG_W = 8.0
#: Below this share a sub-segment is too thin to hold its label; the value
#: stays in the CSV twin.
_LABEL_FLOOR = 0.035

#: The tier border. Heavier than v4's 1 pt white segment gap so it reads as a
#: grouping, and ink so it survives a stripe crossing it.
TIER_EDGE = _INK_2
TIER_EDGE_PT = 1.3

#: White separator between cell-label sub-segments inside one tier: lighter
#: than the tier border, so the tier still reads as the unit.
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
    COMPARE: "Where the prediction landed",
    POOLED: "Where the prediction landed, datasets pooled",
}


def count_col(tier: str, label: str | None = None) -> str:
    """The `accuracy.csv` count column for a tier, or a tier x cell label."""
    if tier == FAILED:
        return "n_failed"
    return f"n_{tier}" if label is None else f"n_{tier}_cell_{label}"


def segments() -> list[tuple[str, str | None]]:
    """Every drawn segment bottom-up: `(tier, cell label)`, `(failed, None)` last."""
    return [(t, lab) for t in TIERS for lab in C.CELL_LABELS] + [(FAILED, None)]


def _add_shares(table: pd.DataFrame) -> pd.DataFrame:
    """Exact shares -- the drawn geometry. Rounding lives in labels and sort keys."""
    for tier, lab in segments():
        col = count_col(tier, lab)
        table[f"share_{col}"] = table[col] / table["n_tgs"]
    for tier in TIERS:
        table[f"share_{count_col(tier)}"] = table[count_col(tier)] / table["n_tgs"]
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


#: The sort key, all descending: cumulative tolerance down the ring ladder.
_RANK_KEYS: tuple[str, ...] = ("within_ring0", "within_ring1", "within_ring2", "within_beyond")


def _with_rank_keys(table: pd.DataFrame) -> pd.DataFrame:
    """Cumulative counts, divided and rounded **once**."""
    out = table.copy()
    running = 0
    for key, tier in zip(_RANK_KEYS, TIERS):
        running = running + out[count_col(tier)]
        out[key] = (running / out["n_tgs"]).round(ACCURACY_DECIMALS)
    return out


def _sorted_methods(frame: pd.DataFrame, tiebreak: dict[str, int]) -> list[str]:
    f = frame.assign(_tie=frame["method"].map(tiebreak).fillna(len(tiebreak)))
    return f.sort_values([*_RANK_KEYS, "_tie"], ascending=[False] * len(_RANK_KEYS) + [True])[
        "method"
    ].tolist()


def method_order(table: pd.DataFrame) -> list[str]:
    """The pooled order across datasets: legend, manifest, last tiebreak."""
    means = _with_rank_keys(table).groupby("method", as_index=False)[list(_RANK_KEYS)].mean()
    return _sorted_methods(means, {m: i for i, m in enumerate(sorted(means["method"]))})


def panel_order(table: pd.DataFrame, dataset: str) -> list[str]:
    """One dataset's methods down the tolerance ladder. Each panel ranks itself."""
    pooled = {m: i for i, m in enumerate(method_order(table))}
    keyed = _with_rank_keys(table)
    return _sorted_methods(keyed[keyed["dataset"] == dataset], pooled)


def _contrast_ink(face: str) -> str:
    """White or ink by the fill's luminance -- for labels and stripes alike."""
    r, g, b = (int(face[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.55 else _INK


def _draw_bar(ax, x: float, row: pd.Series, nd: int) -> None:
    """One method's stack: per tier, the cell-label sub-segments, white
    separators between them, then the tier border; every sub-segment labelled.
    Then the tier rail to its right."""
    half = _BAR_FRAC / 2
    tiers: list[tuple[str, float, float]] = []
    bottom = 0.0
    for tier in STACK:
        face = TIER_INK[tier]
        ink = _contrast_ink(face)
        tier_bottom = bottom
        parts = [None] if tier == FAILED else list(C.CELL_LABELS)
        drawn: list[tuple[float, float]] = []
        for lab in parts:
            v = float(row[f"share_{count_col(tier, lab)}"])
            if v > 0:
                ax.bar(
                    x, v, bottom=bottom, width=_BAR_FRAC,
                    facecolor=face,
                    hatch=CELL_HATCH[lab] if lab else "",
                    # The hatch takes the edge colour; linewidth 0 keeps the
                    # sub-segment unstroked -- separators and border are drawn
                    # as their own marks, so no segment gains apparent width.
                    edgecolor=ink, linewidth=0, zorder=3,
                )
                drawn.append((bottom, v))
            bottom += v
        # White separators between cell labels: interior boundaries only, as
        # lines across the bar rather than strokes around each sub-segment.
        for seg_bottom, _ in drawn[1:]:
            ax.hlines(
                seg_bottom, x - half, x + half, colors=_SURFACE,
                linewidth=SEPARATOR_PT, zorder=3.5,
            )
        height = bottom - tier_bottom
        if height > 0:
            ax.bar(
                x, height, bottom=tier_bottom, width=_BAR_FRAC,
                facecolor="none", edgecolor=TIER_EDGE, linewidth=TIER_EDGE_PT, zorder=4,
            )
        for seg_bottom, v in drawn:
            if v >= _LABEL_FLOOR:
                ax.text(
                    x, seg_bottom + v / 2, f"{round(v, nd) * 100:.0f}%",
                    ha="center", va="center", fontsize=7.5, color=ink, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.12", facecolor=face, edgecolor="none"),
                )
        if tier != FAILED:
            tiers.append((tier, tier_bottom, height))
    _draw_rail(ax, x, tiers, nd)


def _spread(ys: list[float], gap: float) -> list[float]:
    """Nudge sorted label positions apart to at least `gap`, inside [0, 1]."""
    out = list(ys)
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + gap)
    overflow = out[-1] - (1 - gap / 2) if out else 0
    if overflow > 0:
        out = [y - overflow for y in out]
        for i in range(len(out) - 2, -1, -1):
            out[i] = min(out[i], out[i + 1] - gap)
    return out


def _draw_rail(ax, x: float, tiers: list[tuple[str, float, float]], nd: int) -> None:
    """The tier rail right of the bar, with each tier's total beside it.

    "No answer" has no rail: it has no cell-label breakdown, so its in-bar
    label already is its total. Totals are text ink, not the tier colour --
    the rail segment beside each one carries the identity.
    """
    rail_x = x + _BAR_FRAC / 2 + _RAIL_GAP + _RAIL_W / 2
    shown = [(t, b, h) for t, b, h in tiers if h > 0]
    for tier, bottom, height in shown:
        trim = min(_RAIL_SEG_GAP / 2, height / 4)
        ax.bar(
            rail_x, height - 2 * trim, bottom=bottom + trim, width=_RAIL_W,
            facecolor=TIER_INK[tier], edgecolor=TIER_EDGE, linewidth=0.8, zorder=3,
        )
    labelled = [(t, b, h) for t, b, h in shown if round(h, nd) > 0]
    ys = _spread([b + h / 2 for _, b, h in labelled], _RAIL_LABEL_GAP)
    for (tier, bottom, height), y in zip(labelled, ys):
        ax.text(
            rail_x + _RAIL_W / 2 + _RAIL_TEXT_PAD, y, f"{round(height, nd) * 100:.0f}%",
            ha="left", va="center", fontsize=7.5, fontweight="bold", color=_INK, zorder=5,
        )


def _legend_handles():
    from matplotlib.patches import Patch

    tiers = [
        Patch(facecolor=TIER_INK[t], edgecolor=TIER_EDGE, linewidth=0.8, label=TIER_LABELS[t])
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
    """One panel per dataset, one bar per method."""
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
            ax.set_xticklabels(
                [method_label(m) for m in ranked], rotation=35, ha="right", fontsize=9, color=_INK_2
            )
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
        inset = (fig_w - panels_w) / 2 / fig_w
        fig.tight_layout(rect=(inset, 0, 1 - inset, 0.885))

        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / (png_name or NAMES[COMPARE][0].format(slug=grid_slug(nside)))
        fig.savefig(png, dpi=dpi, facecolor=_SURFACE)
        plt.close(fig)
    return png


def csv_columns() -> list[str]:
    """The CSV twin's columns, in order."""
    counts = [count_col(t, lab) for t, lab in segments()]
    tiers = [count_col(t) for t in TIERS]
    return [
        "run_id", "dataset", "method", "n_tgs", "n_solved", *tiers, *counts,
        *[f"share_{c}" for c in (*tiers, *counts)],
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
        "stack": [count_col(t, lab) for t, lab in segments()],
        "encoding": {
            "colour": {"ring tier": TIER_INK, "labels": TIER_LABELS},
            "stripe": {"cell label": CELL_HATCH, "labels": CELL_LEGEND},
            "tier_border": f"{TIER_EDGE} {TIER_EDGE_PT} pt around every tier, 'no answer' included",
            "labels": "each tier x cell-label share at whole percent, above the label floor",
            "separator": f"white {SEPARATOR_PT} pt between cell labels inside a tier",
            "rail": "per-tier strip right of each bar, tier fill and border; tier total beside it in ink",
        },
        "order_note": (
            "each panel ranked by its own cumulative (ring0, <=ring1, <=ring2, answered) "
            "shares at whole percent, descending; `methods` is the pooled order"
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
