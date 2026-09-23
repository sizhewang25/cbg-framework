"""Outcome composition bars, one figure per HEALPix rung.

Each bar is one method's **entire** target set partitioned into where its
prediction landed relative to the truth's cell:

    in the cell | one ring out | two rings out | further | never answered

So accuracy is the bottom segment, the failure share is the top, and everything
between is visible rather than implied. v3's three segments (correct / wrong /
failed) collapsed the middle three into "wrong", which is exactly the
information the ring metric exists to expose.

## Each panel ranks itself

Bars within a panel are ordered by that dataset's own `in the cell` share,
descending, so a panel reads as a leaderboard. The cost is that a method does
not keep one x slot across panels — comparing a method between datasets means
reading its label rather than its position. That is bought deliberately,
because the orders genuinely differ: Octant-Spline leads as01 while
Octant-Hull leads as02 and as03, and a single pooled order would hide it.

## One figure per rung, not one figure with a resolution axis

The rung *is* the tolerance the reader is choosing. Putting four rungs in one
panel would ask them to compare five methods and four tolerances at once, and
the honest comparison — "at 51 km cells, who wins?" — would be buried. Each
figure states its cell size in the title, and `accuracy_by_resolution.csv`
carries the curve for anyone who wants it as a line.

## Colour is a sequential ramp, and it was computed rather than chosen

The four *placed* outcomes are ordered — in the cell beats one ring out beats
two — so they take a **single-hue ramp, light to dark**: relative luminance
0.437, 0.253, 0.127, 0.056. Lightness is the one separator that survives every
colour-vision deficiency *and* greyscale print at once.

"No answer" is a **light grey and sits outside that ramp**, which is the
honest placement: it is not a worse *placement*, it is the absence of one, so
giving it a rank on the precision ramp would claim an ordering it does not
have. Light grey is the conventional reading for absent, and it is legible
here precisely because the ramp runs light-to-dark and so vacated the light
end.

Rejected by measurement, not taste, using the dataviz skill's
`validate_palette.js`:

* **Green ramp + red for "further out"** — the obvious good-to-bad reading.
  Red against the ramp's mid-green is Delta E **1.8 under protanopia**: a
  protanope cannot tell the second ring from a total miss.
* **A mid grey for "no answer"** — the conventional neutral, and the worst
  option here. Mid grey is exactly where green lands under deuteranopia;
  every one tested collided with a ramp step at Delta E **2.6-5.0**.

Among light greys, separation from the lightest green and contrast against the
surface pull opposite ways, so the choice was made on the measured curve
(see `SEGMENT_INK`): `#d8d7cf` keeps Delta E **10.8** — real headroom over the
8 threshold — at 1.44:1. The weak contrast is covered by a hairline edge on
that slot alone, plus the in-place label and the CSV twin.

**No hatch on any outcome.** That channel is reserved for the traffic-weighted
arm drawn beside a mesh bar (`WEIGHTED_HATCH`), and spending it on an outcome
would leave the mesh-vs-weighted distinction with nowhere to go.

The ramp's light end is 2.10:1 against the surface, which the skill flags as
requiring relief. Both reliefs ship: every segment above a visibility floor is
labelled in place, and the CSV twin beside the PNG is the table view.

## Not interactive

A static PNG for a paper, like v3's. The skill's hover layer applies to HTML
and SVG charts; there is no hover channel here, which is why the direct labels
and the CSV are load-bearing rather than decorative.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v4.modules import classify as C
from scripts.analysis.v4.modules import healpix as H
from scripts.analysis.v4.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    MissingArtifactError,
    RunPaths,
    grid_slug,
)

#: Bottom of the stack to top, and the reading order of the claim.
SEGMENTS: tuple[str, ...] = C.OUTCOME_COUNTS

SEGMENT_LABELS = {
    "n_ring0": "in the cell",
    "n_ring1": "1 ring out",
    "n_ring2": "2 rings out",
    "n_beyond": "further out",
    "n_failed": "no answer",
}

#: Single-hue ordinal ramp for the four *placed* outcomes, **light = most
#: precise**, darkening as the prediction lands further out. "No answer" is a
#: light grey and deliberately sits **outside** that ramp.
#:
#: The four greens are monotone in lightness (relative luminance 0.437, 0.253,
#: 0.127, 0.056), which is the separator that survives every colour-vision
#: deficiency and greyscale print at once. The grey does **not** extend that
#: ordering — it is lighter than all four — and that is the point: "no answer"
#: is not a worse placement, it is the absence of one, so putting it on the
#: precision ramp would claim a rank it does not have. Light grey is the
#: conventional reading for absent, and it is legible here because the ramp
#: vacated the light end.
#:
#: The trade-off is measured, not guessed. Against the lightest green, grey
#: separation and surface contrast pull opposite ways:
#:
#:     #e1e0d9  dE 13.5 (protan)   1.32:1 vs white
#:     #d8d7cf  dE 10.8            1.44:1      <- chosen
#:     #cfcec6  dE  8.1            1.58:1
#:     #b5b4ad  dE  2.6   FAIL     2.08:1
#:
#: `#d8d7cf` keeps real headroom over the dE 8 threshold while staying dark
#: enough to read; the weak contrast is covered by `_FAILED_EDGE` plus the
#: in-place label and the CSV twin.
#:
#: Rejected by the same measurement: a ramp plus **red** for "further out" —
#: red against the mid-green is dE 1.8 under protanopia, so a protanope could
#: not separate "two rings out" from a total miss. And any **mid** grey: that
#: is exactly where green lands under deuteranopia (dE 2.6-5.0).
SEGMENT_INK = {
    "n_ring0": "#79bf9b",
    "n_ring1": "#3f9a6f",
    "n_ring2": "#17724a",
    "n_beyond": "#0b4d2c",
    "n_failed": "#d8d7cf",
}

#: A hairline edge on the grey slot only. At 1.44:1 the fill alone does not
#: delineate against a white surface, and the hatch channel is reserved, so the
#: definition comes from an edge instead. Not a stripe.
_FAILED_EDGE = "#b5b4ad"

#: Reserved for the traffic-weighted arm drawn beside a mesh bar, so it must
#: not be spent on an outcome. None exists yet; the constant records the claim
#: on the channel so a later figure does not take it for something else.
WEIGHTED_HATCH = "//"

#: Display names. Shared vocabulary with v3's figures on purpose: the metric
#: changed, the methods did not, and inventing a second set of names would make
#: the two layers look like they scored different things.
METHOD_LABELS = {
    "shortest_ping": "Shortest-Ping",
    "million_scale_cbg": "SoI",
    "vanilla_cbg": "Vanilla",
    "octant_cbg_hull": "Octant-Hull",
    "octant_cbg_spl": "Octant-Spline",
    "spotter_cbg": "Spotter",
    "spotter_hybrid_cbg": "Spotter-Hybrid",
}


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


_SURFACE = "#ffffff"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"

#: Bar thickness as a share of the slot. The skill caps bars at 24 px and says
#: to leave the slot's remainder as air rather than filling it.
_BAR_FRAC = 0.62

#: A 2 px surface gap separates touching segments, one consistent width up the
#: stack. Expressed in points because matplotlib strokes in points; at the
#: figure's 150 dpi, 1.0 pt is ~2 px.
_GAP_PT = 1.0

#: Below this share a segment is too thin to hold its own label without the
#: text overflowing the mark. The skill's rule for an interior segment with no
#: free end is to drop the label and let the legend and table carry it, which is
#: what happens here — the value is never lost, only moved.
_LABEL_FLOOR = 0.055

FIGURE_PNG = "outcome_bars.{slug}.png"
FIGURE_CSV = "outcome_bars.{slug}.csv"
FIGURE_MANIFEST = "outcome_bars.{slug}.manifest.json"

#: Where cross-dataset figures land — keyed by the dataset set, so a two-run
#: comparison cannot overwrite a three-run one.
CROSS_KIND = "cls-accuracy"


def dataset_slug(run_ids: list[str]) -> str:
    """`as01-...-mesh, as02-...` -> `as01+as02+as03`.

    Keyed on the datasets rather than on a count, so the directory names the
    comparison it holds.
    """
    heads = sorted({r.split("-")[0] for r in run_ids})
    return "+".join(heads)


def cross_dir(run_ids: list[str], *, analysis_root: Path | None = None) -> Path:
    out = (
        (analysis_root or DEFAULT_ANALYSIS_ROOT)
        / "_cross"
        / CROSS_KIND
        / dataset_slug(run_ids)
    )
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_rung(run: RunPaths, nside: int, *, analysis_root: Path | None = None):
    """One run's `accuracy.csv` at one rung."""
    path = run.cls_accuracy_dir(nside, root=analysis_root) / C.ACCURACY_CSV
    if not path.exists():
        raise MissingArtifactError(
            f"{path} missing; run `classify --run-id {run.run_id}` first"
        )
    df = pd.read_csv(path)
    df.insert(0, "run_id", run.run_id)
    df.insert(1, "dataset", run.run_id.split("-")[0])
    return df


def build_table(
    runs: list[RunPaths],
    nside: int,
    *,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
) -> pd.DataFrame:
    """Long frame: one row per (dataset, method) with shares that sum to 1."""
    frames = [load_rung(r, nside, analysis_root=analysis_root) for r in runs]
    table = pd.concat(frames, ignore_index=True)
    if methods:
        table = table[table["method"].isin(methods)]
        if table.empty:
            raise ValueError(f"none of {methods} are scored at nside={nside}")
    C.guard_partition(table)
    for seg in SEGMENTS:
        table[f"share_{seg}"] = table[seg] / table["n_targets"]
    return table


def method_order(table: pd.DataFrame) -> list[str]:
    """Best mean `in the cell` share first — the pooled order across datasets.

    Used for the legend and the manifest. Each *panel* ranks itself; see
    `panel_order`.
    """
    means = table.groupby("method")["share_n_ring0"].mean()
    return means.sort_values(ascending=False).index.tolist()


def panel_order(table: pd.DataFrame, dataset: str) -> list[str]:
    """One dataset's methods, best `in the cell` share first.

    **Each panel ranks itself**, so a bar's height decreases left to right
    within every panel and the panel reads as a leaderboard for that dataset.

    The cost is deliberate and worth naming: a method does *not* keep the same
    x slot across panels, so comparing one method between datasets means
    reading its label rather than its position. The ranking within a dataset is
    the question these bars answer, and on these three datasets the order
    genuinely differs -- Octant-Spline leads as01 while Octant-Hull leads as02
    and as03 -- so a single pooled order would hide the thing worth seeing.

    Ties break on the pooled order, so two methods level on a dataset still
    appear in a stable, reproducible sequence rather than whatever order the
    rows happened to arrive in.
    """
    pooled = {m: i for i, m in enumerate(method_order(table))}
    sub = table[table["dataset"] == dataset]
    return (
        sub.assign(_tie=sub["method"].map(pooled))
        .sort_values(["share_n_ring0", "_tie"], ascending=[False, True])["method"]
        .tolist()
    )


def _label_ink(face: str) -> str:
    """White or ink by the fill's luminance, so text inside a segment always
    clears contrast — the one place a label may sit on a colour."""
    r, g, b = (int(face[i : i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if lum < 0.55 else _INK


def render(
    table: pd.DataFrame,
    nside: int,
    out_dir: Path,
    *,
    order: list[str] | None = None,
    dpi: int = 150,
) -> Path:
    """One panel per dataset, one bar per method, segments stacked bottom-up.

    Each panel is ranked by its own `in the cell` share, descending, so every
    panel reads as that dataset's leaderboard. `order` overrides that with a
    single shared sequence when a caller wants position comparable across
    panels instead.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    datasets = sorted(table["dataset"].unique())
    pinned = list(order) if order else None
    slug = grid_slug(nside)

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(4.6 * len(datasets), 5.4),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    fig.patch.set_facecolor(_SURFACE)

    for ax, ds in zip(axes, datasets):
        order = pinned or panel_order(table, ds)
        sub = table[table["dataset"] == ds].set_index("method")
        ax.set_facecolor(_SURFACE)
        xs = np.arange(len(order))
        bottoms = np.zeros(len(order))

        for seg in SEGMENTS:
            vals = np.array(
                [
                    sub.loc[m, f"share_{seg}"] if m in sub.index else 0.0
                    for m in order
                ]
            )
            face = SEGMENT_INK[seg]
            ax.bar(
                xs,
                vals,
                bottom=bottoms,
                width=_BAR_FRAC,
                facecolor=face,
                # The 2 px surface gap between touching segments, one width up
                # the whole stack. No hatch on any outcome: that channel is
                # reserved for the mesh-vs-weighted distinction. The grey slot
                # takes a visible edge instead, because at 1.44:1 its fill does
                # not delineate itself against the surface.
                edgecolor=_FAILED_EDGE if seg == "n_failed" else _SURFACE,
                linewidth=_GAP_PT,
                zorder=3,
            )
            for x, v, b in zip(xs, vals, bottoms):
                if v >= _LABEL_FLOOR:
                    ax.text(
                        x,
                        b + v / 2,
                        f"{v * 100:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=_label_ink(face),
                        zorder=4,
                    )
            bottoms += vals

        n = int(sub["n_targets"].iloc[0]) if len(sub) else 0
        ax.set_title(f"{ds.upper()}  ·  n={n}", fontsize=11, color=_INK, pad=10)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [method_label(m) for m in order],
            rotation=35,
            ha="right",
            fontsize=9,
            color=_INK_2,
        )
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_yticklabels([f"{int(v * 100)}%" for v in np.arange(0, 1.01, 0.2)])
        ax.grid(True, axis="y", color=_GRID, linewidth=0.8, linestyle="-", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # `sharey` already hides the tick labels on the trailing panels, and a
        # spine with no labels beside it reads as a stray vertical rule.
        ax.spines["left"].set_visible(ax is axes[0])
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_AXIS)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=_MUTED, labelsize=9)
        if ax is not axes[0]:
            # `sharey` drops the labels but leaves the tick marks, which then
            # read as a stray dashed rule with nothing to measure against.
            ax.tick_params(axis="y", length=0)

    axes[0].set_ylabel("Share of targets", fontsize=11, color=_INK_2)

    # Legend always present: five segments, so identity is never colour-alone
    # even before the in-place labels. Segment order, not method order — the
    # panels each rank themselves.
    handles = [
        Patch(
            facecolor=SEGMENT_INK[s],
            edgecolor=_FAILED_EDGE if s == "n_failed" else _SURFACE,
            linewidth=0.8,
            label=SEGMENT_LABELS[s],
        )
        for s in SEGMENTS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(SEGMENTS),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.965),
        labelcolor=_INK_2,
    )
    fig.suptitle(
        f"Where the prediction landed  ·  HEALPix nside={nside} "
        f"({H.nominal_cell_km(nside):.0f} km cells)",
        fontsize=13,
        color=_INK,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / FIGURE_PNG.format(slug=slug)
    fig.savefig(png, dpi=dpi, facecolor=_SURFACE)
    plt.close(fig)
    return png


def build_for_runs(
    runs: list[RunPaths],
    *,
    nsides: tuple[int, ...] = H.NSIDE_LADDER,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
) -> list[Path]:
    """One figure, CSV twin and manifest per rung. Returns the PNG paths."""
    run_ids = [r.run_id for r in runs]
    out_dir = cross_dir(run_ids, analysis_root=analysis_root)
    rungs = sorted({H.validate_nside(n) for n in nsides}, reverse=True)
    tables = {
        n: build_table(runs, n, methods=methods, analysis_root=analysis_root)
        for n in rungs
    }
    written: list[Path] = []
    for nside in rungs:
        table = tables[nside]
        slug = grid_slug(nside)
        keep = [
            "run_id", "dataset", "method", "n_targets", *SEGMENTS,
            *[f"share_{s}" for s in SEGMENTS],
            "accuracy_ring0", "accuracy_nearest_seed_retired",
            "error_km_p50", "error_km_p90",
        ]
        table[[c for c in keep if c in table.columns]].to_csv(
            out_dir / FIGURE_CSV.format(slug=slug), index=False
        )
        png = render(table, nside, out_dir)
        (out_dir / FIGURE_MANIFEST.format(slug=slug)).write_text(
            json.dumps(
                {
                    "figure": png.name,
                    "csv": FIGURE_CSV.format(slug=slug),
                    "grid": H.describe(nside),
                    "runs": run_ids,
                    "methods": method_order(table),
                    "panel_order": {
                        ds: panel_order(table, ds)
                        for ds in sorted(table["dataset"].unique())
                    },
                    "method_order_note": (
                        "`methods` is the pooled order across datasets; each "
                        "panel is ranked by its OWN 'in the cell' share "
                        "descending, so a method does not keep one x slot "
                        "across panels"
                    ),
                    "segments": list(SEGMENTS),
                    "segment_labels": SEGMENT_LABELS,
                    "palette": {
                        "ink": SEGMENT_INK,
                        "validated": (
                            "dataviz validate_palette.js --ordinal: monotone "
                            "lightness, adjacent dL >= 0.06, light end 2.10:1, "
                            "hue spread 5deg — all pass. A green+red scheme was "
                            "rejected (red vs mid-green is dE 1.8 under "
                            "protanopia) and so was a grey fifth fill (dE "
                            "1.6-4.5 against the ramp under deuteranopia); "
                            "'never answered' is therefore unfilled."
                        ),
                    },
                    "weighted_arm": (
                        "absent by design — no traffic-weighted run exists, and "
                        "a placeholder would be fabricated data"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        written.append(png)
    return written
