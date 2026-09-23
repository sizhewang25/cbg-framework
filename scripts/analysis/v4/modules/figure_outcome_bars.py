"""Outcome composition bars, one figure per HEALPix rung.

Each bar is one method's **entire** target set partitioned into where its
prediction landed relative to the truth's cell:

    in the cell | one ring out | two rings out | further | never answered

So accuracy is the bottom segment, the failure share is the top, and everything
between is visible rather than implied. v3's three segments (correct / wrong /
failed) collapsed the middle three into "wrong", which is exactly the
information the ring metric exists to expose.

## One figure per rung, not one figure with a resolution axis

The rung *is* the tolerance the reader is choosing. Putting four rungs in one
panel would ask them to compare five methods and four tolerances at once, and
the honest comparison — "at 51 km cells, who wins?" — would be buried. Each
figure states its cell size in the title, and `accuracy_by_resolution.csv`
carries the curve for anyone who wants it as a line.

## Colour is a sequential ramp, and it was computed rather than chosen

The segments are ordered — in the cell beats one ring out beats two, and "no
answer" is the worst outcome of all — so the stack is a **single-hue ramp,
light to dark**, ending in charcoal. Lightness therefore falls monotonically
bottom to top (relative luminance 0.437, 0.253, 0.127, 0.056, 0.024), and
lightness is the one separator that survives every colour-vision deficiency
*and* greyscale print at once.

Three designs were **rejected by measurement**, not taste, using the dataviz
skill's `validate_palette.js`:

* **Green ramp + red for "further out"** — the obvious good-to-bad reading.
  Red against the ramp's mid-green is Delta E **1.8 under protanopia**: a
  protanope cannot tell the second ring from a total miss.
* **A mid grey for "no answer"** — the conventional neutral. Mid grey is
  exactly where green lands under deuteranopia, and every mid grey tested
  collided with some ramp step at Delta E **4.5-4.7**.
* **A near-white grey**, which clears CVD but sits at 1.29:1 against the
  surface and effectively vanishes.

Charcoal `#2b2b29` is what survives: Delta E **8.8** from the darkest green
under deuteranopia, adequate contrast, and it extends the ramp rather than
interrupting it. Making the grey the *darkest* step is what buys the
separation, and it also reads correctly — the bar gets darker as the outcome
gets worse.

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

#: Single-hue ordinal ramp, **light = most precise**, darkening as the
#: prediction lands further out, and ending in a charcoal grey for "no answer".
#:
#: The whole stack is monotone in lightness bottom to top (0.437, 0.253, 0.127,
#: 0.056, 0.024 relative luminance), which is the one separator that survives
#: every colour-vision deficiency *and* greyscale print at once. That the grey
#: is the darkest step rather than a mid tone is what buys it: a mid grey is
#: where green lands under deuteranopia, and every mid grey tested collided
#: with some ramp step at dE 4.5-4.7. Charcoal clears the darkest green at
#: dE 8.8 (deutan) — measured with the dataviz validator, not judged.
#:
#: Rejected by the same measurement: a green ramp plus **red** for "further
#: out". Red against the ramp's mid-green is dE 1.8 under protanopia, so a
#: protanope could not separate "two rings out" from a total miss.
SEGMENT_INK = {
    "n_ring0": "#79bf9b",
    "n_ring1": "#3f9a6f",
    "n_ring2": "#17724a",
    "n_beyond": "#0b4d2c",
    "n_failed": "#2b2b29",
}

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
    """Best mean `in the cell` share first, so the panels share one x order and
    a reader compares the same column across datasets."""
    means = table.groupby("method")["share_n_ring0"].mean()
    return means.sort_values(ascending=False).index.tolist()


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

    `order` is passed in rather than derived here so the whole rung set shares
    one x order. Ranking each rung independently moved a method between slots
    from figure to figure -- Spotter sat 5th at nside 128 and 4th at nside 16 --
    which is exactly the comparison the four figures exist to support.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    datasets = sorted(table["dataset"].unique())
    order = list(order) if order else method_order(table)
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
                edgecolor=_SURFACE,
                # The 2 px surface gap between touching segments, one width up
                # the whole stack. No hatch on any outcome: that channel is
                # reserved for the mesh-vs-weighted distinction.
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
    # even before the in-place labels.
    handles = [
        Patch(
            facecolor=SEGMENT_INK[s],
            edgecolor=_SURFACE,
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
    # One x order for the whole set, taken from the FINEST rung: that is the
    # strictest test, so it ranks the methods on the hardest question rather
    # than on whichever tolerance happens to be plotted.
    shared_order = method_order(tables[rungs[0]])
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
        png = render(table, nside, out_dir, order=shared_order)
        (out_dir / FIGURE_MANIFEST.format(slug=slug)).write_text(
            json.dumps(
                {
                    "figure": png.name,
                    "csv": FIGURE_CSV.format(slug=slug),
                    "grid": H.describe(nside),
                    "runs": run_ids,
                    "methods": shared_order,
                    "method_order_note": (
                        "ranked once by 'in the cell' at the finest rung and "
                        "reused for every rung, so a method keeps its x slot "
                        "across the figure set"
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
