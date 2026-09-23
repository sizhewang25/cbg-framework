"""Variant -> hue, fixed at import from a constant order.

Lives here rather than in `pareto.py` (where it was written) because it is keyed
on `label_for` and `PUBLISHED_METHODS`, and because every module that draws a
variant needs it. `pareto` imports it back; the dependency only runs one way.
"""

from __future__ import annotations

from scripts.analysis.v3.modules.diagram.common.labels import (
    PUBLISHED_METHODS,
    label_for,
)


#: Fixed variant -> hue, assigned by **identity** (`PUBLISHED_METHODS`) and
#: never by rank in the current selection, so `--method` cannot repaint the
#: survivors.
#:
#: Validated with the dataviz skill's `validate_palette.js` against the
#: reference 8-hue categorical theme, `--pairs all` on white — the right check
#: here, since every variant is visible at once. Every 6/7/8-slot prefix of that
#: theme FAILS (green vs orange is dE 3.2 under protanopia), and an exhaustive
#: search over its hues found exactly two passing 6-subsets; this is the better
#: one, worst dE 6.9 (deutan) / 7.6 (tritan). Orange is the hue that had to go.
#:
#: dE 6.9 sits in the 6-8 band that is legal *only* with secondary encoding.
#: That is satisfied three times over: each variant owns its own x column (they
#: never interleave spatially), the legend names every one, and the CSV is the
#: table view. Aqua/yellow/magenta are also below 3:1 on white (2.82/2.17/2.69),
#: a contrast WARN that obliges visible labels or a table view — the legend and
#: CSV again. No 6-subset of this theme clears 3:1 for all six (only five hues
#: do), so at six variants that is unavoidable rather than a shortcut; the 2 px
#: cost line and ringed >=8 px markers give each variant more ink than a dot.
_VARIANT_HUES: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

#: Past the palette's capacity a 9th series is never a generated hue — it folds
#: into one "other" bucket. as7018 carries 11 ablation arms on top of the five
#: published variants, and they belong in that bucket.
_C_OTHER = "#898781"

_C_GRID = "#e1e0d9"
#: Axis spines. Darker than `_C_GRID` so a spine reads as a boundary and a
#: gridline as background. Was defined locally in `pareto.py`; hoisted here when
#: `figure_error_cdf` became the second figure to need it.
_C_AXIS = "#c3c2b7"
_C_INK = "#0b0b0b"
_C_INK_2 = "#52514e"
_C_MUTED = "#898781"
_SURFACE = "#ffffff"


def _build_label_hues() -> dict[str, str]:
    """Display label -> hue, fixed once from `PUBLISHED_METHODS`.

    Keyed on the *label* rather than the combo id so `octant_cbg_spl` and
    `octant_cbg` land on one hue: they are one paper variant whose id differs
    per run, which is why `LABELS` already maps both onto "Octant-Spline CBG".
    Two hues would invent a distinction the runs do not contain. That aliasing
    is what lets `PUBLISHED_METHODS` name one id per variant and still colour
    either spelling.

    Driven by `PUBLISHED_METHODS` rather than `PREFERRED_ORDER` because the two
    answer different questions and only one of them is capacity-bounded.
    `PREFERRED_ORDER` is a sort key and grows whenever an arm needs a place to
    sort; the palette has exactly six validated hues. Reading the sort key here
    meant that adding `spotter_hybrid_cbg` to it consumed the sixth hue and
    pushed `spotter_cbg` -- a published variant -- into the grey bucket. The
    published set is fixed at six by the same argument that fixed the palette at
    six, so keying on it makes the capacities match by construction.

    An arm outside that set gets `_C_OTHER`. That is the documented behaviour
    for as7018's ablation arms and it applies to `spotter_hybrid_cbg` too:
    promoting it to a hue of its own needs a 7-hue re-validation first.
    """
    hues: dict[str, str] = {}
    for method in PUBLISHED_METHODS:
        label = label_for(method)
        if label in hues:
            continue
        if len(hues) < len(_VARIANT_HUES):
            hues[label] = _VARIANT_HUES[len(hues)]
    return hues


#: Computed once, at import, from a constant order — never from the methods
#: present in a given call. This is what makes colour stable under `--method`.
_LABEL_HUES: dict[str, str] = _build_label_hues()


def method_colors(methods) -> dict[str, str]:
    """Variant -> hue, stable under filtering.

    Each hue is pinned to a variant's *identity* via `_LABEL_HUES`, which is
    built from a fixed order at import time. Filtering the method pool with
    `--method` therefore cannot repaint the survivors — colour follows the
    entity, never its rank in the current selection, and the same variant is
    the same colour in every figure of a sweep.

    Anything `PUBLISHED_METHODS` does not name — as7018's 11 ablation arms,
    and `spotter_hybrid_cbg` — folds into the single `_C_OTHER` bucket rather
    than being handed a generated hue, because no palette distinguishes 17
    series and only six of these hues are validated.
    """
    return {m: _LABEL_HUES.get(label_for(m), _C_OTHER) for m in methods}
