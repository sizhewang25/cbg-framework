"""Method -> hue, fixed at import by identity; the one palette every figure uses.

`method_colors` is the entry point for anything that draws a published method,
`method_family` groups the ones that share a hue family (the two Octant
variants), and `SERIES_HUES` is for categorical series that are *not* methods.

Lives here rather than in `pareto.py` (where it was written) because it is keyed
on `label_for` and `PUBLISHED_METHODS`, and because every module that draws a
variant needs it. `pareto` imports it back; the dependency only runs one way.
"""

from __future__ import annotations

from scripts.analysis.v3.modules.classify import SHORTEST_PING
from scripts.analysis.v3.modules.diagram.common.labels import (
    PUBLISHED_METHODS,
    label_for,
)


#: Method -> hue, by display label, assigned by **identity** and never by rank
#: in the current selection, so `--method` cannot repaint the survivors.
#:
#: Designed rather than dealt. The previous palette walked a six-hue tuple in
#: `PUBLISHED_METHODS` order, which put SoI on aqua and Octant-Hull on green —
#: two greens for two unrelated methods — while the two Octant variants landed
#: on unrelated hues. Here hue carries **family** and lightness carries the
#: variant within it: Octant-Hull is a mid green, Octant-Spline a light green
#: of the same hue, and SoI moves to violet, the one hue nothing else is near.
#: Shortest-Ping, Vanilla and Spotter keep their hues, so every figure that
#: quoted them stays readable against its old self.
#:
#: Validated with the dataviz skill's `validate_palette.js --mode light
#: --surface #ffffff --pairs all` — all pairs, because a CDF, a scatter or a
#: frontier puts every method on one axis at once:
#:
#: * CVD separation PASSES: worst dE 8.6 (Spotter vs Octant-Hull, deutan), up
#:   from 6.9 in the 6-8 band that the old palette only met with secondary
#:   encoding. Tritan (reported, not gated) is 6.3, down from 7.6.
#: * Normal-vision floor 16.3 (Shortest-Ping vs SoI), over the hard 15.
#: * The Octant pair is itself 30+ dE apart in every simulation — a lightness
#:   step survives colour-vision deficiency where a hue step would not, which
#:   is what lets the two share a family and stay separable.
#: * Every hue is >= dE 8.7 from `_C_OTHER` / `_C_MUTED` grey under every
#:   simulation, so no method reads as the "other" bucket or as chrome. A
#:   teal Octant pair scored higher on CVD but its dark step was dE 1.6 from
#:   that grey under deuteranopia — the reason green won.
#: * Contrast WARN: Vanilla's yellow (2.17:1) and Octant-Spline's light green
#:   (2.02:1) sit below 3:1 on white. That obliges visible labels or a table
#:   view; every figure drawing methods carries a legend and a CSV twin. No
#:   light family step can clear 3:1 and stay separable from its dark sibling.
METHOD_HUES: dict[str, str] = {
    label_for(SHORTEST_PING): "#2a78d6",  # blue
    label_for("million_scale_cbg"): "#4a3aa7",  # violet
    label_for("vanilla_cbg"): "#eda100",  # yellow
    label_for("octant_cbg_hull"): "#17890b",  # green, mid
    label_for("octant_cbg_spl"): "#58cd78",  # green, light
    label_for("spotter_cbg"): "#e34948",  # red
}

#: Label -> family. Two methods share a family exactly when they share a hue
#: family in `METHOD_HUES`, so a figure that wants to group them (a legend
#: bracket, a shared band, one hatch) reads it here instead of re-deriving it
#: from hexes.
METHOD_FAMILIES: dict[str, str] = {
    label_for(SHORTEST_PING): "shortest-ping",
    label_for("million_scale_cbg"): "soi",
    label_for("vanilla_cbg"): "vanilla",
    label_for("octant_cbg_hull"): "octant",
    label_for("octant_cbg_spl"): "octant",
    label_for("spotter_cbg"): "spotter",
}

#: The method hues as a tuple, in `PUBLISHED_METHODS` order. Kept for the
#: callers that ask "is this colour a method's?" (the outcome-bar and
#: proximity figures assert their own inks are not). **Not** a sequence to
#: deal from: slots 4 and 5 are one family, so cycling it over unrelated
#: series would pair two of them visually. Use `SERIES_HUES` for that.
_VARIANT_HUES: tuple[str, ...] = tuple(
    METHOD_HUES[label_for(m)] for m in PUBLISHED_METHODS
)

#: A general-purpose categorical order for series that are **not** methods —
#: ASes, VP groups, fit families — where every slot must look unrelated to
#: every other. Six distinct hues, validated `--pairs all` on white (worst CVD
#: dE 6.9, in the band that needs secondary encoding: the legend). It shares
#: hexes with `METHOD_HUES`, so never use both in one figure.
SERIES_HUES: tuple[str, ...] = (
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
    """Display label -> hue, checked once at import against `PUBLISHED_METHODS`.

    Keyed on the *label* rather than the combo id so `octant_cbg_spl` and
    `octant_cbg` land on one hue: they are one paper variant whose id differs
    per run, which is why `LABELS` already maps both onto "Octant-Spline CBG".

    The mapping is explicit (`METHOD_HUES`), so there is no order to get wrong;
    what can still drift is the *set*. A published variant with no hue would
    silently fall into the grey bucket, and a hue for a non-published arm would
    hand an ablation a published variant's standing — `spotter_hybrid_cbg`
    stays in `_C_OTHER` until it is added here and the palette re-validated.
    """
    published = {label_for(m) for m in PUBLISHED_METHODS}
    if published != set(METHOD_HUES) or published != set(METHOD_FAMILIES):
        raise RuntimeError(
            f"palette covers {sorted(METHOD_HUES)} but PUBLISHED_METHODS labels "
            f"are {sorted(published)}; add or remove the hue and re-run "
            "validate_palette.js --pairs all before changing either"
        )
    return dict(METHOD_HUES)


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


def method_family(method: str) -> str | None:
    """The method's hue family (`"octant"` for both Octant variants), or `None`
    for anything outside the published set — which is also what folds it into
    `_C_OTHER`."""
    return METHOD_FAMILIES.get(label_for(method))
