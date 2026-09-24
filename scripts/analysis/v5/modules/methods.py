"""Method display names and hues. Ported unchanged from v4.

The names and hexes are v4's (and v3's) on purpose: the methods are the
same, so a reader moving between layers sees the same variant in the same
colour.

Colour is pinned to a method's **identity**, never to its rank in the current
selection, so `--method` cannot repaint the survivors and the same variant is
the same colour in every figure.
"""

from __future__ import annotations

from scripts.analysis.v5.modules.status import SHORTEST_PING

#: Display names. Shared vocabulary with v3's figures on purpose: the metric
#: changed, the methods did not, and inventing a second set of names would make
#: the two layers look like they scored different things.
#:
#: Shorter than v3's, which suffix every variant with " CBG". These labels are
#: written onto circles and under bars where the surrounding figure already
#: says everything but the baseline is a CBG variant, and "Octant-Spline CBG"
#: does not fit on a circle holding 20% of the population.
METHOD_LABELS: dict[str, str] = {
    SHORTEST_PING: "Shortest-Ping",
    "million_scale_cbg": "SoI",
    "vanilla_cbg": "Vanilla",
    "octant_cbg_hull": "Octant-Hull",
    # Both spellings of the one paper variant: the as0* runs name it
    # `octant_cbg_spl`, the as7018 run `octant_cbg`. Mapping them onto one
    # label is also what puts them on one hue below.
    "octant_cbg_spl": "Octant-Spline",
    "octant_cbg": "Octant-Spline",
    "spotter_cbg": "Spotter",
    "spotter_hybrid_cbg": "Spotter-Hybrid",
    # The density MTL on its previous H3 grid, preserved on disk by
    # `rename-combo` when it moved to HEALPix. In no config and not runnable;
    # `combo_ids` globs the output tree, so it is scored and plotted whenever
    # the backup is present. Naming it is what keeps the pair readable as a
    # pair -- without an entry `method_label` falls through to
    # "spotter h3 cbg".
    "spotter_h3_cbg": "Spotter (H3)",
}


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


#: Label -> hue. Keyed on the *label* rather than the combo id so
#: `octant_cbg_spl` and `octant_cbg` land on one colour: they are one paper
#: variant whose id differs per run.
#:
#: Carried over unchanged from v3's validated palette (`validate_palette.js
#: --mode light --surface #ffffff --pairs all`): hue carries family and
#: lightness carries the variant within it, so the two Octant arms share a
#: green and separate by 30+ dE under every colour-vision simulation. Worst
#: cross-method CVD separation is dE 8.6 (Spotter vs Octant-Hull, deutan) and
#: the normal-vision floor is 16.3. Re-validate before changing any hex.
#:
#: Deliberately identical to v3's hexes. The two layers score different things
#: and their numbers must not be mixed, but a reader moving between a v3 figure
#: and a v4 one is looking at the same six *methods*, and repainting them would
#: cost that recognition for nothing.
LABEL_HUES: dict[str, str] = {
    "Shortest-Ping": "#2a78d6",  # blue
    "SoI": "#4a3aa7",  # violet
    "Vanilla": "#eda100",  # yellow
    "Octant-Hull": "#17890b",  # green, mid
    "Octant-Spline": "#58cd78",  # green, light
    "Spotter": "#e34948",  # red
}

#: Anything `LABEL_HUES` does not name folds into one grey bucket rather than
#: being handed a generated hue: no palette separates a dozen series, and an
#: ablation arm should not be given a published variant's standing by accident.
OTHER_HUE = "#898781"


def method_colors(methods) -> dict[str, str]:
    """Method id -> hue, stable under filtering."""
    return {m: LABEL_HUES.get(method_label(m), OTHER_HUE) for m in methods}
