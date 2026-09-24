"""Method terms, labels and hues -- the one lookup table every v5 figure uses.

Methods are named by a **short term** everywhere in v5: axis labels, legends,
manifests and prose. `METHOD_TERMS` is the lookup table from term to full
name; `METHOD_LABELS` maps each benchmark combo id onto its term.

| term  | full name             | combo id(s)                       |
|-------|-----------------------|-----------------------------------|
| OCT-H | Octant-Hull CBG       | octant_cbg_hull                   |
| OCT-S | Octant-Spline CBG     | octant_cbg_spl, octant_cbg        |
| SOI   | Speed-of-Internet CBG | million_scale_cbg                 |
| S-P   | Shortest-Ping         | shortest_ping                     |
| SPO   | Spotter CBG           | spotter_cbg                       |
| VAN   | Vanilla CBG           | vanilla_cbg                       |

Colour is pinned to a method's **identity**, never to its rank in the current
selection, so `--method` cannot repaint the survivors and the same variant is
the same colour in every figure. The hexes are v4's (and v3's), re-keyed on the
terms.
"""

from __future__ import annotations

from scripts.analysis.v5.modules.status import SHORTEST_PING

#: Term -> full name. The lookup table figures and manifests cite.
METHOD_TERMS: dict[str, str] = {
    "OCT-H": "Octant-Hull CBG",
    "OCT-S": "Octant-Spline CBG",
    "SOI": "Speed-of-Internet CBG",
    "S-P": "Shortest-Ping",
    "SPO": "Spotter CBG",
    "VAN": "Vanilla CBG",
}

#: Combo id -> term.
METHOD_LABELS: dict[str, str] = {
    SHORTEST_PING: "S-P",
    "million_scale_cbg": "SOI",
    "vanilla_cbg": "VAN",
    "octant_cbg_hull": "OCT-H",
    # Both spellings of the one paper variant: the as0* runs name it
    # `octant_cbg_spl`, the as7018 run `octant_cbg`. One term, so one hue.
    "octant_cbg_spl": "OCT-S",
    "octant_cbg": "OCT-S",
    "spotter_cbg": "SPO",
    # Not published variants, and not in the lookup table: named so they stay
    # readable when present on disk, without being given a published term.
    "spotter_hybrid_cbg": "SPO-hybrid",
    "spotter_h3_cbg": "SPO (H3)",
}


def method_label(method: str) -> str:
    """The term for a combo id; unknown ids fall back to their spaced id."""
    return METHOD_LABELS.get(method, method.replace("_", " "))


def method_term_table(methods) -> dict[str, str]:
    """`term -> full name` for the terms `methods` use, for a manifest."""
    terms = {method_label(m) for m in methods}
    return {t: METHOD_TERMS[t] for t in METHOD_TERMS if t in terms}


#: Term -> hue. Keyed on the term rather than the combo id so `octant_cbg_spl`
#: and `octant_cbg` land on one colour.
#:
#: v3's validated palette (`validate_palette.js --mode light --surface #ffffff
#: --pairs all`), unchanged: hue carries family and lightness the variant
#: within it, so OCT-H and OCT-S share a green and separate by 30+ dE under
#: every colour-vision simulation. Re-validate before changing any hex.
LABEL_HUES: dict[str, str] = {
    "S-P": "#2a78d6",  # blue
    "SOI": "#4a3aa7",  # violet
    "VAN": "#eda100",  # yellow
    "OCT-H": "#17890b",  # green, mid
    "OCT-S": "#58cd78",  # green, light
    "SPO": "#e34948",  # red
}

#: Anything `LABEL_HUES` does not name folds into one grey bucket rather than
#: being handed a generated hue.
OTHER_HUE = "#898781"


def method_colors(methods) -> dict[str, str]:
    """Method id -> hue, stable under filtering."""
    return {m: LABEL_HUES.get(method_label(m), OTHER_HUE) for m in methods}
