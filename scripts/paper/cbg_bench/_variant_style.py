"""Fixed display style for the four fundamental CBG variants.

Single source of truth so every figure renders the same combo with the same
label, color, and ordering. Import these instead of hand-rolling per script.

The four variants form two clean within-pair LTD contrasts that share a
geometry stack:
  - vanilla_cbg  / million_scale_cbg : spherical_circle + boundary_vertex_mean (disks)
  - octant_cbg_nofil / spotter_cbg_nofil : planar_annulus + monte_carlo_medoid (annuli)
"""

from __future__ import annotations

# Canonical left-to-right / legend order.
VARIANT_ORDER: list[str] = [
    "vanilla_cbg",
    "million_scale_cbg",
    "octant_cbg_nofil",
    "spotter_cbg_nofil",
]

# combo_id -> display label (fixed by the task spec).
VARIANT_LABELS: dict[str, str] = {
    "vanilla_cbg": "Vanilla CBG",
    "million_scale_cbg": "SOI CBG",
    "octant_cbg_nofil": "Octant CBG",
    "spotter_cbg_nofil": "Spotter CBG",
}

# combo_id -> hex color. Tableau-10 picks; distinct and colorblind-tolerant.
VARIANT_COLORS: dict[str, str] = {
    "vanilla_cbg": "#1f77b4",       # blue
    "million_scale_cbg": "#ff7f0e",  # orange
    "octant_cbg_nofil": "#2ca02c",   # green
    "spotter_cbg_nofil": "#d62728",  # red
}

# The two clean within-pair contrasts (deliverable 4 — one figure each).
# (combo_x, combo_y): x is the "reference" variant, y the comparison.
VARIANT_PAIRS: list[tuple[str, str]] = [
    ("vanilla_cbg", "million_scale_cbg"),
    ("octant_cbg_nofil", "spotter_cbg_nofil"),
]


def label(combo_id: str) -> str:
    """Display label for a combo_id (falls back to the raw id)."""
    return VARIANT_LABELS.get(combo_id, combo_id)


def color(combo_id: str) -> str:
    """Hex color for a combo_id (falls back to matplotlib gray)."""
    return VARIANT_COLORS.get(combo_id, "#7f7f7f")
