"""The method term lookup: every published combo id has a term and a name."""

from __future__ import annotations

from scripts.analysis.v5.modules import methods as MT

PUBLISHED = {
    "octant_cbg_hull": ("OCT-H", "Octant-Hull CBG"),
    "octant_cbg_spl": ("OCT-S", "Octant-Spline CBG"),
    "octant_cbg": ("OCT-S", "Octant-Spline CBG"),
    "million_scale_cbg": ("SOI", "Speed-of-Internet CBG"),
    "shortest_ping": ("S-P", "Shortest-Ping"),
    "spotter_cbg": ("SPO", "Spotter CBG"),
    "vanilla_cbg": ("VAN", "Vanilla CBG"),
}


def test_every_published_method_maps_to_its_term_and_name():
    for combo, (term, name) in PUBLISHED.items():
        assert MT.method_label(combo) == term
        assert MT.METHOD_TERMS[term] == name


def test_every_term_has_a_hue_and_both_octant_ids_share_one():
    assert set(MT.METHOD_TERMS) == set(MT.LABEL_HUES)
    colours = MT.method_colors(["octant_cbg_spl", "octant_cbg"])
    assert colours["octant_cbg_spl"] == colours["octant_cbg"]


def test_term_table_lists_only_the_terms_in_use():
    assert MT.method_term_table(["vanilla_cbg", "shortest_ping"]) == {
        "S-P": "Shortest-Ping",
        "VAN": "Vanilla CBG",
    }
