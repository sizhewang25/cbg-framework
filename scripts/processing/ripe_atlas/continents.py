"""ISO 3166-1 alpha-2 country code → continent.

Hand-curated. Covers every code present in the RIPE Atlas probe + anchor
corpus plus the common dependencies/territories likely to appear later.
Single source of truth — imported by `select_probes_and_anchors.py` and by
the `visualize_probes_anchors.ipynb` continent section.

`continent_of(code)` returns "Unknown" for missing codes so callers can
surface that as a sanity warning instead of silently mis-bucketing.
"""

from __future__ import annotations


CONTINENT_OF: dict[str, str] = {
    # Africa
    **dict.fromkeys((
        "DZ", "AO", "BJ", "BW", "BF", "BI", "CM", "CV", "CF", "TD", "KM", "CG",
        "CD", "DJ", "EG", "GQ", "ER", "SZ", "ET", "GA", "GM", "GH", "GN", "GW",
        "CI", "KE", "LS", "LR", "LY", "MG", "MW", "ML", "MR", "MU", "YT", "MA",
        "MZ", "NA", "NE", "NG", "RE", "RW", "SH", "ST", "SN", "SC", "SL", "SO",
        "ZA", "SS", "SD", "TZ", "TG", "TN", "UG", "EH", "ZM", "ZW",
    ), "Africa"),
    # Antarctica
    **dict.fromkeys(("AQ", "BV", "GS", "HM", "TF"), "Antarctica"),
    # Asia
    **dict.fromkeys((
        "AF", "AM", "AZ", "BH", "BD", "BT", "BN", "KH", "CN", "CY", "GE", "HK",
        "IN", "ID", "IR", "IQ", "IL", "JP", "JO", "KZ", "KP", "KR", "KW", "KG",
        "LA", "LB", "MO", "MY", "MV", "MN", "MM", "NP", "OM", "PK", "PS", "PH",
        "QA", "SA", "SG", "LK", "SY", "TW", "TJ", "TH", "TL", "TR", "TM", "AE",
        "UZ", "VN", "YE",
    ), "Asia"),
    # Europe
    **dict.fromkeys((
        "AL", "AD", "AT", "BY", "BE", "BA", "BG", "HR", "CZ", "DK", "EE", "FO",
        "FI", "FR", "DE", "GI", "GR", "GG", "HU", "IS", "IE", "IM", "IT", "JE",
        "XK", "LV", "LI", "LT", "LU", "MT", "MD", "MC", "ME", "NL", "MK", "NO",
        "PL", "PT", "RO", "RU", "SM", "RS", "SK", "SI", "ES", "SE", "CH", "UA",
        "GB", "VA", "AX", "SJ",
    ), "Europe"),
    # North America (incl. Caribbean + Central America)
    **dict.fromkeys((
        "AG", "AI", "AW", "BS", "BB", "BZ", "BM", "CA", "KY", "CR", "CU", "CW",
        "DM", "DO", "SV", "GL", "GD", "GP", "GT", "HT", "HN", "JM", "MQ", "MX",
        "MS", "NI", "PA", "PR", "BL", "KN", "LC", "MF", "PM", "VC", "SX", "TT",
        "TC", "US", "VG", "VI",
    ), "North America"),
    # Oceania
    **dict.fromkeys((
        "AS", "AU", "CK", "FJ", "PF", "GU", "KI", "MH", "FM", "NR", "NC", "NZ",
        "NU", "NF", "MP", "PW", "PG", "PN", "WS", "SB", "TK", "TO", "TV", "UM",
        "VU", "WF",
    ), "Oceania"),
    # South America
    **dict.fromkeys((
        "AR", "BO", "BR", "CL", "CO", "EC", "FK", "GF", "GY", "PY", "PE", "SR",
        "UY", "VE",
    ), "South America"),
}


def continent_of(code: str | None) -> str:
    """Map an ISO 3166-1 alpha-2 code to its continent name.

    Returns "Unknown" for `None`, empty string, or any code not in the table.
    Callers should surface "Unknown" as a sanity warning rather than ignore it.
    """
    if not code:
        return "Unknown"
    return CONTINENT_OF.get(code, "Unknown")


# Approximate continent bounding boxes (lat_min, lat_max, lon_min, lon_max).
# Used to sanity-check a country_code-derived continent against the actual
# coordinates. Motivating case: a RIPE Atlas probe in Guadeloupe is tagged
# `country_code="FR"` (Guadeloupe is a French overseas department) → the
# country_code check labels it Europe, but its (16°N, -62°W) coords belong
# to North America. Bounds are intentionally generous so islands at the
# edges (Iceland, Cyprus, Aleutians, etc.) aren't false-positived.
_CONTINENT_BBOX: dict[str, tuple[float, float, float, float]] = {
    "Africa":        (-35,  38,  -18,   52),
    "Asia":          (-10,  82,   25,  180),
    "Europe":        ( 30,  72,  -25,   60),
    "North America": (  5,  72, -170,  -50),
    "South America": (-56,  13,  -82,  -34),
    "Antarctica":    (-90, -60, -180,  180),
    # Oceania straddles the antimeridian; handled specially in
    # `continent_bbox_contains` below.
}


def continent_bbox_contains(continent: str, lat: float, lon: float) -> bool:
    """Approximate test: does `(lat, lon)` fall inside `continent`'s bbox?

    Use as a sanity guard after `continent_of(country_code)` — catches probes
    whose country_code is a parent country's ISO (e.g. `FR` for Guadeloupe,
    `US` for Hawaii) but whose physical location is on a different continent.

    Returns `True` for unknown / catch-all continents (`"Unknown"`, `"Global"`)
    so callers don't have to special-case them.
    """
    if continent == "Oceania":
        # Pacific wraps the antimeridian. Hawaii (~ -155°) is sometimes
        # bucketed under Oceania; accept either the western sweep (≥ 110°)
        # or the eastern Pacific (≤ -130°).
        return -50 <= lat <= 30 and (lon >= 110 or lon <= -130)
    bbox = _CONTINENT_BBOX.get(continent)
    if bbox is None:
        return True
    lat_min, lat_max, lon_min, lon_max = bbox
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
