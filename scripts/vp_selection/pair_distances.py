"""Pair-distance generators for VP selection.

`compute_geodesic_distances` produces the haversine distance graph the Prim-style
selection strategies (`scripts/vp_selection/strategies.py`) maximize over.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Tuple

from scripts.libs.cbg.rtt_model import haversine_distance


def compute_geodesic_distances(
    coords: Mapping[str, Tuple[float, float]],
) -> dict[tuple[str, str], float]:
    """Pairwise great-circle distances (km) over a VP pool.

    Returns `{(a, b): km}` with `a < b` lexicographic for every unordered pair.
    No self-pair entries. Empty or single-element pools return an empty dict.
    """
    pairs: dict[tuple[str, str], float] = {}
    for a, b in combinations(sorted(coords), 2):
        lat_a, lon_a = coords[a]
        lat_b, lon_b = coords[b]
        pairs[(a, b)] = haversine_distance(lat_a, lon_a, lat_b, lon_b)
    return pairs
