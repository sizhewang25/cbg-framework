"""Seeds: sites grouped by complete linkage, one seed per group.

A **seed** is the spherical centroid of a group of sites whose pairwise
distances are all at most `grid_km`. The seeds generate the cell partition:
a cell is the Voronoi cell of one seed, bounded by the landmass.

## Why group sites at all

Two sites closer than one grid are one place at that granularity. Without
grouping, EWR and JFK (~33 km apart) would split the New York area into two
slivers of cells, and a prediction landing between them would be `wrong` on
what is at most a naming difference. Grouping at `grid_km` makes the cell
partition as coarse as the grid partition it sits beside.

## Why complete linkage

It caps the group **diameter** -- the largest pairwise distance -- which is the
quantity "these sites are within one grid of each other" is about. Single
linkage would chain sites 40 km apart into a group of any length; a cap on the
distance to the centroid is a different quantity (a group's diameter can reach
twice its radius), and an earlier answer space in this repo was caught by
exactly that difference.

## Logical, not geospatial

A group depends only on the distances between sites and on `grid_km`. It does
not depend on where the HEALPix grid boundaries fall: two sites in one grid can
sit in two groups, and two sites in adjacent grids can share one. That
independence is the point of having a second partition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.v5.modules.geodesy import pairwise_km, spherical_centroid


def group_sites(site_lat, site_lon, diameter_km: float) -> np.ndarray:
    """Seed id per site: complete linkage, every group's diameter <= `diameter_km`.

    Ids are dense, `0..K-1`, ordered by each group's smallest input position,
    so they follow the site ids (which are sorted keys) and not scipy's
    internal labelling. The caller passes sites in `site_id` order.
    """
    lat = np.asarray(site_lat, dtype=float).ravel()
    lon = np.asarray(site_lon, dtype=float).ravel()
    n = lat.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    d = pairwise_km(lat, lon)
    np.fill_diagonal(d, 0.0)
    # Symmetrise: the chord form is symmetric up to the last ulp, and
    # squareform refuses anything that is not exactly so.
    d = (d + d.T) / 2.0
    labels = fcluster(
        linkage(squareform(d, checks=False), method="complete"),
        t=float(diameter_km),
        criterion="distance",
    )
    first_seen: dict[int, int] = {}
    for label in labels:
        first_seen.setdefault(int(label), len(first_seen))
    return np.array([first_seen[int(x)] for x in labels], dtype=np.int64)


def build_seeds(sites: pd.DataFrame, diameter_km: float) -> tuple[np.ndarray, pd.DataFrame]:
    """`(seed id per site, seeds frame)` from a `sites` frame.

    `sites` needs `site_id, site_lat, site_lon, n_tgs`, sorted by `site_id`.
    The seeds frame is `seed_id, seed_lat, seed_lon, n_sites, n_tgs,
    seed_diameter_km, nearest_seed_km`.
    """
    seed_of_site = group_sites(sites["site_lat"], sites["site_lon"], diameter_km)
    rows = []
    for seed_id in range(int(seed_of_site.max()) + 1 if len(seed_of_site) else 0):
        member = sites.loc[seed_of_site == seed_id]
        lat, lon = spherical_centroid(member["site_lat"], member["site_lon"])
        span = pairwise_km(member["site_lat"], member["site_lon"])
        rows.append(
            {
                "seed_id": seed_id,
                "seed_lat": lat,
                "seed_lon": lon,
                "n_sites": int(len(member)),
                "n_tgs": int(member["n_tgs"].sum()),
                "seed_diameter_km": round(float(span.max()), 3),
            }
        )
    seeds = pd.DataFrame(
        rows,
        columns=["seed_id", "seed_lat", "seed_lon", "n_sites", "n_tgs", "seed_diameter_km"],
    )
    if len(seeds) > 1:
        mesh = pairwise_km(seeds["seed_lat"], seeds["seed_lon"])
        np.fill_diagonal(mesh, np.inf)
        seeds["nearest_seed_km"] = np.round(mesh.min(axis=1), 3)
    else:
        seeds["nearest_seed_km"] = np.nan
    return seed_of_site, seeds
