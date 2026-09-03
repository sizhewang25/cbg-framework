"""Fitting `n` circles so their overlaps match the observed intersections.

Radii are fixed by the set sizes — `circle_radii` puts the share in the *area*,
never the radius — so only the centres are free. The objective is the weighted
squared error over **every** combination rather than just the pairs: fitting
pairs alone is analytically tidy and leaves the three- and four-way regions to
whatever falls out, which is the failure the ring template was chosen to avoid.

Nothing here draws. `EulerLayout` carries the fit and the two numbers that judge
it (`placed`, `pair_error`), and `euler_fit_table` is the row-by-row audit trail
`plot` points the reader at.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

from scripts.analysis.v3.modules.diagram.common.labels import (
    RING_LETTERS,
    label_for,
    region_key,
)


#: Circle centres are searched inside this half-width, in the same units as the
#: radii (where the whole target population has area 1). Wide enough that six
#: disjoint circles fit, tight enough to keep the sampling grid dense.
EULER_BOX = 1.6

#: Region areas are measured by sampling this many points per side over
#: `EULER_BOX`. 320 puts ~10 samples in a region holding 0.1% of the population,
#: which is the resolution the objective needs; the reported fit is re-measured
#: at `EULER_FIT_GRID`.
EULER_GRID = 320

#: Denser grid for the numbers that reach the figure and the CSV, so the quoted
#: misplacement is not an artifact of the optimizer's own sampling.
EULER_FIT_GRID = 600

#: Squared-error weight on combinations with **no** targets. Above 1 because
#: "these four methods are never all right together" is a fact about the data,
#: and a layout that draws that region anyway is making one up; the asymmetry
#: buys the topology at a small cost in area accuracy.
EULER_EMPTY_WEIGHT = 8.0

#: Weight on a term pulling the centres toward the origin. Small enough to be
#: swamped by any real area error (the pooled operator runs sit at a loss of
#: ~4.5e-3, this contributes ~6e-6) and large enough to pick the compact layout
#: when two are equally faithful — which is every layout, once all the circles
#: that must be disjoint are.
COMPACTNESS = 1e-6

#: Restarts of the local search, the first from the MDS seed and the rest jittered
#: around it. Three because on the pooled operator runs restarts 4-6 never
#: improved the fit by more than 0.3pp of misplacement and each one costs ~8s.
EULER_RESTARTS = 3


def set_shares(membership: pd.DataFrame, order: list[str]) -> "np.ndarray":
    """Share of the target population each method gets right, in ring order."""
    return np.array([float(membership[m].mean()) for m in order])


def combination_shares(membership: pd.DataFrame, order: list[str]) -> "np.ndarray":
    """Exact-combination shares indexed by bitmask: `1 << i` is `order[i]`.

    The array twin of `intersection_table` — same disjoint reading, same
    denominator, addressable by the bitmask arithmetic the layout needs. Index 0
    is the "no method correct" share, which lives outside every circle and is
    therefore excluded from every fit term below.
    """
    n = len(order)
    cols = np.column_stack([membership[m].to_numpy(dtype=bool) for m in order])
    keys = (cols * (1 << np.arange(n))).sum(axis=1)
    return np.bincount(keys, minlength=1 << n) / len(membership)


def circle_radii(shares: "np.ndarray") -> "np.ndarray":
    """Radii whose **areas** are the set shares — never the radii themselves.

    Encoding a quantity as a radius exaggerates it by squaring; area is what a
    reader integrates when comparing two circles, so area is what carries the
    number.
    """
    return np.sqrt(np.asarray(shares, dtype=float) / math.pi)


def lens_area(r1: float, r2: float, d: float) -> float:
    """Area shared by two circles of radii `r1`, `r2` whose centres are `d` apart."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a = max(-1.0, min(1.0, (d * d + r1 * r1 - r2 * r2) / (2 * d * r1)))
    b = max(-1.0, min(1.0, (d * d + r2 * r2 - r1 * r1) / (2 * d * r2)))
    return (
        r1 * r1 * math.acos(a)
        + r2 * r2 * math.acos(b)
        - 0.5 * math.sqrt(
            max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
        )
    )


#: Gap left between two circles whose sets never co-occur, as a fraction of the
#: smaller radius. Zero would put them tangent, which reads as "they just barely
#: touch" — the one relationship the data is certain they do not have.
DISJOINT_MARGIN = 0.06


def separation_for_overlap(r1: float, r2: float, target: float) -> float:
    """Centre distance at which two circles share exactly `target` area.

    The three cases are the three Euler relationships, and the first two are
    where this function earns its place — they are the ones the reader is
    entitled to read straight off the picture:

    * `target == 0` -> **disjoint**, pushed past tangency by `DISJOINT_MARGIN`;
    * `target >= min(area)` -> **contained**, one circle fully inside the other;
    * otherwise a bisection on `lens_area`, which is monotone in `d`.
    """
    from scipy.optimize import brentq

    lo, hi = abs(r1 - r2), r1 + r2
    if target <= 0:
        return hi + DISJOINT_MARGIN * min(r1, r2)
    if target >= math.pi * min(r1, r2) ** 2:
        return lo
    return float(brentq(lambda d: lens_area(r1, r2, d) - target, lo + 1e-12, hi - 1e-12))


def _mds_centres(radii: "np.ndarray", pair_target: "np.ndarray") -> "np.ndarray":
    """Starting positions: classical MDS on the ideal pairwise distances.

    Every pair's distance is individually solvable (`separation_for_overlap`);
    what is not simultaneously satisfiable is all of them at once in the plane.
    MDS gives the least-squares compromise, which is a far better start than any
    ring or random scatter — the local search that follows only has to fix the
    higher-order regions.
    """
    n = len(radii)
    d = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        d[i, j] = d[j, i] = separation_for_overlap(
            radii[i], radii[j], float(pair_target[i, j])
        )
    j_mat = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * j_mat @ (d ** 2) @ j_mat
    vals, vecs = np.linalg.eigh(gram)
    take = np.argsort(vals)[::-1][:2]
    return vecs[:, take] * np.sqrt(np.maximum(vals[take], 0.0))


def _area_sampler(n: int, radii: "np.ndarray", grid: int):
    """Region areas by point sampling, as a function of the centres.

    Shapely booleans over `2**n - 1` combinations are exact but far too slow to
    sit inside an optimizer loop. A fixed grid turns the whole partition into one
    `bincount` over per-point bitmasks: every point is charged to exactly the
    combination that contains it, so the areas are disjoint and sum to the union
    by construction — the same invariant `intersection_table` has.
    """
    lin = (np.arange(grid) + 0.5) / grid * 2 * EULER_BOX - EULER_BOX
    gx, gy = (a.ravel() for a in np.meshgrid(lin, lin))
    cell = (2 * EULER_BOX / grid) ** 2
    bits = (1 << np.arange(n)).astype(np.int64)
    r2 = (radii ** 2)[:, None]

    def areas(flat_centres) -> "np.ndarray":
        c = np.asarray(flat_centres, dtype=float).reshape(n, 2)
        inside = (gx[None, :] - c[:, 0:1]) ** 2 + (gy[None, :] - c[:, 1:2]) ** 2 <= r2
        return np.bincount((inside * bits[:, None]).sum(axis=0), minlength=1 << n) * cell

    return areas


class EulerLayout:
    """A fitted layout plus everything needed to judge it.

    `observed` and `drawn` are both bitmask-indexed exact-combination shares, so
    they are directly comparable term by term and `misplaced` is a distance
    between two distributions over the same regions.
    """

    def __init__(self, order, centres, radii, observed, drawn, loss):
        self.order = list(order)
        self.centres = centres
        self.radii = radii
        self.observed = observed
        self.drawn = drawn
        self.loss = float(loss)

    @property
    def misplaced(self) -> float:
        """Share of targets the picture puts in the wrong region.

        Total variation between the observed and drawn region distributions:
        move that much probability mass and the two agree. Index 0 (no method
        correct) is excluded — it is outside every circle in both, so it is not
        a region the layout can get wrong.
        """
        return 0.5 * float(np.abs(self.drawn[1:] - self.observed[1:]).sum())

    @property
    def placed(self) -> float:
        """The complement: share of targets whose region the picture gets right."""
        return 1.0 - self.misplaced

    def pair_error(self) -> float:
        """Largest absolute error over the 2-set intersections, in share units.

        Reported separately from `misplaced` because pairs are the relationship
        a reader actually reads off a Euler diagram; the higher-order regions
        are the ones circles cannot control.
        """
        worst = 0.0
        n = len(self.order)
        for i, j in combinations(range(n), 2):
            both = (1 << i) | (1 << j)
            keys = [k for k in range(1, 1 << n) if k & both == both]
            worst = max(
                worst, abs(float(self.drawn[keys].sum() - self.observed[keys].sum()))
            )
        return worst


def fit_euler_layout(
    membership: pd.DataFrame,
    order: list[str],
    *,
    restarts: int = EULER_RESTARTS,
    grid: int = EULER_GRID,
    fit_grid: int = EULER_FIT_GRID,
) -> EulerLayout:
    """Place `n` circles so their overlaps match the observed intersections.

    Radii are fixed by the set sizes, so only the centres are free: `2n`
    parameters against `2**n - 1` regions. The objective is the weighted squared
    error over **every** combination, not just the pairs — fitting pairs alone is
    analytically tidy and produces a picture whose three- and four-way regions
    are whatever happens to fall out, which is precisely the failure the printed
    ring template was chosen to avoid. Combinations observed empty are weighted
    up (`EULER_EMPTY_WEIGHT`) so the layout separates circles rather than
    inventing a region.

    Deterministic: the MDS start is a function of the data and the jitters are
    seeded by restart index, so the same membership matrix always yields the same
    figure.
    """
    from scipy.optimize import minimize

    n = len(order)
    if not 2 <= n <= len(RING_LETTERS):
        raise ValueError(f"an Euler layout takes 2-{len(RING_LETTERS)} sets, got {n}")
    if len(membership) == 0:
        raise ValueError("no targets to lay out")

    observed = combination_shares(membership, order)
    radii = circle_radii(set_shares(membership, order))
    if not (radii > 0).all():
        dead = [m for m, r in zip(order, radii) if r <= 0]
        raise ValueError(
            f"{dead} classified no target correctly, so there is no circle to "
            f"draw for them; drop them with --method"
        )

    cols = np.column_stack([membership[m].to_numpy(dtype=bool) for m in order])
    pair_target = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        pair_target[i, j] = pair_target[j, i] = float((cols[:, i] & cols[:, j]).mean())

    areas = _area_sampler(n, radii, grid)
    weights = np.where(observed > 0, 1.0, EULER_EMPTY_WEIGHT)[1:]

    def loss(flat) -> float:
        diff = areas(flat)[1:] - observed[1:]
        # Areas are only measured inside the sampling box, so a circle that
        # wandered out would be scored on the part that stayed in. The bounds
        # below make that unreachable; the compactness term breaks the ties that
        # sent it there — with every pair already disjoint the objective is
        # exactly flat, and a flat plateau is where a direct search wanders.
        pull = COMPACTNESS * float((np.asarray(flat) ** 2).sum())
        return float((weights * diff * diff).sum()) + pull

    limit = EULER_BOX - float(radii.max())
    bounds = [(-limit, limit)] * (2 * n)
    start = np.clip(_mds_centres(radii, pair_target).ravel(), -limit, limit)
    best = None
    for attempt in range(max(1, restarts)):
        x0 = start if attempt == 0 else np.clip(
            start + np.random.default_rng(attempt).normal(0.0, 0.1, start.size),
            -limit, limit,
        )
        res = minimize(
            loss, x0, method="Powell", bounds=bounds,
            options={"maxiter": 20000, "xtol": 1e-4, "ftol": 1e-7},
        )
        if best is None or res.fun < best.fun:
            best = res

    centres = np.asarray(best.x, dtype=float).reshape(n, 2)
    centres -= centres.mean(axis=0)  # centre the figure, which changes nothing
    drawn = _area_sampler(n, radii, fit_grid)(centres.ravel())
    return EulerLayout(order, centres, radii, observed, drawn, best.fun)


def euler_fit_table(layout: EulerLayout, n_targets: int) -> pd.DataFrame:
    """Observed against drawn, one row per combination — the figure's audit trail.

    Rows for combinations that are empty in the data but drawn anyway are the
    ones to read first: they are the regions the picture asserts and the data
    denies. `delta` is drawn minus observed, so those are exactly the positive
    rows with `n_targets == 0`.
    """
    order = layout.order
    n = len(order)
    rows = []
    for key in range(1, 1 << n):
        members = [order[i] for i in range(n) if key >> i & 1]
        obs, drawn = float(layout.observed[key]), float(layout.drawn[key])
        if obs == 0 and drawn < 1e-4:
            continue
        rows.append(
            {
                "methods": "|".join(label_for(m) for m in members),
                "region": region_key(order, set(members)),
                "n_methods": len(members),
                "n_targets": int(round(obs * n_targets)),
                "observed_share": round(obs, 4),
                "drawn_share": round(drawn, 4),
                "delta": round(drawn - obs, 4),
            }
        )
    return (
        pd.DataFrame(
            rows,
            columns=["methods", "region", "n_methods", "n_targets",
                     "observed_share", "drawn_share", "delta"],
        )
        .sort_values(["observed_share", "drawn_share"], ascending=False)
        .reset_index(drop=True)
    )
