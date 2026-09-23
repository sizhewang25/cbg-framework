"""Does the density MTL's coarse-to-fine descent find the same mode as a full
global pass?

`GaussianDensityMTL` evaluates Spotter's surface over the whole globe at a
coarse resolution, carries the top `top_k` cells forward (widened by
`neighbor_ring`), descends into their children, and repeats. That is an
**approximation**: a genuine secondary mode sitting outside the carried set is
gone, and nothing downstream can tell. A "basin miss" is a target whose pruned
argmax lands more than `miss_km` from the argmax an exhaustive global pass finds.

This harness is how `top_k` and `neighbor_ring` get chosen. Commit 9e9df7d set
`top_k=8, neighbor_ring=1` on exactly this measurement, but the script was never
committed -- only its result table, in the commit message -- so the numbers could
not be re-derived when the grid changed. Hence this file.

## What it measures against

The reference is `GaussianDensityMTL` itself with `coarse_resolution` set equal
to `resolution`, which the class documents as its single-global-pass, no-pruning
mode. So the comparison exercises the shipped code path on both sides and
differs only in the pruning, rather than comparing the shipped code against a
re-implementation that could drift from it.

## Why the constraints are synthetic

Real `mu`/`sigma` are not recoverable from the benchmark's output: the parquet
stores `echoed_upper_km`/`echoed_lower_km`, and that band is explicitly not
invertible. So the constraint set is built from the run's **real VP geometry and
real target positions** -- which is what determines whether the surface is
multi-modal -- with `mu` inflated off the true distance by a heavy-tailed draw.

A Pareto factor, not a Gaussian one: RTT error is one-sided (queueing and
circuitous routing inflate delay, nothing deflates it below the speed of light),
and it is the long tail that creates the spurious far-away modes pruning can
lose. A symmetric noise model makes this test look easier than it is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scripts.framework.geometry import haversine
from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.mtl.gaussian_density import GaussianDensityMTL
from scripts.framework.v2.types import Coord, Distance, Latency, VpId

#: A prediction further than this from the reference argmax is a different
#: basin rather than a quantisation difference. 50 km is one nside-128 cell
#: pitch, and was the threshold 9e9df7d used on H3-4's 45 km cells.
DEFAULT_MISS_KM = 50.0

#: Tail weight of the RTT inflation. Lower is heavier; 1.5 has infinite
#: variance, which is the regime that produces far secondary modes.
_PARETO_SHAPE = 1.5
_PARETO_SCALE = 0.35

#: sigma(d) = SIGMA_BASE + SIGMA_SLOPE * d, clipped. Shaped after the fitted
#: Spotter curves rather than taken from one: the harness asks whether pruning
#: finds the mode of *a* plausible surface, not whether one fold's fit is right.
_SIGMA_BASE_KM = 150.0
_SIGMA_SLOPE = 0.35
_SIGMA_CLIP_KM = (100.0, 2500.0)


@dataclass(frozen=True)
class SweepRow:
    """One `(top_k, neighbor_ring)` setting's result."""

    grid: str
    resolution: int
    coarse_resolution: int
    top_k: int
    neighbor_ring: int
    n_targets: int
    n_misses: int
    miss_km: float
    p50_km: float
    p90_km: float
    max_km: float
    n_unsolved: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def load_geometry(
    setup_dir: Path, fold: str = "fold_0", combo: Optional[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(vps, targets)` for one setup: `vps.csv` plus a fold's target coords.

    Any combo's `targets.parquet` will do -- only `target_lat`/`target_lon` are
    read, and every combo in a fold scores the same population -- so `combo`
    defaults to whichever sorts first rather than naming one that may have been
    renamed out from under the caller.
    """
    vp_path = setup_dir / "vps.csv"
    if not vp_path.exists():
        raise FileNotFoundError(f"{vp_path} missing; cannot build VP geometry")
    vps = pd.read_csv(vp_path)

    fold_dir = setup_dir / fold
    candidates = sorted(
        d for d in fold_dir.glob("*") if (d / "targets.parquet").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"no */targets.parquet under {fold_dir}")
    chosen = next((d for d in candidates if d.name == combo), None) if combo else None
    if combo and chosen is None:
        raise FileNotFoundError(
            f"combo {combo!r} not under {fold_dir}; have "
            f"{[d.name for d in candidates]}"
        )
    picked = chosen or candidates[0]
    targets = pd.read_parquet(
        picked / "targets.parquet", columns=["target_id", "target_lat", "target_lon"]
    )
    return vps, targets


def constraints_for(
    target_lat: float,
    target_lon: float,
    vps: pd.DataFrame,
    rng: np.random.Generator,
) -> list[LTDResult]:
    """One distribution-carrying `LTDResult` per VP, mu inflated off the truth."""
    out: list[LTDResult] = []
    inflation = 1.0 + rng.pareto(_PARETO_SHAPE, size=len(vps)) * _PARETO_SCALE
    for (row, infl) in zip(vps.itertuples(index=False), inflation):
        d = haversine((row.vp_lat, row.vp_lon), (target_lat, target_lon))
        sigma = float(
            np.clip(_SIGMA_BASE_KM + _SIGMA_SLOPE * d, *_SIGMA_CLIP_KM)
        )
        mu = float(d * infl)
        out.append(
            LTDResult(
                success=True,
                vp_id=VpId(str(row.vp_id)),
                vp_coord=Coord(float(row.vp_lat), float(row.vp_lon)),
                latency=Latency(max(1.0, mu / 100.0)),
                tg_distance=Distance(
                    upper_km=mu + sigma,
                    lower_km=max(0.0, mu - sigma),
                    mu_km=mu,
                    sigma_km=sigma,
                ),
            )
        )
    return out


def _argmax(mtl: GaussianDensityMTL, cons: list[LTDResult]) -> Optional[Coord]:
    """The MAP cell centre, i.e. what `density_argmax` would return."""
    res = mtl.multilaterate(cons)
    if not res.success or res.density is None or not res.density.cells:
        return None
    best = max(
        range(len(res.density.log_density)), key=lambda i: res.density.log_density[i]
    )
    return res.density.cells[best]


def sweep(
    vps: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    resolution: int,
    coarse_resolution: int,
    settings: tuple[tuple[int, int], ...],
    grid_kwargs: Optional[dict] = None,
    n_targets: int = 50,
    seed: int = 20260923,
    miss_km: float = DEFAULT_MISS_KM,
    progress=None,
) -> list[SweepRow]:
    """Compare each `(top_k, neighbor_ring)` against the global pass.

    The **same** constraint set is reused across settings, drawn once per
    target from `seed`. That is the point: the settings must be compared on
    identical surfaces, or a difference in the draw shows up as a difference in
    pruning.

    `grid_kwargs` is forwarded verbatim to `GaussianDensityMTL`, so this stays
    usable if a second tessellation is ever added -- the sweep is about pruning,
    not about which grid is being pruned. `resolution` and `coarse_resolution`
    are whatever that grid calls a resolution (an nside, today).
    """
    grid_kwargs = dict(grid_kwargs or {})
    rows = targets if len(targets) <= n_targets else targets.sample(
        n_targets, random_state=seed
    )
    rng = np.random.default_rng(seed)
    per_target = [
        constraints_for(float(r.target_lat), float(r.target_lon), vps, rng)
        for r in rows.itertuples(index=False)
    ]

    if progress:
        progress(
            f"reference: global pass at resolution={resolution} "
            f"over {len(per_target)} targets"
        )
    reference_mtl = GaussianDensityMTL(
        resolution=resolution, coarse_resolution=resolution, **grid_kwargs
    )
    reference = [_argmax(reference_mtl, c) for c in per_target]

    out: list[SweepRow] = []
    for top_k, ring in settings:
        mtl = GaussianDensityMTL(
            resolution=resolution,
            coarse_resolution=coarse_resolution,
            top_k=top_k,
            neighbor_ring=ring,
            **grid_kwargs,
        )
        gaps: list[float] = []
        unsolved = 0
        for cons, ref in zip(per_target, reference):
            got = _argmax(mtl, cons)
            if ref is None or got is None:
                unsolved += 1
                continue
            gaps.append(haversine((got.lat, got.lon), (ref.lat, ref.lon)))
        arr = np.asarray(gaps, dtype=float) if gaps else np.zeros(0)
        row = SweepRow(
            grid=str(grid_kwargs.get("grid", "unspecified")),
            resolution=resolution,
            coarse_resolution=coarse_resolution,
            top_k=top_k,
            neighbor_ring=ring,
            n_targets=int(arr.size),
            n_misses=int((arr > miss_km).sum()) if arr.size else 0,
            miss_km=miss_km,
            p50_km=round(float(np.percentile(arr, 50)), 3) if arr.size else 0.0,
            p90_km=round(float(np.percentile(arr, 90)), 3) if arr.size else 0.0,
            max_km=round(float(arr.max()), 3) if arr.size else 0.0,
            n_unsolved=unsolved,
        )
        out.append(row)
        if progress:
            progress(
                f"  top_k={top_k} ring={ring}: {row.n_misses}/{row.n_targets} "
                f"misses >{miss_km:g} km, max {row.max_km:g} km"
            )
    return out


def format_table(rows: list[SweepRow]) -> str:
    """The sweep as a fixed-width table, the shape 9e9df7d's message used."""
    head = (
        f"{'top_k':>6} {'ring':>5} {'misses':>12} "
        f"{'p50_km':>8} {'p90_km':>8} {'max_km':>9} {'unsolved':>9}"
    )
    lines = [head]
    for r in rows:
        lines.append(
            f"{r.top_k:>6} {r.neighbor_ring:>5} "
            f"{r.n_misses:>4}/{r.n_targets:<7} "
            f"{r.p50_km:>8.1f} {r.p90_km:>8.1f} {r.max_km:>9.1f} {r.n_unsolved:>9}"
        )
    return "\n".join(lines)


def write_report(rows: list[SweepRow], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([r.as_dict() for r in rows], indent=2) + "\n")
    return out_path
