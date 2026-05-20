"""Pipeline-setup helpers for v2 integration tests.

Each `*_pipeline()` factory returns a fully-prepared LTDModel — fitted
via its public `fit(samples)` API — plus an observation list whose
predicted constraints overlap a fixed `TARGET` coord. That keeps the
combinations test purely about composition: it can iterate over every
valid (LTD × MTL × CTR) triple and assert SUCCESS without re-deriving
calibration per combo.

Geometry:
    TARGET = (0, 0). Four VPs sit roughly at the compass points, jittered
    to distances {480, 510, 490, 520} km so no spherical-circle crossing
    falls exactly on another disk's boundary (the symmetric layout puts
    crossings on co-incident boundaries, and SphericalCircleMTL's
    inside-all filter then rejects them at FP precision). With Circle
    LTDs latencies map to a 700 km radius — each disk includes TARGET.
    With Annulus LTDs latencies map to a band [400, 600] km — each
    target distance lies inside every annulus.

FitSample synthesis:
    Probes are placed `d_km` north of each VP along the meridian, so
    haversine(vp_coord, probe_coord) == d_km exactly. That lets every
    LTD see distances it can reason about cleanly:
      * LowEnvelopeLTD (LP best-line):  latency = 0.02·d + 5
      * BoundedSplineLTD (per-VP Octant): parallel ±100 km hull, μ=50·rtt
      * NormalDistLTD   (pooled Spotter): same band, pooled across VPs
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from scripts.framework.v2.ltd.base import FitSample
from scripts.framework.v2.ltd.bounded_spline import BoundedSplineLTD
from scripts.framework.v2.ltd.low_envelope import LowEnvelopeLTD
from scripts.framework.v2.ltd.normal_dist import NormalDistLTD
from scripts.framework.v2.ltd.speed_of_internet import SpeedOfInternetLTD
from scripts.framework.v2.types import Coord, Latency, VpId

# Earth radius (km) used by haversine — 1° latitude ≈ this/57.296 km.
_KM_PER_DEG = 6371.0 * np.pi / 180.0  # ≈ 111.195

TARGET: Coord = Coord(lat=0.0, lon=0.0)


def _deg(km: float) -> float:
    return km / _KM_PER_DEG


# Four jittered VPs. Each is within [480, 520] km of TARGET so disks of
# radius 700 km and annuli [400, 600] km both contain the origin.
VPS: List[Tuple[VpId, Coord]] = [
    (VpId("vp-n"), Coord(lat=_deg(480.0), lon=0.0)),
    (VpId("vp-s"), Coord(lat=-_deg(510.0), lon=0.0)),
    (VpId("vp-e"), Coord(lat=0.0, lon=_deg(490.0))),
    (VpId("vp-w"), Coord(lat=0.0, lon=-_deg(520.0))),
]


def observations(latency_ms: float) -> List[Tuple[VpId, Coord, Latency]]:
    """Return one obs tuple per VP, all carrying the same RTT."""
    return [(vp_id, vp_coord, Latency(latency_ms)) for vp_id, vp_coord in VPS]


def _probe_at(vp_coord: Coord, distance_km: float) -> Coord:
    """Coord `distance_km` north of vp_coord along the meridian.

    haversine(vp_coord, returned) == distance_km exactly: pure-meridian
    distance on a sphere is R · Δlat_radians.
    """
    return Coord(lat=vp_coord.lat + _deg(distance_km), lon=vp_coord.lon)


def _check_fit(ltd_name: str, fit_result) -> None:
    if not fit_result.success:
        raise AssertionError(
            f"{ltd_name}.fit failed: error={fit_result.error}, args={fit_result.args}"
        )


# ----- Circle-family LTDs ---------------------------------------------------


def speed_of_internet_pipeline() -> Tuple[SpeedOfInternetLTD, list]:
    """Stateless. At RTT=7 ms, radius = 100 · 7 = 700 km > 500 km target distance.

    SpeedOfInternetLTD._fit is a no-op (no per-VP state), but we still call
    `ltd.fit([])` so every pipeline goes through the same public entry point.
    """
    ltd = SpeedOfInternetLTD()
    _check_fit("SpeedOfInternetLTD", ltd.fit([]))
    return ltd, observations(7.0)


def low_envelope_pipeline() -> Tuple[LowEnvelopeLTD, list]:
    """Per-VP LP best-line fit: latency = 0.02·d + 5 → RTT=19 ms predicts 700 km.

    Eight (probe, latency) rows per VP; the wrapper's `_fit` partitions
    by vp_id, computes haversine from vp_coord → probe_coord, and fits
    one RTTDistanceModel per VP.
    """
    distances_km = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
    slope, intercept = 0.02, 5.0

    samples: list[FitSample] = []
    for vp_id, vp_coord in VPS:
        for d in distances_km:
            samples.append(
                FitSample(
                    vp_id=vp_id,
                    vp_coord=vp_coord,
                    probe_coord=_probe_at(vp_coord, d),
                    latency=Latency(slope * d + intercept),
                )
            )

    ltd = LowEnvelopeLTD()
    _check_fit("LowEnvelopeLTD", ltd.fit(samples))
    return ltd, observations(19.0)


# ----- Annulus-family LTDs --------------------------------------------------


def normal_dist_pipeline() -> Tuple[NormalDistLTD, list]:
    """Pooled-normal fit on a parallel ±100 km band centered at μ=50·rtt.

    Hyperparameters are loosened (5 bins, 2 per bin, deg-1 μ, deg-0 σ,
    cutoff_min_points=1) so the pooled polynomial fit converges on this
    compact integration sample set. With 4 VPs × 10 RTTs × 4 probes =
    160 pooled rows, the band recovers as ≈ [400, 600] km at RTT=10.
    """
    rtt_values = list(np.linspace(5.0, 50.0, 10))
    spread_km = 100.0
    n_per_rtt = 4

    samples: list[FitSample] = []
    for vp_id, vp_coord in VPS:
        for rtt in rtt_values:
            center = 50.0 * rtt
            for d in np.linspace(center - spread_km, center + spread_km, n_per_rtt):
                samples.append(
                    FitSample(
                        vp_id=vp_id,
                        vp_coord=vp_coord,
                        probe_coord=_probe_at(vp_coord, float(d)),
                        latency=Latency(float(rtt)),
                    )
                )

    ltd = NormalDistLTD(
        n_bins=5,
        min_per_bin=2,
        deg_mu=1,
        deg_sigma=0,
        cutoff_min_points=1,
    )
    _check_fit("NormalDistLTD", ltd.fit(samples))
    return ltd, observations(10.0)


def bounded_spline_pipeline() -> Tuple[BoundedSplineLTD, list]:
    """Per-VP Octant spline with parallel ±100 km hull band, μ=50·rtt.

    Five RTTs × two probes (lower/upper hull) per VP. At RTT=10 the
    fitted spline + hull yields ≈ [400, 600] km. No explicit δ is
    needed; the wrapper handles per-VP δ search internally.
    """
    rtt_values = [10.0, 20.0, 30.0, 40.0, 50.0]

    samples: list[FitSample] = []
    for vp_id, vp_coord in VPS:
        for rtt in rtt_values:
            for d in (50.0 * rtt - 100.0, 50.0 * rtt + 100.0):
                samples.append(
                    FitSample(
                        vp_id=vp_id,
                        vp_coord=vp_coord,
                        probe_coord=_probe_at(vp_coord, d),
                        latency=Latency(rtt),
                    )
                )

    ltd = BoundedSplineLTD(
        target_coverage=0.8,
        cutoff_min_points=1,
        spline_n_knots=4,
        bin_size_ms=1000,
    )
    _check_fit("BoundedSplineLTD", ltd.fit(samples))
    return ltd, observations(10.0)
