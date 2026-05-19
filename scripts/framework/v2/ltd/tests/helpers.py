"""Shared test fixtures for v2 LTD wrapper tests.

Two flavors of fixture:

1. `make_fitted_*_model` factories return library-level objects
   (RTTDistanceModel / OctantRTTModel / SpotterRTTModel). The v2 wrappers
   build these internally via _fit, but unit tests inject them directly
   into the private attribute (`ltd._submodels`, `ltd._model`) to assert
   prediction behavior without paying the haversine + fit roundtrip.

2. `make_fit_samples_for_*` factories build list[FitSample] inputs for
   the integration tests that exercise the real `fit(samples)` path.
   Probes are placed along a meridian north of the VP coord, so
   haversine(vp, probe) ~= the target distance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from scripts.framework.v2.ltd.base import FitSample
from scripts.framework.v2.types import Coord, Latency, VpId
from scripts.libs.cbg.rtt_model import RTTDistanceModel

if TYPE_CHECKING:
    from scripts.libs.octant_simple.octant_model import OctantRTTModel
    from scripts.libs.spotter.spotter_model import SpotterRTTModel


# Earth radius (km) used by haversine_distance() — 1 deg of latitude ≈ this/57.296 km
_KM_PER_DEG_LAT = 6371.0 * np.pi / 180.0  # ≈ 111.195


ANCHOR_COORDS: dict[VpId, Coord] = {
    VpId("anchor-a"): Coord(40.0, -74.0),
    VpId("anchor-b"): Coord(41.0, -75.0),
    VpId("anchor-c"): Coord(42.0, -76.0),
    VpId("anchor-d"): Coord(43.0, -77.0),
}


def make_fitted_low_envelope_model(
    anchor_ip: str = "anchor-a",
    slope: float = 0.02,
    intercept: float = 5.0,
) -> RTTDistanceModel:
    """Fit a real LP model where RTT = 0.02 * distance + 5.

    At RTT = 25 ms, predict_distance returns (25 - 5) / 0.02 = 1000 km.
    At RTT < 5 ms, predict_distance returns a non-physical (negative) radius.
    """
    coord = ANCHOR_COORDS[VpId(anchor_ip)]
    distances = np.array([100, 200, 300, 400, 500, 600, 700, 800], dtype=float)
    rtts = slope * distances + intercept
    model = RTTDistanceModel(
        anchor_ip=anchor_ip, anchor_lat=coord.lat, anchor_lon=coord.lon
    )
    ok = model.fit(distances, rtts)
    if not ok:
        raise AssertionError(model.fit_message)
    return model


def make_unfitted_low_envelope_model(anchor_ip: str = "anchor-a") -> RTTDistanceModel:
    """Build a real but unfitted LP RTT-distance model."""
    coord = ANCHOR_COORDS[VpId(anchor_ip)]
    return RTTDistanceModel(
        anchor_ip=anchor_ip, anchor_lat=coord.lat, anchor_lon=coord.lon
    )


def make_fitted_octant_model(
    anchor_ip: str = "anchor-a",
    *,
    fit_spline: bool = True,
) -> "OctantRTTModel":
    """Fit a real Octant model with an easily checked parallel distance band.

    For every RTT x, the training set contains two points:
      lower distance = 50 * x - 100
      upper distance = 50 * x + 100

    Thus at RTT=20, the hand-derived hull bounds are 900..1100 km. The
    50 km/ms slope sits well below the 100 km/ms speed-of-internet ceiling,
    so every probe survives the baseline filter in `OctantRTTModel.fit`.
    """
    from scripts.libs.octant_simple.octant_model import OctantRTTModel

    coord = ANCHOR_COORDS[VpId(anchor_ip)]
    rtt_values = np.array([10, 20, 30, 40, 50], dtype=float)
    rtts = np.repeat(rtt_values, 2)
    distances = np.ravel(
        [[50.0 * rtt - 100.0, 50.0 * rtt + 100.0] for rtt in rtt_values]
    )
    model = OctantRTTModel(
        anchor_ip=anchor_ip,
        anchor_lat=coord.lat,
        anchor_lon=coord.lon,
    )
    ok = model.fit(
        rtts,
        distances,
        cutoff_min_points=1,
        fit_spline=fit_spline,
        spline_n_knots=4,
        bin_size_ms=1000,
    )
    if not ok:
        raise AssertionError(model.fit_message)
    return model


def make_fitted_degenerate_octant_model(anchor_ip: str = "anchor-a") -> "OctantRTTModel":
    """Fit a real Octant model whose lower and upper bounds are identical."""
    from scripts.libs.octant_simple.octant_model import OctantRTTModel

    coord = ANCHOR_COORDS[VpId(anchor_ip)]
    rtts = np.array([10, 20, 30, 40, 50], dtype=float)
    distances = 100.0 * rtts
    model = OctantRTTModel(
        anchor_ip=anchor_ip,
        anchor_lat=coord.lat,
        anchor_lon=coord.lon,
    )
    ok = model.fit(
        rtts,
        distances,
        cutoff_min_points=1,
        fit_spline=True,
        spline_n_knots=2,
        bin_size_ms=1000,
    )
    if not ok:
        raise AssertionError(model.fit_message)
    return model


def make_unfitted_octant_model(anchor_ip: str = "anchor-a") -> "OctantRTTModel":
    """Build a real but unfitted Octant RTT-distance model."""
    from scripts.libs.octant_simple.octant_model import OctantRTTModel

    coord = ANCHOR_COORDS[VpId(anchor_ip)]
    return OctantRTTModel(
        anchor_ip=anchor_ip, anchor_lat=coord.lat, anchor_lon=coord.lon
    )


def make_fitted_spotter_model(
    *,
    p_mu: Optional[np.ndarray] = None,
    p_sigma: Optional[np.ndarray] = None,
    k: float = 2.0,
    rtt_min: float = 0.0,
    rtt_max: float = 100.0,
    cutoff_rtt: float = 0.0,
) -> "SpotterRTTModel":
    """Build a hand-constructed fitted Spotter model with a parallel +/- k*sigma band.

    Defaults: mu(d) = 50 * d, sigma(d) = 50, k = 2 -> band of width 200 km.
    At RTT=20: inner=900, outer=1100. At RTT=30: inner=1400, outer=1600. The 50
    km/ms slope keeps every probe below the 2/3*c speed-of-internet line, so the
    new baseline clip in predict_distance_bounds is a no-op at these RTTs.

    `cutoff_rtt=0.0` (the default) preserves the legacy `rtt > rtt_max -> None`
    gate for tests that exercise that branch; set `cutoff_rtt > 0` to test the
    new flat-extension regime.
    """
    from scripts.libs.spotter.spotter_model import SpotterRTTModel

    if p_mu is None:
        p_mu = np.array([50.0, 0.0])
    if p_sigma is None:
        p_sigma = np.array([50.0])
    return SpotterRTTModel(
        p_mu=np.asarray(p_mu, dtype=float),
        p_sigma=np.asarray(p_sigma, dtype=float),
        k=k,
        rtt_min=rtt_min,
        rtt_max=rtt_max,
        cutoff_rtt=cutoff_rtt,
        fitted=True,
    )


def make_fitted_degenerate_spotter_model() -> "SpotterRTTModel":
    """Fitted Spotter model whose inner = outer (zero-width band)."""
    from scripts.libs.spotter.spotter_model import SpotterRTTModel

    return SpotterRTTModel(
        p_mu=np.array([50.0, 0.0]),
        p_sigma=np.array([0.0]),
        k=2.0,
        rtt_min=0.0,
        rtt_max=100.0,
        fitted=True,
    )


def make_unfitted_spotter_model() -> "SpotterRTTModel":
    """Build a real but unfitted Spotter RTT-distance model."""
    from scripts.libs.spotter.spotter_model import SpotterRTTModel

    return SpotterRTTModel()


# ---- FitSample factories (integration tests for the real fit() path) ----


def _probe_at_distance(vp_coord: Coord, distance_km: float) -> Coord:
    """Place a probe `distance_km` north of vp_coord along the meridian.

    haversine(vp_coord, returned_coord) == distance_km to within ~1e-6 km
    because pure-meridian distance on a sphere is exactly R * Δlat_radians.
    """
    delta_lat_deg = distance_km / _KM_PER_DEG_LAT
    return Coord(lat=vp_coord.lat + delta_lat_deg, lon=vp_coord.lon)


def make_low_envelope_fit_samples(
    vp_id: str = "anchor-a",
    slope: float = 0.02,
    intercept: float = 5.0,
    distances_km: Optional[list[float]] = None,
) -> list[FitSample]:
    """Build FitSamples whose LP fit recovers (slope, intercept).

    Latency = slope * distance + intercept; probes are placed at each target
    distance along the meridian. At slope=0.02, intercept=5: RTT=25 → 1000 km.
    """
    if distances_km is None:
        distances_km = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
    coord = ANCHOR_COORDS[VpId(vp_id)]
    return [
        FitSample(
            vp_id=VpId(vp_id),
            vp_coord=coord,
            probe_coord=_probe_at_distance(coord, d),
            latency=Latency(slope * d + intercept),
        )
        for d in distances_km
    ]


def make_bounded_spline_fit_samples(
    vp_id: str = "anchor-a",
    rtt_values: Optional[list[float]] = None,
) -> list[FitSample]:
    """Build FitSamples that yield a parallel ± 100 km hull at each RTT.

    For every RTT x, the sample set includes two probes:
      lower distance = 50 * x - 100
      upper distance = 50 * x + 100
    so at RTT=20 the Octant hull bounds come out to [900, 1100] km. The
    50 km/ms slope keeps every probe above the 2/3·c speed-of-internet line.
    """
    if rtt_values is None:
        rtt_values = [10.0, 20.0, 30.0, 40.0, 50.0]
    coord = ANCHOR_COORDS[VpId(vp_id)]
    samples: list[FitSample] = []
    for rtt in rtt_values:
        for d in (50.0 * rtt - 100.0, 50.0 * rtt + 100.0):
            samples.append(
                FitSample(
                    vp_id=VpId(vp_id),
                    vp_coord=coord,
                    probe_coord=_probe_at_distance(coord, d),
                    latency=Latency(rtt),
                )
            )
    return samples


def make_normal_dist_fit_samples(
    vp_id: str = "anchor-a",
    n_per_rtt: int = 4,
    rtt_values: Optional[list[float]] = None,
    spread_km: float = 100.0,
) -> list[FitSample]:
    """Build pooled-fit samples with a parallel band of width 2 * spread_km.

    For each RTT x, place n_per_rtt probes at distances evenly spread in
    [50*x - spread_km, 50*x + spread_km]. The pooled Spotter fit then sees a
    centered mean ≈ 50*x with spread ≈ spread_km. The 50 km/ms slope keeps
    every probe below the 2/3*c speed-of-internet line so all rows survive
    the baseline filter in SpotterRTTModel.fit.
    """
    if rtt_values is None:
        rtt_values = list(np.linspace(5.0, 50.0, 10))
    coord = ANCHOR_COORDS[VpId(vp_id)]
    samples: list[FitSample] = []
    for rtt in rtt_values:
        center = 50.0 * rtt
        for d in np.linspace(center - spread_km, center + spread_km, n_per_rtt):
            samples.append(
                FitSample(
                    vp_id=VpId(vp_id),
                    vp_coord=coord,
                    probe_coord=_probe_at_distance(coord, float(d)),
                    latency=Latency(float(rtt)),
                )
            )
    return samples
