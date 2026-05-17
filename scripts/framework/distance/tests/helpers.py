"""Shared test helpers for RTT-distance wrapper tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scripts.libs.cbg_feasibility.rtt_model import RTTDistanceModel

if TYPE_CHECKING:
    from scripts.libs.octant.octant_model import OctantRTTModel
    from scripts.libs.spotter.spotter_model import SpotterRTTModel


ANCHOR_COORDS = {
    "anchor-a": (40.0, -74.0),
    "anchor-b": (41.0, -75.0),
    "anchor-c": (42.0, -76.0),
    "anchor-d": (43.0, -77.0),
}


def make_fitted_low_envelope_model(
    anchor_ip: str = "anchor-a",
    slope: float = 0.02,
    intercept: float = 5.0,
) -> RTTDistanceModel:
    """Fit a real LP model where RTT = 0.02 * distance + 5."""
    lat, lon = ANCHOR_COORDS[anchor_ip]
    distances = np.array([100, 200, 300, 400, 500, 600, 700, 800], dtype=float)
    rtts = slope * distances + intercept
    model = RTTDistanceModel(anchor_ip=anchor_ip, anchor_lat=lat, anchor_lon=lon)
    ok = model.fit(distances, rtts, method="lp", bin_size_km=100.0)
    if not ok:
        raise AssertionError(model.fit_message)
    return model


def make_unfitted_low_envelope_model(anchor_ip: str = "anchor-a") -> RTTDistanceModel:
    """Build a real but unfitted LP RTT-distance model."""
    lat, lon = ANCHOR_COORDS[anchor_ip]
    return RTTDistanceModel(anchor_ip=anchor_ip, anchor_lat=lat, anchor_lon=lon)


def make_fitted_octant_model(
    anchor_ip: str = "anchor-a",
    *,
    fit_spline: bool = True,
) -> OctantRTTModel:
    """Fit a real Octant model with an easily checked parallel distance band.

    For every RTT x, the training set contains two points:
      lower distance = 100 * x - 100
      upper distance = 100 * x + 100

    Thus at RTT=20, the hand-derived hull bounds are 1900..2100 km.
    """
    from scripts.libs.octant.octant_model import OctantRTTModel

    lat, lon = ANCHOR_COORDS[anchor_ip]
    rtt_values = np.array([10, 20, 30, 40, 50], dtype=float)
    rtts = np.repeat(rtt_values, 2)
    distances = np.ravel([
        [100.0 * rtt - 100.0, 100.0 * rtt + 100.0]
        for rtt in rtt_values
    ])
    model = OctantRTTModel(
        anchor_ip=anchor_ip,
        anchor_lat=lat,
        anchor_lon=lon,
        cutoff_variant="none",
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


def make_fitted_degenerate_octant_model(anchor_ip: str = "anchor-a") -> OctantRTTModel:
    """Fit a real Octant model whose lower and upper bounds are identical."""
    from scripts.libs.octant.octant_model import OctantRTTModel

    lat, lon = ANCHOR_COORDS[anchor_ip]
    rtts = np.array([10, 20, 30, 40, 50], dtype=float)
    distances = 100.0 * rtts
    model = OctantRTTModel(
        anchor_ip=anchor_ip,
        anchor_lat=lat,
        anchor_lon=lon,
        cutoff_variant="none",
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


def make_unfitted_octant_model(anchor_ip: str = "anchor-a") -> OctantRTTModel:
    """Build a real but unfitted Octant RTT-distance model."""
    from scripts.libs.octant.octant_model import OctantRTTModel

    lat, lon = ANCHOR_COORDS[anchor_ip]
    return OctantRTTModel(anchor_ip=anchor_ip, anchor_lat=lat, anchor_lon=lon)


def make_fitted_spotter_model(
    *,
    p_mu: np.ndarray = None,
    p_sigma: np.ndarray = None,
    k: float = 2.0,
    rtt_min: float = 0.0,
    rtt_max: float = 100.0,
) -> SpotterRTTModel:
    """Build a hand-constructed fitted Spotter model with a parallel +/- k*sigma band.

    Defaults: mu(d) = 100 * d, sigma(d) = 50, k = 2 -> band of width 200 km.
    At RTT=20: inner=1900, outer=2100. At RTT=30: inner=2900, outer=3100.
    """
    from scripts.libs.spotter.spotter_model import SpotterRTTModel

    if p_mu is None:
        p_mu = np.array([100.0, 0.0])
    if p_sigma is None:
        p_sigma = np.array([50.0])
    return SpotterRTTModel(
        p_mu=np.asarray(p_mu, dtype=float),
        p_sigma=np.asarray(p_sigma, dtype=float),
        k=k,
        rtt_min=rtt_min,
        rtt_max=rtt_max,
        fitted=True,
    )


def make_fitted_degenerate_spotter_model() -> SpotterRTTModel:
    """Build a fitted Spotter model whose inner = outer (zero-width band)."""
    from scripts.libs.spotter.spotter_model import SpotterRTTModel

    return SpotterRTTModel(
        p_mu=np.array([100.0, 0.0]),
        p_sigma=np.array([0.0]),
        k=2.0,
        rtt_min=0.0,
        rtt_max=100.0,
        fitted=True,
    )


def make_unfitted_spotter_model() -> SpotterRTTModel:
    """Build a real but unfitted Spotter RTT-distance model."""
    from scripts.libs.spotter.spotter_model import SpotterRTTModel

    return SpotterRTTModel()
