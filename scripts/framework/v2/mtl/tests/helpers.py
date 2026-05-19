"""Shared fixtures for v2 MTL wrapper tests.

`ltd_result` builds a successful LTDResult for cases where the MTL code only
reads vp_coord / tg_distance / vp_id.

`ltd_result_with_latency` returns a duck-typed namespace carrying an extra
`latency` attribute, so PlanarAnnulusWeightedMTL — which reads
`getattr(r, "latency", None)` — can be exercised before the real
`LTDResult.latency` field lands. Once that field is added, the namespace can
be deleted and the factory can return an LTDResult directly.
"""

from __future__ import annotations

from types import SimpleNamespace

from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.types import Coord, Distance, Latency, VpId


def ltd_result(
    vp_id: str,
    lat: float,
    lon: float,
    upper_km: float,
    lower_km: float = 0.0,
) -> LTDResult:
    return LTDResult(
        success=True,
        vp_id=VpId(vp_id),
        vp_coord=Coord(lat=lat, lon=lon),
        tg_distance=Distance(upper_km=upper_km, lower_km=lower_km),
    )


def ltd_result_with_latency(
    vp_id: str,
    lat: float,
    lon: float,
    upper_km: float,
    lower_km: float,
    latency: float,
) -> SimpleNamespace:
    """LTDResult-shaped namespace with a `latency` attribute.

    The weighted-annulus wrapper reads latency via `getattr`, so any object
    exposing the right attributes works at runtime. SimpleNamespace keeps the
    test honest about the duck-typing until LTDResult gains a real field.
    """
    return SimpleNamespace(
        success=True,
        vp_id=VpId(vp_id),
        vp_coord=Coord(lat=lat, lon=lon),
        tg_distance=Distance(upper_km=upper_km, lower_km=lower_km),
        latency=Latency(latency),
    )
