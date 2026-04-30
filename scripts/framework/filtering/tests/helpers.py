"""Shared helpers for filtering tests."""

from __future__ import annotations

from scripts.framework.types import CircleConstraint


def circle(
    vp_ip: str,
    *,
    lat: float = 0.0,
    lon: float = 0.0,
    rtt_ms: float = 1.0,
    radius_km: float = 100.0,
    inner_radius_km: float = 0.0,
    weight: float = 1.0,
) -> CircleConstraint:
    return CircleConstraint(
        vp_lat=lat,
        vp_lon=lon,
        vp_ip=vp_ip,
        rtt_ms=rtt_ms,
        radius_km=radius_km,
        inner_radius_km=inner_radius_km,
        weight=weight,
    )
