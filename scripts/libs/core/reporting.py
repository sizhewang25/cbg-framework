"""Reporting helpers for CBG evaluation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from scripts.libs.core.combinations import PipelineSpec
from scripts.libs.core.evaluate import ProbeResult


@dataclass(frozen=True)
class ResultCounts:
    """Disjoint outcome counts for evaluated probes."""

    total_probes: int
    estimated_count: int
    intersection_count: int
    fallback_count: int
    no_estimate_count: int
    other_estimate_count: int
    multilateration_success_count: int


def count_result_outcomes(results: Sequence[ProbeResult]) -> ResultCounts:
    """Count probe outcomes using disjoint buckets for CDF reporting."""
    total = len(results)
    estimated = sum(1 for r in results if r.error_km is not None)
    fallback = sum(
        1 for r in results if r.error_km is not None and r.fallback_used
    )
    intersection = sum(
        1
        for r in results
        if r.error_km is not None and r.did_intersect and not r.fallback_used
    )
    other_estimate = max(0, estimated - intersection - fallback)
    no_estimate = total - estimated
    multilateration_success = sum(1 for r in results if r.did_intersect)
    return ResultCounts(
        total_probes=total,
        estimated_count=estimated,
        intersection_count=intersection,
        fallback_count=fallback,
        no_estimate_count=no_estimate,
        other_estimate_count=other_estimate,
        multilateration_success_count=multilateration_success,
    )


def format_intersection_fallback_total(results: Sequence[ProbeResult]) -> str:
    """Format the count equation shown in error-CDF labels."""
    counts = count_result_outcomes(results)
    parts = [
        f"{counts.intersection_count:,} I",
        f"{counts.fallback_count:,} F",
    ]
    if counts.other_estimate_count:
        parts.append(f"{counts.other_estimate_count:,} O")
    if counts.no_estimate_count:
        parts.append(f"{counts.no_estimate_count:,} N")
    return f"{' + '.join(parts)} = {counts.total_probes:,}"


def count_fitted_anchors(
    spec: PipelineSpec,
    anchor_coords: Mapping[str, tuple],
    lp_models: Optional[Mapping[str, object]] = None,
    octant_models: Optional[Mapping[str, object]] = None,
) -> int:
    """Count anchors with a usable RTT-distance model for a pipeline spec.

    The theoretical speed-of-Internet model has no fitting step, so every
    anchor with coordinates is considered modelled.
    """
    anchor_ips = set(anchor_coords)
    if spec.distance == "speed_of_internet":
        return len(anchor_ips)
    if spec.distance == "low_envelope":
        models = lp_models or {}
    elif spec.distance == "bounded_spline":
        models = octant_models or {}
    else:
        return 0
    return sum(
        1
        for anchor_ip in anchor_ips
        if getattr(models.get(anchor_ip), "fitted", False)
    )
