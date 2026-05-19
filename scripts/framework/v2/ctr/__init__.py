"""Centroid selection stage."""

from scripts.framework.v2.ctr.base import CTRMethod, CTRResult
from scripts.framework.v2.ctr.boundary_vertex_mean import BoundaryVertexMeanCTR
from scripts.framework.v2.ctr.geometric_centroid import GeometricCentroidCTR
from scripts.framework.v2.ctr.geometric_median import GeometricMedianCTR
from scripts.framework.v2.ctr.monte_carlo_median import MonteCarloMedianCTR

__all__ = [
    "CTRMethod",
    "CTRResult",
    "BoundaryVertexMeanCTR",
    "GeometricCentroidCTR",
    "GeometricMedianCTR",
    "MonteCarloMedianCTR",
]
