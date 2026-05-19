"""framework v2 — Circle / Annulus typed CBG pipeline.

See scripts/framework/docs/framework_v2_design.md for the design rationale.

Top-level entry points:
    CBGModel(ltd, mtl, ctr)            direct composition (validates families)
    CBGModel.from_config(ltd, mtl, ctr) registry-driven composition

Subclass one of the family bases when implementing a new stage:
    LTDModel  → CircleLTDModel | AnnulusLTDModel
    MTLMethod → CircleMTLMethod | AnnulusMTLMethod
    CTRMethod → (no family split)
"""

from scripts.framework.v2.ctr.base import CTRMethod, CTRResult
from scripts.framework.v2.ctr.boundary_vertex_mean import BoundaryVertexMeanCTR
from scripts.framework.v2.ctr.geometric_centroid import GeometricCentroidCTR
from scripts.framework.v2.ctr.geometric_median import GeometricMedianCTR
from scripts.framework.v2.ctr.monte_carlo_median import MonteCarloMedianCTR
from scripts.framework.v2.ltd.base import (
    AnnulusLTDModel,
    CircleLTDModel,
    FitSample,
    FittingResult,
    LTDModel,
    LTDResult,
)
from scripts.framework.v2.ltd.bounded_spline import BoundedSplineLTD
from scripts.framework.v2.ltd.low_envelope import LowEnvelopeLTD
from scripts.framework.v2.ltd.normal_dist import NormalDistLTD
from scripts.framework.v2.ltd.speed_of_internet import SpeedOfInternetLTD
from scripts.framework.v2.model import CBGModel, GeoResult, IncompatibleStagesError
from scripts.framework.v2.mtl.base import (
    AnnulusMTLMethod,
    CircleMTLMethod,
    Intersection,
    MTLMethod,
    MTLResult,
)
from scripts.framework.v2.mtl.planar_annulus import PlanarAnnulusMTL
from scripts.framework.v2.mtl.planar_annulus_weighted import PlanarAnnulusWeightedMTL
from scripts.framework.v2.mtl.planar_circle import PlanarCircleMTL
from scripts.framework.v2.mtl.spherical_circle import SphericalCircleMTL
from scripts.framework.v2.registry import (
    CTR_REGISTRY,
    LTD_REGISTRY,
    MTL_REGISTRY,
    register_ctr,
    register_ltd,
    register_mtl,
)
from scripts.framework.v2.types import (
    Coord,
    Distance,
    Error,
    GeoStatus,
    Latency,
    VpId,
)

__all__ = [
    # types
    "Coord",
    "Distance",
    "Error",
    "GeoStatus",
    "Latency",
    "VpId",
    # ltd
    "AnnulusLTDModel",
    "CircleLTDModel",
    "FitSample",
    "FittingResult",
    "LTDModel",
    "LTDResult",
    "BoundedSplineLTD",
    "LowEnvelopeLTD",
    "NormalDistLTD",
    "SpeedOfInternetLTD",
    # mtl
    "AnnulusMTLMethod",
    "CircleMTLMethod",
    "Intersection",
    "MTLMethod",
    "MTLResult",
    "PlanarAnnulusMTL",
    "PlanarAnnulusWeightedMTL",
    "PlanarCircleMTL",
    "SphericalCircleMTL",
    # ctr
    "CTRMethod",
    "CTRResult",
    "BoundaryVertexMeanCTR",
    "GeometricCentroidCTR",
    "GeometricMedianCTR",
    "MonteCarloMedianCTR",
    # composition
    "CBGModel",
    "GeoResult",
    "IncompatibleStagesError",
    # registry
    "CTR_REGISTRY",
    "LTD_REGISTRY",
    "MTL_REGISTRY",
    "register_ctr",
    "register_ltd",
    "register_mtl",
]
