"""Modular CBG Geolocation Framework.

HuggingFace-style pipeline with pluggable core phases:
  Phase 1: Distance estimation (RTT → radius)
  Optional preprocessing: Filtering (remove redundant constraints)
  Phase 2: Multilateration (intersect → region)
  Phase 3: Centroid selection (region → single point)

Usage::

    from scripts.framework import CBGPipeline

    # Million-Scale CBG (default)
    pipe = CBGPipeline.from_config()

    # Vanilla CBG (LP bestline)
    pipe = CBGPipeline.from_config(distance="low_envelope")
    pipe.distance.fit(df_asn=df_asn)

    # Custom combination
    pipe = CBGPipeline.from_config(
        distance="low_envelope",
        filtering="redundant_circle",  # use filtering="none" to disable
        multilateration="planar_circle",
        centroid="geometric_centroid",
    )

    location, circles = pipe.geolocate(measurements, anchor_coords)
    result = pipe.geolocate_with_metadata(measurements, anchor_coords)

Available components (string names for from_config):

    Distance:       speed_of_internet, low_envelope, bounded_spline
    Filtering:      redundant_circle, none
    Multilateration: spherical_circle, planar_circle, planar_annulus,
                    planar_annulus_weighted
    Centroid:       boundary_vertex_mean, geometric_centroid,
                    monte_carlo_median, geometric_median
"""

from scripts.framework.pipeline import CBGPipeline
from scripts.framework.registry import (
    CENTROID_REGISTRY,
    DISTANCE_REGISTRY,
    FILTERING_REGISTRY,
    MULTILATERATION_REGISTRY,
)
from scripts.framework.types import CircleConstraint, GeolocationResult, MultilatResult

# Import all variant modules to trigger @register_* decorators.
# Each import activates the decorator on the class, populating the registries.
import scripts.framework.distance.speed_of_internet  # noqa: F401
import scripts.framework.filtering.redundant_circle  # noqa: F401
import scripts.framework.filtering.none  # noqa: F401
import scripts.framework.multilateration.spherical_circle  # noqa: F401
import scripts.framework.centroid.boundary_vertex_mean  # noqa: F401

# Deferred imports — these have heavier dependencies (LP models, Octant, Shapely).
# They are imported lazily to avoid import errors when dependencies are missing.
try:
    import scripts.framework.distance.low_envelope  # noqa: F401
except ImportError:
    pass
try:
    import scripts.framework.distance.bounded_spline  # noqa: F401
except ImportError:
    pass
try:
    import scripts.framework.multilateration.planar_circle  # noqa: F401
except ImportError:
    pass
try:
    import scripts.framework.multilateration.planar_annulus_weighted  # noqa: F401
except ImportError:
    pass
try:
    import scripts.framework.multilateration.planar_annulus  # noqa: F401
except ImportError:
    pass
try:
    import scripts.framework.centroid.geometric  # noqa: F401
except ImportError:
    pass
try:
    import scripts.framework.centroid.monte_carlo_median  # noqa: F401
except ImportError:
    pass
try:
    import scripts.framework.centroid.geometric_median  # noqa: F401
except ImportError:
    pass

__all__ = [
    "CBGPipeline",
    "CircleConstraint",
    "GeolocationResult",
    "MultilatResult",
    "DISTANCE_REGISTRY",
    "FILTERING_REGISTRY",
    "MULTILATERATION_REGISTRY",
    "CENTROID_REGISTRY",
]
