# framework v2 — Design

Status: abstract bases, types, registry, and composition implemented at `scripts/framework/v2/`. Concrete LTD/MTL/CTR classes not yet built. Coexists with v1 (`scripts/framework/` excluding `v2/`).

## Goals

Cleaner OOP separation of the three CBG pipeline stages with:
- Explicit, type-safe data classes between stages (vs v1's `MultilatResult` having both `vertices` and `region` fields, only one of which is set).
- `fit() → FittingResult` instead of v1's "fit mutates `self`" pattern.
- Named types (`Latency`, `Coord`, `Distance`) instead of raw floats and tuples.
- Annular constraints as a first-class concept (`Distance` is a range, not a scalar).
- Top-level `GeoResult` with a tri-valued status that distinguishes success / fallback / error.

Filtering is **not** a first-class stage. In v1 it was leaky (built into `planar_annulus_weighted`, incompatible with annular distance, etc.). Where filtering is needed, it lives as preprocessing inside an `MTLMethod` implementation.

## Design decisions

| # | Decision | Why |
|---|---|---|
| 1 | `Distance` is a range `{upper_km, lower_km}`, not a scalar | Octant / bounded-spline produces annuli; a scalar would silently drop one of v1's three distance models |
| 2 | One abstract `LTDModel` interface supports both per-VP (mainstream) and global (future) implementations | The two model shapes are an implementation detail; callers shouldn't care |
| 3 | `MTLResult.intersection: Polygon \| List[Coord] \| None` | Preserves v1's dual representation: spherical multilateration produces lat/lon vertices, planar multilateration produces Shapely geometries |
| 4 | `success: bool; error: Optional[Error] = None` per stage result | Beats stringly-typed status; `Error` is a shared enum |
| 5 | `CBGModel.geolocate` owns the fallback (closest VP by min latency) and returns `GeoResult{coord, status, error}` where `status ∈ {SUCCESS, FALLBACK, ERROR}` | Fallback rate is a real metric the benchmark layer reports; needs a place to live |
| 6 | LTD/MTL family bases (`CircleLTDModel ⊂ AnnulusLTDModel`, `CircleMTLMethod` / `AnnulusMTLMethod`) document the geometry of each stage but no longer gate composition: every (LTD × MTL) pair is legal. A disk is just an annulus with inner radius 0, so Circle LTDs flow into Annulus MTLs cleanly; Circle MTLs read only the outer bound so any LTD flows the other way too. | Replaces v1's hard-coded `if` ladder in `from_config`. Concrete impls pick a family for clarity, not enforcement. |

## Shared types

```python
# framework_v2/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import NewType, Optional, Union

Latency = NewType("Latency", float)       # milliseconds
VpId    = NewType("VpId", str)            # VP identifier (typically IP)

@dataclass(frozen=True)
class Coord:
    lat: float
    lon: float

@dataclass(frozen=True)
class Distance:
    upper_km: float
    lower_km: float = 0.0                 # 0 → full disk, >0 → annulus
    @property
    def is_annular(self) -> bool:
        return self.lower_km > 0.0

class Error(Enum):
    INSUFFICIENT_DATA        = "insufficient_data"
    NUMERICAL_FAILURE        = "numerical_failure"
    VP_NOT_FITTED            = "vp_not_fitted"
    RTT_OUT_OF_RANGE         = "rtt_out_of_range"
    NO_INTERSECTION          = "no_intersection"
    INSUFFICIENT_CONSTRAINTS = "insufficient_constraints"
    EMPTY_REGION             = "empty_region"
    DEGENERATE_REGION        = "degenerate_region"
    ALL_PHASES_FAILED        = "all_phases_failed"

class GeoStatus(Enum):
    SUCCESS  = "success"      # main pipeline produced a coord
    FALLBACK = "fallback"     # all 3 phases failed, fallback produced a coord
    ERROR    = "error"        # no coord at all
```

## Type hierarchy

Decision #6 expresses Circle/Annulus compatibility in the type system:

```
LTDModel (abstract)
└── AnnulusLTDModel (abstract)             produces Distance(lower_km >= 0, annular allowed)
    └── CircleLTDModel (abstract)          specialization: lower_km always 0
        └── (concrete circle impls)

MTLMethod (abstract)
├── CircleMTLMethod (abstract)             reads only tg_distance.upper_km
│   └── ...
└── AnnulusMTLMethod (abstract)            reads both lower_km and upper_km
    └── ...

CTRMethod (abstract)                       orthogonal to circle/annulus
```

Compatibility rule: **all four LTD × MTL combinations are legal.** A disk is just an annulus with inner radius 0, so `CircleLTDModel` is a subclass of `AnnulusLTDModel`; Circle LTDs flow into Annulus MTLs as `inner_radius_km=0`, and Annulus LTDs flow into Circle MTLs with the inner bound silently discarded (Circle MTLs read only `upper_km`).

| LTD ↓ \ MTL → | `CircleMTLMethod` | `AnnulusMTLMethod` |
|---|:-:|:-:|
| `CircleLTDModel` | native | inner=0 (polygon-shape output) |
| `AnnulusLTDModel` | inner bound discarded | native |

The registration decorators (`register_ltd`, `register_mtl`) reject any class that subclasses `LTDModel` / `MTLMethod` directly without picking a family — so typos surface at import.

## Stage 1 — Latency-to-distance model (`LTDModel`)

Per decision **#2**, the abstract interface is identical for per-VP and global models. The concrete class decides whether to partition training data by VP (mainstream) or pool it (future global). `predict` still takes a `VpId` so per-VP implementations can look up the right submodel; global implementations ignore that argument for the prediction itself and only use it to attach `vp_coord` to the result.

```python
# framework_v2/ltd/base.py
from abc import ABC, abstractmethod

@dataclass(frozen=True)
class FitSample:
    vp_id:       VpId
    vp_coord:    Coord
    probe_coord: Coord                    # known target coord at training time
    latency:     Latency

@dataclass(frozen=True)
class FittingResult:
    success: bool
    error:   Optional[Error] = None
    args:    Optional[dict]  = None       # fitted params (per-VP dict or global)

@dataclass(frozen=True)
class LTDResult:
    success:     bool
    error:       Optional[Error] = None
    vp_id:       Optional[VpId]      = None
    vp_coord:    Optional[Coord]     = None
    tg_distance: Optional[Distance]  = None

class LTDModel(ABC):
    """Two shapes both implement this interface — per-VP (mainstream)
    and global (future). Do not subclass directly; subclass one of the
    family bases below."""
    @abstractmethod
    def fit(self, samples: list[FitSample]) -> FittingResult: ...

    @abstractmethod
    def predict(self, vp_id: VpId, vp_coord: Coord, latency: Latency) -> LTDResult: ...

    def predict_all(
        self,
        obs: list[tuple[VpId, Coord, Latency]],
    ) -> list[LTDResult]:
        return [self.predict(vp, c, lat) for vp, c, lat in obs]


class AnnulusLTDModel(LTDModel, ABC):
    """Produces possibly-annular constraints (Distance.lower_km may be > 0)."""


class CircleLTDModel(AnnulusLTDModel, ABC):
    """Specialization of AnnulusLTDModel where Distance.lower_km is always 0."""
```

> **Why `predict` takes `vp_coord` explicitly rather than looking it up:** the model would otherwise have to cache VP coords from fit time, which couples fitting to prediction. The `CBGModel` layer knows the live VP coords from the observation set; passing them in keeps `LTDModel` stateless about geometry.

## Stage 2 — Multilateration (`MTLMethod`)

Per decision **#3**, `intersection` is a union covering both planar Shapely geometries and spherical vertex lists.

```python
# framework_v2/mtl/base.py
from shapely.geometry.base import BaseGeometry as ShapelyGeometry

Intersection = Union[ShapelyGeometry, list[Coord], None]

@dataclass(frozen=True)
class MTLResult:
    success:      bool
    error:        Optional[Error] = None
    intersection: Intersection    = None

class MTLMethod(ABC):
    """Do not subclass directly; subclass one of the family bases below."""
    @abstractmethod
    def multilaterate(self, results: list[LTDResult]) -> MTLResult: ...


class CircleMTLMethod(MTLMethod, ABC):
    """Consumes disk constraints. Must be paired with a CircleLTDModel."""


class AnnulusMTLMethod(MTLMethod, ABC):
    """Requires annular constraints. Must be paired with an AnnulusLTDModel."""
```

> Centroid implementations must handle both branches of the union. v1's `BoundaryVertexMeanCentroid` already does — that logic can be lifted directly.

## Stage 3 — Centroid (`CTRMethod`)

```python
# framework_v2/ctr/base.py
@dataclass(frozen=True)
class CTRResult:
    success:  bool
    error:    Optional[Error] = None
    tg_coord: Optional[Coord] = None

class CTRMethod(ABC):
    @abstractmethod
    def select_centroid(self, mtl: MTLResult) -> CTRResult: ...
```

## Composition — `CBGModel`

Per decision **#5**, `CBGModel.geolocate` owns the fallback (closest VP by min latency) and returns a top-level `GeoResult`.

```python
# framework_v2/model.py

@dataclass(frozen=True)
class GeoResult:
    coord:  Optional[Coord]
    status: GeoStatus                     # SUCCESS | FALLBACK | ERROR
    error:  Optional[Error] = None

class CBGModel:
    def __init__(self, latency_distance_model, multilateration_method, centroid_method,
                 *, enable_fallback: bool = True):
        self.ltd = latency_distance_model
        self.mtl = multilateration_method
        self.ctr = centroid_method
        self.enable_fallback = enable_fallback

    def fit(self, samples): ...

    def geolocate(self, obs):
        # 3-stage pipeline + closest-VP fallback when any stage fails
        # see scripts/framework/v2/model.py for full implementation
        ...

    @classmethod
    def from_config(cls, ltd: str, mtl: str, ctr: str, *, ltd_kwargs=None, ...):
        # registry-driven composition; raises KeyError on unknown names.
        ...
```

## Registry

`scripts/framework/v2/registry.py` exposes three dicts and three decorators:

```python
LTD_REGISTRY: dict[str, type[LTDModel]]
MTL_REGISTRY: dict[str, type[MTLMethod]]
CTR_REGISTRY: dict[str, type[CTRMethod]]

@register_ltd("speed_of_internet")
class SpeedOfInternetLTD(CircleLTDModel): ...
```

The decorator enforces family-base subclassing at import time. A class registered under `register_ltd` that subclasses `LTDModel` directly (not Circle/Annulus) raises `TypeError` when the module is imported — bugs surface immediately rather than at `CBGModel` construction.

The registries are intentionally separate from v1's `DISTANCE_REGISTRY` etc. No shared dicts; no silent name collisions across frameworks.

## Layout

```
scripts/framework/v2/
  __init__.py                 re-exports the headline public API
  types.py                    Coord, Distance, Latency, VpId, Error, GeoStatus
  registry.py                 LTD/MTL/CTR registries + register_* decorators
  model.py                    CBGModel, GeoResult
  ltd/
    __init__.py
    base.py                   LTDModel, CircleLTDModel, AnnulusLTDModel + data classes
  mtl/
    __init__.py
    base.py                   MTLMethod, CircleMTLMethod, AnnulusMTLMethod + data classes
  ctr/
    __init__.py
    base.py                   CTRMethod + CTRResult
```

Concrete impls (to be added under each of `ltd/`, `mtl/`, `ctr/`) will register themselves via the decorators, and `v2/__init__.py` will need to import them eagerly so the decorators run at package init.

## Open items

- **Trace / diagnostics on `GeoResult`.** The benchmark layer currently inspects intermediate fields like `multilateration_success`, `circles_used`, `fallback_reason` from v1's `GeolocationResult`. v2's minimal `GeoResult` doesn't carry these. Either (a) add an optional `trace: Optional[GeoTrace]` field, or (b) expose intermediate results separately.
- **Concrete impls.** Port v1's three distance models (`speed_of_internet` → `SpeedOfInternetLTD : CircleLTDModel`, `low_envelope` → `LowEnvelopeLTD : CircleLTDModel`, `bounded_spline` → `BoundedSplineLTD : AnnulusLTDModel`) and four multilateration methods (`spherical_circle`, `planar_circle` as `CircleMTLMethod`; `planar_annulus`, `planar_annulus_weighted` as `AnnulusMTLMethod`). Centroid methods port 1:1. The fitters in `scripts/libs/cbg_feasibility/rtt_model.py`, `scripts/libs/million_scale/evaluate_million_scale.py`, and `scripts/libs/octant/octant_evaluation.py` are reusable as-is.
- **Adapter shim or replacement?** v2 may either coexist with v1 indefinitely (benchmark gets a `--framework {v1,v2}` flag) or replace it after parity.
