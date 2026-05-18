"""CBGModel — composition of LTDModel + MTLMethod + CTRMethod with owned fallback.

geolocate runs the three-stage pipeline; if any stage fails and fallback is
enabled, returns the coord of the lowest-latency VP with status=FALLBACK.
If fallback is disabled or there are no observations, returns status=ERROR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from scripts.framework.v2.ctr.base import CTRMethod
from scripts.framework.v2.ltd.base import (
    AnnulusLTDModel,
    CircleLTDModel,
    FitSample,
    FittingResult,
    LTDModel,
)
from scripts.framework.v2.mtl.base import (
    AnnulusMTLMethod,
    CircleMTLMethod,
    MTLMethod,
)
from scripts.framework.v2.registry import CTR_REGISTRY, LTD_REGISTRY, MTL_REGISTRY
from scripts.framework.v2.types import Coord, Error, GeoStatus, Latency, VpId


class IncompatibleStagesError(TypeError):
    """Raised when a CBGModel is constructed from stages whose Circle/Annulus
    families don't match."""


@dataclass(frozen=True)
class GeoResult:
    """Top-level pipeline outcome for one probe.

    status == SUCCESS:  the three-stage pipeline produced coord.
    status == FALLBACK: pipeline failed, coord is the lowest-latency VP's
                       location. `error` documents the failure that
                       triggered the fallback.
    status == ERROR:    no coord. `error` describes why.
    """

    coord: Optional[Coord]
    status: GeoStatus
    error: Optional[Error] = None


class CBGModel:
    def __init__(
        self,
        latency_distance_model: LTDModel,
        multilateration_method: MTLMethod,
        centroid_method: CTRMethod,
        *,
        enable_fallback: bool = True,
    ) -> None:
        self._validate_family_pairing(latency_distance_model, multilateration_method)
        self.ltd = latency_distance_model
        self.mtl = multilateration_method
        self.ctr = centroid_method
        self.enable_fallback = enable_fallback

    @staticmethod
    def _validate_family_pairing(ltd: LTDModel, mtl: MTLMethod) -> None:
        if isinstance(mtl, AnnulusMTLMethod) and not isinstance(ltd, AnnulusLTDModel):
            raise IncompatibleStagesError(
                f"{type(mtl).__name__} requires an AnnulusLTDModel; "
                f"{type(ltd).__name__} produces disk constraints only"
            )
        if isinstance(mtl, CircleMTLMethod) and not isinstance(ltd, CircleLTDModel):
            raise IncompatibleStagesError(
                f"{type(mtl).__name__} consumes disk constraints; "
                f"{type(ltd).__name__} produces annular constraints"
            )

    def fit(self, samples: list[FitSample]) -> FittingResult:
        return self.ltd.fit(samples)

    def geolocate(
        self,
        obs: list[tuple[VpId, Coord, Latency]],
    ) -> GeoResult:
        """Run the three-stage pipeline on one probe's observations.

        Each entry of `obs` is (vp_id, vp_coord, measured_latency).
        """
        ltd_results = self.ltd.predict_all(obs)
        ok = [r for r in ltd_results if r.success]

        last_error: Optional[Error] = None

        mtl_result = self.mtl.multilaterate(ok)
        if mtl_result.success:
            ctr_result = self.ctr.select_centroid(mtl_result)
            if ctr_result.success and ctr_result.tg_coord is not None:
                return GeoResult(coord=ctr_result.tg_coord, status=GeoStatus.SUCCESS)
            last_error = ctr_result.error
        else:
            last_error = mtl_result.error

        if self.enable_fallback and obs:
            nearest = min(obs, key=lambda x: x[2])
            return GeoResult(
                coord=nearest[1],
                status=GeoStatus.FALLBACK,
                error=last_error,
            )

        return GeoResult(
            coord=None,
            status=GeoStatus.ERROR,
            error=last_error or Error.ALL_PHASES_FAILED,
        )

    @classmethod
    def from_config(
        cls,
        ltd: str,
        mtl: str,
        ctr: str,
        *,
        ltd_kwargs: Optional[dict] = None,
        mtl_kwargs: Optional[dict] = None,
        ctr_kwargs: Optional[dict] = None,
        enable_fallback: bool = True,
    ) -> "CBGModel":
        if ltd not in LTD_REGISTRY:
            raise KeyError(
                f"LTD model {ltd!r} not registered. Available: {sorted(LTD_REGISTRY)}"
            )
        if mtl not in MTL_REGISTRY:
            raise KeyError(
                f"MTL method {mtl!r} not registered. Available: {sorted(MTL_REGISTRY)}"
            )
        if ctr not in CTR_REGISTRY:
            raise KeyError(
                f"CTR method {ctr!r} not registered. Available: {sorted(CTR_REGISTRY)}"
            )
        return cls(
            latency_distance_model=LTD_REGISTRY[ltd](**(ltd_kwargs or {})),
            multilateration_method=MTL_REGISTRY[mtl](**(mtl_kwargs or {})),
            centroid_method=CTR_REGISTRY[ctr](**(ctr_kwargs or {})),
            enable_fallback=enable_fallback,
        )
