"""CBGModel — composition of LTDModel + MTLMethod + CTRMethod with owned fallback.

geolocate runs the three-stage pipeline; if any stage fails and fallback is
enabled, returns the coord of the lowest-latency VP with status=FALLBACK.
If fallback is disabled or there are no observations, returns status=ERROR.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, ContextManager, Optional

from scripts.framework.v2.ctr.base import CTRMethod, CTRResult
from scripts.framework.v2.ltd.base import (
    AnnulusLTDModel,
    FitSample,
    FittingResult,
    LTDModel,
    LTDResult,
)
from scripts.framework.v2.mtl.base import (
    AnnulusMTLMethod,
    MTLMethod,
    MTLResult,
)
from scripts.framework.v2.registry import CTR_REGISTRY, LTD_REGISTRY, MTL_REGISTRY
from scripts.framework.v2.types import Coord, Error, GeoStatus, Latency, VpId

# Instrumentation hook contract for CBGModel.geolocate.
#
# `instrument(stage)` returns a context manager wrapping the call to the
# stage-named method. Benchmarks plug timing / memory profilers in here without
# the framework taking a dependency on either.
#
# Stage names (load-bearing — benchmarks dispatch on them):
#   "ltd" — wraps self.ltd.predict_all(obs)
#   "mtl" — wraps self.mtl.multilaterate(ok_ltd_results)
#   "ctr" — wraps self.ctr.select_centroid(mtl_result); only entered if MTL succeeded
StageInstrument = Callable[[str], ContextManager[None]]


class IncompatibleStagesError(TypeError):
    """Raised when a CBGModel pairs an AnnulusMTLMethod with a non-annulus LTD.

    The Annulus → Circle direction (AnnulusLTDModel feeding a CircleMTLMethod)
    is permitted: Circle MTLs only read `tg_distance.upper_km`, so the inner
    bound is silently discarded — annular constraints degrade cleanly to disks.
    The Circle LTD → Annulus MTL direction is still rejected because annular
    MTLs are designed for inner-bound information their LTD partner doesn't
    produce; allowing it would mask an LTD selection error.
    """


@dataclass(frozen=True)
class GeoResult:
    """Top-level pipeline outcome for one probe.

    status == SUCCESS:  the three-stage pipeline produced coord.
    status == FALLBACK: pipeline failed, coord is the lowest-latency VP's
                       location. `error` documents the failure that
                       triggered the fallback.
    status == ERROR:    no coord. `error` describes why.

    The per-stage results (`ltd_results`, `mtl_result`, `ctr_result`) are
    retained for inspection on every status. Stages that didn't run (e.g.
    CTR when MTL failed) are left as their default empty/None values.
    """

    coord: Optional[Coord]
    status: GeoStatus
    error: Optional[Error] = None
    ltd_results: tuple[LTDResult, ...] = ()
    mtl_result: Optional[MTLResult] = None
    ctr_result: Optional[CTRResult] = None


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
        """Reject only the unsafe direction.

        AnnulusMTLMethod needs annular semantics from its LTD partner — pairing
        it with a CircleLTDModel would silently degrade to a disk MTL and mask
        the LTD selection. CircleMTLMethod is permissive: it consumes only
        `tg_distance.upper_km`, so an AnnulusLTDModel can feed it (the inner
        bound is discarded, the pipeline still runs).
        """
        if isinstance(mtl, AnnulusMTLMethod) and not isinstance(ltd, AnnulusLTDModel):
            raise IncompatibleStagesError(
                f"{type(mtl).__name__} requires an AnnulusLTDModel; "
                f"{type(ltd).__name__} produces disk constraints only"
            )

    def fit(self, samples: list[FitSample]) -> FittingResult:
        return self.ltd.fit(samples)

    def geolocate(
        self,
        obs: list[tuple[VpId, Coord, Latency]],
        *,
        instrument: Optional[StageInstrument] = None,
    ) -> GeoResult:
        """Run the three-stage pipeline on one probe's observations.

        Each entry of `obs` is (vp_id, vp_coord, measured_latency).

        `instrument`, if provided, is called as `instrument(stage_name)` for
        each stage that actually runs. The returned context manager wraps the
        call. See StageInstrument above for the stage-name contract. Default
        (None) is a no-op nullcontext, so this is fully backward-compatible.
        """
        cm: StageInstrument = instrument if instrument is not None else (lambda _stage: nullcontext())

        with cm("ltd"):
            ltd_results = tuple(self.ltd.predict_all(obs))
        ok = [r for r in ltd_results if r.success]

        last_error: Optional[Error] = None
        ctr_result: Optional[CTRResult] = None

        with cm("mtl"):
            mtl_result = self.mtl.multilaterate(ok)
        if mtl_result.success:
            with cm("ctr"):
                ctr_result = self.ctr.select_centroid(mtl_result)
            if ctr_result.success and ctr_result.tg_coord is not None:
                return GeoResult(
                    coord=ctr_result.tg_coord,
                    status=GeoStatus.SUCCESS,
                    ltd_results=ltd_results,
                    mtl_result=mtl_result,
                    ctr_result=ctr_result,
                )
            last_error = ctr_result.error
        else:
            last_error = mtl_result.error

        if self.enable_fallback and obs:
            nearest = min(obs, key=lambda x: x[2])
            return GeoResult(
                coord=nearest[1],
                status=GeoStatus.FALLBACK,
                error=last_error,
                ltd_results=ltd_results,
                mtl_result=mtl_result,
                ctr_result=ctr_result,
            )

        return GeoResult(
            coord=None,
            status=GeoStatus.ERROR,
            error=last_error or Error.ALL_PHASES_FAILED,
            ltd_results=ltd_results,
            mtl_result=mtl_result,
            ctr_result=ctr_result,
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
