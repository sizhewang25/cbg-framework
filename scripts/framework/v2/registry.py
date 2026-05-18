"""Three registries — one per stage — with family-checking decorators.

Each register_* decorator validates at import time that the registered class
subclasses the right abstract family base. A class that subclasses LTDModel
directly (without picking CircleLTDModel or AnnulusLTDModel) raises a
TypeError when its module is imported — bugs surface immediately rather than
at CBGModel composition.
"""

from __future__ import annotations

from typing import Callable, Type

from scripts.framework.v2.ctr.base import CTRMethod
from scripts.framework.v2.ltd.base import AnnulusLTDModel, CircleLTDModel, LTDModel
from scripts.framework.v2.mtl.base import AnnulusMTLMethod, CircleMTLMethod, MTLMethod

LTD_REGISTRY: dict[str, Type[LTDModel]] = {}
MTL_REGISTRY: dict[str, Type[MTLMethod]] = {}
CTR_REGISTRY: dict[str, Type[CTRMethod]] = {}


def register_ltd(name: str) -> Callable[[Type[LTDModel]], Type[LTDModel]]:
    """Register an LTDModel subclass under `name`. The class must subclass
    CircleLTDModel or AnnulusLTDModel, not LTDModel directly."""

    def deco(cls: Type[LTDModel]) -> Type[LTDModel]:
        if not issubclass(cls, (CircleLTDModel, AnnulusLTDModel)):
            raise TypeError(
                f"{cls.__name__} must subclass CircleLTDModel or AnnulusLTDModel "
                f"(subclassing LTDModel directly is not allowed)"
            )
        if name in LTD_REGISTRY:
            raise ValueError(f"Duplicate LTD registration: {name!r}")
        LTD_REGISTRY[name] = cls
        return cls

    return deco


def register_mtl(name: str) -> Callable[[Type[MTLMethod]], Type[MTLMethod]]:
    """Register an MTLMethod subclass under `name`. The class must subclass
    CircleMTLMethod or AnnulusMTLMethod, not MTLMethod directly."""

    def deco(cls: Type[MTLMethod]) -> Type[MTLMethod]:
        if not issubclass(cls, (CircleMTLMethod, AnnulusMTLMethod)):
            raise TypeError(
                f"{cls.__name__} must subclass CircleMTLMethod or AnnulusMTLMethod "
                f"(subclassing MTLMethod directly is not allowed)"
            )
        if name in MTL_REGISTRY:
            raise ValueError(f"Duplicate MTL registration: {name!r}")
        MTL_REGISTRY[name] = cls
        return cls

    return deco


def register_ctr(name: str) -> Callable[[Type[CTRMethod]], Type[CTRMethod]]:
    """Register a CTRMethod subclass under `name`."""

    def deco(cls: Type[CTRMethod]) -> Type[CTRMethod]:
        if not issubclass(cls, CTRMethod):
            raise TypeError(f"{cls.__name__} must subclass CTRMethod")
        if name in CTR_REGISTRY:
            raise ValueError(f"Duplicate CTR registration: {name!r}")
        CTR_REGISTRY[name] = cls
        return cls

    return deco
