"""Registry mechanism for pluggable CBG pipeline components."""

from __future__ import annotations

from typing import Dict, Type

DISTANCE_REGISTRY: Dict[str, Type] = {}
FILTERING_REGISTRY: Dict[str, Type] = {}
MULTILATERATION_REGISTRY: Dict[str, Type] = {}
CENTROID_REGISTRY: Dict[str, Type] = {}


def _make_register(registry: dict):
    """Create a registration decorator for the given registry dict."""

    def register(name: str):
        def decorator(cls):
            if name in registry:
                raise ValueError(
                    f"Duplicate registration: {name!r} already registered "
                    f"as {registry[name].__name__}"
                )
            registry[name] = cls
            return cls

        return decorator

    return register


register_distance = _make_register(DISTANCE_REGISTRY)
register_filtering = _make_register(FILTERING_REGISTRY)
register_multilateration = _make_register(MULTILATERATION_REGISTRY)
register_centroid = _make_register(CENTROID_REGISTRY)
