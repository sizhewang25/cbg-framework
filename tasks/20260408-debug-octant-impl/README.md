# Debug Octant Implementation

This task tracks the accuracy regression investigation for recent Octant
geolocation changes.

The immediate goal is to pinpoint which implementation change, or combination
of changes, caused the final Octant error accuracy to get worse relative to the
April 1, 2026 baseline.

Baseline reference:

- Commit `a9bb3d6` from 2026-04-01

Current investigation scope:

- [octant_geolocation.py](../../scripts/analysis/octant/octant_geolocation.py)
- [evaluate_million_scale.py](../../scripts/analysis/million_scale/evaluate_million_scale.py)
- [pyproject.toml](../../pyproject.toml)
- [poetry.lock](../../poetry.lock)

The working hypothesis is that several individually reasonable changes may have
interacted in a way that degraded the final Octant CDF and median error.

This task is for debugging and attribution, not for implementing a fix yet.
