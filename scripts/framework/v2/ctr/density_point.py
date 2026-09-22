"""Density-aware point estimate — Spotter's §III-B "maximum".

§III-B names three ways to collapse the probability surface to a single
location: the centre of the estimated region, the maximum, or the mean of the
distribution. **Only the maximum is implemented.**

All three were built and measured; `density_mean` and `density_region_center`
were removed once `density_argmax` was chosen as the shipped estimator, along
with a fourth (`density_mle`, a continuous refinement of the argmax by least
squares on `min Σ zᵢ²`). If one is needed again, the history is in
notes/2026-09-22-spotter-faithful-vs-harness-conclusion.md, and note that
`density_mle` additionally required `DensityField.constraints`, which was
dropped with it.

The estimator that remains is the MAP up to the grid's own resolution, and it
is the only one of the four invariant to how far the MTL truncated the field:
adding or dropping low-probability cells cannot move the maximum. That is what
makes it the right default under coarse-to-fine pruning, and it is the reason
it was the one kept.

It refuses to run on an `MTLResult` with no `density`, rather than falling back
to the `intersection` coords: a density-blind reading of a density field is
what the geometric CTRs already do, and silently substituting it here would
make the combo id a lie about which estimator produced the number.
"""

from __future__ import annotations

from typing import Optional

from scripts.framework.v2.ctr.base import CTRMethod, CTRResult
from scripts.framework.v2.mtl.base import DensityField, MTLResult
from scripts.framework.v2.registry import register_ctr
from scripts.framework.v2.types import Error


def _require_density(mtl: MTLResult) -> Optional[DensityField]:
    if mtl.density is None or not mtl.density.cells:
        return None
    return mtl.density


def _no_density() -> CTRResult:
    return CTRResult(success=False, error=Error.INSUFFICIENT_DATA)


@register_ctr("density_argmax")
class DensityArgmaxCTR(CTRMethod):
    """The single most probable cell's centre — §III-B's "the maximum".

    The MAP estimate up to the grid's own resolution. An H3-4 cell is ~20 km
    edge, so that quantisation is the estimator's floor; measured against a
    continuous solve on the same objective it cost ~21 km at p5 and ~20 km at
    p50 on as01.

    Reads `log_density` directly rather than `probabilities()`: the normalised
    form is a monotone transform of it, so the argmax is identical and the
    exponentiation is wasted work.
    """

    def _select_centroid(self, mtl: MTLResult) -> CTRResult:
        field = _require_density(mtl)
        if field is None:
            return _no_density()
        best = max(range(len(field.log_density)), key=lambda i: field.log_density[i])
        return CTRResult(success=True, tg_coord=field.cells[best])
