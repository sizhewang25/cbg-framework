"""PlanarAnnulusWeightedMTL — Octant weighted face decomposition.

Each constraint contributes a weight derived from its RTT to faces of the
planar arrangement of all annulus boundaries. The top-weighted faces are
unioned until their cumulative weight clears `weight_threshold * Σwᵢ`,
yielding the feasible region.

`weight_mode` selects the decay family. Only ratios matter downstream — the
face argmax is invariant under `wᵢ → c·wᵢ` — so a family is characterised by
its dynamic range over the RTTs present, not by its absolute values:

  `exp`        `exp(-rtt_ms / weight_tau_ms)` — negative-exponential decay,
               a.k.a. a Boltzmann/softmax weight at temperature `weight_tau_ms`.
               Keys off the RTT *difference*, so it carries a dimensional
               constant: rescaling every RTT by c changes the answer.
  `inv_power`  `rtt_ms ** -weight_k` — the inverse-power (gravity) function.
               Keys off the RTT *ratio*, hence scale-free: `rtt → c·rtt` sends
               every weight to `c**-weight_k · wᵢ`, which the argmax discards.
  `uniform`    `1.0` — every surviving constraint votes equally. The control
               arm; note it still requires latency (see below).

`weight_tau_ms` applies only to `exp`, `weight_k` only to `inv_power`. The
defaults (`exp`, τ=50 ms) reproduce this class's original behaviour, which
several published configs depend on.

At small τ the exponential underflows rather than merely steepening: with
`exp(-72/1) ≈ 5e-32` summed against `exp(-3) ≈ 0.05`, the ratio is below
float64 epsilon, so at τ=1 every constraint beyond ~36 ms of the minimum RTT
contributes exactly zero. Read that setting as a hard top-k, not as a very
steep exponential.

`highest_weight_only=True` bypasses the cumulative-weight aggregation and
returns just the single top-weighted face — deterministic, always Polygon,
and avoids the MultiPolygon-discards-weights failure mode where downstream
area-biased sampling drifts off the densest overlap. `weight_threshold`
has no effect in that branch.

The original RTT is required for the weight. v2's `LTDResult` does not yet
carry latency; this wrapper reads `getattr(r, "latency", None)` so it stays
forward-compatible. Once `LTDResult.latency` lands, the happy path works
without further code changes. Until then, the wrapper short-circuits with
`Error.INSUFFICIENT_DATA` whenever any result lacks latency — and it does so
under `uniform` too, even though that mode never reads the value. Relaxing it
there would let the control admit VPs the other modes drop, so the arms would
no longer share a participant set, which is the only thing that makes them
comparable.

Wraps scripts/libs/octant/octant_geolocation.compute_feasible_region_weighted.
"""

from __future__ import annotations

import math
from dataclasses import replace

from scripts.framework.geometry import filter_redundant_outer_disks
from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.mtl.base import AnnulusMTLMethod, MTLResult
from scripts.framework.v2.mtl._annulus_common import (
    annular_constraint_from_ltd,
    wrap_region_as_mtl_result,
)
from scripts.framework.v2.registry import register_mtl
from scripts.framework.v2.types import Error
from scripts.libs.octant.octant_geolocation import (
    compute_feasible_region_weighted,
)

#: Accepted `weight_mode` values. Config kwargs reach `__init__` unvalidated
#: (`scripts/framework/v2/model.py` hands `mtl_kwargs` straight to the
#: constructor, and the `benchmark:` config block has no schema), so an
#: unrecognised mode has to raise here or it would fall through to the `exp`
#: default and silently produce a duplicate sweep arm.
_WEIGHT_MODES = ("exp", "inv_power", "uniform")

#: Floor applied before the inverse-power division, and only there: `exp` is
#: already finite at rtt=0 (it is 1.0), so flooring it would move the default
#: arm off the value every published as0* run scored.
#:
#: 0.1 ms is chosen to never bind on real data while still bounding the
#: dynamic range. The three production datasets hold no zero RTTs at all
#: (minima 0.584, 0.706, 0.619 ms) but do hold 127 / 104 / 91 sub-millisecond
#: rows, so a 1.0 ms floor would silently clip ~0.2% of the population and
#: make the arm something other than `rtt**-k`. Going much lower is worse in
#: the other direction: at k=3 a floor of 1e-9 would give a single zero-RTT
#: sample a weight of 1e27, which is enough to push every other constraint
#: below float64's relative epsilon and degenerate the region to whatever
#: that one annulus contains.
_RTT_FLOOR_MS = 0.1

#: `weight_tau_ms` when the exponential mode does not name one. Every published
#: as0* run passed 50.0 explicitly; this default is what makes a `run.json`
#: written before `weight_mode` existed replay as the run that produced it.
_DEFAULT_TAU_MS = 50.0


def _reject_unused(**supplied: float | None) -> None:
    """Refuse kwargs the selected `weight_mode` would ignore.

    Ignoring them silently would be worse than it looks. `mtl_kwargs` is
    copied verbatim into every fold's `run.json` and replayed from there by
    the analysis layer — `scripts/analysis/v5/modules/map_mtl.py` rebuilds the
    MTL as `MTL_REGISTRY[name](**json.loads(mtl_kwargs_json))` — so a
    silently-dropped key becomes a permanent record of a parameter that never
    took effect, indistinguishable from one that did.

    The realistic failure is copy-paste: every shipped as0* combo carries
    `weight_tau_ms: 50.0`, so a new arm written by editing one of them will
    bring the tau along into a mode that has no use for it.
    """
    named = sorted(k for k, v in supplied.items() if v is not None)
    if named:
        raise ValueError(
            f"{', '.join(named)} not accepted for this weight_mode: it "
            f"ignores them, and mtl_kwargs is recorded in run.json verbatim "
            f"and replayed from there, so an ignored key is indistinguishable "
            f"from an applied one. Remove the key or change weight_mode."
        )


@register_mtl("planar_annulus_weighted")
class PlanarAnnulusWeightedMTL(AnnulusMTLMethod):
    """Octant weighted feasible region via planar face decomposition.

    `enable_circle_filter` drops constraints whose outer disk fully contains
    another's outer disk before the weighted face decomposition. Same heuristic
    as PlanarAnnulusMTL; the cumulative weight Σwᵢ used by the threshold is
    recomputed over the kept constraints.
    """

    def __init__(
        self,
        weight_threshold: float = 0.5,
        weight_tau_ms: float | None = None,
        n_pts: int = 64,
        enable_circle_filter: bool = True,
        highest_weight_only: bool = False,
        weight_mode: str = "exp",
        weight_k: float | None = None,
    ) -> None:
        if weight_mode not in _WEIGHT_MODES:
            raise ValueError(
                f"weight_mode must be one of {list(_WEIGHT_MODES)}, got "
                f"{weight_mode!r}. The benchmark `mtl_kwargs:` block has no "
                f"schema, so this constructor is the only thing between a typo "
                f"and a sweep arm silently scoring the default exponential "
                f"under another arm's label."
            )

        if weight_mode == "exp":
            _reject_unused(weight_k=weight_k)
            tau = _DEFAULT_TAU_MS if weight_tau_ms is None else float(weight_tau_ms)
            if not math.isfinite(tau) or tau <= 0.0:
                raise ValueError(
                    f"weight_tau_ms must be finite and > 0, got {weight_tau_ms!r}. "
                    f"0 divides by zero; a negative tau silently inverts the "
                    f"ranking so the farthest vantage point wins."
                )
            self.weight_tau_ms, self.weight_k = tau, None
        elif weight_mode == "inv_power":
            _reject_unused(weight_tau_ms=weight_tau_ms)
            if weight_k is None:
                raise ValueError(
                    "weight_mode='inv_power' requires weight_k "
                    "(w = rtt ** -weight_k). It has no default on purpose: the "
                    "sweep scores k in {1, 2, 3} and none of them is the "
                    "obvious one to assume."
                )
            k = float(weight_k)
            if not math.isfinite(k) or k <= 0.0:
                raise ValueError(
                    f"weight_k must be finite and > 0, got {weight_k!r}. The "
                    f"minus sign is already in the formula "
                    f"(w = rtt ** -weight_k), so a negative value would make "
                    f"the slowest vantage point heaviest, and 0 is "
                    f"weight_mode='uniform' spelled obscurely."
                )
            self.weight_tau_ms, self.weight_k = None, k
        else:  # uniform
            _reject_unused(weight_tau_ms=weight_tau_ms, weight_k=weight_k)
            self.weight_tau_ms, self.weight_k = None, None

        self.weight_threshold = weight_threshold
        self.n_pts = n_pts
        self.enable_circle_filter = enable_circle_filter
        self.highest_weight_only = highest_weight_only
        self.weight_mode = weight_mode

    def _weight(self, rtt_ms: float) -> float:
        """Per-constraint weight for `rtt_ms` under the configured family.

        Split out from `_multilaterate` so each family is testable without
        building geometry — the geometry is what makes this MTL expensive,
        and none of it depends on which family is selected.
        """
        if self.weight_mode == "uniform":
            return 1.0
        if self.weight_mode == "inv_power":
            # The floor applies here and nowhere else — see _RTT_FLOOR_MS.
            return max(rtt_ms, _RTT_FLOOR_MS) ** (-self.weight_k)
        return math.exp(-rtt_ms / self.weight_tau_ms)

    def _multilaterate(self, results: list[LTDResult]) -> MTLResult:
        if not results:
            return MTLResult(success=False, error=Error.INSUFFICIENT_DATA)

        if self.enable_circle_filter:
            centers = [(r.vp_coord.lat, r.vp_coord.lon) for r in results]
            radii = [r.tg_distance.upper_km for r in results]
            keep = filter_redundant_outer_disks(centers, radii)
            results = [results[k] for k in keep]

        # VPs that survived the filter and decide the region (recording only).
        participating = tuple(r.vp_id for r in results)

        constraints = []
        for r in results:
            rtt_ms = getattr(r, "latency", None)
            if rtt_ms is None:
                return MTLResult(
                    success=False, error=Error.INSUFFICIENT_DATA,
                    participating_vp_ids=participating,
                )
            constraints.append(
                annular_constraint_from_ltd(
                    r,
                    rtt_ms=float(rtt_ms),
                    weight=self._weight(float(rtt_ms)),
                )
            )

        region = compute_feasible_region_weighted(
            constraints,
            weight_threshold=self.weight_threshold,
            n_pts=self.n_pts,
            highest_weight_only=self.highest_weight_only,
        )
        return replace(
            wrap_region_as_mtl_result(region),
            participating_vp_ids=participating,
        )
