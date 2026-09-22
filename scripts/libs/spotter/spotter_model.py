"""Spotter (Laki et al. 2011) pooled RTT-distance model.

Implements the pipeline laid out in notes/2026-05-17-spotter-normality-check.md:

    1. fit_mu_sigma(rtt, dist) -> p_mu(d) and p_log_sigma(d).
       Spotter's central claim is that the conditional distribution
       f_d(s) = N(mu(d), sigma(d)^2) is *landmark-independent*, so a single
       pooled pair describes all anchors.

       mu is a **monotonically non-decreasing, non-negative cubic fitted to the
       raw pairs** -- no binning. sigma is fitted **in log space** from mu's
       residuals, so it is positive unconditionally. Both departures from the
       original binned `np.polyfit` are explained in `fit_mu_sigma`.

    2. calibrate_k(rtt, dist, p_mu, p_log_sigma, target_coverage) -> k
       Optional empirical step. k = quantile(|z|, target_coverage) on
       the calibration set; distribution-free, the Spotter analogue of
       Octant's coverage-driven delta search. Skipped when no
       target_coverage is supplied -- the model then keeps k = 1.0,
       matching the paper's published band.

    3. SpotterRTTModel.predict_distance_bounds(rtt) -> (inner, outer)
       Symmetric annulus [max(0, mu - k*sigma), max(0, mu + k*sigma)],
       with the outer bound clipped by the 2/3*c speed-of-internet line
       (signal can't travel faster than light). Default k = 1.0 yields
       the paper's mu(d) +/- sigma(d) band (Figure 3a); passing
       target_coverage to .fit() widens the band to the calibrated
       k * sigma. Above `cutoff_rtt` the polynomial extrapolation is
       unsafe, so mu and sigma are held flat at their cutoff value --
       Octant-style graceful degradation.

CAVEAT (load-bearing). On ping_10k_to_anchors the landmark-independence
claim FAILS: per-anchor Q-Q curves S-off the diagonal due to probe-side
last-mile heterogeneity. On anchors_meshed_pings it approximately holds.
See the note for the panel-by-panel mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import nnls

from scripts.libs.octant.octant_model import sentinel_extension_distance


SPEED_OF_LIGHT_KM_MS = 300.0
THEORETICAL_SLOPE = 2.0 / (SPEED_OF_LIGHT_KM_MS * (2.0 / 3.0))  # ~ 0.01 ms/km


def compute_cutoff_rtt(
    rtts: np.ndarray,
    bin_size_ms: float = 5.0,
    cutoff_min_points: int = 30,
) -> float:
    """Right edge of the last RTT bin that contains >= cutoff_min_points.

    Mirrors the cutoff scan in octant/octant_model.py. Scans bins
    left-to-right; clamped to max(rtts). Returns 0.0 on empty input. When no
    bin meets the threshold cutoff_rtt stays at min(rtts), then is clamped to
    max(rtts) -- so it is always finite and within the data range.
    """
    rtts = np.asarray(rtts, dtype=float)
    if rtts.size == 0:
        return 0.0
    min_rtt = float(rtts.min())
    max_rtt = float(rtts.max())
    cutoff_rtt = min_rtt
    for bin_start in np.arange(min_rtt, max_rtt, bin_size_ms):
        bin_count = int(
            np.sum((rtts >= bin_start) & (rtts < bin_start + bin_size_ms))
        )
        if bin_count >= cutoff_min_points:
            cutoff_rtt = bin_start + bin_size_ms
    return min(cutoff_rtt, max_rtt)


#: Additive correction for estimating `log sigma` from `log |residual|`.
#:
#: For r ~ N(0, sigma^2), E[log|r|] = log sigma - (gamma + log 2)/2, so the raw
#: mean of log|r| under-states log sigma by this constant. Adding it back makes
#: the stage-2 regression unbiased for log sigma. Verified numerically: the
#: empirical mean of log|Z| over 4M standard normal draws is -0.63500 against
#: this constant's -0.63518.
LOG_ABS_NORMAL_BIAS = (np.euler_gamma + np.log(2.0)) / 2.0

#: Guards log(0) when a residual lands exactly on the fitted curve.
_LOG_SIGMA_FLOOR_KM = 1e-9

#: The monotonicity constraint is imposed on [0, MONOTONE_MARGIN * max(rtt)].
#: Beyond the data the cubic is unconstrained, so the margin is what stops it
#: turning over just outside the support -- which is where the observed
#: pathology lives (the fit is monotone *inside* the range; it dives outside).
#: 1.5 covers the extrapolation any caller can reach, because `cutoff_rtt`
#: already clamps evaluation to the last dense bin.
DEFAULT_MONOTONE_MARGIN = 1.5


def _monotone_nonneg_cubic(
    rtt: np.ndarray, dist: np.ndarray, hi: float
) -> np.ndarray:
    """Least-squares cubic that is non-decreasing AND non-negative on [0, hi].

    Solved as a non-negative least-squares problem rather than a constrained
    one. Write the derivative in the degree-2 Bernstein basis on [0, hi]:

        p'(x) = w0*B0(u) + w1*B1(u) + w2*B2(u),   u = x / hi

    The Bernstein basis is non-negative on the interval, so `w >= 0` makes
    `p' >= 0` there. Integrating gives p in terms of (c, w0, w1, w2), and since
    p(0) = c, requiring `c >= 0` as well makes p non-negative for all x >= 0 --
    an increasing function that starts non-negative cannot go below zero. So
    *plain* `nnls` over all four coefficients delivers both shape constraints at
    once, with no split intercept and no inequality solver.

    Why not SLSQP with `p'(x_k) >= 0` on a grid: measured, it fails to converge
    on saturating data. `nnls` is exact and always terminates.

    Why the analytic expansion below rather than sampling the curve and calling
    `np.polyfit`: the round-trip is lossy. Measured, it left `min p' = -0.77`
    just outside the sampled range, i.e. it silently broke the one property this
    function exists to guarantee.

    Note the constraint is *sufficient*, not necessary: some cubics that are
    monotone on [0, hi] have no non-negative Bernstein derivative
    representation, so this is a mild conservatism, not an exact QP.
    """
    u = rtt / hi
    # Integrals of the Bernstein basis: I_j(u) = \int_0^u B_j(v) dv.
    i0 = u - u**2 + u**3 / 3.0
    i1 = u**2 - (2.0 / 3.0) * u**3
    i2 = u**3 / 3.0
    design = np.column_stack(
        [np.ones_like(u), hi * i0, hi * i1, hi * i2]
    )
    coef, _ = nnls(design, dist)
    c, w0, w1, w2 = (float(v) for v in coef)

    # p(x) = c + hi * [ w0*u + (w1-w0)*u^2 + (w0 - 2*w1 + w2)/3 * u^3 ],  u = x/hi
    # Substituting u = x/hi turns each hi*u^j into x^j / hi^(j-1).
    a3 = (w0 - 2.0 * w1 + w2) / (3.0 * hi**2)
    a2 = (w1 - w0) / hi
    a1 = w0
    a0 = c
    return np.array([a3, a2, a1, a0], dtype=float)


#: Smallest sample count `fit` will attempt. The cubic needs 4 points and the
#: log-sigma polynomial `deg_sigma + 1`; below that the system is
#: under-determined. Replaces the old `min_per_bin` gate, which conflated "how
#: many points per bin" with "how many points at all" back when the fit
#: consumed bins.
MIN_FIT_SAMPLES = 4


def _reject_binning_kwargs(**kwargs) -> None:
    """Raise on any surviving bin-summary parameter.

    `n_bins` and `min_per_bin` configured the binning that `fit_mu_sigma` no
    longer does: mu is fitted to the raw pairs and sigma to its residuals.
    Silently ignoring them would leave 61 config files asserting a bandwidth
    that has no effect, which is the failure mode this rejection exists to
    prevent. Removing them from the signature outright would instead surface as
    an opaque `TypeError`, so they are kept, defaulted to None, and refused
    with an explanation.
    """
    offenders = {k: v for k, v in kwargs.items() if v is not None}
    if offenders:
        named = ", ".join(f"{k}={v!r}" for k, v in sorted(offenders.items()))
        raise ValueError(
            f"{named}: binning parameters are no longer supported. mu(d) is "
            f"fitted to the raw (rtt, distance) pairs under a monotonicity "
            f"constraint and sigma(d) to its residuals in log space, so there "
            f"is no binning to configure. Remove these keys from the caller "
            f"(including any ltd_kwargs in YAML)."
        )


def fit_mu_sigma(
    rtt: np.ndarray,
    dist: np.ndarray,
    *,
    deg_mu: int = 3,
    deg_sigma: int = 2,
    monotone_margin: float = DEFAULT_MONOTONE_MARGIN,
    n_bins: Optional[int] = None,
    min_per_bin: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit mu(d) monotone on raw pairs, then log sigma(d) from its residuals.

    **mu is fitted to the RAW (rtt, distance) pairs, monotonically.** Regression
    needs no binning, which removes bin count and bin width as unexamined
    bandwidth knobs. The monotonicity is the real content: more delay cannot
    mean less distance, and an unconstrained `np.polyfit` has no such guarantee
    -- notes/2026-05-17-spotter-normality-check.md records mu(d) extrapolating
    to -60,000 km on a narrow-support slice. `cutoff_rtt` bounds that damage
    downstream; this attacks it at source. See `_monotone_nonneg_cubic`.

    Note this also changes the *weighting*, not just the smoothing. Fitting bin
    means weighted every bin equally; fitting raw pairs weights every pair
    equally. On as01 bin occupancy spanned 31 to 2,769 points, so the two differ
    materially in sparse RTT regions. The raw fit is the standard estimator of
    E[distance | delay], which is what Spotter's f_d is defined as, so this
    moves toward the paper rather than away from it.

    **sigma is fitted in LOG space, from mu's residuals.** `exp` is positive
    unconditionally, which removes the pathology where a deg-2 sigma polynomial
    dips below zero and prediction has to refuse. Binning the raw distance also
    inflated sigma-hat by leaking the local slope into it -- at the old
    `n_bins=40` the inflation measured +18.3% against a known truth, matching
    sqrt(sigma^2 + slope^2 * width^2 / 12). Residuals carry no such term.

    `deg_mu` is accepted but must be 3: the Bernstein construction is
    cubic-specific. `n_bins` / `min_per_bin` are **deprecated and rejected** --
    see `_reject_binning_kwargs`.

    Args:
        rtt: RTT values in ms.
        dist: Great-circle distances in km, aligned with `rtt`.
        deg_mu: Must be 3.
        deg_sigma: Polynomial degree for log sigma(d).
        monotone_margin: mu is constrained on [0, monotone_margin * max(rtt)].
        n_bins: Deprecated. Must be None.
        min_per_bin: Deprecated. Must be None.

    Returns:
        (p_mu, p_log_sigma).

        `p_mu` is standard `np.polyval` coefficients for mu in km.
        `p_log_sigma` is `np.polyval` coefficients for **log sigma** -- callers
        must exponentiate, and should prefer `SpotterRTTModel.sigma_at`.
    """
    _reject_binning_kwargs(n_bins=n_bins, min_per_bin=min_per_bin)
    if deg_mu != 3:
        raise ValueError(
            f"deg_mu must be 3 (the monotone fit is cubic), got {deg_mu}"
        )
    rtt = np.asarray(rtt, dtype=float)
    dist = np.asarray(dist, dtype=float)
    if monotone_margin <= 0:
        raise ValueError(f"monotone_margin must be positive, got {monotone_margin}")

    hi = float(rtt.max()) * float(monotone_margin)
    if not np.isfinite(hi) or hi <= 0:
        raise ValueError("rtt must contain a positive finite maximum")
    p_mu = _monotone_nonneg_cubic(rtt, dist, hi)

    # Post-condition. The construction guarantees this analytically; the check
    # is here because "by construction" is what stops being true when someone
    # edits the expansion above.
    grid = np.linspace(0.0, hi, 1024)
    if float(np.polyval(np.polyder(p_mu), grid).min()) < -1e-6:
        raise ValueError("monotone cubic fit produced a decreasing segment")

    residual = dist - np.polyval(p_mu, rtt)
    p_log_sigma = np.polyfit(
        rtt,
        np.log(np.abs(residual) + _LOG_SIGMA_FLOOR_KM) + LOG_ABS_NORMAL_BIAS,
        deg_sigma,
    )
    return p_mu, p_log_sigma


def sigma_km(p_log_sigma: np.ndarray, rtt):
    """sigma in km from log-sigma coefficients: `exp(polyval(p_log_sigma, rtt))`.

    The **one** place the log parameterisation is undone. Module-level and
    shape-preserving (scalar in, float out; array in, array out) because the
    callers that need sigma are split between the two: `predict_*` want a
    scalar, while `calibrate_k` and the diagnostic/figure modules evaluate it
    over whole columns. An earlier scalar-only helper on the model failed for
    exactly that reason -- every array caller bypassed it and hand-rolled
    `np.exp(np.polyval(...))`, which is the duplication this exists to stop.

    Overflow yields `inf` rather than raising; callers filter on `isfinite`.
    """
    with np.errstate(over="ignore"):
        out = np.exp(np.polyval(p_log_sigma, rtt))
    return float(out) if np.isscalar(rtt) or np.ndim(rtt) == 0 else out


def calibrate_k(
    rtt: np.ndarray,
    dist: np.ndarray,
    p_mu: np.ndarray,
    p_log_sigma: np.ndarray,
    target_coverage: float = 0.95,
) -> float:
    """Empirical confidence multiplier: k = quantile(|z|, target_coverage).

    z = (dist - mu(d)) / sigma(d). Distribution-free; the Spotter analogue of
    Octant's coverage-driven delta search in scripts/libs/octant/octant_model.py.

    `p_log_sigma` holds coefficients for **log sigma**, so sigma is recovered by
    exponentiating. The old `sigma > 0` filter is gone -- `exp` cannot be
    non-positive -- but the finiteness filter stays, because `exp` of a large
    extrapolated value overflows to `inf`, which would silently zero out z.
    """
    rtt = np.asarray(rtt, dtype=float)
    dist = np.asarray(dist, dtype=float)
    mu = np.polyval(p_mu, rtt)
    sig = sigma_km(p_log_sigma, rtt)
    mask = np.isfinite(mu) & np.isfinite(sig) & (sig > 0)
    z = (dist[mask] - mu[mask]) / sig[mask]
    z = z[np.isfinite(z)]
    return float(np.quantile(np.abs(z), target_coverage))


@dataclass
class SpotterRTTModel:
    """Pooled Spotter RTT->distance model.

    One (p_mu, p_log_sigma, k) shared across all anchors.
    predict_distance_bounds produces a symmetric annulus
    [mu(d) - k*sigma(d), mu(d) + k*sigma(d)] clipped at 0 on the inner side and
    at the 2/3*c baseline on the outer. Default k = 1.0 reproduces the paper's
    mu(d) +/- sigma(d) band (Figure 3a); passing `target_coverage` to .fit()
    switches to a calibrated k = quantile(|z|, target_coverage), the Spotter
    analogue of Octant's delta-search.

    **`p_log_sigma` holds coefficients for log sigma, not sigma.** Always read
    it through `sigma_at`. The field was renamed from `p_sigma` precisely so
    that code written against the old meaning fails loudly instead of computing
    `exp(50)` and returning a plausible-looking band.

    Above `cutoff_rtt` both curves are held flat at the cutoff value. This
    matters MORE under the log parameterisation, not less: the old sigma decayed
    polynomially past the data (and could go negative), whereas `exp` of a
    growing quadratic diverges exponentially. On as01, sigma at 200 ms is
    8.3e3 km extrapolated against 2.4e3 km at the 91.9 ms support edge.

    **The three accessors deliberately disagree about out-of-range RTTs.** They
    share one evaluator (`_curves_at`) and differ only in range policy, which is
    a property of what each is for:

    | method | below `rtt_min` | above `cutoff_rtt` |
    |---|---|---|
    | `predict_distance` | `None` | `None` above `rtt_max` |
    | `predict_mu_sigma` | clamp to `rtt_min` | clamp to `cutoff_rtt` |
    | `predict_distance_bounds` | line through the origin | flat, then sentinel extension |

    `predict_distance` refuses because a bare mu with no band has no honest
    reading outside the data. `predict_mu_sigma` clamps because a density needs
    a finite `(mu, sigma)` at every RTT it is asked about, and a pessimistic one
    beats a missing one. `predict_distance_bounds` uses the origin line because
    a *bound* below the calibration range is still meaningful even when the
    polynomial is not. Do not "unify" these without changing what the callers
    mean.
    """

    p_mu: Optional[np.ndarray] = None
    p_log_sigma: Optional[np.ndarray] = None
    k: float = 1.0
    rtt_min: float = 0.0
    rtt_max: float = 0.0
    cutoff_rtt: float = 0.0
    sentinel_rtt: float = 10000.0
    fitted: bool = False
    fit_message: str = ""
    metadata: dict = field(default_factory=dict)

    def fit(
        self,
        rtt: np.ndarray,
        dist: np.ndarray,
        *,
        deg_mu: int = 3,
        deg_sigma: int = 2,
        target_coverage: Optional[float] = None,
        bin_size_ms: float = 5.0,
        cutoff_min_points: int = 30,
        n_bins: Optional[int] = None,
        min_per_bin: Optional[int] = None,
    ) -> bool:
        """Fit the pooled mu(d), log sigma(d) polynomials.

        Drops physically impossible rows (rtt < THEORETICAL_SLOPE * dist)
        first. Computes a per-fit `cutoff_rtt` (right edge of the last dense
        RTT bin) so prediction can stop extrapolating into the sparse tail --
        `bin_size_ms` / `cutoff_min_points` configure *that* scan only, and are
        unrelated to the binning the fit itself no longer does.

        `n_bins` / `min_per_bin` are deprecated and rejected; see
        `_reject_binning_kwargs`.

        When `target_coverage` is None (default), `self.k` is left at 1.0
        and the predicted band is the paper's mu +/- sigma. When set, k is
        calibrated empirically via `calibrate_k` so the band covers that
        fraction of the training residuals -- the Spotter analogue of
        Octant's delta-search.

        Returns True on success; on failure sets fit_message and returns False.
        """
        rtt = np.asarray(rtt, dtype=float)
        dist = np.asarray(dist, dtype=float)
        valid = (
            np.isfinite(rtt)
            & np.isfinite(dist)
            & (rtt > 0)
            & (dist > 0)
            & (rtt >= THEORETICAL_SLOPE * dist)
        )
        rtt = rtt[valid]
        dist = dist[valid]
        need = max(MIN_FIT_SAMPLES, deg_sigma + 1)
        if len(rtt) < need:
            self.fit_message = (
                f"Too few valid points: {len(rtt)} (need >= {need})"
            )
            self.fitted = False
            return False
        try:
            p_mu, p_log_sigma = fit_mu_sigma(
                rtt, dist,
                deg_mu=deg_mu,
                deg_sigma=deg_sigma,
                n_bins=n_bins,
                min_per_bin=min_per_bin,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.fit_message = f"Polynomial fit failed: {exc}"
            self.fitted = False
            return False
        self.p_mu = p_mu
        self.p_log_sigma = p_log_sigma
        if target_coverage is None:
            self.k = 1.0
        else:
            self.k = calibrate_k(
                rtt, dist, p_mu, p_log_sigma, target_coverage=target_coverage
            )
        self.rtt_min = float(rtt.min())
        self.rtt_max = float(rtt.max())
        self.cutoff_rtt = compute_cutoff_rtt(
            rtt, bin_size_ms=bin_size_ms, cutoff_min_points=cutoff_min_points
        )
        self.metadata = {
            "n_pairs": int(len(rtt)),
            "cutoff_rtt": float(self.cutoff_rtt),
            "k": float(self.k),
        }
        if target_coverage is not None:
            self.metadata["target_coverage"] = float(target_coverage)
        self.fitted = True
        self.fit_message = "ok"
        return True

    def sigma_at(self, rtt):
        """sigma(rtt) in km. Thin bound form of `sigma_km`.

        No clamping -- callers decide their own evaluation point, and the
        `predict_*` methods clamp to the calibrated range before calling.
        """
        if self.p_log_sigma is None:
            raise ValueError("model has no p_log_sigma; fit it first")
        return sigma_km(self.p_log_sigma, rtt)

    def _curves_at(self, eval_rtt: float) -> Tuple[float, float]:
        """Both curves at an ALREADY-CLAMPED rtt: (mu_km, sigma_km).

        Split out because the three public accessors below disagree about what
        to do outside the calibrated range but agree exactly on how to evaluate
        inside it. Keeping the evaluation in one place means a change of
        parameterisation touches one line rather than three call sites that are
        easy to update inconsistently.

        No range logic and no guards: that is the caller's job, and the reason
        this is private.
        """
        return (
            float(np.polyval(self.p_mu, eval_rtt)),
            float(sigma_km(self.p_log_sigma, eval_rtt)),
        )

    def predict_distance(self, rtt: float) -> Optional[float]:
        """Return mu(rtt) only (no band)."""
        if not self.fitted or self.p_mu is None:
            return None
        if rtt < self.rtt_min or rtt > self.rtt_max:
            return None
        return float(np.polyval(self.p_mu, rtt))

    def predict_mu_sigma(self, rtt: float) -> Optional[Tuple[float, float]]:
        """The (mu, sigma) of f_d = N(mu(d), sigma(d)^2) at this RTT.

        Spotter's model is a *distribution*, and a density-based evaluator needs
        it directly rather than the [mu - k*sigma, mu + k*sigma] band that
        `predict_distance_bounds` derives from it (the band is not invertible:
        the inner side clamps at 0 and the outer is clipped by 2/3*c).

        The polynomial is evaluated at `clip(rtt, rtt_min, cutoff_rtt)` — the
        same refusal-to-extrapolate that the bounds path applies, expressed as
        one clamp:

        - Above `cutoff_rtt`: held flat. This is now the more important of the
          two clamps: `exp` of a growing quadratic diverges exponentially in the
          sparse tail, where the old linear-space sigma merely drifted.
        - Below `rtt_min`: also held flat. The bounds path instead switches to a
          line through the origin, which is a statement about *limits* and has
          no (mu, sigma) reading -- a Gaussian centred on a shrinking radius is
          not what that construction means. Clamping keeps the density finite
          and defined; it makes the model deliberately pessimistic (too wide,
          too far) for sub-calibration RTTs rather than confidently wrong.

        Returns None only when unfitted or when the evaluation is non-finite.
        Sigma can no longer be non-positive -- `exp` forbids it -- so the old
        `sigma <= 0` rejection is gone. `mu` is likewise non-negative by
        construction now; the `max(0.0, ...)` is retained as belt and braces
        against a hand-constructed `p_mu`.
        """
        if not self.fitted or self.p_mu is None or self.p_log_sigma is None:
            return None
        hi = self.cutoff_rtt if self.cutoff_rtt > 0 else self.rtt_max
        eval_rtt = rtt
        if self.rtt_min > 0:
            eval_rtt = max(eval_rtt, self.rtt_min)
        if hi > 0:
            eval_rtt = min(eval_rtt, hi)
        mu, sigma = self._curves_at(eval_rtt)
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
            return None
        return max(0.0, mu), sigma

    def predict_distance_bounds(
        self, rtt: float
    ) -> Optional[Tuple[float, float]]:
        """Symmetric annulus (inner, outer) = mu(rtt) +/- k * sigma(rtt).

        Default k = 1.0 reproduces the paper's published band (Figure 3a);
        a calibrated k (set by fitting with target_coverage) widens the
        band to that empirical coverage of the training residuals.

        Three regimes, mirroring octant's hull conventions:

        - Below `rtt_min`: line through origin. inner = 0; outer scales
          linearly from 0 at rtt=0 to outer(rtt_min) at rtt=rtt_min. The
          polynomial isn't safe to extrapolate below the calibration
          range, but the line-through-origin gives a usable bound.
        - Above `cutoff_rtt` (when set): mu and sigma held flat at the
          cutoff value -- the deg-3 / deg-2 polynomials are not safe to
          extrapolate into the sparse tail. Inner stays at inner(cutoff);
          outer extends from outer(cutoff) toward a fictitious sentinel
          z = (sentinel_rtt, sentinel_rtt / THEORETICAL_SLOPE) on the
          2/3*c bound -- the Octant paper's smooth-transition construction
          (see octant.sentinel_extension_distance). Raises
          ValueError if rtt > sentinel_rtt.
        - Otherwise: plain polynomial evaluation.

        The outer bound is always clipped by the 2/3*c speed-of-internet
        line (rtt / THEORETICAL_SLOPE). If mu already exceeds that line
        in the polynomial regime, outer < inner and the caller sees the
        band as degenerate -- the right signal that the polynomial is in
        the unphysical regime at this RTT.

        Returns None when the model is unfitted, or when cutoff_rtt is
        unset (==0) and rtt > rtt_max (legacy gate for hand-constructed
        test fixtures).
        """
        if not self.fitted or self.p_mu is None or self.p_log_sigma is None:
            return None
        if rtt < self.rtt_min:
            if self.rtt_min <= 0:
                return None
            mu_min, sigma_min = self._curves_at(self.rtt_min)
            outer_at_min = min(
                max(0.0, mu_min + self.k * sigma_min),
                self.rtt_min / THEORETICAL_SLOPE,
            )
            return 0.0, (outer_at_min / self.rtt_min) * rtt
        if self.cutoff_rtt > 0:
            eval_rtt = min(rtt, self.cutoff_rtt)
        else:
            if rtt > self.rtt_max:
                return None
            eval_rtt = rtt
        mu, sigma = self._curves_at(eval_rtt)
        inner = max(0.0, mu - self.k * sigma)
        outer = max(0.0, mu + self.k * sigma)
        if self.cutoff_rtt > 0 and rtt > self.cutoff_rtt:
            outer = sentinel_extension_distance(
                rtt,
                self.cutoff_rtt,
                outer,
                THEORETICAL_SLOPE,
                self.sentinel_rtt,
            )
        outer = min(outer, rtt / THEORETICAL_SLOPE)
        return inner, outer
