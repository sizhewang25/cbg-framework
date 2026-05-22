"""Spotter (Laki et al. 2011) pooled RTT-distance model.

Implements the three-step calibration laid out in
notes/2026-05-17-spotter-normality-check.md:

    1. fit_mu_sigma(rtt, dist) -> polynomial fits p_mu(d), p_sigma(d)
       over RTT bins. Spotter's central claim is that the conditional
       distribution f_d(s) = N(mu(d), sigma(d)^2) is *landmark-independent*,
       so a single pooled pair describes all anchors.

    2. calibrate_k(rtt, dist, p_mu, p_sigma, target_coverage) -> k
       Empirical: k = quantile(|z|, target_coverage) on the calibration set.
       Distribution-free; matches Octant's coverage-driven delta search in
       spirit. Avoids assuming sigma_z = 1 (the note's panel (b) shows
       sigma_z = 0.894 on probes->anchors, so the parametric k = Phi^-1(.)
       would be wrong by ~12 %).

    3. SpotterRTTModel.predict_distance_bounds(rtt) -> (inner, outer)
       Symmetric annulus [max(0, mu - k*sigma), max(0, mu + k*sigma)],
       with the outer bound clipped by the 2/3*c speed-of-internet line
       (signal can't travel faster than light). Above `cutoff_rtt` the
       polynomial extrapolation is unsafe, so mu and sigma are held flat
       at their cutoff value -- Octant-style graceful degradation.

CAVEAT (load-bearing). On ping_10k_to_anchors the landmark-independence
claim FAILS: per-anchor Q-Q curves S-off the diagonal due to probe-side
last-mile heterogeneity. On anchors_meshed_pings it approximately holds
(sigma_z = 0.964). See the note for the panel-by-panel mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from scripts.libs.octant_simple.octant_model import sentinel_extension_distance


SPEED_OF_LIGHT_KM_MS = 300.0
THEORETICAL_SLOPE = 2.0 / (SPEED_OF_LIGHT_KM_MS * (2.0 / 3.0))  # ~ 0.01 ms/km


def compute_cutoff_rtt(
    rtts: np.ndarray,
    bin_size_ms: float = 5.0,
    cutoff_min_points: int = 30,
) -> float:
    """Right edge of the last RTT bin that contains >= cutoff_min_points.

    Mirrors the cutoff scan in octant_simple/octant_model.py. Scans bins
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


def fit_mu_sigma(
    rtt: np.ndarray,
    dist: np.ndarray,
    n_bins: int = 40,
    min_per_bin: int = 30,
    deg_mu: int = 3,
    deg_sigma: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin RTT, fit polynomials to per-bin mean and std of distance.

    Mirrors scripts/libs/cbg_feasibility/spotter_normality_check.fit_mu_sigma.

    Args:
        rtt: RTT values in ms.
        dist: Great-circle distances in km, aligned with `rtt`.
        n_bins: Number of equal-width RTT bins.
        min_per_bin: Bins with fewer points are dropped.
        deg_mu: Polynomial degree for mu(d).
        deg_sigma: Polynomial degree for sigma(d).

    Returns:
        (p_mu, p_sigma, centers, mus, sigmas), polynomial coeffs
        highest-degree first (np.polyfit convention).
    """
    rtt = np.asarray(rtt, dtype=float)
    dist = np.asarray(dist, dtype=float)
    edges = np.linspace(rtt.min(), rtt.max(), n_bins + 1)
    centers, mus, sigmas = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (rtt >= lo) & (rtt < hi)
        if mask.sum() < min_per_bin:
            continue
        centers.append(0.5 * (lo + hi))
        mus.append(float(np.mean(dist[mask])))
        sigmas.append(float(np.std(dist[mask])))
    centers = np.asarray(centers)
    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)
    p_mu = np.polyfit(centers, mus, deg_mu)
    p_sigma = np.polyfit(centers, sigmas, deg_sigma)
    return p_mu, p_sigma, centers, mus, sigmas


def calibrate_k(
    rtt: np.ndarray,
    dist: np.ndarray,
    p_mu: np.ndarray,
    p_sigma: np.ndarray,
    target_coverage: float = 0.95,
) -> float:
    """Empirical confidence multiplier: k = quantile(|z|, target_coverage).

    z = (dist - mu(d)) / sigma(d). Points with non-positive sigma are dropped.
    """
    rtt = np.asarray(rtt, dtype=float)
    dist = np.asarray(dist, dtype=float)
    mu = np.polyval(p_mu, rtt)
    sig = np.polyval(p_sigma, rtt)
    mask = (sig > 0) & np.isfinite(mu) & np.isfinite(sig)
    z = (dist[mask] - mu[mask]) / sig[mask]
    z = z[np.isfinite(z)]
    return float(np.quantile(np.abs(z), target_coverage))


@dataclass
class SpotterRTTModel:
    """Pooled Spotter RTT->distance model.

    One (p_mu, p_sigma, k) shared across all anchors. predict_distance_bounds
    produces a symmetric annulus [mu(d) - k*sigma(d), mu(d) + k*sigma(d)]
    clipped at 0 on the inner side and at the 2/3*c baseline on the outer.
    Above `cutoff_rtt` the polynomial is held flat at the cutoff -- the
    extrapolation is not safe, so the model falls back to a constant-width
    band rather than letting deg-3 mu / deg-2 sigma diverge.
    """

    p_mu: Optional[np.ndarray] = None
    p_sigma: Optional[np.ndarray] = None
    k: float = 0.0
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
        n_bins: int = 40,
        min_per_bin: int = 30,
        deg_mu: int = 3,
        deg_sigma: int = 2,
        target_coverage: float = 0.95,
        bin_size_ms: float = 5.0,
        cutoff_min_points: int = 30,
    ) -> bool:
        """Fit the pooled mu(d), sigma(d) polynomials and calibrate k.

        Drops physically impossible rows (rtt < THEORETICAL_SLOPE * dist)
        before binning. Computes a per-fit `cutoff_rtt` (right edge of the
        last dense bin) so prediction can stop extrapolating into the sparse
        tail. Returns True on success; on failure sets fit_message and
        returns False.
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
        if len(rtt) < max(min_per_bin, deg_mu + 1, deg_sigma + 1):
            self.fit_message = (
                f"Too few valid points: {len(rtt)} (need >= {min_per_bin})"
            )
            self.fitted = False
            return False
        try:
            p_mu, p_sigma, centers, _, _ = fit_mu_sigma(
                rtt, dist,
                n_bins=n_bins,
                min_per_bin=min_per_bin,
                deg_mu=deg_mu,
                deg_sigma=deg_sigma,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.fit_message = f"Polynomial fit failed: {exc}"
            self.fitted = False
            return False
        if len(centers) < max(deg_mu + 1, deg_sigma + 1):
            self.fit_message = (
                f"Too few populated bins: {len(centers)}"
            )
            self.fitted = False
            return False
        self.p_mu = p_mu
        self.p_sigma = p_sigma
        self.k = calibrate_k(rtt, dist, p_mu, p_sigma, target_coverage=target_coverage)
        self.rtt_min = float(rtt.min())
        self.rtt_max = float(rtt.max())
        self.cutoff_rtt = compute_cutoff_rtt(
            rtt, bin_size_ms=bin_size_ms, cutoff_min_points=cutoff_min_points
        )
        self.metadata = {
            "n_pairs": int(len(rtt)),
            "n_bins_used": int(len(centers)),
            "target_coverage": float(target_coverage),
            "cutoff_rtt": float(self.cutoff_rtt),
        }
        self.fitted = True
        self.fit_message = "ok"
        return True

    def predict_distance(self, rtt: float) -> Optional[float]:
        """Return mu(rtt) only (no band)."""
        if not self.fitted or self.p_mu is None:
            return None
        if rtt < self.rtt_min or rtt > self.rtt_max:
            return None
        return float(np.polyval(self.p_mu, rtt))

    def predict_distance_bounds(
        self, rtt: float
    ) -> Optional[Tuple[float, float]]:
        """Symmetric annulus (inner, outer) = mu(rtt) +/- k*sigma(rtt).

        Three regimes, mirroring octant_simple's hull conventions:

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
          (see octant_simple.sentinel_extension_distance). Raises
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
        if not self.fitted or self.p_mu is None or self.p_sigma is None:
            return None
        if rtt < self.rtt_min:
            if self.rtt_min <= 0:
                return None
            mu_min = float(np.polyval(self.p_mu, self.rtt_min))
            sigma_min = float(np.polyval(self.p_sigma, self.rtt_min))
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
        mu = float(np.polyval(self.p_mu, eval_rtt))
        sigma = float(np.polyval(self.p_sigma, eval_rtt))
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
