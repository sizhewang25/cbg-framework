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
       Symmetric annulus: [max(0, mu - k*sigma), max(0, mu + k*sigma)].
       Returns None outside the calibration RTT range -- the deg-3 mu(d)
       and deg-2 sigma(d) polynomials are not safe to extrapolate (see
       note section "Methodology notes / Differences from Spotter").

CAVEAT (load-bearing). On ping_10k_to_anchors the landmark-independence
claim FAILS: per-anchor Q-Q curves S-off the diagonal due to probe-side
last-mile heterogeneity. On anchors_meshed_pings it approximately holds
(sigma_z = 0.964). See the note for the panel-by-panel mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


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
    produces a symmetric annulus [mu(d) - k*sigma(d), mu(d) + k*sigma(d)],
    clipped at 0. Outside the calibration RTT range, returns None rather than
    extrapolate the polynomials.
    """

    p_mu: Optional[np.ndarray] = None
    p_sigma: Optional[np.ndarray] = None
    k: float = 0.0
    rtt_min: float = 0.0
    rtt_max: float = 0.0
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
    ) -> bool:
        """Fit the pooled mu(d), sigma(d) polynomials and calibrate k.

        Returns True on success; on failure sets fit_message and returns False.
        """
        rtt = np.asarray(rtt, dtype=float)
        dist = np.asarray(dist, dtype=float)
        valid = np.isfinite(rtt) & np.isfinite(dist) & (rtt > 0)
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
        self.metadata = {
            "n_pairs": int(len(rtt)),
            "n_bins_used": int(len(centers)),
            "target_coverage": float(target_coverage),
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

        Returns None when rtt is outside [rtt_min, rtt_max] -- the polynomial
        fits are not safe to extrapolate (see note section "Methodology notes").
        Inner is clamped at 0; outer is clamped at 0 as a safety net for
        pathological polynomials.
        """
        if not self.fitted or self.p_mu is None or self.p_sigma is None:
            return None
        if rtt < self.rtt_min or rtt > self.rtt_max:
            return None
        mu = float(np.polyval(self.p_mu, rtt))
        sigma = float(np.polyval(self.p_sigma, rtt))
        inner = max(0.0, mu - self.k * sigma)
        outer = max(0.0, mu + self.k * sigma)
        return inner, outer
