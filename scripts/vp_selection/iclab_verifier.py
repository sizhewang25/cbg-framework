"""ICLab CBG-style accept/reject verifier (Niaki 2020 §App B, Cho 2024 §III).

Given a target's *claimed country* and a set of landmark→target RTT
measurements, returns `"accept"` if every landmark's implied one-way
propagation speed is at or below the calibrated speed limit, else `"reject"`.

The check is strict `>` against the limit:

    owtt        = rtt / 2                              # one-way time, ms
    implied_v   = dist(landmark, claimed_country) / owtt   # km/ms
    if implied_v > speed_limit:  reject

Distances must be precomputed via
`scripts.vp_selection.country_borders.precompute_landmark_country_distances`
(or the equivalent), with keys `(landmark_id, country_iso2)`.

The verifier is intentionally tolerant of bad data: zero/negative RTTs are
skipped (would divide by zero) and missing distance-lookup entries are
skipped (the verifier degrades toward ACCEPT rather than rejecting on
uncertainty — matches Cho's monotonicity argument).
"""

from __future__ import annotations

from typing import Literal, Mapping

Verdict = Literal["accept", "reject"]


def iclab_verify(
    landmark_rtts: Mapping[str, float],
    claimed_country: str,
    distances: Mapping[tuple[str, str], float],
    speed_limit_km_per_ms: float,
) -> Verdict:
    """Run the ICLab CBG verifier on one (target, claim) pair.

    `landmark_rtts` maps `landmark_id -> min RTT (ms)` for the landmarks
    measuring this target. `distances` is the precomputed
    `(landmark_id, country_iso2) -> km` lookup.
    """
    for lm_id, rtt in landmark_rtts.items():
        if rtt <= 0:
            continue
        d = distances.get((lm_id, claimed_country))
        if d is None:
            continue
        owtt = rtt / 2.0
        implied_v = d / owtt
        if implied_v > speed_limit_km_per_ms:
            return "reject"
    return "accept"
