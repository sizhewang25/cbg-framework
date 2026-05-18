"""Per-anchor LP best-line RTT-to-distance model for CBG.

Calibrates a single line `rtt = slope · distance + intercept` per anchor by
solving the LP from Gueye et al. (IMC 2004) — the tightest line that lies
below every observation, with `slope >= 2/3·c` and `intercept >= 0`. Inverting
gives `distance = (rtt - intercept) / slope`, the upper bound on the target
distance for an observed RTT.

This is the simplified, production-only mirror of
`scripts/libs/cbg_feasibility/rtt_model.py`. The legacy module keeps its
binned-percentile fit, multi-stage outlier filter, and diagnostic bin fields
for the exploratory scripts that depend on them.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linprog


EARTH_RADIUS_KM = 6371.0
SPEED_OF_LIGHT_KM_S = 300_000.0
SPEED_OF_LIGHT_KM_MS = SPEED_OF_LIGHT_KM_S / 1000.0  # 300 km/ms
SPEED_RATIO = 2 / 3
# RTT lower bound at 2/3 c: rtt = 2·d / (2/3·c) = 0.01 ms/km
THEORETICAL_SLOPE = 2 / (SPEED_OF_LIGHT_KM_MS * SPEED_RATIO)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points in degrees, in km."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


@dataclass
class RTTDistanceModel:
    """Per-anchor LP best-line model. `predict_distance(rtt)` returns the
    upper bound on the great-circle distance from the anchor consistent
    with an observed RTT."""

    anchor_ip: str
    anchor_lat: float
    anchor_lon: float
    slope: Optional[float] = None
    intercept: Optional[float] = None
    n_measurements: int = 0
    fitted: bool = False
    fit_message: str = ""

    def fit(
        self,
        distances: np.ndarray,
        rtts: np.ndarray,
        baseline_slope: Optional[float] = None,
        enable_baseline_filter: bool = True,
    ) -> bool:
        """Fit the per-anchor LP best line.

        Pipeline: drop invalid rows → optional baseline filter → LP solve.
        """
        if baseline_slope is None:
            baseline_slope = THEORETICAL_SLOPE

        distances = np.asarray(distances, dtype=float)
        rtts = np.asarray(rtts, dtype=float)
        self.n_measurements = len(distances)

        if len(distances) != len(rtts):
            self.slope = None
            self.intercept = None
            self.fitted = False
            self.fit_message = "distance and rtt arrays must have same length"
            return False

        valid = (
            (distances > 0) & (rtts > 0) & np.isfinite(distances) & np.isfinite(rtts)
        )
        distances = distances[valid]
        rtts = rtts[valid]

        if enable_baseline_filter:
            distances, rtts = self.filter_baseline(distances, rtts, baseline_slope)

        result = self.fit_bestline_lp(distances, rtts, baseline_slope)
        self.slope = result["slope"]
        self.intercept = result["intercept"]
        self.fitted = result["success"]
        self.fit_message = result["message"]
        return self.fitted

    @staticmethod
    def filter_baseline(
        distances: np.ndarray,
        rtts: np.ndarray,
        baseline_slope: float = THEORETICAL_SLOPE,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Drop rows below the speed-of-light line (`rtt < baseline_slope · distance`).

        Physical sanity check: any RTT below 2·d/(2/3·c) is impossible at the
        chosen propagation speed and indicates a mislabeled coordinate or a
        measurement artifact. Returns the surviving (distances, rtts).
        """
        distances = np.asarray(distances, dtype=float)
        rtts = np.asarray(rtts, dtype=float)
        mask = rtts >= baseline_slope * distances
        return distances[mask], rtts[mask]

    @staticmethod
    def fit_bestline_lp(
        distances: np.ndarray,
        rtts: np.ndarray,
        baseline_slope: float = THEORETICAL_SLOPE,
    ) -> Dict[str, Any]:
        """Solve the CBG best-line LP.

        Inputs are assumed already filtered (no NaN, no sub-baseline points).
        See `fit` for the full pipeline.

            Minimize Σ [d_j - (m·g_j + b)]   (total slack above the line)
            s.t.     m·g_j + b <= d_j  for all j
                     m >= baseline_slope
                     b >= 0

        Returns dict: slope, intercept, n_points, success, violations, message.
        """
        distances = np.asarray(distances, dtype=float)
        rtts = np.asarray(rtts, dtype=float)
        n_points = len(distances)

        if n_points < 3:
            return {
                "slope": None,
                "intercept": None,
                "n_points": n_points,
                "success": False,
                "violations": 0,
                "message": f"need at least 3 points, got {n_points}",
            }

        # LP minimizes variables x = [m, b] in Σ [d_j - (m·g_j + b)]
        # i.e., -> Maximize m·Σg + n·b ↔ minimize -Σg·m - n·b.
        # c is the objective coefficient 1-D matrix of x: [[-Σg, -n]]
        c = np.array([-float(np.sum(distances)), -float(n_points)])
        # Inequality: A_ub · x ≤ b_ub
        # A_ub is the coefficient matrix on the left: each row is [gj, 1], shape (n, 2)
        A_ub = np.column_stack([distances, np.ones(n_points)])
        # b_ub is the coefficient matrix on the right: each row is [dj], shape (n,)
        b_ub = rtts
        # m bound: >= basline slope, b bound: >= 0
        bounds = [(baseline_slope, None), (0.0, None)]

        try:
            # solver backend: highs (C++, fast, modern)
            # result.x: The optimal decision vector. 
            # - result.x[0] is our slope; 
            # - result.x[1] is our intercept.
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        except Exception as e:
            return {
                "slope": None,
                "intercept": None,
                "n_points": n_points,
                "success": False,
                "violations": 0,
                "message": f"LP error: {e}",
            }

        if not result.success:
            return {
                "slope": None,
                "intercept": None,
                "n_points": n_points,
                "success": False,
                "violations": 0,
                "message": f"LP did not converge: {result.message}",
            }

        slope = float(result.x[0])
        intercept = float(result.x[1])
        predicted = slope * distances + intercept
        violations = int(np.sum(rtts < predicted - 1e-3))

        msg = f"LP converged: {n_points} points"
        if violations > 0:
            msg += f", {violations} violations"

        return {
            "slope": slope,
            "intercept": intercept,
            "n_points": n_points,
            "success": True,
            "violations": violations,
            "message": msg,
        }

    def predict_distance(self, rtt: float) -> Optional[float]:
        if not self.fitted:
            return None
        if self.slope is None or self.slope <= 0:
            return 0.0
        return max(0.0, (rtt - self.intercept) / self.slope)

    def save(self, filepath: Path) -> None:
        """Persist the fitted model as JSON for checkpoint recovery."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "RTTDistanceModel":
        with open(filepath) as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor_ip": self.anchor_ip,
            "anchor_lat": self.anchor_lat,
            "anchor_lon": self.anchor_lon,
            "slope": self.slope,
            "intercept": self.intercept,
            "n_measurements": self.n_measurements,
            "fitted": self.fitted,
            "fit_message": self.fit_message,
        }

    def __repr__(self) -> str:
        if self.fitted:
            return (
                f"RTTDistanceModel(anchor={self.anchor_ip}, "
                f"slope={self.slope:.6f} ms/km, "
                f"intercept={self.intercept:.2f} ms, "
                f"n={self.n_measurements})"
            )
        return (
            f"RTTDistanceModel(anchor={self.anchor_ip}, "
            f"fitted=False, msg='{self.fit_message}')"
        )
