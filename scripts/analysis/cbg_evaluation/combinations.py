"""Pipeline combination registry for systematic CBG evaluation.

Defines the 9 valid pipeline configurations (3 distance × 3 multilateration paths)
and the ablation diff pairs for Error-Diff CDF analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class PipelineSpec:
    """One pipeline combination to evaluate."""

    combo_id: str       # e.g. "A1", "B2", "C3"
    label: str          # Human-readable label for plots
    distance: str       # framework registry name
    filtering: str
    multilateration: str
    centroid: str
    color: str          # Plot color (hex or named)
    linestyle: str      # matplotlib linestyle
    needs_lp_fit: bool = False
    needs_octant_fit: bool = False


# ---------------------------------------------------------------------------
# 9 combinations across 3 paths
# ---------------------------------------------------------------------------
# Path A: spherical → arithmetic_mean  (solid lines)
# Path B: shapely  → geometric_centroid (dashed lines)
# Path C: shapely  → arithmetic_mean   (dashdot lines)
# ---------------------------------------------------------------------------

COMBINATIONS: List[PipelineSpec] = [
    # Path A: spherical + arithmetic_mean
    PipelineSpec(
        "A1", "SoI + Spherical + Arith",
        "speed_of_internet", "redundant_circle", "spherical", "arithmetic_mean",
        "#0072B2", "-",
    ),
    PipelineSpec(
        "A2", "LP + Spherical + Arith",
        "low_envelope", "redundant_circle", "spherical", "arithmetic_mean",
        "#000000", "-",
        needs_lp_fit=True,
    ),
    PipelineSpec(
        "A3", "Spline + Spherical + Arith",
        "bounded_spline", "redundant_circle", "spherical", "arithmetic_mean",
        "#009E73", "-",
        needs_octant_fit=True,
    ),
    # Path B: shapely + geometric_centroid
    PipelineSpec(
        "B1", "SoI + Shapely + Geom",
        "speed_of_internet", "redundant_circle", "shapely", "geometric_centroid",
        "#0072B2", "--",
    ),
    PipelineSpec(
        "B2", "LP + Shapely + Geom",
        "low_envelope", "redundant_circle", "shapely", "geometric_centroid",
        "#000000", "--",
        needs_lp_fit=True,
    ),
    PipelineSpec(
        "B3", "Spline + Shapely + Geom",
        "bounded_spline", "redundant_circle", "shapely", "geometric_centroid",
        "#009E73", "--",
        needs_octant_fit=True,
    ),
    # Path C: shapely + arithmetic_mean
    PipelineSpec(
        "C1", "SoI + Shapely + Arith",
        "speed_of_internet", "redundant_circle", "shapely", "arithmetic_mean",
        "#0072B2", "-.",
    ),
    PipelineSpec(
        "C2", "LP + Shapely + Arith",
        "low_envelope", "redundant_circle", "shapely", "arithmetic_mean",
        "#000000", "-.",
        needs_lp_fit=True,
    ),
    PipelineSpec(
        "C3", "Spline + Shapely + Arith",
        "bounded_spline", "redundant_circle", "shapely", "arithmetic_mean",
        "#009E73", "-.",
        needs_octant_fit=True,
    ),
]

SPECS_BY_ID: Dict[str, PipelineSpec] = {s.combo_id: s for s in COMBINATIONS}

# ---------------------------------------------------------------------------
# Diff pairs for one-dimensional ablation analysis
# ---------------------------------------------------------------------------
# Each tuple: (combo_id_A, combo_id_B)
# Error-Diff CDF plots: error_A - error_B per probe
# Negative delta → A is better

DIFF_PAIRS: List[Tuple[str, str]] = [
    # Distance ablation (hold spherical + redundant_circle + arithmetic_mean)
    ("A1", "A2"),   # SoI vs LP
    ("A2", "A3"),   # LP vs Spline
    ("A1", "A3"),   # SoI vs Spline
    # Multilateration ablation (hold SoI + arith)
    ("A1", "C1"),   # spherical vs shapely
    # Centroid ablation (hold SoI + shapely)
    ("C1", "B1"),   # arith vs geom
    # Distance + centroid combined
    ("A1", "B1"),   # spherical+arith vs shapely+geom (same SoI distance)
]
