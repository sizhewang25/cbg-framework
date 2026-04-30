"""Pipeline combination registry for systematic CBG evaluation.

Defines the 18 valid pipeline configurations and ablation diff pairs.
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
# 18 combinations across 7 paths
# ---------------------------------------------------------------------------
# Path A: spherical_circle → arithmetic_mean           (3 distance models)
# Path B: spherical_circle → geometric_centroid         (3 distance models)
# Path C: planar_circle    → arithmetic_mean            (3 distance models)
# Path D: planar_circle    → geometric_centroid          (3 distance models)
# Path E: planar_annulus   → arithmetic_mean            (spline only)
# Path F: planar_annulus   → geometric_centroid          (spline only)
# Path G: planar_annulus   → monte_carlo_median          (spline only)
# Path H: planar_circle    → monte_carlo_median          (3 distance models)
# ---------------------------------------------------------------------------

COMBINATIONS: List[PipelineSpec] = [
    # Path A: spherical_circle + arithmetic_mean
    PipelineSpec(
        "A1", "SoI + spherical_circle + Arith",
        "speed_of_internet", "redundant_circle", "spherical_circle", "arithmetic_mean",
        "#0072B2", "-",
    ),
    PipelineSpec(
        "A2", "LP + spherical_circle + Arith",
        "low_envelope", "redundant_circle", "spherical_circle", "arithmetic_mean",
        "#000000", "-",
        needs_lp_fit=True,
    ),
    PipelineSpec(
        "A3", "Spline + spherical_circle + Arith",
        "bounded_spline", "redundant_circle", "spherical_circle", "arithmetic_mean",
        "#009E73", "-",
        needs_octant_fit=True,
    ),
    # Path B: spherical_circle + geometric_centroid
    PipelineSpec(
        "B1", "SoI + spherical_circle + Geom",
        "speed_of_internet", "redundant_circle", "spherical_circle", "geometric_centroid",
        "#0072B2", "--",
    ),
    PipelineSpec(
        "B2", "LP + spherical_circle + Geom",
        "low_envelope", "redundant_circle", "spherical_circle", "geometric_centroid",
        "#000000", "--",
        needs_lp_fit=True,
    ),
    PipelineSpec(
        "B3", "Spline + spherical_circle + Geom",
        "bounded_spline", "redundant_circle", "spherical_circle", "geometric_centroid",
        "#009E73", "--",
        needs_octant_fit=True,
    ),
    # Path C: planar_circle + arithmetic_mean
    PipelineSpec(
        "C1", "SoI + planar_circle + Arith",
        "speed_of_internet", "redundant_circle", "planar_circle", "arithmetic_mean",
        "#0072B2", "-.",
    ),
    PipelineSpec(
        "C2", "LP + planar_circle + Arith",
        "low_envelope", "redundant_circle", "planar_circle", "arithmetic_mean",
        "#000000", "-.",
        needs_lp_fit=True,
    ),
    PipelineSpec(
        "C3", "Spline + planar_circle + Arith",
        "bounded_spline", "redundant_circle", "planar_circle", "arithmetic_mean",
        "#009E73", "-.",
        needs_octant_fit=True,
    ),
    # Path D: planar_circle + geometric_centroid
    PipelineSpec(
        "D1", "SoI + planar_circle + Geom",
        "speed_of_internet", "redundant_circle", "planar_circle", "geometric_centroid",
        "#0072B2", ":",
    ),
    PipelineSpec(
        "D2", "LP + planar_circle + Geom",
        "low_envelope", "redundant_circle", "planar_circle", "geometric_centroid",
        "#000000", ":",
        needs_lp_fit=True,
    ),
    PipelineSpec(
        "D3", "Spline + planar_circle + Geom",
        "bounded_spline", "redundant_circle", "planar_circle", "geometric_centroid",
        "#009E73", ":",
        needs_octant_fit=True,
    ),
    # Path E: planar_annulus + arithmetic_mean (spline only)
    PipelineSpec(
        "E3", "Spline + planar_annulus + Arith",
        "bounded_spline", "none", "planar_annulus", "arithmetic_mean",
        "#009E73", "-",
        needs_octant_fit=True,
    ),
    # Path F: planar_annulus + geometric_centroid (spline only)
    PipelineSpec(
        "F3", "Spline + planar_annulus + Geom",
        "bounded_spline", "none", "planar_annulus", "geometric_centroid",
        "#009E73", "-",
        needs_octant_fit=True,
    ),
    # Path G: planar_annulus + monte_carlo_median (spline only)
    PipelineSpec(
        "G3", "Spline + planar_annulus + MC Median",
        "bounded_spline", "none", "planar_annulus", "monte_carlo_median",
        "#009E73", "-",
        needs_octant_fit=True,
    ),
    # Path H: planar_circle + monte_carlo_median (all 3 distance models)
    PipelineSpec(
        "H1", "SoI + planar_circle + MC Median",
        "speed_of_internet", "redundant_circle", "planar_circle", "monte_carlo_median",
        "#0072B2", "-",
    ),
    PipelineSpec(
        "H2", "LP + planar_circle + MC Median",
        "low_envelope", "redundant_circle", "planar_circle", "monte_carlo_median",
        "#000000", "-",
        needs_lp_fit=True,
    ),
    PipelineSpec(
        "H3", "Spline + planar_circle + MC Median",
        "bounded_spline", "redundant_circle", "planar_circle", "monte_carlo_median",
        "#009E73", "-",
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
    # Distance ablation (hold spherical_circle + arithmetic_mean)
    ("A1", "A2"),   # SoI vs LP
    ("A2", "A3"),   # LP vs Spline
    ("A1", "A3"),   # SoI vs Spline
    # Centroid ablation (hold SoI + spherical_circle)
    ("A1", "B1"),   # arith vs geom (spherical_circle)
    # Centroid ablation (hold SoI + planar_circle)
    ("C1", "D1"),   # arith vs geom (planar_circle)
    # Multilateration ablation (hold SoI + arith)
    ("A1", "C1"),   # spherical_circle vs planar_circle (arith)
    # Multilateration ablation (hold SoI + geom)
    ("B1", "D1"),   # spherical_circle vs planar_circle (geom)
    # MC median vs geometric centroid (hold SoI + planar_circle)
    ("H1", "D1"),   # mc_median vs geom (planar_circle)
    # MC median vs arithmetic mean (hold SoI + planar_circle)
    ("H1", "C1"),   # mc_median vs arith (planar_circle)
    # planar_annulus vs planar_circle (hold spline + geom)
    ("F3", "D3"),   # planar_annulus vs planar_circle (geom, spline)
    # planar_annulus vs planar_circle (hold spline + arith)
    ("E3", "C3"),   # planar_annulus vs planar_circle (arith, spline)
]
