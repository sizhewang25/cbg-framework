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
# Path A: spherical → arithmetic_mean           (3 distance models)
# Path B: spherical → geometric_centroid         (3 distance models)
# Path C: shapely   → arithmetic_mean            (3 distance models)
# Path D: shapely   → geometric_centroid          (3 distance models)
# Path E: unweighted_annulus → arithmetic_mean    (spline only)
# Path F: unweighted_annulus → geometric_centroid  (spline only)
# Path G: unweighted_annulus → monte_carlo_median  (spline only)
# Path H: shapely   → monte_carlo_median          (3 distance models)
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
    # Path B: spherical + geometric_centroid
    PipelineSpec(
        "B1", "SoI + Spherical + Geom",
        "speed_of_internet", "redundant_circle", "spherical", "geometric_centroid",
        "#0072B2", "--",
    ),
    PipelineSpec(
        "B2", "LP + Spherical + Geom",
        "low_envelope", "redundant_circle", "spherical", "geometric_centroid",
        "#000000", "--",
        needs_lp_fit=True,
    ),
    PipelineSpec(
        "B3", "Spline + Spherical + Geom",
        "bounded_spline", "redundant_circle", "spherical", "geometric_centroid",
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
    # Path D: shapely + geometric_centroid
    PipelineSpec(
        "D1", "SoI + Shapely + Geom",
        "speed_of_internet", "redundant_circle", "shapely", "geometric_centroid",
        "#0072B2", ":",
    ),
    PipelineSpec(
        "D2", "LP + Shapely + Geom",
        "low_envelope", "redundant_circle", "shapely", "geometric_centroid",
        "#000000", ":",
        needs_lp_fit=True,
    ),
    PipelineSpec(
        "D3", "Spline + Shapely + Geom",
        "bounded_spline", "redundant_circle", "shapely", "geometric_centroid",
        "#009E73", ":",
        needs_octant_fit=True,
    ),
    # Path E: unweighted_annulus + arithmetic_mean (spline only)
    PipelineSpec(
        "E3", "Spline + Annulus + Arith",
        "bounded_spline", "redundant_circle", "unweighted_annulus", "arithmetic_mean",
        "#009E73", "-",
        needs_octant_fit=True,
    ),
    # Path F: unweighted_annulus + geometric_centroid (spline only)
    PipelineSpec(
        "F3", "Spline + Annulus + Geom",
        "bounded_spline", "redundant_circle", "unweighted_annulus", "geometric_centroid",
        "#009E73", "-",
        needs_octant_fit=True,
    ),
    # Path G: unweighted_annulus + monte_carlo_median (spline only)
    PipelineSpec(
        "G3", "Spline + Annulus + MC Median",
        "bounded_spline", "redundant_circle", "unweighted_annulus", "monte_carlo_median",
        "#009E73", "-",
        needs_octant_fit=True,
    ),
    # Path H: shapely + monte_carlo_median (all 3 distance models)
    PipelineSpec(
        "H1", "SoI + Shapely + MC Median",
        "speed_of_internet", "redundant_circle", "shapely", "monte_carlo_median",
        "#0072B2", "-",
    ),
    PipelineSpec(
        "H2", "LP + Shapely + MC Median",
        "low_envelope", "redundant_circle", "shapely", "monte_carlo_median",
        "#000000", "-",
        needs_lp_fit=True,
    ),
    PipelineSpec(
        "H3", "Spline + Shapely + MC Median",
        "bounded_spline", "redundant_circle", "shapely", "monte_carlo_median",
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
    # Distance ablation (hold spherical + arithmetic_mean)
    ("A1", "A2"),   # SoI vs LP
    ("A2", "A3"),   # LP vs Spline
    ("A1", "A3"),   # SoI vs Spline
    # Centroid ablation (hold SoI + spherical)
    ("A1", "B1"),   # arith vs geom (spherical)
    # Centroid ablation (hold SoI + shapely)
    ("C1", "D1"),   # arith vs geom (shapely)
    # Multilateration ablation (hold SoI + arith)
    ("A1", "C1"),   # spherical vs shapely (arith)
    # Multilateration ablation (hold SoI + geom)
    ("B1", "D1"),   # spherical vs shapely (geom)
    # MC median vs geometric centroid (hold SoI + shapely)
    ("H1", "D1"),   # mc_median vs geom (shapely)
    # MC median vs arithmetic mean (hold SoI + shapely)
    ("H1", "C1"),   # mc_median vs arith (shapely)
    # Annulus vs shapely (hold spline + geom)
    ("F3", "D3"),   # annulus vs shapely (geom, spline)
    # Annulus vs shapely (hold spline + arith)
    ("E3", "C3"),   # annulus vs shapely (arith, spline)
]
