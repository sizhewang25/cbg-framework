"""Pipeline combination registry for systematic CBG evaluation.

Defines the valid pipeline configurations and ablation diff pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PipelineSpec:
    """One pipeline combination to evaluate."""

    combo_id: str       # e.g. "S1", "L2", "B4"
    label: str          # Human-readable label for plots
    distance: str       # framework registry name
    filtering: str
    multilateration: str
    centroid: str
    color: str          # Plot color (hex or named)
    linestyle: str      # matplotlib linestyle
    needs_lp_fit: bool = False
    needs_octant_fit: bool = False
    multilateration_kwargs: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# 10 combinations across 4 comparison groups
# ---------------------------------------------------------------------------
# S1/S2: SoI filtering ablation
# L1/L2: LP filtering ablation
# B1/B2/B5: Bounded-spline weighted/unweighted annulus with Monte Carlo median
# B3/B4/B6: Bounded-spline weighted/unweighted annulus with geometric centroid
# ---------------------------------------------------------------------------

COMBINATIONS: List[PipelineSpec] = [
    # Literature proposals
    PipelineSpec(
        "Original CBG", "LP + no filtering + spherical_circle + geometric_centroid",
        "low_envelope", "none", "spherical_circle", "geometric_centroid",
        "#000000", "--",
        needs_lp_fit=True,
    ),

    # SoI filtering ablation
    PipelineSpec(
        "Million-scale CBG", "SoI + redundant_circle + spherical_circle + boundary_vertex_mean",
        "speed_of_internet", "redundant_circle", "spherical_circle", "boundary_vertex_mean",
        "#0072B2", "-",
    ),
    PipelineSpec(
        "M2", "SoI + no filtering + spherical_circle + boundary_vertex_mean",
        "speed_of_internet", "none", "spherical_circle", "boundary_vertex_mean",
        "#0072B2", "--",
    ),
    PipelineSpec(
        "M3", "SoI + redundant_circle + spherical_circle + geometric_centroid",
        "speed_of_internet", "redundant_circle", "spherical_circle", "geometric_centroid",
        "#0072B2", ":",
    ),
    # # LP filtering ablation
    # PipelineSpec(
    #     "L1", "LP + redundant_circle + spherical_circle + boundary_vertex_mean",
    #     "low_envelope", "redundant_circle", "spherical_circle", "boundary_vertex_mean",
    #     "#000000", "-",
    #     needs_lp_fit=True,
    # ),
    # PipelineSpec(
    #     "L2", "LP + no filtering + spherical_circle + boundary_vertex_mean",
    #     "low_envelope", "none", "spherical_circle", "boundary_vertex_mean",
    #     "#000000", "--",
    #     needs_lp_fit=True,
    # ),
    # # Bounded-spline weighted/unweighted annulus, Monte Carlo median
    # PipelineSpec(
    #     "B1", "Spline + planar_annulus_weighted@0.9 + monte_carlo_median",
    #     "bounded_spline", "none", "planar_annulus_weighted", "monte_carlo_median",
    #     "#009E73", "-",
    #     needs_octant_fit=True,
    #     multilateration_kwargs={"weight_threshold": 0.9},
    # ),
    # PipelineSpec(
    #     "B2", "Spline + planar_annulus + monte_carlo_median",
    #     "bounded_spline", "none", "planar_annulus", "monte_carlo_median",
    #     "#009E73", "--",
    #     needs_octant_fit=True,
    # ),
    # PipelineSpec(
    #     "B5", "Spline + planar_annulus_weighted@0.5 + monte_carlo_median",
    #     "bounded_spline", "none", "planar_annulus_weighted", "monte_carlo_median",
    #     "#009E73", ":",
    #     needs_octant_fit=True,
    #     multilateration_kwargs={"weight_threshold": 0.5},
    # ),
    # # Bounded-spline weighted/unweighted annulus, geometric centroid
    # PipelineSpec(
    #     "B3", "Spline + planar_annulus_weighted@0.9 + geometric_centroid",
    #     "bounded_spline", "none", "planar_annulus_weighted", "geometric_centroid",
    #     "#D55E00", "-",
    #     needs_octant_fit=True,
    #     multilateration_kwargs={"weight_threshold": 0.9},
    # ),
    # PipelineSpec(
    #     "B4", "Spline + planar_annulus + geometric_centroid",
    #     "bounded_spline", "none", "planar_annulus", "geometric_centroid",
    #     "#D55E00", "--",
    #     needs_octant_fit=True,
    # ),
    # PipelineSpec(
    #     "B6", "Spline + planar_annulus_weighted@0.5 + geometric_centroid",
    #     "bounded_spline", "none", "planar_annulus_weighted", "geometric_centroid",
    #     "#D55E00", ":",
    #     needs_octant_fit=True,
    #     multilateration_kwargs={"weight_threshold": 0.5},
    # ),
]

SPECS_BY_ID: Dict[str, PipelineSpec] = {s.combo_id: s for s in COMBINATIONS}

# ---------------------------------------------------------------------------
# Diff pairs for one-dimensional ablation analysis
# ---------------------------------------------------------------------------
# Each tuple: (combo_id_A, combo_id_B)
# Error-Diff CDF plots: error_A - error_B per probe
# Negative delta → A is better

DIFF_PAIRS: List[Tuple[str, str]] = [
    # Filtering ablations
    # ("S1", "S2"),   # SoI redundant filtering vs no filtering
    # ("L1", "L2"),   # LP redundant filtering vs no filtering
    # Distance ablations, holding spherical_circle + boundary_vertex_mean
    # ("S1", "L1"),   # SoI vs LP with redundant filtering
    # ("L1", "B1"),   # SoI vs LP with no filtering
    # ("S1", "B1"),   # SoI vs LP with no filtering
    # Weighted-region ablations, holding centroid
    # ("B1", "B2"),   # weighted@0.9 vs unweighted annulus, Monte Carlo median
    # ("B3", "B4"),   # weighted@0.9 vs unweighted annulus, geometric centroid
    # Weighted-threshold ablations, holding centroid
    # ("B1", "B5"),   # weighted@0.9 vs weighted@0.5, Monte Carlo median
    # ("B3", "B6"),   # weighted@0.9 vs weighted@0.5, geometric centroid
    # Centroid ablations, holding annulus weighting
    # ("B1", "B3"),   # Monte Carlo median vs geometric centroid, weighted@0.9 annulus
    # ("B2", "B4"),   # Monte Carlo median vs geometric centroid, unweighted annulus
    # ("B5", "B6"),   # Monte Carlo median vs geometric centroid, weighted@0.5 annulus
]
