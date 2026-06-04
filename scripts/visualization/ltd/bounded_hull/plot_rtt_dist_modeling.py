"""Visualize a fitted bounded-hull LTD per VP.

Four layers on every plot: the RTT-vs-distance scatter, the theoretical 2/3·c
baseline, the per-VP convex hull, and the cutoff construction (a vertical
cutoff marker plus the sentinel/flat extension lines above the cutoff). The
annular feasible region is shaded — between the upper and lower convex hulls
*below* the cutoff, and between the sentinel (outer) and flat (inner) extension
lines *above* it.

This is the spline-free counterpart of the `bounded_spline` visualization: the
underlying model is still a `BoundedSplineLTD` (its per-VP `OctantRTTModel`
supplies the hull, cutoff, and extension via `predict_distance_bounds`), but
the spline center and δ-scaled band are not drawn. The shaded band is exactly
`predict_distance_bounds(rtt, delta=None)`, which returns the bare hull bounds
below the cutoff and the extension bounds above it.

Run as a script to validate on `vultr_pings_us_only.csv` — fits one
BoundedSplineLTD across all anchors (each `dst_ip` is a VP) and writes one
PNG per anchor to the `outputs/` subfolder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from scripts.framework.v2.ltd.base import FitSample
from scripts.framework.v2.ltd.bounded_spline import BoundedSplineLTD
from scripts.framework.v2.types import Coord, Latency, VpId
from scripts.libs.cbg.rtt_model import THEORETICAL_SLOPE, haversine_distance
from scripts.libs.octant.octant_model import OctantRTTModel


# ---------------------------------------------------------------------------
# Plotting primitives
# ---------------------------------------------------------------------------


def plot_rtt_distance(
    ax: Axes,
    distances: np.ndarray,
    rtts: np.ndarray,
    submodel: Optional[OctantRTTModel] = None,
    *,
    title: Optional[str] = None,
) -> Axes:
    """Draw scatter + 2/3·c baseline + convex hull + shaded annular band.

    `submodel` supplies the hull vertices and the cutoff. The shaded band is
    `submodel.predict_distance_bounds(rtt, delta=None)` — the bare hull bounds
    below `cutoff_rtt` and the sentinel/flat extension bounds above it.
    """
    distances = np.asarray(distances, dtype=float)
    rtts = np.asarray(rtts, dtype=float)

    ax.scatter(
        rtts, distances, s=24, marker="+", c="black", linewidths=0.8,
        alpha=0.5, label="_nolegend_",
    )

    rtt_max = float(rtts.max()) if rtts.size else 1.0
    rtt_axis = np.linspace(0.0, rtt_max, 100)
    ax.plot(
        rtt_axis, rtt_axis / THEORETICAL_SLOPE, color="black",
        linestyle="--", linewidth=1.2, label="SOI Line",
    )

    if submodel is not None and submodel.fitted:
        cutoff = submodel.cutoff_rtt
        rtt_lo = float(min(submodel.hull_upper_rtts[0], submodel.hull_lower_rtts[0]))
        # Single grid for the shaded band and the extension lines. The explicit
        # `[cutoff, cutoff + 1e-6]` pair pins the transition so the below/above
        # split lands exactly at the cutoff.
        rtt_grid = np.linspace(rtt_lo, max(rtt_max, rtt_lo + 1e-6), 200)
        if cutoff > rtt_lo and cutoff < rtt_grid[-1]:
            rtt_grid = np.sort(np.concatenate(
                [rtt_grid, [cutoff, cutoff + 1e-6]]
            ))

        # Draw the *actual* convex hull from its stored monotone-chain
        # vertices — the upper and lower chains span the full data range and
        # wrap every point, including the sparse cluster above the cutoff.
        ax.plot(submodel.hull_upper_rtts, submodel.hull_upper_distances,
                color="black", linestyle="-", linewidth=1.5,
                label="Convex Hull")
        ax.plot(submodel.hull_lower_rtts, submodel.hull_lower_distances,
                color="black", linestyle="-", linewidth=1.5, label="_nolegend_")

        # Annular bounds: hull below the cutoff, sentinel/flat extension above.
        inner = np.empty_like(rtt_grid)
        outer = np.empty_like(rtt_grid)
        for i, r in enumerate(rtt_grid):
            inner[i], outer[i] = submodel.predict_distance_bounds(
                float(r), delta=None
            )

        # Above the cutoff the band boundaries are the cutoff extension lines
        # (sentinel-connecting outer line + flat inner line); draw them dotted,
        # matching the cutoff marker.
        if cutoff > 0:
            above = rtt_grid > cutoff
        else:
            above = np.zeros_like(rtt_grid, dtype=bool)
        if above.any():
            ax.plot(rtt_grid[above], outer[above], color="black",
                    linestyle=":", linewidth=1.0, label="_nolegend_")
            ax.plot(rtt_grid[above], inner[above], color="black",
                    linestyle=":", linewidth=1.0, label="_nolegend_")

        # Shade the whole feasible region: between the hulls before the cutoff
        # and between the extension lines after it.
        ax.fill_between(rtt_grid, inner, outer, color="gray", alpha=0.3)

        if cutoff > 0:
            ax.axvline(cutoff, color="black", linestyle=":", linewidth=1.0,
                       label="Cutoff")

    ax.set_xlabel("RTT (ms)")
    ax.set_ylabel("Distance (km)")
    if title:
        ax.set_title(title)
    ax.set_xlim(0.0, 125.0)
    ax.set_ylim(0.0, 8000.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    return ax


def plot_bounded_hull_vp(
    model: BoundedSplineLTD,
    samples: list[FitSample],
    vp_id: VpId,
    ax: Optional[Axes] = None,
    *,
    title: Optional[str] = None,
) -> Axes:
    """Plot scatter, baseline, convex hull, and shaded band for one VP.

    Filters `samples` to those matching `vp_id`, recomputes haversine
    distances, and overlays the fitted per-VP submodel's hull + cutoff.
    """
    vp_samples = [s for s in samples if s.vp_id == vp_id]
    distances = np.array(
        [
            haversine_distance(
                s.vp_coord.lat, s.vp_coord.lon,
                s.probe_coord.lat, s.probe_coord.lon,
            )
            for s in vp_samples
        ],
        dtype=float,
    )
    rtts = np.array([float(s.latency) for s in vp_samples], dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    return plot_rtt_distance(
        ax, distances, rtts,
        submodel=model._submodels.get(vp_id),
        title=title,
    )


# ---------------------------------------------------------------------------
# Validation driver
# ---------------------------------------------------------------------------


def _load_vultr_samples(csv_path: Path) -> list[FitSample]:
    """Load Vultr probe→anchor pings as FitSamples (anchor = VP)."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[
        "dst_ip", "min_rtt",
        "anchor_latitude", "anchor_longitude",
        "probe_latitude", "probe_longitude",
    ])
    df = df[df["min_rtt"] > 0]

    return [
        FitSample(
            vp_id=VpId(row.dst_ip),
            vp_coord=Coord(row.anchor_latitude, row.anchor_longitude),
            probe_coord=Coord(row.probe_latitude, row.probe_longitude),
            latency=Latency(float(row.min_rtt)),
        )
        for row in df.itertuples(index=False)
    ]


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    csv_path = (
        Path(__file__).resolve().parents[4]
        / "datasets" / "vultr_pings_us_only.csv"
    )

    print(f"Loading samples from {csv_path}")
    samples = _load_vultr_samples(csv_path)
    print(f"Loaded {len(samples)} samples across "
          f"{len({s.vp_id for s in samples})} anchors")

    model = BoundedSplineLTD(target_coverage=0.9)
    result = model.fit(samples)
    if not result.success:
        raise RuntimeError(f"fit failed: {result.error}")

    fitted_vps = result.args["vps_fitted"]
    attempted = result.args["vps_attempted"]
    print(f"Fitted {len(fitted_vps)}/{len(attempted)} VPs")

    for vp_id in attempted:
        submodel = model._submodels.get(vp_id)
        cutoff = submodel.cutoff_rtt if submodel and submodel.fitted else float("nan")
        print(f"  {vp_id}: fitted={submodel.fitted if submodel else False}, "
              f"cutoff_rtt={cutoff:.2f} ms")

        fig, ax = plt.subplots(figsize=(9, 6))
        plot_bounded_hull_vp(model, samples, vp_id, ax=ax)
        out_path = output_dir / f"scatter_{str(vp_id).replace('.', '_')}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved {out_path.name}")


if __name__ == "__main__":
    main()
