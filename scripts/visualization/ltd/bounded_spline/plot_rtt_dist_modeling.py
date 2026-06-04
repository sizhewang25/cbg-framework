"""Visualize a fitted BoundedSplineLTD per VP.

Six layers on every plot: the RTT-vs-distance scatter, the theoretical 2/3·c
baseline, the per-VP upper and lower convex hulls, the per-VP spline center,
and the per-VP δ-band tuned for `target_coverage = 0.9`.

The hull lines and the δ-band both come from `predict_distance_bounds` —
called with `delta=None` for the bare hulls and with the per-VP `delta` for
the band. That keeps the visualization aligned with whatever extension the
model uses above `cutoff_rtt` (currently the Octant sentinel construction).

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
    delta: Optional[float] = None,
    *,
    title: Optional[str] = None,
) -> Axes:
    """Draw scatter + 2/3·c baseline + hulls + spline + δ-band on `ax`.

    `submodel` supplies the hull arrays and spline knots; `delta` widens the
    spline-anchored band via `submodel.predict_distance_bounds(rtt, delta)`.
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
        has_spline = (
            submodel.spline_rtt_knots is not None
            and submodel.spline_dist_knots is not None
        )
        rtt_lo = (
            float(min(submodel.spline_rtt_knots))
            if has_spline
            else float(min(submodel.hull_upper_rtts[0], submodel.hull_lower_rtts[0]))
        )
        # Single grid for both the hull lines (delta=None) and the δ-band
        # (delta=delta). Both come from `predict_distance_bounds`, so they
        # agree above `cutoff_rtt` with whatever extension the model uses
        # (currently the Octant sentinel construction). Insert an explicit
        # `[cutoff, cutoff + 1e-6]` pair so the inner-band discontinuity at
        # cutoff renders as a near-vertical snap rather than a slanted
        # segment — the linspace's ~1 ms spacing would otherwise smear it.
        rtt_grid = np.linspace(rtt_lo, max(rtt_max, rtt_lo + 1e-6), 200)
        if cutoff > rtt_lo and cutoff < rtt_grid[-1]:
            rtt_grid = np.sort(np.concatenate(
                [rtt_grid, [cutoff, cutoff + 1e-6]]
            ))

        # Draw the *actual* convex hull from its stored monotone-chain
        # vertices — the upper and lower chains span the full data range and
        # wrap every point, including the sparse cluster above the cutoff.
        # Reconstructing it from `predict_distance_bounds(delta=None)` would
        # instead render the sentinel (outer) / flat (inner) extension above
        # the cutoff, which is the model's prediction construction, not the
        # geometric hull.
        ax.plot(submodel.hull_upper_rtts, submodel.hull_upper_distances,
                color="black", linestyle="-", linewidth=1.5,
                label="Convex Hull")
        ax.plot(submodel.hull_lower_rtts, submodel.hull_lower_distances,
                color="black", linestyle="-", linewidth=1.5, label="_nolegend_")

        if has_spline:
            spline_hi = cutoff if cutoff > 0 else rtt_max
            spline_grid = np.linspace(rtt_lo, max(spline_hi, rtt_lo + 1e-6), 200)
            centers = np.array(
                [submodel.predict_distance(float(r)) for r in spline_grid]
            )
            ax.plot(spline_grid, centers, color="black", linestyle="-.",
                    linewidth=1.0, label="Spline Approximation")

            if delta is not None:
                inner_band = np.empty_like(rtt_grid)
                outer_band = np.empty_like(rtt_grid)
                for i, r in enumerate(rtt_grid):
                    inner_band[i], outer_band[i] = submodel.predict_distance_bounds(
                        float(r), delta=delta
                    )
                # The Scaled Spline (dash-dot) is only the data-driven band
                # *below* the cutoff. At/above the cutoff the band degenerates
                # into the cutoff construction — the sentinel-connecting outer
                # line and the flat inner line — which we draw dotted, matching
                # the cutoff marker. Splitting at `cutoff` (strict <, >) also
                # drops the single discontinuity point, so the vertical snap
                # at the cutoff is no longer rendered.
                if cutoff > 0:
                    below = rtt_grid < cutoff
                    above = rtt_grid > cutoff
                else:
                    below = np.ones_like(rtt_grid, dtype=bool)
                    above = np.zeros_like(rtt_grid, dtype=bool)
                ax.plot(rtt_grid[below], outer_band[below], color="black",
                        linestyle="-.", linewidth=2.0, label="Scaled Spline")
                ax.plot(rtt_grid[below], inner_band[below], color="black",
                        linestyle="-.", linewidth=2.0, label="_nolegend_")
                if above.any():
                    ax.plot(rtt_grid[above], outer_band[above], color="black",
                            linestyle=":", linewidth=1.0, label="_nolegend_")
                    ax.plot(rtt_grid[above], inner_band[above], color="black",
                            linestyle=":", linewidth=1.0, label="_nolegend_")
                ax.fill_between(rtt_grid, inner_band, outer_band,
                                color="gray", alpha=0.3)

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


def plot_bounded_spline_vp(
    model: BoundedSplineLTD,
    samples: list[FitSample],
    vp_id: VpId,
    ax: Optional[Axes] = None,
    *,
    title: Optional[str] = None,
) -> Axes:
    """Plot scatter, baseline, hulls, spline, and δ-band for one VP.

    Filters `samples` to those matching `vp_id`, recomputes haversine
    distances, and overlays the fitted per-VP submodel + per-VP δ.
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
        delta=model._deltas.get(vp_id),
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
        delta = model._deltas.get(vp_id)
        cutoff = submodel.cutoff_rtt if submodel and submodel.fitted else float("nan")
        delta_str = f"{delta:.3f}" if delta is not None else "—"
        print(f"  {vp_id}: fitted={submodel.fitted if submodel else False}, "
              f"δ={delta_str}, cutoff_rtt={cutoff:.2f} ms")

        fig, ax = plt.subplots(figsize=(9, 6))
        plot_bounded_spline_vp(model, samples, vp_id, ax=ax)
        out_path = output_dir / f"scatter_{str(vp_id).replace('.', '_')}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved {out_path.name}")


if __name__ == "__main__":
    main()
