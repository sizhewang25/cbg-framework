"""Visualize a fitted BoundedSplineLTD per VP.

Six layers on every plot: the RTT-vs-distance scatter, the theoretical 2/3·c
baseline, the per-VP upper and lower convex hulls, the per-VP spline center,
and the per-VP δ-band tuned for `sample_coverage = 0.9`.

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
from scripts.libs.octant_simple.octant_model import OctantRTTModel


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
        rtts, distances, s=14, alpha=0.35, c="gray", edgecolors="none",
        label=f"Measurements (n={len(distances)})",
    )

    rtt_max = float(rtts.max()) if rtts.size else 1.0
    rtt_axis = np.linspace(0.0, rtt_max, 100)
    ax.plot(
        rtt_axis, rtt_axis / THEORETICAL_SLOPE, "k--", linewidth=1.5, alpha=0.6,
        label=f"2/3·c baseline ({THEORETICAL_SLOPE:.4f} ms/km)",
    )

    if submodel is not None and submodel.fitted:
        # `hull_upper_*` = upper envelope in (rtt, distance) space = outer
        # distance bound for a given RTT (top edge of the scatter with RTT
        # on x and distance on y). `hull_lower_*` = inner distance bound.
        # Labels match the predict_distance_bounds `(inner_km, outer_km)` API.
        if submodel.hull_upper_distances:
            ax.plot(
                submodel.hull_upper_rtts, submodel.hull_upper_distances,
                "g-", linewidth=1.8, alpha=0.85, label="Outer hull (max dist)",
            )
        if submodel.hull_lower_distances:
            ax.plot(
                submodel.hull_lower_rtts, submodel.hull_lower_distances,
                "g--", linewidth=1.8, alpha=0.85, label="Inner hull (min dist)",
            )

        has_spline = (
            submodel.spline_rtt_knots is not None
            and submodel.spline_dist_knots is not None
        )
        if has_spline:
            rtt_lo = float(min(submodel.spline_rtt_knots))
            data_max = float(rtts.max()) if rtts.size else 0.0
            spline_hi = (
                submodel.cutoff_rtt
                if submodel.cutoff_rtt > 0
                else data_max
            )
            spline_grid = np.linspace(rtt_lo, max(spline_hi, rtt_lo + 1e-6), 200)
            centers = np.array(
                [submodel.predict_distance(float(r)) for r in spline_grid]
            )
            ax.plot(
                spline_grid, centers, color="darkorange", linewidth=2.0,
                label="Spline center",
            )

            if delta is not None:
                band_hi = max(data_max, spline_hi)
                rtt_grid = np.linspace(rtt_lo, max(band_hi, rtt_lo + 1e-6), 200)
                inner = np.empty_like(rtt_grid)
                outer = np.empty_like(rtt_grid)
                for i, r in enumerate(rtt_grid):
                    inner[i], outer[i] = submodel.predict_distance_bounds(
                        float(r), delta=delta
                    )
                ax.fill_between(
                    rtt_grid, inner, outer,
                    color="darkorange", alpha=0.20,
                    label=f"δ-band (δ={delta:.3f}, coverage≈0.9)",
                )

        if submodel.cutoff_rtt > 0:
            ax.axvline(
                submodel.cutoff_rtt, color="purple", linestyle=":",
                linewidth=1.0, alpha=0.6,
                label=f"cutoff_rtt = {submodel.cutoff_rtt:.1f} ms",
            )

    ax.set_xlabel("RTT (ms)")
    ax.set_ylabel("Distance (km)")
    if title:
        ax.set_title(title)
    ax.set_xlim(0.0, 200.0)
    d_max = float(distances.max()) if distances.size else 1.0
    ax.set_ylim(0.0, d_max * 1.1)
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
        title=title or f"VP {vp_id}",
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
        / "scripts" / "libs" / "cbg_feasibility" / "data"
        / "vultr_pings_us_only.csv"
    )

    print(f"Loading samples from {csv_path}")
    samples = _load_vultr_samples(csv_path)
    print(f"Loaded {len(samples)} samples across "
          f"{len({s.vp_id for s in samples})} anchors")

    model = BoundedSplineLTD(sample_coverage=0.9)
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
        plot_bounded_spline_vp(
            model, samples, vp_id, ax=ax,
            title=f"Bounded-spline fit — anchor {vp_id}",
        )
        out_path = output_dir / f"scatter_{str(vp_id).replace('.', '_')}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved {out_path.name}")


if __name__ == "__main__":
    main()
