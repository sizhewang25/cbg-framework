"""Visualize a fitted NormalDistLTD pooled across all VPs.

Five layers on every plot: the RTT-vs-distance scatter, the theoretical
2/3·c baseline, the pooled μ(rtt) center curve, and the symmetric
(μ ± σ) band — the paper's published band (Laki et al. 2011 Fig 3a).

NormalDistLTD is a pooled (Spotter) model — one (μ, σ) shared across
all VPs. The same overlay is drawn on every per-anchor scatter so the
viewer can eyeball how well the pooled fit explains each anchor's points.

Run as a script to validate on `vultr_pings_us_only.csv` — fits one
NormalDistLTD across all anchors (each `dst_ip` is a VP) and writes one
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
from scripts.framework.v2.ltd.normal_dist import NormalDistLTD
from scripts.framework.v2.types import Coord, Latency, VpId
from scripts.libs.cbg.rtt_model import THEORETICAL_SLOPE, haversine_distance
from scripts.libs.spotter.spotter_model import SpotterRTTModel


# ---------------------------------------------------------------------------
# Plotting primitives
# ---------------------------------------------------------------------------


def plot_rtt_distance(
    ax: Axes,
    distances: np.ndarray,
    rtts: np.ndarray,
    model: Optional[SpotterRTTModel] = None,
    *,
    title: Optional[str] = None,
) -> Axes:
    """Draw scatter + 2/3·c baseline + pooled μ center + (μ ± σ) band on `ax`.

    `model` is the pooled SpotterRTTModel; the same overlay is reused on every
    per-anchor scatter (Spotter's pooled-normal claim).
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

    if model is not None and model.fitted:
        cutoff = model.cutoff_rtt if model.cutoff_rtt > 0 else model.rtt_max
        rtt_grid = np.linspace(model.rtt_min, max(rtt_max, model.rtt_min + 1e-6), 200)
        # Pin the transition exactly at the cutoff so the below/above split
        # lands cleanly on it.
        if model.cutoff_rtt > 0 and model.rtt_min < cutoff < rtt_grid[-1]:
            rtt_grid = np.sort(np.concatenate(
                [rtt_grid, [cutoff, cutoff + 1e-6]]
            ))

        # Center: μ drawn only up to the cutoff — not extended beyond it.
        eval_grid = np.minimum(rtt_grid, cutoff)
        mu_line = np.maximum(0.0, np.polyval(model.p_mu, eval_grid))
        below = rtt_grid <= cutoff
        ax.plot(rtt_grid[below], mu_line[below], color="black", linestyle="-.",
                linewidth=1.2, label="μ(rtt)")

        # Band lines come straight from predict_distance_bounds so the
        # baseline clip + flat-extension are visualized exactly as the
        # model delivers them at predict time (no raw polynomial drawn
        # beyond the cutoff). Skip RTTs where the model returns None.
        inner_line = np.full_like(rtt_grid, np.nan)
        outer_line = np.full_like(rtt_grid, np.nan)
        for i, r in enumerate(rtt_grid):
            bounds = model.predict_distance_bounds(float(r))
            if bounds is not None:
                inner_line[i], outer_line[i] = bounds

        # Below the cutoff the band is the (μ ± σ) envelope (solid); above it
        # the boundaries are the sentinel (outer) + flat (inner) extension
        # lines, drawn dotted to match the cutoff marker.
        if model.cutoff_rtt > 0:
            above = rtt_grid > cutoff
        else:
            above = np.zeros_like(rtt_grid, dtype=bool)
        band_below = ~above
        ax.plot(rtt_grid[band_below], outer_line[band_below], color="black",
                linestyle="-", linewidth=1.5, label="μ(rtt) ± k·σ (k=1)")
        ax.plot(rtt_grid[band_below], inner_line[band_below], color="black",
                linestyle="-", linewidth=1.5, label="_nolegend_")
        if above.any():
            ax.plot(rtt_grid[above], outer_line[above], color="black",
                    linestyle=":", linewidth=1.0, label="_nolegend_")
            ax.plot(rtt_grid[above], inner_line[above], color="black",
                    linestyle=":", linewidth=1.0, label="_nolegend_")
        ax.fill_between(rtt_grid, inner_line, outer_line,
                        color="gray", alpha=0.3)

        if model.cutoff_rtt > 0:
            ax.axvline(model.cutoff_rtt, color="black", linestyle=":",
                       linewidth=1.0, label="Cutoff")

    ax.set_xlabel("RTT (ms)")
    ax.set_ylabel("Distance (km)")
    if title:
        ax.set_title(title)
    ax.set_xlim(0.0, 125.0)
    ax.set_ylim(0.0, 8000.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    return ax


def plot_normal_dist_vp(
    model: NormalDistLTD,
    samples: list[FitSample],
    vp_id: VpId,
    ax: Optional[Axes] = None,
    *,
    title: Optional[str] = None,
) -> Axes:
    """Plot scatter, baseline, and pooled (μ, σ) band for one VP.

    Filters `samples` to those matching `vp_id`, recomputes haversine
    distances, and overlays the (shared) pooled SpotterRTTModel.
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
        model=model._model,
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

    model = NormalDistLTD()
    result = model.fit(samples)
    if not result.success:
        raise RuntimeError(f"fit failed: {result.error}")

    print(f"Fitted pooled model: "
          f"rtt_range=[{result.args['rtt_min']:.2f}, "
          f"{result.args['rtt_max']:.2f}] ms, "
          f"cutoff_rtt={result.args['cutoff_rtt']:.2f} ms")

    vp_ids = sorted({s.vp_id for s in samples})
    for vp_id in vp_ids:
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_normal_dist_vp(model, samples, vp_id, ax=ax)
        out_path = output_dir / f"scatter_{str(vp_id).replace('.', '_')}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path.name}")


if __name__ == "__main__":
    main()
