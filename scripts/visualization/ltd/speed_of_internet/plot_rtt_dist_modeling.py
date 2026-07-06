"""Visualize the theoretical 2/3·c (speed-of-internet) baseline per VP.

One layer on every plot: the RTT-vs-distance scatter overlaid with only the
THEORETICAL_SLOPE line — no fitted submodel, no low-envelope overlay.

Run as a script to validate on `vultr_pings_us_only.csv` — loads one scatter
across all anchors (each `dst_ip` is a VP) and writes one PNG per anchor to
the `outputs/` subfolder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from scripts.framework.v2.ltd.base import FitSample
from scripts.framework.v2.types import Coord, Latency, VpId
from scripts.libs.cbg.rtt_model import (
    THEORETICAL_SLOPE,
    haversine_distance,
)


# ---------------------------------------------------------------------------
# Plotting primitives
# ---------------------------------------------------------------------------


def plot_rtt_distance(
    ax: Axes,
    distances: np.ndarray,
    rtts: np.ndarray,
    *,
    title: Optional[str] = None,
    max_rtt_ms: Optional[float] = None,
    max_dist_km: Optional[float] = None,
) -> Axes:
    """Draw scatter + 2/3·c theoretical baseline on `ax`.

    `max_rtt_ms` clips the RTT (x) axis and `max_dist_km` the Distance (y)
    axis; both trim the scatter accordingly.
    """
    distances = np.asarray(distances, dtype=float)
    rtts = np.asarray(rtts, dtype=float)

    mask = np.ones(rtts.shape, dtype=bool)
    if max_rtt_ms is not None:
        mask &= rtts <= max_rtt_ms
    if max_dist_km is not None:
        mask &= distances <= max_dist_km
    plot_d, plot_r = distances[mask], rtts[mask]

    ax.scatter(plot_r, plot_d, s=10, c="black", marker="+", linewidths=0.6, alpha=0.5)

    d_max = float(distances.max()) if distances.size else 1.0
    d_grid = np.linspace(0.0, max_dist_km if max_dist_km is not None else d_max, 100)

    ax.plot(
        THEORETICAL_SLOPE * d_grid, d_grid, color="black",
        linestyle="--", linewidth=1.2, label="SOI Line",
    )

    ax.set_xlabel("RTT (ms)")
    ax.set_ylabel("Distance (km)")
    if title:
        ax.set_title(title)
    if max_dist_km is not None:
        ax.set_ylim(0.0, max_dist_km)
    else:
        ax.set_ylim(0.0, d_max * 1.05)
    if max_rtt_ms is not None:
        ax.set_xlim(0.0, max_rtt_ms)
    ax.legend(loc="upper left", fontsize=11)
    return ax


def plot_soi_vp(
    samples: list[FitSample],
    vp_id: VpId,
    ax: Optional[Axes] = None,
    *,
    max_rtt_ms: Optional[float] = None,
    max_dist_km: Optional[float] = None,
    title: Optional[str] = None,
) -> Axes:
    """Plot scatter and SOI baseline for one VP.

    Filters `samples` to those matching `vp_id` and recomputes haversine
    distances before calling `plot_rtt_distance`.
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
        title=title,
        max_rtt_ms=max_rtt_ms,
        max_dist_km=max_dist_km,
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
    vp_ids = list({s.vp_id for s in samples})
    print(f"Loaded {len(samples)} samples across {len(vp_ids)} anchors")

    for vp_id in vp_ids:
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_soi_vp(samples, vp_id, ax=ax, max_rtt_ms=125, max_dist_km=8000)
        out_path = output_dir / f"scatter_{str(vp_id).replace('.', '_')}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path.name}")


if __name__ == "__main__":
    main()
