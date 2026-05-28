"""Visualize a fitted LowEnvelopeLTD per VP.

Three layers on every plot: the RTT-vs-distance scatter, the theoretical
2/3·c baseline, and the per-VP LP low-envelope line (slope·d + intercept).

Run as a script to validate on `vultr_pings_us_only.csv` — fits one
LowEnvelopeLTD across all anchors (each `dst_ip` is a VP) and writes one
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
from scripts.framework.v2.ltd.low_envelope import LowEnvelopeLTD
from scripts.framework.v2.types import Coord, Latency, VpId
from scripts.libs.cbg.rtt_model import (
    THEORETICAL_SLOPE,
    RTTDistanceModel,
    haversine_distance,
)


# ---------------------------------------------------------------------------
# Plotting primitives
# ---------------------------------------------------------------------------


def plot_rtt_distance(
    ax: Axes,
    distances: np.ndarray,
    rtts: np.ndarray,
    submodel: Optional[RTTDistanceModel] = None,
    *,
    title: Optional[str] = None,
    max_rtt_ms: Optional[float] = None,
) -> Axes:
    """Draw scatter + 2/3·c baseline + low-envelope line on `ax`.

    `submodel` supplies the LP line (`rtt = slope·distance + intercept`).
    `max_rtt_ms` clips the y-axis and trims the scatter accordingly.
    """
    distances = np.asarray(distances, dtype=float)
    rtts = np.asarray(rtts, dtype=float)

    if max_rtt_ms is not None:
        mask = rtts <= max_rtt_ms
        plot_d, plot_r = distances[mask], rtts[mask]
    else:
        plot_d, plot_r = distances, rtts

    ax.scatter(
        plot_r, plot_d, s=14, alpha=0.35, c="gray", edgecolors="none",
        label=f"Measurements (n={len(plot_d)})",
    )

    d_max = float(distances.max()) if distances.size else 1.0
    d_grid = np.linspace(0.0, d_max, 100)

    ax.plot(
        THEORETICAL_SLOPE * d_grid, d_grid, "k--", linewidth=1.5, alpha=0.6,
        label=f"2/3·c baseline ({THEORETICAL_SLOPE:.4f} ms/km)",
    )

    if submodel is not None and submodel.fitted:
        line = submodel.slope * d_grid + submodel.intercept
        ax.plot(
            line, d_grid, "r-", linewidth=2.0,
            label=(
                f"Low envelope: {submodel.slope:.4f}·d "
                f"+ {submodel.intercept:.2f}"
            ),
        )

    ax.set_xlabel("RTT (ms)")
    ax.set_ylabel("Distance (km)")
    if title:
        ax.set_title(title)
    ax.set_ylim(0.0, d_max * 1.05)
    if max_rtt_ms is not None:
        ax.set_xlim(0.0, max_rtt_ms)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    return ax


def plot_low_envelope_vp(
    model: LowEnvelopeLTD,
    samples: list[FitSample],
    vp_id: VpId,
    ax: Optional[Axes] = None,
    *,
    max_rtt_ms: Optional[float] = None,
    title: Optional[str] = None,
) -> Axes:
    """Plot scatter, baseline, and low-envelope line for one VP.

    Filters `samples` to those matching `vp_id`, recomputes haversine
    distances, and overlays the fitted per-VP submodel's LP line.
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
        title=title or f"VP {vp_id}",
        max_rtt_ms=max_rtt_ms,
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

    model = LowEnvelopeLTD()
    result = model.fit(samples)
    if not result.success:
        raise RuntimeError(f"fit failed: {result.error}")

    fitted_vps = result.args["vps_fitted"]
    print(f"Fitted {len(fitted_vps)}/{len(result.args['vps_attempted'])} VPs")

    for vp_id in result.args["vps_attempted"]:
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_low_envelope_vp(
            model, samples, vp_id, ax=ax, max_rtt_ms=200,
            title=f"Low-envelope fit — anchor {vp_id}",
        )
        out_path = output_dir / f"scatter_{str(vp_id).replace('.', '_')}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path.name}")


if __name__ == "__main__":
    main()
