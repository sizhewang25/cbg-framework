"""Plot a target catalog against the large-only airport reference set on a map.

The closest-airport eval metric (scripts/benchmark/v2/airport_eval.py) snaps each
target to the nearest *large* hub from the committed slim set (large-only — see
notes/2026-06-18-closest-airport-eval-decisions.md). This script renders the two
layers that metric operates on, so you can eyeball coverage: **large airports**
as the faint background reference grid and the **targets** on top, coloured by
whether a large hub sits within the eval's 40 km "same airport" radius. A side
panel plots the CDF of the target→nearest-hub distance — the answer-space
quantization floor the metric works against.

Targets are read from a `dump_csv_targets` output (`targets.csv` or
`targets.json`) — i.e. the canonical `target_id, target_lat, target_lon, ...`
record list. Airports come from the committed slim parquet
(`datasets/static_datasets/ourairports_iata.parquet`, large-only); if it is
missing the raw OurAirports CSV is downloaded and filtered to large hubs on the
fly.

CLI:
    python -m scripts.visualization.airport.plot_targets_airports \\
        --targets datasets/vultr_pings_us_canonical/targets.csv
    python -m scripts.visualization.airport.plot_targets_airports \\
        --targets path/to/targets.json --extent -130 -65 24 50 --out /tmp/t.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from scripts.benchmark.v2.airports import (
    AIRPORT_TYPES,
    DEFAULT_AIRPORTS_PARQUET,
    AirportIndex,
    download_ourairports_csv,
    filter_airports,
)

# Eval's "same airport" radius (km): a target whose nearest large hub is within
# this distance is one the metric can resolve exactly. Mirrors the within-40km
# match in airport_eval.py.
_MATCH_KM = 40.0


def _load_airports(src_csv: Path | None) -> pd.DataFrame:
    """Large-only airport set: the committed slim parquet, or a fresh
    download filtered to large hubs when the parquet / `--src-csv` is given."""
    if src_csv is not None:
        raw = pd.read_csv(src_csv, low_memory=False)
        return filter_airports(raw, types=AIRPORT_TYPES)
    if DEFAULT_AIRPORTS_PARQUET.exists():
        return pd.read_parquet(DEFAULT_AIRPORTS_PARQUET)
    print("Slim parquet absent; downloading OurAirports CSV ...")
    raw = pd.read_csv(download_ourairports_csv(), low_memory=False)
    return filter_airports(raw, types=AIRPORT_TYPES)


def _load_targets(path: Path) -> pd.DataFrame:
    """Read a `dump_csv_targets` output (.csv or .json) into a frame with
    `target_lat` / `target_lon` (+ whatever else is present)."""
    if path.suffix.lower() == ".json":
        df = pd.DataFrame(json.loads(path.read_text()))
    else:
        df = pd.read_csv(path)
    missing = [c for c in ("target_lat", "target_lon") if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} missing columns {missing}; expected a dump_csv_targets output")
    df = df.dropna(subset=["target_lat", "target_lon"]).copy()
    df["target_lat"] = df["target_lat"].astype(float)
    df["target_lon"] = df["target_lon"].astype(float)
    return df.reset_index(drop=True)


def plot_targets_airports(
    targets: pd.DataFrame,
    airports: pd.DataFrame,
    out_path: Path,
    *,
    extent: tuple[float, float, float, float] | None = None,
) -> Path:
    """Render large airports (faint reference grid) + targets (coloured by
    within-40km-of-a-hub) on a PlateCarree map, with a CDF of the
    target→nearest-hub distance alongside."""
    # Tag each target with its nearest large hub distance so we can split the
    # scatter into resolvable (≤40 km) vs. far-from-any-hub.
    index = AirportIndex(airports)
    _, km = index.query_many(targets["target_lat"].to_numpy(), targets["target_lon"].to_numpy())
    near = km <= _MATCH_KM
    n_near = int(near.sum())
    n_far = int((~near).sum())

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.12)

    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="#eaf2f8")
    ax.add_feature(cfeature.LAND, facecolor="#f6f4ef")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#999999")
    ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="#cccccc")

    # Background reference grid: every large hub, faint and small.
    ax.scatter(
        airports["longitude_deg"], airports["latitude_deg"],
        transform=ccrs.PlateCarree(),
        s=10, c="#7f7f7f", marker="^", zorder=2, edgecolors="none", alpha=0.45,
        label=f"large airports ({len(airports):,})",
    )

    # Targets on top, split by resolvability.
    ax.scatter(
        targets.loc[near, "target_lon"], targets.loc[near, "target_lat"],
        transform=ccrs.PlateCarree(),
        s=14, c="#2ca02c", zorder=4, edgecolors="none", alpha=0.75,
        label=f"target ≤{_MATCH_KM:.0f} km from hub ({n_near:,})",
    )
    ax.scatter(
        targets.loc[~near, "target_lon"], targets.loc[~near, "target_lat"],
        transform=ccrs.PlateCarree(),
        s=14, c="#d62728", zorder=5, edgecolors="none", alpha=0.8,
        label=f"target >{_MATCH_KM:.0f} km from hub ({n_far:,})",
    )

    pct = 100.0 * n_near / len(targets) if len(targets) else 0.0
    ax.set_title(
        f"{len(targets):,} targets vs {len(airports):,} large hubs — "
        f"{pct:.0f}% within {_MATCH_KM:.0f} km of a hub",
        fontsize=13,
    )
    ax.legend(loc="lower left", framealpha=0.9, fontsize=10, markerscale=1.4)

    # Side panel: CDF of the target→nearest-large-hub distance. This is the
    # answer-space quantization floor the metric works against — how far each
    # target sits from the best hub it could ever snap to.
    axb = fig.add_subplot(gs[0, 1])
    finite = km[np.isfinite(km)]
    if len(finite):
        s = np.sort(finite)
        cdf = np.arange(1, len(s) + 1) / len(s)
        axb.plot(s, cdf, color="#1f77b4", linewidth=2.0)
        p50, p90 = np.percentile(s, [50, 90])
        axb.axvline(_MATCH_KM, color="green", linestyle=":", alpha=0.6)
        axb.text(_MATCH_KM, 0.02, f" {_MATCH_KM:.0f} km", color="green",
                 fontsize=8, rotation=90, va="bottom", ha="left")
        axb.text(0.04, 0.97, f"p50 {p50:,.0f} km\np90 {p90:,.0f} km",
                 transform=axb.transAxes, fontsize=8, va="top", ha="left",
                 family="monospace",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    axb.set_xscale("log")
    x_fmt = ScalarFormatter()
    x_fmt.set_scientific(False)
    axb.xaxis.set_major_formatter(x_fmt)
    axb.set_xlim(1, max(1000.0, float(finite.max()) if len(finite) else 1000.0))
    axb.set_ylim(0, 1)
    axb.set_title("target → nearest hub", fontsize=11)
    axb.set_xlabel("distance (km)")
    axb.set_ylabel("CDF")
    axb.grid(True, which="both", alpha=0.3)
    axb.spines[["top", "right"]].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets", type=Path,
        default=Path("datasets/vultr_pings_us_canonical/targets.csv"),
        help="dump_csv_targets output (targets.csv or targets.json).",
    )
    parser.add_argument(
        "--src-csv", type=Path, default=None,
        help="Raw OurAirports airports.csv for the airport layer. "
             "If omitted, uses the committed slim parquet (or downloads).",
    )
    parser.add_argument(
        "--extent", type=float, nargs=4, default=None,
        metavar=("MINLON", "MAXLON", "MINLAT", "MAXLAT"),
        help="Crop the map to a bounding box. Default: global.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "targets_airports.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    if not args.targets.exists():
        raise SystemExit(f"targets file not found at {args.targets}")

    targets = _load_targets(args.targets)
    airports = _load_airports(args.src_csv)
    extent = tuple(args.extent) if args.extent is not None else None
    out = plot_targets_airports(targets, airports, args.out, extent=extent)
    print(f"Wrote {out} ({len(targets):,} targets, {len(airports):,} large airports)")


if __name__ == "__main__":
    main()
