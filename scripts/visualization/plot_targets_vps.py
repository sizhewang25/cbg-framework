"""Plot a target catalog together with the vantage points (VPs) on a map.

This reuses the map-rendering logic of
`scripts.visualization.airport.plot_targets_airports` but drops the airport
reference layer and the nearest-hub CDF: it simply draws **VPs** and
**targets** on a PlateCarree map so you can eyeball their relative geographic
coverage.

Both layers are read from canonical benchmark CSV/JSON record lists:
  * targets — a `dump_csv_targets` output (`target_id, target_lat, target_lon, ...`)
  * VPs     — a `dump_csv_vps` output (`vp_id, vp_lat, vp_lon, ...`)

CLI:
    python -m scripts.visualization.plot_targets_vps \\
        --targets datasets/vultr_pings_us_canonical/targets.csv \\
        --vps datasets/vultr_pings_us_canonical/vps.csv
    python -m scripts.visualization.plot_targets_vps \\
        --targets t.json --vps vps.json --extent -130 -65 24 50 --out /tmp/tv.png
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
import pandas as pd


def _load_points(path: Path, lat_col: str, lon_col: str) -> pd.DataFrame:
    """Read a canonical CSV/JSON record list into a frame with `lat_col` /
    `lon_col` (+ whatever else is present), dropping rows missing coords."""
    if path.suffix.lower() == ".json":
        df = pd.DataFrame(json.loads(path.read_text()))
    else:
        df = pd.read_csv(path)
    missing = [c for c in (lat_col, lon_col) if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} missing columns {missing}")
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    df[lat_col] = df[lat_col].astype(float)
    df[lon_col] = df[lon_col].astype(float)
    return df.reset_index(drop=True)


def plot_targets_vps(
    targets: pd.DataFrame,
    vps: pd.DataFrame,
    out_path: Path,
    *,
    extent: tuple[float, float, float, float] | None = None,
) -> Path:
    """Render VPs (blue triangles) and targets (orange dots) on a PlateCarree
    map. Returns the output path."""
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="#eaf2f8")
    ax.add_feature(cfeature.LAND, facecolor="#f6f4ef")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#999999")
    ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="#cccccc")

    ax.scatter(
        vps["vp_lon"], vps["vp_lat"],
        transform=ccrs.PlateCarree(),
        s=28, c="#1f77b4", marker="^", zorder=3, edgecolors="white", linewidths=0.3,
        alpha=0.85, label=f"VPs ({len(vps):,})",
    )
    ax.scatter(
        targets["target_lon"], targets["target_lat"],
        transform=ccrs.PlateCarree(),
        s=14, c="#ff7f0e", zorder=4, edgecolors="none", alpha=0.75,
        label=f"targets ({len(targets):,})",
    )

    ax.set_title(f"{len(targets):,} targets vs {len(vps):,} VPs", fontsize=13)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=10, markerscale=1.4)

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
        "--vps", type=Path,
        default=Path("datasets/vultr_pings_us_canonical/vps.csv"),
        help="dump_csv_vps output (vps.csv or vps.json).",
    )
    parser.add_argument(
        "--extent", type=float, nargs=4, default=None,
        metavar=("MINLON", "MAXLON", "MINLAT", "MAXLAT"),
        help="Crop the map to a bounding box. Default: global.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "targets_vps.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    if not args.targets.exists():
        raise SystemExit(f"targets file not found at {args.targets}")
    if not args.vps.exists():
        raise SystemExit(f"vps file not found at {args.vps}")

    targets = _load_points(args.targets, "target_lat", "target_lon")
    vps = _load_points(args.vps, "vp_lat", "vp_lon")
    extent = tuple(args.extent) if args.extent is not None else None
    out = plot_targets_vps(targets, vps, args.out, extent=extent)
    print(f"Wrote {out} ({len(targets):,} targets, {len(vps):,} VPs)")


if __name__ == "__main__":
    main()
