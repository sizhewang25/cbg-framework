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


US_MAINLAND_EXTENT = (-125.0, -66.0, 24.0, 50.0)


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


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    by_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        col = by_lower.get(name.lower())
        if col is not None:
            return col
    return None


def _load_vp_target_pairs(path: Path) -> pd.DataFrame:
    """Load pair measurements and normalize columns to vp/target coords.

    Supported columns are case-insensitive variants of:
      - vp_lat, vp_lon
      - target_lat, target_lon
    """
    if path.suffix.lower() == ".json":
        df = pd.DataFrame(json.loads(path.read_text()))
    else:
        df = pd.read_csv(path)

    vp_lat_col = _pick_column(df, ("vp_lat",))
    vp_lon_col = _pick_column(df, ("vp_lon",))
    target_lat_col = _pick_column(df, ("target_lat",))
    target_lon_col = _pick_column(df, ("target_lon",))

    if None in (vp_lat_col, vp_lon_col, target_lat_col, target_lon_col):
        raise SystemExit(
            f"{path} must include vp_lat/vp_lon/target_lat/target_lon columns "
            "(any case is accepted)."
        )

    norm = pd.DataFrame(
        {
            "vp_lat": pd.to_numeric(df[vp_lat_col], errors="coerce"),
            "vp_lon": pd.to_numeric(df[vp_lon_col], errors="coerce"),
            "target_lat": pd.to_numeric(df[target_lat_col], errors="coerce"),
            "target_lon": pd.to_numeric(df[target_lon_col], errors="coerce"),
        }
    )
    norm = norm.dropna(subset=["vp_lat", "vp_lon", "target_lat", "target_lon"])
    norm = norm.drop_duplicates(subset=["vp_lat", "vp_lon", "target_lat", "target_lon"])
    return norm.reset_index(drop=True)


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
        targets["target_lon"], targets["target_lat"],
        transform=ccrs.PlateCarree(),
        s=14, c="#ff7f0e", zorder=3, edgecolors="none", alpha=0.5,
        label=f"targets ({len(targets):,})",
    )
    ax.scatter(
        vps["vp_lon"], vps["vp_lat"],
        transform=ccrs.PlateCarree(),
        s=28, c="#1f77b4", marker="^", zorder=4, edgecolors="white", linewidths=0.3,
        alpha=0.6, label=f"VPs ({len(vps):,})",
    )

    ax.set_title(f"{len(targets):,} targets vs {len(vps):,} VPs", fontsize=13)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=10, markerscale=1.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_vp_target_flows(
    pairs: pd.DataFrame,
    out_path: Path,
    *,
    extent: tuple[float, float, float, float] | None = None,
) -> Path:
    """Render unique VP-target traffic pairs as curved geodesic flows on a map."""
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

    for row in pairs.itertuples(index=False):
        ax.plot(
            [row.vp_lon, row.target_lon],
            [row.vp_lat, row.target_lat],
            transform=ccrs.Geodetic(),
            color="#4f6d7a",
            linewidth=0.5,
            alpha=0.2,
            zorder=2,
        )

    vp_points = pairs[["vp_lat", "vp_lon"]].drop_duplicates()
    target_points = pairs[["target_lat", "target_lon"]].drop_duplicates()

    ax.scatter(
        vp_points["vp_lon"],
        vp_points["vp_lat"],
        transform=ccrs.PlateCarree(),
        s=30,
        c="#1f77b4",
        marker="^",
        zorder=4,
        edgecolors="white",
        linewidths=0.35,
        alpha=0.7,
        label=f"VPs ({len(vp_points):,})",
    )
    ax.scatter(
        target_points["target_lon"],
        target_points["target_lat"],
        transform=ccrs.PlateCarree(),
        s=16,
        c="#d62728",
        zorder=5,
        edgecolors="white",
        linewidths=0.25,
        alpha=0.7,
        label=f"targets ({len(target_points):,})",
    )

    ax.set_title(
        f"Unique VP-target flows: {len(pairs):,} pairs",
        fontsize=13,
    )
    ax.legend(loc="lower left", framealpha=0.9, fontsize=10, markerscale=1.3)

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
    parser.add_argument(
        "--pairs", type=Path, default=None,
        help=(
            "Path to pair measurements containing VP and target coordinates "
            "(e.g., VP_LAT/VP_LON/TARGET_LAT/TARGET_LON). "
            "When provided, plots unique VP-target traffic flows."
        ),
    )
    parser.add_argument(
        "--us-only", action="store_true",
        help="Zoom to US mainland extent (-125 -66 24 50).",
    )
    args = parser.parse_args()

    extent = tuple(args.extent) if args.extent is not None else None
    if args.us_only and extent is None:
        extent = US_MAINLAND_EXTENT

    if args.pairs is not None:
        if not args.pairs.exists():
            raise SystemExit(f"pairs file not found at {args.pairs}")
        pairs = _load_vp_target_pairs(args.pairs)
        out = plot_vp_target_flows(pairs, args.out, extent=extent)
        vp_count = len(pairs[["vp_lat", "vp_lon"]].drop_duplicates())
        target_count = len(pairs[["target_lat", "target_lon"]].drop_duplicates())
        print(
            f"Wrote {out} ({len(pairs):,} unique VP-target pairs, "
            f"{vp_count:,} VPs, {target_count:,} targets)"
        )
        return

    if not args.targets.exists():
        raise SystemExit(f"targets file not found at {args.targets}")
    if not args.vps.exists():
        raise SystemExit(f"vps file not found at {args.vps}")

    targets = _load_points(args.targets, "target_lat", "target_lon")
    vps = _load_points(args.vps, "vp_lat", "vp_lon")
    out = plot_targets_vps(targets, vps, args.out, extent=extent)
    print(f"Wrote {out} ({len(targets):,} targets, {len(vps):,} VPs)")


if __name__ == "__main__":
    main()
