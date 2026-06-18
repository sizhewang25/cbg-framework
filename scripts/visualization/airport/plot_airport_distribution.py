"""Plot the world distribution of active scheduled-service airports.

The closest-airport eval metric (scripts/benchmark/v2/airport_eval.py) snaps
predictions and ground truth to the nearest *large* hub (the committed slim set
is large-only — see notes/2026-06-18-closest-airport-eval-decisions.md). This
script takes a wider view for context: it renders **both large and medium**
scheduled-service airports on a world map, coloured by type, so you can see what
the metric keeps (large) versus what it leaves out (medium), and the regional
density differences that drive the metric.

All airports shown pass the same gates as the eval set — non-blank IATA code,
non-blank municipality, and `scheduled_service == 'yes'` — only the `type`
filter is widened to {large, medium}. The set is derived from the raw
OurAirports CSV (downloaded by default, or `--src-csv`), not the large-only slim
parquet. A longitude-band density panel accompanies the map.

CLI:
    python -m scripts.visualization.airport.plot_airport_distribution
    python -m scripts.visualization.airport.plot_airport_distribution \\
        --src-csv path/to/airports.csv --out /tmp/airports.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd

from scripts.benchmark.v2.airports import download_ourairports_csv, filter_airports

# The distribution view contrasts large hubs (the eval set) against medium
# airports (excluded), so it always works on the widened type set.
_VIZ_TYPES = ("large_airport", "medium_airport")

# Per type: (marker size, color, z-order) — large hubs sit on top so they stay
# legible in dense regions like Western Europe and the US Northeast.
_TYPE_STYLE = {
    "large_airport": dict(s=18, color="#d62728", zorder=3, label="large"),
    "medium_airport": dict(s=4, color="#1f77b4", zorder=2, label="medium"),
}

# Rough continent assignment from the leading character bands of longitude is
# unreliable; instead we bin by ISO country → continent using a tiny prefix map
# only for the side panel. Kept deliberately coarse — it's a context histogram,
# not an analysis axis.
_CONTINENT_OF_LON = [
    (-170, -30, "Americas"),
    (-30, 60, "Europe/Africa"),
    (60, 180, "Asia/Oceania"),
]


def _load(src_csv: Path) -> pd.DataFrame:
    """Load the raw OurAirports CSV and filter to large+medium scheduled-service
    airports (same IATA/municipality/scheduled gates as the eval set)."""
    raw = pd.read_csv(src_csv, low_memory=False)
    return filter_airports(raw, types=_VIZ_TYPES)


def _lon_band(lon: float) -> str:
    for lo, hi, name in _CONTINENT_OF_LON:
        if lo <= lon < hi:
            return name
    return "Asia/Oceania"


def plot_distribution(df: pd.DataFrame, out_path: Path) -> Path:
    """Render the airport scatter on a PlateCarree world map + a density panel."""
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.12)

    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="#eaf2f8")
    ax.add_feature(cfeature.LAND, facecolor="#f6f4ef")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#999999")
    ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="#cccccc")

    for atype, style in _TYPE_STYLE.items():
        sub = df[df["type"] == atype]
        if sub.empty:  # e.g. the hub-level (large-only) set has no medium rows
            continue
        ax.scatter(
            sub["longitude_deg"], sub["latitude_deg"],
            transform=ccrs.PlateCarree(),
            s=style["s"], c=style["color"], zorder=style["zorder"],
            edgecolors="none", alpha=0.7,
            label=f"{style['label']} ({len(sub):,})",
        )

    kinds = " vs ".join(sorted(t.replace("_airport", "") for t in df["type"].unique()))
    ax.set_title(
        f"Scheduled-service airports ({kinds}) — {len(df):,} total "
        "(OurAirports; eval set = large only)",
        fontsize=13,
    )
    ax.legend(loc="lower left", framealpha=0.9, fontsize=10, markerscale=1.5)

    # Side panel: coarse longitude-band counts for quick density context.
    axb = fig.add_subplot(gs[0, 1])
    bands = df["longitude_deg"].map(_lon_band).value_counts()
    order = ["Americas", "Europe/Africa", "Asia/Oceania"]
    counts = [int(bands.get(b, 0)) for b in order]
    axb.barh(order, counts, color="#888888")
    for y, c in enumerate(counts):
        axb.text(c, y, f" {c:,}", va="center", fontsize=9)
    axb.set_title("by longitude band", fontsize=11)
    axb.set_xlabel("airports")
    axb.invert_yaxis()
    axb.spines[["top", "right"]].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-csv", type=Path, default=None,
        help="Raw OurAirports airports.csv. If omitted, downloads the latest.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "airport_distribution.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    src_csv = args.src_csv
    if src_csv is None:
        print("Downloading OurAirports CSV ...")
        src_csv = download_ourairports_csv()
    elif not src_csv.exists():
        raise SystemExit(f"CSV not found at {src_csv}")

    df = _load(src_csv)
    by_type = df["type"].value_counts()
    out = plot_distribution(df, args.out)
    print(
        f"Wrote {out} ({len(df):,} airports: "
        f"{by_type.get('large_airport', 0):,} large, "
        f"{by_type.get('medium_airport', 0):,} medium)"
    )


if __name__ == "__main__":
    main()
