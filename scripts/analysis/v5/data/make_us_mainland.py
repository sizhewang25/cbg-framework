"""Rebuild `us_mainland.geojson`, the unbuffered landmass v5 clips cells to.

One-shot, not part of any pipeline: the GeoJSON is vendored so v5 does not
depend on a machine's cartopy cache. Run it only to regenerate that file.

Source is Natural Earth `ne_110m_admin_0_countries`, `ADM0_A3 == "USA"`,
keeping the single polygon that intersects the lower-48 bounding box. Alaska
and Hawaii are separate parts and are dropped. The country polygon runs the
border through the Great Lakes, so the US side of every lake is inland -- the
`_lakes` variant of the layer would cut them out and is deliberately not used.

The 110m scale is ~10 km coarse at the coast, which is inside the smallest
buffer v5 applies (one nside-128 grid, 50.9 km).

    python -m scripts.analysis.v5.data.make_us_mainland [--shp PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import shapely
from shapely.geometry import box, mapping

DEFAULT_SHP = (
    Path.home()
    / ".local/share/cartopy/shapefiles/natural_earth/cultural/ne_110m_admin_0_countries.shp"
)
OUT = Path(__file__).resolve().parent / "us_mainland.geojson"

#: Lower-48 bounding box, (lon_min, lat_min, lon_max, lat_max).
CONUS_BBOX = (-125.0, 24.0, -66.0, 50.0)


def main() -> None:
    import geopandas as gpd

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shp", type=Path, default=DEFAULT_SHP)
    args = ap.parse_args()

    countries = gpd.read_file(args.shp)
    usa = countries.loc[countries["ADM0_A3"] == "USA", "geometry"].iloc[0]
    parts = [g for g in getattr(usa, "geoms", [usa]) if g.intersects(box(*CONUS_BBOX))]
    if len(parts) != 1:
        raise SystemExit(f"expected one lower-48 polygon, found {len(parts)}")
    mainland = shapely.make_valid(parts[0])

    OUT.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "name": "US mainland (lower 48)",
                            "source": "Natural Earth ne_110m_admin_0_countries",
                            "adm0_a3": "USA",
                            "crs": "EPSG:4326",
                        },
                        "geometry": mapping(mainland),
                    }
                ],
            }
        )
        + "\n"
    )
    print(f"wrote {OUT} ({len(mainland.exterior.coords)} exterior vertices)")


if __name__ == "__main__":
    main()
