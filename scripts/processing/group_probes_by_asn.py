"""
Group RIPE Atlas probes by ASN/organization and analyze distribution.
Filter for US probes, show top-10 ASN distribution, and generate HTML maps.
"""

import json
import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_FILE = PROJECT_ROOT / "datasets" / "reproducibility_datasets" / "atlas" / "reproducibility_probes.json"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "us_probes.csv"


def load_probes_to_dataframe(input_path: Path) -> pd.DataFrame:
    """Load the probes JSON file into a pandas DataFrame."""
    with open(input_path, "r") as f:
        data = json.load(f)

    # Flatten the nested structure
    rows = []
    for probe in data:
        row = {
            "id": probe.get("id"),
            "address_v4": probe.get("address_v4"),
            "address_v6": probe.get("address_v6"),
            "asn_v4": probe.get("asn_v4"),
            "asn_v6": probe.get("asn_v6"),
            "country_code": probe.get("country_code"),
            "description": probe.get("description"),
            "is_anchor": probe.get("is_anchor"),
            "is_public": probe.get("is_public"),
            "prefix_v4": probe.get("prefix_v4"),
            "prefix_v6": probe.get("prefix_v6"),
            "type": probe.get("type"),
        }

        # Extract coordinates from geometry
        geometry = probe.get("geometry", {})
        if geometry and geometry.get("coordinates"):
            coords = geometry["coordinates"]
            row["longitude"] = coords[0]
            row["latitude"] = coords[1]
        else:
            row["longitude"] = None
            row["latitude"] = None

        # Extract status
        status = probe.get("status", {})
        row["status_name"] = status.get("name") if status else None

        # Extract tags as comma-separated string
        tags = probe.get("tags", [])
        row["tags"] = ",".join([t.get("slug", "") for t in tags]) if tags else ""

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def get_top_asns(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Group IPs by ASN and return top N by count."""
    asn_counts = df.groupby("asn_v4").size().reset_index(name="count")
    asn_counts = asn_counts.sort_values("count", ascending=False).head(n)
    return asn_counts


def generate_html_map(df: pd.DataFrame, asn: int, output_path: Path):
    """Generate an HTML map for probes of a specific ASN."""
    asn_df = df[df["asn_v4"] == asn].copy()

    if asn_df.empty:
        return

    # Filter out rows with missing coordinates
    asn_df = asn_df.dropna(subset=["latitude", "longitude"])
    if asn_df.empty:
        return

    # Calculate center of the map
    center_lat = asn_df["latitude"].mean()
    center_lon = asn_df["longitude"].mean()

    # Generate HTML with Leaflet.js
    markers_js = ""
    for _, row in asn_df.iterrows():
        popup = f"Probe {row['id']}<br>{row['address_v4']}<br>{row['description'] or 'No description'}"
        # Escape quotes in popup
        popup = popup.replace('"', '\\"').replace("'", "\\'")
        markers_js += f'L.marker([{row["latitude"]}, {row["longitude"]}]).addTo(map).bindPopup("{popup}");\n'

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>AS{asn} - US Probes Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}
        .info {{
            padding: 10px 15px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
        }}
        .info h4 {{ margin: 0 0 5px; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 4);

        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '&copy; OpenStreetMap contributors'
        }}).addTo(map);

        // Add info control
        var info = L.control({{position: 'topright'}});
        info.onAdd = function(map) {{
            var div = L.DomUtil.create('div', 'info');
            div.innerHTML = '<h4>AS{asn}</h4>Probes: {len(asn_df)}';
            return div;
        }};
        info.addTo(map);

        // Add markers
        {markers_js}
    </script>
</body>
</html>
"""

    output_path.write_text(html_content)
    print(f"  Generated: {output_path.name}")


def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {INPUT_FILE}")
    df = load_probes_to_dataframe(INPUT_FILE)

    print(f"\nTotal probes loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Filter for US probes only
    us_df = df[df["country_code"] == "US"].copy()
    print(f"US probes: {len(us_df)}")

    # Save US probes to CSV
    us_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved US probes to: {OUTPUT_FILE}")

    # Show top 10 ASNs by probe count (US only)
    print("\n" + "=" * 60)
    print("Top 10 US ASNs by Number of Probes:")
    print("=" * 60)

    top_asns = get_top_asns(us_df, n=10)
    for _, row in top_asns.iterrows():
        print(f"{int(row['count']):4d}  AS{int(row['asn_v4'])}")

    # Distribution summary
    print("\n" + "=" * 60)
    print("US Distribution Summary:")
    print("=" * 60)
    asn_counts = us_df.groupby("asn_v4").size()
    print(f"Total US probes: {len(us_df)}")
    print(f"Unique US ASNs: {len(asn_counts)}")
    print(f"Probes in top 10 ASNs: {top_asns['count'].sum()} ({100*top_asns['count'].sum()/len(us_df):.1f}%)")

    # Generate HTML maps for top 10 ASNs
    print("\n" + "=" * 60)
    print("Generating HTML maps for top 10 ASNs:")
    print("=" * 60)

    for _, row in top_asns.iterrows():
        asn = int(row["asn_v4"])
        map_path = OUTPUT_DIR / f"probes_map_AS{asn}.html"
        generate_html_map(us_df, asn, map_path)

    print(f"\nAll maps saved to: {OUTPUT_DIR}")

    return us_df


if __name__ == "__main__":
    main()
