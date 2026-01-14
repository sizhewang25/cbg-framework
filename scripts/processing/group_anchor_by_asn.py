"""
Group RIPE Atlas anchors by ASN/organization and analyze distribution.
Filter for US anchors, show top-10 org distribution, and generate HTML maps.
"""

import json
import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_FILE = PROJECT_ROOT / "datasets" / "static_datasets" / "ip_info_geo_anchors.json"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "us_anchors.csv"


def load_anchors_to_dataframe(input_path: Path) -> pd.DataFrame:
    """Load the IP info JSON file into a pandas DataFrame."""
    with open(input_path, "r") as f:
        data = json.load(f)

    # Convert dict of dicts to DataFrame
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "ip_key"
    df = df.reset_index(drop=True)

    # Parse lat/lon from 'loc' column
    if "loc" in df.columns:
        loc_split = df["loc"].str.split(",", expand=True)
        df["latitude"] = pd.to_numeric(loc_split[0], errors="coerce")
        df["longitude"] = pd.to_numeric(loc_split[1], errors="coerce")

    return df


def extract_asn(org: str) -> str:
    """Extract ASN number from org string like 'AS24173 Netnam Company'."""
    if pd.isna(org):
        return None
    parts = org.split(" ", 1)
    if parts and parts[0].startswith("AS"):
        return parts[0]
    return None


def get_top_orgs(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Group IPs by organization and return top N by count."""
    org_counts = df.groupby("org").size().reset_index(name="count")
    org_counts = org_counts.sort_values("count", ascending=False).head(n)
    return org_counts


def generate_html_map(df: pd.DataFrame, org: str, output_path: Path):
    """Generate an HTML map for anchors of a specific organization."""
    org_df = df[df["org"] == org].copy()

    if org_df.empty:
        return

    # Calculate center of the map
    center_lat = org_df["latitude"].mean()
    center_lon = org_df["longitude"].mean()

    # Extract ASN for filename-safe naming
    asn = extract_asn(org) or "unknown"

    # Generate HTML with Leaflet.js
    markers_js = ""
    for _, row in org_df.iterrows():
        popup = f"{row['ip']}<br>{row['city']}, {row['region']}"
        markers_js += f'L.marker([{row["latitude"]}, {row["longitude"]}]).addTo(map).bindPopup("{popup}");\n'

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{org} - US Anchors Map</title>
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
            div.innerHTML = '<h4>{org}</h4>Anchors: {len(org_df)}';
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
    df = load_anchors_to_dataframe(INPUT_FILE)

    print(f"\nTotal anchors loaded: {len(df)}")

    # Extract ASN for additional grouping
    df["asn"] = df["org"].apply(extract_asn)

    # Filter for US anchors only
    us_df = df[df["country"] == "US"].copy()
    print(f"US anchors: {len(us_df)}")

    # Save US anchors to CSV
    us_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved US anchors to: {OUTPUT_FILE}")

    # Show top 10 organizations by IP count (US only)
    print("\n" + "=" * 60)
    print("Top 10 US Organizations by Number of Anchors:")
    print("=" * 60)

    top_orgs = get_top_orgs(us_df, n=10)
    for _, row in top_orgs.iterrows():
        print(f"{row['count']:4d}  {row['org']}")

    # Distribution summary
    print("\n" + "=" * 60)
    print("US Distribution Summary:")
    print("=" * 60)
    org_counts = us_df.groupby("org").size()
    print(f"Total US anchors: {len(us_df)}")
    print(f"Unique US organizations: {len(org_counts)}")
    print(f"Anchors in top 10 orgs: {top_orgs['count'].sum()} ({100*top_orgs['count'].sum()/len(us_df):.1f}%)")

    # Generate HTML maps for top 10 organizations
    print("\n" + "=" * 60)
    print("Generating HTML maps for top 10 organizations:")
    print("=" * 60)

    for _, row in top_orgs.iterrows():
        org = row["org"]
        asn = extract_asn(org) or "unknown"
        # Create safe filename
        safe_name = asn.replace("/", "_").replace("\\", "_")
        map_path = OUTPUT_DIR / f"map_{safe_name}.html"
        generate_html_map(us_df, org, map_path)

    print(f"\nAll maps saved to: {OUTPUT_DIR}")

    return us_df


if __name__ == "__main__":
    main()
