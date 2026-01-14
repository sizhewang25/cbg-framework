"""
Analyze pings to Vultr (AS20473) anchors.
1. Find all Vultr anchor IPs
2. Query ClickHouse for pings from probes to those anchors
3. Enrich with probe/anchor metadata (country, ASN, geo coords)
4. Group probes by ASN and visualize distribution
"""

import json
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.clickhouse import Clickhouse
from default import PROBES_TO_ANCHORS_PING_TABLE

# Paths
OUTPUT_DIR = Path(__file__).parent / "outputs"
PROBES_FILE = PROJECT_ROOT / "datasets" / "reproducibility_datasets" / "atlas" / "reproducibility_probes.json"
ANCHORS_FILE = PROJECT_ROOT / "datasets" / "reproducibility_datasets" / "atlas" / "reproducibility_anchors.json"
US_ANCHORS_FILE = OUTPUT_DIR / "us_anchors.csv"

# Output files
VULTR_ANCHORS_FILE = OUTPUT_DIR / "vultr_anchors.csv"
VULTR_PINGS_FILE = OUTPUT_DIR / "vultr_pings.csv"
VULTR_PINGS_ENRICHED_FILE = OUTPUT_DIR / "vultr_pings_enriched.csv"
VULTR_PINGS_US_ONLY_FILE = OUTPUT_DIR / "vultr_pings_us_only.csv"


def load_vultr_anchors() -> pd.DataFrame:
    """Load Vultr (AS20473) anchor IPs from the US anchors CSV."""
    us_anchors = pd.read_csv(US_ANCHORS_FILE)
    vultr_anchors = us_anchors[us_anchors["asn"] == "AS20473"].copy()
    return vultr_anchors


def load_probes_metadata() -> dict:
    """Load probe metadata from JSON file into a dict keyed by probe_id."""
    with open(PROBES_FILE, "r") as f:
        probes = json.load(f)

    probe_dict = {}
    for probe in probes:
        probe_id = probe.get("id")
        geometry = probe.get("geometry", {})
        coords = geometry.get("coordinates", [None, None]) if geometry else [None, None]

        probe_dict[probe_id] = {
            "probe_id": probe_id,
            "probe_ip": probe.get("address_v4"),
            "probe_asn": probe.get("asn_v4"),
            "probe_country": probe.get("country_code"),
            "probe_longitude": coords[0] if coords else None,
            "probe_latitude": coords[1] if coords else None,
            "probe_description": probe.get("description"),
        }
    return probe_dict


def load_anchors_metadata() -> dict:
    """Load anchor metadata from JSON file into a dict keyed by IP."""
    with open(ANCHORS_FILE, "r") as f:
        anchors = json.load(f)

    anchor_dict = {}
    for anchor in anchors:
        ip = anchor.get("address_v4") or anchor.get("ip_v4")
        geometry = anchor.get("geometry", {})
        coords = geometry.get("coordinates", [None, None]) if geometry else [None, None]

        anchor_dict[ip] = {
            "anchor_ip": ip,
            "anchor_asn": anchor.get("asn_v4"),
            "anchor_country": anchor.get("country_code"),
            "anchor_longitude": coords[0] if coords else None,
            "anchor_latitude": coords[1] if coords else None,
            "anchor_city": anchor.get("city"),
        }
    return anchor_dict


def query_pings_to_vultr(vultr_ips: list) -> pd.DataFrame:
    """Query ClickHouse for pings from probes to Vultr anchors."""
    ch = Clickhouse()

    # Build query to get pings to Vultr anchors
    ip_list = ", ".join([f"toIPv4('{ip}')" for ip in vultr_ips])

    query = f"""
    SELECT
        IPv4NumToString(src) as src_ip,
        IPv4NumToString(dst) as dst_ip,
        prb_id,
        min as min_rtt,
        mean as mean_rtt,
        sent,
        rcvd,
        msm_id,
        date
    FROM {ch.database}.{PROBES_TO_ANCHORS_PING_TABLE}
    WHERE dst IN ({ip_list})
    AND min > 0
    """

    print(f"Executing query for pings to {len(vultr_ips)} Vultr anchors...")
    results = ch.execute(query)

    df = pd.DataFrame(results, columns=[
        "src_ip", "dst_ip", "prb_id", "min_rtt", "mean_rtt",
        "sent", "rcvd", "msm_id", "date"
    ])

    return df


def enrich_pings(pings_df: pd.DataFrame, probes_meta: dict, anchors_meta: dict) -> pd.DataFrame:
    """Enrich ping data with probe and anchor metadata."""
    enriched_rows = []

    for _, row in pings_df.iterrows():
        enriched = dict(row)

        # Add probe metadata
        probe_info = probes_meta.get(row["prb_id"], {})
        enriched.update({
            "probe_asn": probe_info.get("probe_asn"),
            "probe_country": probe_info.get("probe_country"),
            "probe_latitude": probe_info.get("probe_latitude"),
            "probe_longitude": probe_info.get("probe_longitude"),
        })

        # Add anchor metadata
        anchor_info = anchors_meta.get(row["dst_ip"], {})
        enriched.update({
            "anchor_asn": anchor_info.get("anchor_asn"),
            "anchor_country": anchor_info.get("anchor_country"),
            "anchor_latitude": anchor_info.get("anchor_latitude"),
            "anchor_longitude": anchor_info.get("anchor_longitude"),
            "anchor_city": anchor_info.get("anchor_city"),
        })

        enriched_rows.append(enriched)

    return pd.DataFrame(enriched_rows)


def get_top_asns(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Group by probe ASN and return top N by count."""
    # Count unique probes per ASN
    probe_counts = df.groupby("probe_asn")["prb_id"].nunique().reset_index(name="probe_count")
    probe_counts = probe_counts.sort_values("probe_count", ascending=False).head(n)
    return probe_counts


def generate_html_map(df: pd.DataFrame, asn: int, output_path: Path):
    """Generate an HTML map for probes of a specific ASN that pinged Vultr."""
    asn_df = df[df["probe_asn"] == asn].copy()

    if asn_df.empty:
        return

    # Get unique probes
    probes = asn_df.drop_duplicates(subset=["prb_id"])[
        ["prb_id", "probe_latitude", "probe_longitude", "src_ip", "probe_country"]
    ].dropna(subset=["probe_latitude", "probe_longitude"])

    if probes.empty:
        return

    # Calculate center of the map
    center_lat = probes["probe_latitude"].mean()
    center_lon = probes["probe_longitude"].mean()

    # Generate markers for probes
    markers_js = ""
    for _, row in probes.iterrows():
        popup = f"Probe {int(row['prb_id'])}<br>{row['src_ip']}<br>{row['probe_country']}"
        popup = popup.replace('"', '\\"').replace("'", "\\'")
        markers_js += f'L.marker([{row["probe_latitude"]}, {row["probe_longitude"]}]).addTo(map).bindPopup("{popup}");\n'

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>AS{asn} Probes Pinging Vultr Anchors</title>
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
            div.innerHTML = '<h4>AS{asn}</h4>Probes pinging Vultr: {len(probes)}';
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load Vultr anchors
    print("=" * 60)
    print("Step 1: Loading Vultr (AS20473) anchors...")
    print("=" * 60)

    vultr_anchors = load_vultr_anchors()
    print(f"Found {len(vultr_anchors)} Vultr anchors")
    print(vultr_anchors[["ip", "city", "region"]].to_string())

    # Save Vultr anchors
    vultr_anchors.to_csv(VULTR_ANCHORS_FILE, index=False)
    print(f"\nSaved to: {VULTR_ANCHORS_FILE}")

    # Step 2: Query ClickHouse for pings to Vultr anchors
    print("\n" + "=" * 60)
    print("Step 2: Querying ClickHouse for pings to Vultr anchors...")
    print("=" * 60)

    vultr_ips = vultr_anchors["ip"].tolist()
    pings_df = query_pings_to_vultr(vultr_ips)
    print(f"Found {len(pings_df)} ping measurements")
    print(f"From {pings_df['prb_id'].nunique()} unique probes")

    # Save raw pings
    pings_df.to_csv(VULTR_PINGS_FILE, index=False)
    print(f"\nSaved to: {VULTR_PINGS_FILE}")

    # Step 3: Enrich with metadata
    print("\n" + "=" * 60)
    print("Step 3: Enriching with probe/anchor metadata...")
    print("=" * 60)

    probes_meta = load_probes_metadata()
    anchors_meta = load_anchors_metadata()
    print(f"Loaded metadata for {len(probes_meta)} probes and {len(anchors_meta)} anchors")

    enriched_df = enrich_pings(pings_df, probes_meta, anchors_meta)

    # Save enriched pings
    enriched_df.to_csv(VULTR_PINGS_ENRICHED_FILE, index=False)
    print(f"\nSaved enriched data to: {VULTR_PINGS_ENRICHED_FILE}")

    # Filter for US-only (both probe and anchor in US)
    us_only_df = enriched_df[
        (enriched_df["probe_country"] == "US") &
        (enriched_df["anchor_country"] == "US")
    ].copy()
    us_only_df.to_csv(VULTR_PINGS_US_ONLY_FILE, index=False)
    print(f"US-only pings (US probe -> US anchor): {len(us_only_df)} measurements from {us_only_df['prb_id'].nunique()} probes")
    print(f"Saved to: {VULTR_PINGS_US_ONLY_FILE}")

    # Step 4: Group by probe ASN and show distribution
    print("\n" + "=" * 60)
    print("Step 4: Probe ASN Distribution (pinging Vultr anchors):")
    print("=" * 60)

    top_asns = get_top_asns(enriched_df, n=10)
    for _, row in top_asns.iterrows():
        asn = row["probe_asn"]
        count = row["probe_count"]
        asn_str = f"AS{int(asn)}" if pd.notna(asn) else "Unknown"
        print(f"{int(count):4d} probes  {asn_str}")

    # Distribution summary
    print("\n" + "=" * 60)
    print("Distribution Summary:")
    print("=" * 60)
    total_probes = enriched_df["prb_id"].nunique()
    unique_asns = enriched_df["probe_asn"].nunique()
    print(f"Total unique probes pinging Vultr: {total_probes}")
    print(f"Unique probe ASNs: {unique_asns}")
    print(f"Top 10 ASNs coverage: {top_asns['probe_count'].sum()} probes ({100*top_asns['probe_count'].sum()/total_probes:.1f}%)")

    # Country distribution
    print("\n" + "=" * 60)
    print("Probe Country Distribution:")
    print("=" * 60)
    country_counts = enriched_df.groupby("probe_country")["prb_id"].nunique().sort_values(ascending=False).head(10)
    for country, count in country_counts.items():
        print(f"{count:4d} probes  {country}")

    # Step 5: Generate HTML maps
    print("\n" + "=" * 60)
    print("Step 5: Generating HTML maps for top 10 ASNs:")
    print("=" * 60)

    for _, row in top_asns.iterrows():
        asn = row["probe_asn"]
        if pd.notna(asn):
            asn_int = int(asn)
            map_path = OUTPUT_DIR / f"vultr_pings_AS{asn_int}.html"
            generate_html_map(enriched_df, asn_int, map_path)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")

    return enriched_df


if __name__ == "__main__":
    main()
