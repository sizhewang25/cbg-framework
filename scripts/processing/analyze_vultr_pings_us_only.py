"""
Analyze US-only Vultr pings: probe ASN distribution and map visualization.
"""

import json
import pandas as pd
from pathlib import Path

# Paths
OUTPUT_DIR = Path(__file__).parent / "outputs"
US_ONLY_FILE = OUTPUT_DIR / "vultr_pings_us_only.csv"


def load_us_only_pings() -> pd.DataFrame:
    """Load the US-only Vultr pings data."""
    return pd.read_csv(US_ONLY_FILE)


def analyze_by_asn(df: pd.DataFrame) -> pd.DataFrame:
    """Group by probe ASN and compute statistics."""
    probe_by_asn = df.groupby("probe_asn").agg({
        "prb_id": "nunique",
        "min_rtt": ["mean", "min", "max", "median", "count"]
    }).reset_index()
    probe_by_asn.columns = [
        "probe_asn", "probe_count", "avg_rtt", "min_rtt",
        "max_rtt", "median_rtt", "measurement_count"
    ]
    probe_by_asn = probe_by_asn.sort_values("probe_count", ascending=False)
    return probe_by_asn


def generate_html_map(df: pd.DataFrame, asn: int, output_path: Path):
    """Generate an HTML map for US probes of a specific ASN with interactive anchor filtering."""
    asn_df = df[df["probe_asn"] == asn].copy()

    if asn_df.empty:
        return

    # Get unique probes with coordinates and their anchor associations
    probes = asn_df.drop_duplicates(subset=["prb_id"])[
        ["prb_id", "probe_latitude", "probe_longitude", "src_ip"]
    ].dropna(subset=["probe_latitude", "probe_longitude"])

    if probes.empty:
        return

    # Get unique anchors for this ASN's pings
    anchors = asn_df.drop_duplicates(subset=["dst_ip"])[
        ["dst_ip", "anchor_latitude", "anchor_longitude", "anchor_city"]
    ].dropna(subset=["anchor_latitude", "anchor_longitude"])

    # Build probe-to-anchor mapping (which probes pinged which anchors)
    probe_anchor_map = {}
    for _, row in asn_df.iterrows():
        prb_id = int(row["prb_id"])
        dst_ip = row["dst_ip"]
        if prb_id not in probe_anchor_map:
            probe_anchor_map[prb_id] = set()
        probe_anchor_map[prb_id].add(dst_ip)

    # Convert to JSON-friendly format
    probe_anchor_json = {str(k): list(v) for k, v in probe_anchor_map.items()}

    # Build anchor stats (probes per anchor, RTT stats)
    anchor_stats = {}
    for dst_ip in anchors["dst_ip"]:
        anchor_pings = asn_df[asn_df["dst_ip"] == dst_ip]
        anchor_stats[dst_ip] = {
            "probe_count": anchor_pings["prb_id"].nunique(),
            "measurement_count": len(anchor_pings),
            "avg_rtt": round(anchor_pings["min_rtt"].mean(), 1),
            "min_rtt": round(anchor_pings["min_rtt"].min(), 1),
            "max_rtt": round(anchor_pings["min_rtt"].max(), 1),
        }

    # Calculate center of the map (US-centered)
    center_lat = 39.8283
    center_lon = -98.5795

    # RTT stats for this ASN (overall)
    avg_rtt = asn_df["min_rtt"].mean()
    min_rtt = asn_df["min_rtt"].min()
    max_rtt = asn_df["min_rtt"].max()

    # Build probes data as JSON
    probes_data = []
    for _, row in probes.iterrows():
        probes_data.append({
            "prb_id": int(row["prb_id"]),
            "lat": row["probe_latitude"],
            "lon": row["probe_longitude"],
            "ip": row["src_ip"],
        })

    # Build anchors data as JSON
    anchors_data = []
    for _, row in anchors.iterrows():
        stats = anchor_stats.get(row["dst_ip"], {})
        anchors_data.append({
            "ip": row["dst_ip"],
            "lat": row["anchor_latitude"],
            "lon": row["anchor_longitude"],
            "city": row["anchor_city"],
            "probe_count": stats.get("probe_count", 0),
            "measurement_count": stats.get("measurement_count", 0),
            "avg_rtt": stats.get("avg_rtt", 0),
            "min_rtt": stats.get("min_rtt", 0),
            "max_rtt": stats.get("max_rtt", 0),
        })

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>AS{asn} US Probes to Vultr Anchors</title>
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
            max-width: 280px;
        }}
        .info h4 {{ margin: 0 0 5px; }}
        .info p {{ margin: 5px 0; font-size: 12px; }}
        .legend {{
            padding: 10px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
        }}
        .legend-item {{ display: flex; align-items: center; margin: 5px 0; font-size: 12px; }}
        .legend-circle {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .filter-info {{
            padding: 10px 15px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            max-width: 250px;
        }}
        .filter-info h4 {{ margin: 0 0 8px; color: #d63384; }}
        .filter-info p {{ margin: 4px 0; font-size: 11px; }}
        .reset-btn {{
            margin-top: 8px;
            padding: 5px 10px;
            background: #6c757d;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        }}
        .reset-btn:hover {{ background: #5a6268; }}
        .anchor-selected {{
            animation: pulse 1s infinite;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.3); }}
            100% {{ transform: scale(1); }}
        }}
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

        // Data
        var probesData = {json.dumps(probes_data)};
        var anchorsData = {json.dumps(anchors_data)};
        var probeAnchorMap = {json.dumps(probe_anchor_json)};

        // Layer groups
        var probeMarkers = L.layerGroup().addTo(map);
        var anchorMarkers = L.layerGroup().addTo(map);

        // State
        var selectedAnchor = null;
        var allProbeMarkerRefs = {{}};
        var allAnchorMarkerRefs = {{}};

        // Info control
        var info = L.control({{position: 'topright'}});
        info.onAdd = function(map) {{
            this._div = L.DomUtil.create('div', 'info');
            this.update();
            return this._div;
        }};
        info.update = function(anchorData) {{
            if (anchorData) {{
                this._div.innerHTML = '<h4>AS{asn}</h4>' +
                    '<p style="color: #d63384;"><strong>Filtered by: ' + anchorData.city + '</strong></p>' +
                    '<p><strong>Anchor IP:</strong> ' + anchorData.ip + '</p>' +
                    '<p><strong>Probes to this anchor:</strong> ' + anchorData.probe_count + '</p>' +
                    '<p><strong>Measurements:</strong> ' + anchorData.measurement_count + '</p>' +
                    '<p><strong>Avg RTT:</strong> ' + anchorData.avg_rtt + ' ms</p>' +
                    '<p><strong>Min RTT:</strong> ' + anchorData.min_rtt + ' ms</p>' +
                    '<p><strong>Max RTT:</strong> ' + anchorData.max_rtt + ' ms</p>';
            }} else {{
                this._div.innerHTML = '<h4>AS{asn}</h4>' +
                    '<p><strong>Total US Probes:</strong> {len(probes)}</p>' +
                    '<p><strong>Total Measurements:</strong> {len(asn_df)}</p>' +
                    '<p><strong>Avg RTT:</strong> {avg_rtt:.1f} ms</p>' +
                    '<p><strong>Min RTT:</strong> {min_rtt:.1f} ms</p>' +
                    '<p><strong>Max RTT:</strong> {max_rtt:.1f} ms</p>' +
                    '<p style="color: #666; font-style: italic; margin-top: 10px;">Click an anchor (★) to filter probes</p>';
            }}
        }};
        info.addTo(map);

        // Filter info control
        var filterInfo = L.control({{position: 'topleft'}});
        filterInfo.onAdd = function(map) {{
            this._div = L.DomUtil.create('div', 'filter-info');
            this._div.style.display = 'none';
            return this._div;
        }};
        filterInfo.update = function(anchorData) {{
            if (anchorData) {{
                this._div.style.display = 'block';
                this._div.innerHTML = '<h4>★ ' + anchorData.city + '</h4>' +
                    '<p><strong>' + anchorData.probe_count + '</strong> probes shown</p>' +
                    '<button class="reset-btn" onclick="resetFilter()">Show All Probes</button>';
            }} else {{
                this._div.style.display = 'none';
            }}
        }};
        filterInfo.addTo(map);

        // Legend
        var legend = L.control({{position: 'bottomright'}});
        legend.onAdd = function(map) {{
            var div = L.DomUtil.create('div', 'legend');
            div.innerHTML = '<div class="legend-item"><div class="legend-circle" style="background: #3388ff;"></div>Probe</div>' +
                '<div class="legend-item"><div class="legend-circle" style="background: #cccccc;"></div>Probe (filtered out)</div>' +
                '<div class="legend-item"><span style="color: red; font-size: 16px; margin-right: 5px;">★</span>Vultr Anchor (click to filter)</div>';
            return div;
        }};
        legend.addTo(map);

        // Create probe markers
        function createProbeMarkers(filteredAnchorIp) {{
            probeMarkers.clearLayers();
            allProbeMarkerRefs = {{}};

            probesData.forEach(function(probe) {{
                var probeAnchors = probeAnchorMap[probe.prb_id.toString()] || [];
                var isActive = !filteredAnchorIp || probeAnchors.includes(filteredAnchorIp);

                var marker = L.circleMarker([probe.lat, probe.lon], {{
                    radius: isActive ? 6 : 4,
                    fillColor: isActive ? "#3388ff" : "#cccccc",
                    color: isActive ? "#000" : "#999",
                    weight: 1,
                    opacity: isActive ? 1 : 0.5,
                    fillOpacity: isActive ? 0.8 : 0.3
                }});

                var popup = "Probe " + probe.prb_id + "<br>" + probe.ip;
                if (filteredAnchorIp && isActive) {{
                    popup += "<br><em>Pings to selected anchor</em>";
                }}
                marker.bindPopup(popup);
                marker.addTo(probeMarkers);
                allProbeMarkerRefs[probe.prb_id] = marker;
            }});
        }}

        // Create anchor icon
        function createAnchorIcon(isSelected) {{
            return L.divIcon({{
                className: 'anchor-icon' + (isSelected ? ' anchor-selected' : ''),
                html: '<div style="color: ' + (isSelected ? '#ff0000' : 'red') + '; font-size: ' + (isSelected ? '28px' : '20px') + '; text-shadow: 0 0 3px white;">★</div>',
                iconSize: [20, 20],
                iconAnchor: [10, 10]
            }});
        }}

        // Initialize anchor markers (called once)
        function initAnchorMarkers() {{
            anchorsData.forEach(function(anchor) {{
                var marker = L.marker([anchor.lat, anchor.lon], {{
                    icon: createAnchorIcon(false)
                }});

                var popup = "<strong>Vultr Anchor</strong><br>" +
                    anchor.ip + "<br>" +
                    anchor.city + "<br>" +
                    "<em>" + anchor.probe_count + " probes</em>";
                marker.bindPopup(popup);

                marker.on('click', function(e) {{
                    L.DomEvent.stopPropagation(e);
                    if (selectedAnchor === anchor.ip) {{
                        resetFilter();
                    }} else {{
                        filterByAnchor(anchor);
                    }}
                }});

                marker.addTo(anchorMarkers);
                allAnchorMarkerRefs[anchor.ip] = {{ marker: marker, data: anchor }};
            }});
        }}

        // Update anchor markers appearance (called on filter change)
        function updateAnchorMarkers() {{
            Object.keys(allAnchorMarkerRefs).forEach(function(ip) {{
                var ref = allAnchorMarkerRefs[ip];
                var isSelected = selectedAnchor === ip;
                ref.marker.setIcon(createAnchorIcon(isSelected));
            }});
        }}

        // Filter probes by anchor
        function filterByAnchor(anchorData) {{
            selectedAnchor = anchorData.ip;
            createProbeMarkers(anchorData.ip);
            updateAnchorMarkers();
            info.update(anchorData);
            filterInfo.update(anchorData);
        }}

        // Reset filter
        function resetFilter() {{
            selectedAnchor = null;
            createProbeMarkers(null);
            updateAnchorMarkers();
            info.update(null);
            filterInfo.update(null);
        }}

        // Make resetFilter available globally
        window.resetFilter = resetFilter;

        // Initial render
        createProbeMarkers(null);
        initAnchorMarkers();
    </script>
</body>
</html>
"""

    output_path.write_text(html_content)
    print(f"  Generated: {output_path.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("=" * 60)
    print("US-Only Vultr Pings: Probe ASN Analysis")
    print("=" * 60)

    df = load_us_only_pings()

    print(f"\nTotal measurements: {len(df)}")
    print(f"Unique probes: {df['prb_id'].nunique()}")
    print(f"Unique probe ASNs: {df['probe_asn'].nunique()}")

    # Analyze by ASN
    probe_by_asn = analyze_by_asn(df)

    # Show top 15 ASNs
    print("\n" + "=" * 60)
    print("Top 15 US Probe ASNs (by number of probes):")
    print("=" * 60)
    print(f"{'ASN':<12} {'Probes':>8} {'Measurements':>14} {'Avg RTT':>10} {'Min RTT':>10}")
    print("-" * 60)

    for _, row in probe_by_asn.head(15).iterrows():
        asn = f"AS{int(row['probe_asn'])}" if pd.notna(row['probe_asn']) else "Unknown"
        print(f"{asn:<12} {int(row['probe_count']):>8} {int(row['measurement_count']):>14} {row['avg_rtt']:>10.2f} {row['min_rtt']:>10.2f}")

    # Coverage stats
    top10_probes = probe_by_asn.head(10)["probe_count"].sum()
    total_probes = df["prb_id"].nunique()
    print(f"\nTop 10 ASNs: {int(top10_probes)} probes ({100*top10_probes/total_probes:.1f}% of total)")

    # RTT statistics for top 10 ASNs
    print("\n" + "=" * 60)
    print("RTT Statistics for Top 10 ASNs:")
    print("=" * 60)
    top_asns = probe_by_asn.head(10)["probe_asn"].tolist()
    for asn in top_asns:
        asn_df = df[df["probe_asn"] == asn]
        asn_str = f"AS{int(asn)}"
        print(f"{asn_str:<10}: min={asn_df['min_rtt'].min():.1f}ms, "
              f"median={asn_df['min_rtt'].median():.1f}ms, "
              f"max={asn_df['min_rtt'].max():.1f}ms")

    # Generate HTML maps for top 10 ASNs
    print("\n" + "=" * 60)
    print("Generating HTML maps for top 10 US ASNs:")
    print("=" * 60)

    for asn in top_asns:
        if pd.notna(asn):
            asn_int = int(asn)
            map_path = OUTPUT_DIR / f"us_vultr_pings_AS{asn_int}.html"
            generate_html_map(df, asn_int, map_path)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")

    # Save ASN analysis to CSV
    analysis_file = OUTPUT_DIR / "us_vultr_pings_asn_analysis.csv"
    probe_by_asn.to_csv(analysis_file, index=False)
    print(f"ASN analysis saved to: {analysis_file}")

    return probe_by_asn


if __name__ == "__main__":
    main()
