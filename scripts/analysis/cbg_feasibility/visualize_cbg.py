"""
CBG Multilateration Visualization

This script:
1. Loads fitted RTT-distance models for all anchors
2. Selects random probes from the target ASN as "unknown targets"
3. Applies CBG multilateration using calibrated models
4. Generates interactive Folium maps showing:
   - Anchor locations (red stars)
   - Distance constraint circles from each anchor
   - True probe location (green marker)
   - Estimated location from circle intersection (blue marker)
"""

import pandas as pd
import numpy as np
import folium
from folium import plugins
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from dataclasses import dataclass

try:
    from shapely.geometry import Point, Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: shapely not installed. Using fallback centroid method.")

from rtt_model import RTTDistanceModel, haversine_distance, rtt_to_distance_fixed


@dataclass
class CBGResult:
    """Results from CBG multilateration."""
    target_ip: str
    true_lat: float
    true_lon: float
    estimated_lat: Optional[float]
    estimated_lon: Optional[float]
    error_km: Optional[float]
    n_anchors_used: int
    anchor_distances: Dict[str, float]  # anchor_ip -> predicted distance
    anchor_rtts: Dict[str, float]  # anchor_ip -> observed RTT


def load_models(model_dir: Path) -> Dict[str, RTTDistanceModel]:
    """Load all fitted models from directory."""
    models = {}
    for pkl_file in model_dir.glob("*.pkl"):
        if pkl_file.stem.startswith("scatter_"):
            continue  # Skip scatter plots
        model = RTTDistanceModel.load(pkl_file)
        if model.fitted:
            models[model.anchor_ip] = model
    print(f"Loaded {len(models)} fitted models")
    return models


def load_data(data_path: Path, asn: int) -> pd.DataFrame:
    """Load and filter data for target ASN."""
    df = pd.read_csv(data_path)
    df = df[df['probe_asn'] == float(asn)].copy()
    print(f"Loaded {len(df)} measurements for ASN {asn}")
    return df


def get_probe_rtts(df: pd.DataFrame, probe_ip: str) -> Dict[str, float]:
    """Get min RTT from probe to each anchor."""
    probe_data = df[df['src_ip'] == probe_ip]
    rtts = {}
    for _, row in probe_data.iterrows():
        rtts[row['dst_ip']] = row['min_rtt']
    return rtts


def create_circle_polygon(center_lat: float, center_lon: float, radius_km: float, num_points: int = 64) -> Optional['Polygon']:
    """
    Create a polygon approximating a circle on Earth's surface.

    Args:
        center_lat, center_lon: Center coordinates (degrees)
        radius_km: Radius in kilometers
        num_points: Number of points to use for polygon approximation

    Returns:
        shapely.geometry.Polygon object or None if shapely not available
    """
    if not SHAPELY_AVAILABLE:
        return None

    # Approximate conversion factors at this latitude
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(center_lat))

    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        dx_km = radius_km * math.cos(angle)
        dy_km = radius_km * math.sin(angle)

        # Convert back to lat/lon
        lat = center_lat + (dy_km / km_per_deg_lat)
        lon = center_lon + (dx_km / km_per_deg_lon)
        points.append((lon, lat))  # Shapely uses (lon, lat) order

    return Polygon(points)


def find_circles_intersection(circles: List[Tuple[float, float, float]]) -> Tuple[Optional[float], Optional[float]]:
    """
    Find the centroid of the intersection region of multiple circles.

    Uses Shapely for proper geometric intersection.

    Args:
        circles: List of (lat, lon, radius_km) tuples

    Returns:
        (centroid_lat, centroid_lon) or (None, None) if no intersection
    """
    if not SHAPELY_AVAILABLE or not circles:
        return estimate_location_centroid_fallback(circles)

    # Create polygon approximations of circles
    polygons = []
    for center_lat, center_lon, radius_km in circles:
        poly = create_circle_polygon(center_lat, center_lon, radius_km)
        if poly is not None:
            polygons.append(poly)

    if not polygons:
        return None, None

    # Find intersection of all polygons
    intersection = polygons[0]
    for poly in polygons[1:]:
        intersection = intersection.intersection(poly)
        if intersection.is_empty:
            # No common intersection - fall back to weighted centroid
            return estimate_location_centroid_fallback(circles)

    # Get centroid of intersection region
    if intersection.is_empty:
        return estimate_location_centroid_fallback(circles)

    centroid = intersection.centroid
    # Shapely uses (lon, lat) order
    return centroid.y, centroid.x


def estimate_location_centroid_fallback(
    circles: List[Tuple[float, float, float]]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Fallback: Estimate target location as centroid of anchor positions weighted by inverse distance.

    Used when shapely is not available or circles don't intersect.

    Args:
        circles: List of (lat, lon, radius_km) tuples for each anchor

    Returns:
        (estimated_lat, estimated_lon) or (None, None) if no circles
    """
    if not circles:
        return None, None

    # Use inverse radius as weight (closer anchors get more weight)
    total_weight = 0
    weighted_lat = 0
    weighted_lon = 0

    for lat, lon, radius in circles:
        if radius > 0:
            weight = 1.0 / radius
            weighted_lat += lat * weight
            weighted_lon += lon * weight
            total_weight += weight

    if total_weight == 0:
        return None, None

    return weighted_lat / total_weight, weighted_lon / total_weight


def run_cbg_for_probe(
    probe_ip: str,
    probe_lat: float,
    probe_lon: float,
    probe_rtts: Dict[str, float],
    models: Dict[str, RTTDistanceModel]
) -> CBGResult:
    """
    Run CBG multilateration for a single probe.

    Args:
        probe_ip: Target probe IP
        probe_lat, probe_lon: True probe location
        probe_rtts: Dict of anchor_ip -> observed RTT
        models: Dict of anchor_ip -> fitted RTTDistanceModel

    Returns:
        CBGResult with estimated location and error
    """
    circles = []
    anchor_distances = {}
    anchor_rtts = {}

    for anchor_ip, rtt in probe_rtts.items():
        if anchor_ip not in models:
            continue

        model = models[anchor_ip]
        distance = model.predict_distance(rtt)

        if distance is not None and distance > 0:
            circles.append((model.anchor_lat, model.anchor_lon, distance))
            anchor_distances[anchor_ip] = distance
            anchor_rtts[anchor_ip] = rtt

    # Estimate location using circle intersection
    est_lat, est_lon = find_circles_intersection(circles)

    # Calculate error
    error_km = None
    if est_lat is not None and est_lon is not None:
        error_km = haversine_distance(probe_lat, probe_lon, est_lat, est_lon)

    return CBGResult(
        target_ip=probe_ip,
        true_lat=probe_lat,
        true_lon=probe_lon,
        estimated_lat=est_lat,
        estimated_lon=est_lon,
        error_km=error_km,
        n_anchors_used=len(circles),
        anchor_distances=anchor_distances,
        anchor_rtts=anchor_rtts
    )


def create_cbg_map(
    result: CBGResult,
    models: Dict[str, RTTDistanceModel],
    output_path: Path,
    show_fixed_circles: bool = True
) -> None:
    """
    Create interactive Folium map showing CBG multilateration.

    Shows:
    - Anchor locations (red stars)
    - Calibrated distance circles (blue, semi-transparent)
    - Fixed 2/3c distance circles (gray dashed, for comparison)
    - True probe location (green marker)
    - Estimated location (blue marker with error)
    """
    # Center map on US
    center_lat = 39.8283
    center_lon = -98.5795
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles='cartodbpositron')

    # Add title
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 60px; z-index: 1000;
                background-color: white; padding: 10px; border-radius: 5px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
        <h4 style="margin: 0;">CBG Multilateration: {result.target_ip}</h4>
        <p style="margin: 5px 0 0 0; font-size: 12px;">
            Anchors used: {result.n_anchors_used} |
            Error: {result.error_km:.1f} km
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Add anchors and circles
    for anchor_ip, model in models.items():
        # Anchor marker (red star)
        anchor_popup = f"""
        <b>Anchor: {anchor_ip}</b><br>
        Location: ({model.anchor_lat:.2f}, {model.anchor_lon:.2f})<br>
        Slope: {model.slope:.4f} ms/km<br>
        Intercept: {model.intercept:.1f} ms
        """

        folium.Marker(
            location=[model.anchor_lat, model.anchor_lon],
            popup=anchor_popup,
            icon=folium.DivIcon(
                html='<div style="font-size: 20px; color: red;">★</div>',
                icon_size=(20, 20),
                icon_anchor=(10, 10)
            )
        ).add_to(m)

        # Distance circle (calibrated)
        if anchor_ip in result.anchor_distances:
            distance = result.anchor_distances[anchor_ip]
            rtt = result.anchor_rtts[anchor_ip]

            # Calibrated circle (blue)
            folium.Circle(
                location=[model.anchor_lat, model.anchor_lon],
                radius=distance * 1000,  # Convert km to meters
                color='blue',
                fill=True,
                fill_opacity=0.1,
                weight=2,
                popup=f"Calibrated: {distance:.0f} km (RTT: {rtt:.1f} ms)"
            ).add_to(m)

            # Fixed 2/3c circle for comparison (gray dashed)
            if show_fixed_circles:
                fixed_distance = rtt_to_distance_fixed(rtt, speed_fraction=2/3)
                folium.Circle(
                    location=[model.anchor_lat, model.anchor_lon],
                    radius=fixed_distance * 1000,
                    color='gray',
                    fill=False,
                    weight=1,
                    dash_array='5,5',
                    popup=f"Fixed (2/3c): {fixed_distance:.0f} km"
                ).add_to(m)

    # True location (green marker)
    folium.Marker(
        location=[result.true_lat, result.true_lon],
        popup=f"<b>True Location</b><br>{result.target_ip}<br>({result.true_lat:.4f}, {result.true_lon:.4f})",
        icon=folium.Icon(color='green', icon='home')
    ).add_to(m)

    # Estimated location (blue marker)
    if result.estimated_lat is not None:
        folium.Marker(
            location=[result.estimated_lat, result.estimated_lon],
            popup=f"<b>Estimated Location</b><br>({result.estimated_lat:.4f}, {result.estimated_lon:.4f})<br>Error: {result.error_km:.1f} km",
            icon=folium.Icon(color='blue', icon='crosshairs', prefix='fa')
        ).add_to(m)

        # Line from true to estimated
        folium.PolyLine(
            locations=[
                [result.true_lat, result.true_lon],
                [result.estimated_lat, result.estimated_lon]
            ],
            color='red',
            weight=2,
            dash_array='5,5',
            popup=f"Error: {result.error_km:.1f} km"
        ).add_to(m)

    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 30px; right: 30px; z-index: 1000;
                background-color: white; padding: 10px; border-radius: 5px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.3); font-size: 12px;">
        <b>Legend</b><br>
        <span style="color: red;">★</span> Vultr Anchor<br>
        <span style="color: green;">●</span> True Location<br>
        <span style="color: blue;">●</span> Estimated Location<br>
        <span style="color: blue;">━</span> Calibrated Circle<br>
        <span style="color: gray;">┄</span> Fixed (2/3c) Circle
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Fit bounds to show all markers
    all_lats = [result.true_lat] + [m.anchor_lat for m in models.values()]
    all_lons = [result.true_lon] + [m.anchor_lon for m in models.values()]
    if result.estimated_lat:
        all_lats.append(result.estimated_lat)
        all_lons.append(result.estimated_lon)

    m.fit_bounds([[min(all_lats) - 2, min(all_lons) - 2],
                  [max(all_lats) + 2, max(all_lons) + 2]])

    # Save map
    m.save(str(output_path))
    print(f"  Saved map: {output_path.name}")


def run_cbg_analysis(
    df: pd.DataFrame,
    models: Dict[str, RTTDistanceModel],
    output_dir: Path,
    n_samples: int = 5,
    random_seed: int = 42
) -> List[CBGResult]:
    """
    Run CBG analysis on random sample of probes.

    Args:
        df: Filtered dataframe for target ASN
        models: Fitted anchor models
        output_dir: Directory for output maps
        n_samples: Number of probes to sample
        random_seed: Random seed for reproducibility

    Returns:
        List of CBGResult objects
    """
    # Get unique probes with their locations
    probes = df[['src_ip', 'probe_latitude', 'probe_longitude']].drop_duplicates()
    print(f"\nTotal unique probes: {len(probes)}")

    # Sample probes
    np.random.seed(random_seed)
    if len(probes) > n_samples:
        sample_idx = np.random.choice(len(probes), n_samples, replace=False)
        probes = probes.iloc[sample_idx]
    print(f"Sampling {len(probes)} probes for CBG test")

    # Run CBG for each probe
    results = []

    for _, probe in probes.iterrows():
        probe_ip = probe['src_ip']
        probe_lat = probe['probe_latitude']
        probe_lon = probe['probe_longitude']

        print(f"\nProcessing probe {probe_ip}...")

        # Get RTTs from this probe to all anchors
        probe_rtts = get_probe_rtts(df, probe_ip)
        print(f"  RTTs to {len(probe_rtts)} anchors")

        # Run CBG
        result = run_cbg_for_probe(
            probe_ip=probe_ip,
            probe_lat=probe_lat,
            probe_lon=probe_lon,
            probe_rtts=probe_rtts,
            models=models
        )

        if result.error_km is not None:
            print(f"  Estimated location: ({result.estimated_lat:.2f}, {result.estimated_lon:.2f})")
            print(f"  True location: ({result.true_lat:.2f}, {result.true_lon:.2f})")
            print(f"  Error: {result.error_km:.1f} km")
        else:
            print(f"  Could not estimate location (insufficient anchors)")

        results.append(result)

        # Create map
        map_path = output_dir / f"cbg_test_{probe_ip.replace('.', '_')}.html"
        create_cbg_map(result, models, map_path)

    return results


def print_summary(results: List[CBGResult]) -> None:
    """Print summary of CBG results."""
    print("\n" + "=" * 60)
    print("CBG ANALYSIS SUMMARY")
    print("=" * 60)

    valid_results = [r for r in results if r.error_km is not None]

    print(f"Total probes tested: {len(results)}")
    print(f"Successfully geolocated: {len(valid_results)}")

    if valid_results:
        errors = [r.error_km for r in valid_results]
        print(f"\nError Statistics (km):")
        print(f"  Mean: {np.mean(errors):.1f}")
        print(f"  Median: {np.median(errors):.1f}")
        print(f"  Min: {np.min(errors):.1f}")
        print(f"  Max: {np.max(errors):.1f}")
        print(f"  Std: {np.std(errors):.1f}")

        # Accuracy at thresholds
        thresholds = [50, 100, 250, 500, 1000]
        print(f"\nAccuracy at distance thresholds:")
        for thresh in thresholds:
            pct = 100.0 * sum(1 for e in errors if e <= thresh) / len(errors)
            print(f"  ≤{thresh:4d} km: {pct:5.1f}%")

        print(f"\nPer-probe results:")
        for r in valid_results:
            print(f"  {r.target_ip}: {r.error_km:.1f} km ({r.n_anchors_used} anchors)")


def main():
    parser = argparse.ArgumentParser(description='CBG Multilateration Visualization')
    parser.add_argument('--data', type=str, default='data/vultr_pings_us_only.csv',
                        help='Path to input data CSV')
    parser.add_argument('--models', type=str, default=None,
                        help='Path to model directory (default: outputs/vultr-{ASN}-rtt-models)')
    parser.add_argument('--asn', type=int, default=7922,
                        help='Target ASN (default: 7922 Comcast)')
    parser.add_argument('--n-samples', type=int, default=5,
                        help='Number of probes to test (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: same as models)')

    args = parser.parse_args()

    # Set paths
    script_dir = Path(__file__).parent
    data_path = script_dir / args.data

    if args.models:
        model_dir = Path(args.models)
    else:
        model_dir = script_dir / f"outputs/vultr-{args.asn}-rtt-models"

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = model_dir

    print("=" * 60)
    print("CBG MULTILATERATION VISUALIZATION")
    print("=" * 60)
    print(f"Data: {data_path}")
    print(f"Models: {model_dir}")
    print(f"Target ASN: {args.asn}")
    print(f"Samples: {args.n_samples}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Load models
    models = load_models(model_dir)

    if not models:
        print("ERROR: No fitted models found. Run fit_models.py first.")
        return

    # Load data
    df = load_data(data_path, args.asn)

    if len(df) == 0:
        print(f"ERROR: No data for ASN {args.asn}")
        return

    # Run CBG analysis
    results = run_cbg_analysis(
        df=df,
        models=models,
        output_dir=output_dir,
        n_samples=args.n_samples,
        random_seed=args.seed
    )

    # Print summary
    print_summary(results)

    # Save results summary
    summary_path = output_dir / "cbg_results.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'asn': args.asn,
            'n_probes': len(results),
            'results': [
                {
                    'target_ip': r.target_ip,
                    'true_lat': r.true_lat,
                    'true_lon': r.true_lon,
                    'estimated_lat': r.estimated_lat,
                    'estimated_lon': r.estimated_lon,
                    'error_km': r.error_km,
                    'n_anchors_used': r.n_anchors_used,
                    'anchor_distances': r.anchor_distances,
                    'anchor_rtts': r.anchor_rtts
                }
                for r in results
            ]
        }, f, indent=2)
    print(f"\nSaved results: {summary_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
