"""
Parse daylight_map.html to analyze marker and tile positions
Shows where markers appear relative to their tile bounds
"""

import re
import json
from collections import defaultdict

def parse_html():
    """Parse the HTML file to extract markers and tile bounds"""

    html_file = "daylight_map.html"

    print("="*70)
    print("PARSING daylight_map.html")
    print("="*70)

    try:
        with open(html_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"\nERROR: {html_file} not found")
        print("Run: python view.py")
        return

    # Extract markers (they have Lambert93 coordinates in popups)
    print("\nExtracting markers...")
    marker_pattern = r'<b>Daylight:</b>\s*([\d.]+)h<br>\s*<b>Position:</b>\s*Row\s*(\d+),\s*Col\s*(\d+)<br>\s*<b>Lambert93:</b>\s*([\d.]+),\s*([\d.]+)<br>\s*<b>WGS84:</b>\s*([\d.]+)°N,\s*([\d.]+)°E<br>\s*<b>Tile:</b>\s*([^<]+)'

    markers = []
    for match in re.finditer(marker_pattern, content):
        daylight, row, col, lambert_x, lambert_y, lat, lon, tile = match.groups()
        markers.append({
            'daylight': float(daylight),
            'row': int(row),
            'col': int(col),
            'lambert_x': float(lambert_x),
            'lambert_y': float(lambert_y),
            'lat': float(lat),
            'lon': float(lon),
            'tile': tile.strip()
        })

    print(f"Found {len(markers)} markers")

    # Extract image overlays with their bounds
    print("\nExtracting tile bounds...")

    # Look for ImageOverlay with bounds
    # Pattern: bounds: [[south, west], [north, east]]
    bounds_pattern = r'bounds:\s*\[\[\s*([\d.]+),\s*([\d.]+)\s*\],\s*\[\s*([\d.]+),\s*([\d.]+)\s*\]\]'

    tile_bounds = []
    for match in re.finditer(bounds_pattern, content):
        south, west, north, east = match.groups()
        tile_bounds.append({
            'south': float(south),
            'west': float(west),
            'north': float(north),
            'east': float(east)
        })

    print(f"Found {len(tile_bounds)} tile bounds")

    # Analyze marker positions relative to tile bounds
    print("\n" + "="*70)
    print("ANALYZING MARKER POSITIONS")
    print("="*70)

    # Group markers by tile
    markers_by_tile = defaultdict(list)
    for marker in markers:
        markers_by_tile[marker['tile']].append(marker)

    # Test first few tiles
    issues_found = 0
    tiles_tested = 0

    for tile_name in sorted(markers_by_tile.keys())[:10]:
        tile_markers = markers_by_tile[tile_name]

        print(f"\nTile: {tile_name}")
        print(f"  Markers: {len(tile_markers)}")

        # Find the bounds that should contain these markers
        # We need to find which tile boundary contains these marker positions

        # Get marker extent
        marker_lats = [m['lat'] for m in tile_markers]
        marker_lons = [m['lon'] for m in tile_markers]
        marker_south = min(marker_lats)
        marker_north = max(marker_lats)
        marker_west = min(marker_lons)
        marker_east = max(marker_lons)

        print(f"  Marker extent: lat [{marker_south:.6f}, {marker_north:.6f}], lon [{marker_west:.6f}, {marker_east:.6f}]")

        # Find matching tile bounds
        matching_bounds = None
        for bounds in tile_bounds:
            # Check if markers fall roughly within this tile
            lat_overlap = not (marker_north < bounds['south'] or marker_south > bounds['north'])
            lon_overlap = not (marker_east < bounds['west'] or marker_west > bounds['east'])

            if lat_overlap and lon_overlap:
                matching_bounds = bounds
                break

        if matching_bounds:
            print(f"  Tile bounds: lat [{matching_bounds['south']:.6f}, {matching_bounds['north']:.6f}], lon [{matching_bounds['west']:.6f}, {matching_bounds['east']:.6f}]")

            # Check if all markers are within bounds
            markers_outside = 0
            for marker in tile_markers:
                if not (matching_bounds['south'] <= marker['lat'] <= matching_bounds['north'] and
                        matching_bounds['west'] <= marker['lon'] <= matching_bounds['east']):
                    markers_outside += 1
                    if markers_outside <= 3:  # Show first 3
                        print(f"    ✗ Marker OUTSIDE: Row {marker['row']}, Col {marker['col']} at ({marker['lon']:.6f}, {marker['lat']:.6f})")

            if markers_outside > 0:
                print(f"  ⚠ {markers_outside}/{len(tile_markers)} markers OUTSIDE bounds!")
                issues_found += 1
            else:
                print(f"  ✓ All markers within bounds")
        else:
            print(f"  ⚠ No matching tile bounds found")

        tiles_tested += 1

    # Overall summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total markers: {len(markers)}")
    print(f"Total tile bounds: {len(tile_bounds)}")
    print(f"Tiles tested: {tiles_tested}")
    print(f"Tiles with issues: {issues_found}")

    if issues_found > 0:
        print(f"\n⚠ PROBLEM: {issues_found} tiles have markers outside their bounds!")
        print("This explains why markers appear mispositioned.")
    else:
        print(f"\n✓ All tested tiles have markers within bounds")

    # Show a specific example
    print("\n" + "="*70)
    print("DETAILED EXAMPLE: First marker")
    print("="*70)

    if markers:
        m = markers[0]
        print(f"Marker:")
        print(f"  Tile: {m['tile']}")
        print(f"  Position: Row {m['row']}, Col {m['col']}")
        print(f"  Lambert93: ({m['lambert_x']}, {m['lambert_y']})")
        print(f"  WGS84: ({m['lon']:.6f}°E, {m['lat']:.6f}°N)")
        print(f"  Daylight: {m['daylight']}h")

        # Try to find its tile
        for bounds in tile_bounds:
            if bounds['south'] <= m['lat'] <= bounds['north'] and bounds['west'] <= m['lon'] <= bounds['east']:
                print(f"\nFound in tile bounds:")
                print(f"  Lat: [{bounds['south']:.6f}, {bounds['north']:.6f}]")
                print(f"  Lon: [{bounds['west']:.6f}, {bounds['east']:.6f}]")

                # Calculate position within tile (0=south/west, 1=north/east)
                lat_pos = (m['lat'] - bounds['south']) / (bounds['north'] - bounds['south'])
                lon_pos = (m['lon'] - bounds['west']) / (bounds['east'] - bounds['west'])

                print(f"\nPosition within tile:")
                print(f"  Lat: {lat_pos*100:.1f}% from south (0%) to north (100%)")
                print(f"  Lon: {lon_pos*100:.1f}% from west (0%) to east (100%)")

                if m['row'] == 0:
                    print(f"\n  Row 0 should be at NORTH (100%), got {lat_pos*100:.1f}%")
                    if lat_pos > 0.99:
                        print(f"  ✓ Correct!")
                    else:
                        print(f"  ✗ Wrong! Should be near 100%")

                break
        else:
            print(f"\n✗ Marker is NOT within any tile bounds!")

    print("\n" + "="*70)

if __name__ == "__main__":
    parse_html()
