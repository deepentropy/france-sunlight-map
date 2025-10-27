"""
Test if tiles are properly aligned (no gaps)
Checks actual ASC headers and transformation
"""

import glob
import os
from pyproj import Transformer

def read_asc_header(asc_path):
    """Read ASC header"""
    header = {}
    try:
        with open(asc_path, 'r') as f:
            for i in range(6):
                line = f.readline().strip().split()
                if len(line) >= 2:
                    key = line[0].lower()
                    value = line[1]
                    if key in ['ncols', 'nrows']:
                        header[key] = int(value)
                    elif key in ['xllcorner', 'yllcorner', 'cellsize']:
                        header[key] = float(value)
        return header
    except Exception as e:
        print(f"Error reading {asc_path}: {e}")
        return None

def main():
    print("="*70)
    print("TESTING TILE ALIGNMENT")
    print("="*70)

    ASC_DIR = "RGEALTI/1_DONNEES_LIVRAISON_2021-10-00009/RGEALTI_MNT_5M_ASC_LAMB93_IGN69_D074"

    if not os.path.exists(ASC_DIR):
        print(f"\nERROR: ASC directory not found: {ASC_DIR}")
        return

    asc_files = sorted(glob.glob(os.path.join(ASC_DIR, "*.asc")))
    print(f"\nFound {len(asc_files)} ASC files")

    if len(asc_files) == 0:
        print("No ASC files to analyze")
        return

    # Read headers for first 10 tiles
    print("\n" + "="*70)
    print("TILE EXTENTS (Lambert93)")
    print("="*70)

    tiles = []
    for asc_file in asc_files[:10]:
        basename = os.path.basename(asc_file).replace('.asc', '')
        header = read_asc_header(asc_file)

        if header:
            xmin = header['xllcorner']
            ymin = header['yllcorner']
            xmax = xmin + header['ncols'] * header['cellsize']
            ymax = ymin + header['nrows'] * header['cellsize']

            tiles.append({
                'name': basename,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'header': header
            })

            print(f"\n{basename}")
            print(f"  X: {xmin} to {xmax} (width: {xmax-xmin}m)")
            print(f"  Y: {ymin} to {ymax} (height: {ymax-ymin}m)")

    # Check for gaps/overlaps
    print("\n" + "="*70)
    print("CHECKING FOR GAPS/OVERLAPS")
    print("="*70)

    found_adjacent = False
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            t1 = tiles[i]
            t2 = tiles[j]

            # Check if tiles are horizontally adjacent
            if abs(t1['ymin'] - t2['ymin']) < 1 and abs(t1['ymax'] - t2['ymax']) < 1:
                # Same Y range, check X
                if abs(t1['xmax'] - t2['xmin']) < 10:  # t1 left of t2
                    gap = t2['xmin'] - t1['xmax']
                    print(f"\nHorizontal neighbors:")
                    print(f"  {t1['name'][-20:]} | {t2['name'][-20:]}")
                    print(f"  X gap: {gap:.3f}m {'✓ ALIGNED' if abs(gap) < 0.1 else '✗ GAP/OVERLAP'}")
                    found_adjacent = True
                elif abs(t2['xmax'] - t1['xmin']) < 10:  # t2 left of t1
                    gap = t1['xmin'] - t2['xmax']
                    print(f"\nHorizontal neighbors:")
                    print(f"  {t2['name'][-20:]} | {t1['name'][-20:]}")
                    print(f"  X gap: {gap:.3f}m {'✓ ALIGNED' if abs(gap) < 0.1 else '✗ GAP/OVERLAP'}")
                    found_adjacent = True

            # Check if tiles are vertically adjacent
            if abs(t1['xmin'] - t2['xmin']) < 1 and abs(t1['xmax'] - t2['xmax']) < 1:
                # Same X range, check Y
                if abs(t1['ymax'] - t2['ymin']) < 10:  # t1 below t2
                    gap = t2['ymin'] - t1['ymax']
                    print(f"\nVertical neighbors:")
                    print(f"  {t2['name'][-20:]} (above)")
                    print(f"  {t1['name'][-20:]} (below)")
                    print(f"  Y gap: {gap:.3f}m {'✓ ALIGNED' if abs(gap) < 0.1 else '✗ GAP/OVERLAP'}")
                    found_adjacent = True
                elif abs(t2['ymax'] - t1['ymin']) < 10:  # t2 below t1
                    gap = t1['ymin'] - t2['ymax']
                    print(f"\nVertical neighbors:")
                    print(f"  {t1['name'][-20:]} (above)")
                    print(f"  {t2['name'][-20:]} (below)")
                    print(f"  Y gap: {gap:.3f}m {'✓ ALIGNED' if abs(gap) < 0.1 else '✗ GAP/OVERLAP'}")
                    found_adjacent = True

    if not found_adjacent:
        print("\nNo adjacent tiles found in first 10 files")

    # Test WGS84 transformation
    print("\n" + "="*70)
    print("WGS84 TRANSFORMATION TEST")
    print("="*70)

    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    # Test with first tile
    if tiles:
        t = tiles[0]
        header = t['header']
        cellsize = header['cellsize']

        print(f"\nTile: {t['name']}")
        print(f"\nLambert93 EDGES (old, incorrect):")
        print(f"  SW: ({t['xmin']}, {t['ymin']})")
        print(f"  NE: ({t['xmax']}, {t['ymax']})")

        lon_sw_edge, lat_sw_edge = transformer.transform(t['xmin'], t['ymin'])
        lon_ne_edge, lat_ne_edge = transformer.transform(t['xmax'], t['ymax'])

        print(f"\nWGS84 from edges (old, incorrect):")
        print(f"  SW: ({lon_sw_edge:.6f}, {lat_sw_edge:.6f})")
        print(f"  NE: ({lon_ne_edge:.6f}, {lat_ne_edge:.6f})")

        south_edge = min(lat_sw_edge, lat_ne_edge)
        north_edge = max(lat_sw_edge, lat_ne_edge)
        west_edge = min(lon_sw_edge, lon_ne_edge)
        east_edge = max(lon_sw_edge, lon_ne_edge)

        print(f"\nOLD bounds (incorrect): [[{south_edge:.6f}, {west_edge:.6f}], [{north_edge:.6f}, {east_edge:.6f}]]")

        # Now calculate using pixel CENTERS (correct)
        print(f"\n" + "-"*70)
        print("CORRECTED: Using pixel CENTERS")
        print("-"*70)

        xmin_center = header['xllcorner'] + 0.5 * cellsize
        ymin_center = header['yllcorner'] + 0.5 * cellsize
        xmax_center = header['xllcorner'] + (header['ncols'] - 0.5) * cellsize
        ymax_center = header['yllcorner'] + (header['nrows'] - 0.5) * cellsize

        print(f"\nLambert93 pixel CENTERS (correct):")
        print(f"  SW: ({xmin_center}, {ymin_center})")
        print(f"  NE: ({xmax_center}, {ymax_center})")

        lon_sw, lat_sw = transformer.transform(xmin_center, ymin_center)
        lon_ne, lat_ne = transformer.transform(xmax_center, ymax_center)

        print(f"\nWGS84 from pixel centers (correct):")
        print(f"  SW: ({lon_sw:.6f}, {lat_sw:.6f})")
        print(f"  NE: ({lon_ne:.6f}, {lat_ne:.6f})")

        south = min(lat_sw, lat_ne)
        north = max(lat_sw, lat_ne)
        west = min(lon_sw, lon_ne)
        east = max(lon_sw, lon_ne)

        print(f"\nNEW bounds (correct): [[{south:.6f}, {west:.6f}], [{north:.6f}, {east:.6f}]]")

        # Verify transformation is monotonic
        if lat_ne > lat_sw:
            print(f"✓ Latitude increases north (correct)")
        else:
            print(f"✗ WARNING: Latitude decreases north!")

        if lon_ne > lon_sw:
            print(f"✓ Longitude increases east (correct)")
        else:
            print(f"✗ WARNING: Longitude decreases east!")

        # Test a specific point
        print(f"\n" + "="*70)
        print("TESTING SPECIFIC POINT (Row 0, Col 20)")
        print("="*70)

        i, j = 0, 20  # Row 0, Col 20

        x_lambert = header['xllcorner'] + (j + 0.5) * cellsize
        y_lambert = header['yllcorner'] + (header['nrows'] - i - 0.5) * cellsize

        print(f"Lambert93: ({x_lambert}, {y_lambert})")

        lon, lat = transformer.transform(x_lambert, y_lambert)
        print(f"WGS84: ({lon:.6f}, {lat:.6f})")

        # Check if point is within OLD bounds (should FAIL)
        print(f"\nChecking against OLD bounds (edges):")
        if south_edge <= lat <= north_edge and west_edge <= lon <= east_edge:
            print(f"✓ Point is within OLD bounds")
        else:
            print(f"✗ Point is OUTSIDE OLD bounds (expected!)")
            print(f"  lat {lat:.6f} should be in [{south_edge:.6f}, {north_edge:.6f}]")
            print(f"  lon {lon:.6f} should be in [{west_edge:.6f}, {east_edge:.6f}]")

        # Check if point is within NEW bounds (should PASS)
        print(f"\nChecking against NEW bounds (pixel centers):")
        if south <= lat <= north and west <= lon <= east:
            print(f"✓ Point is within NEW bounds (correct!)")
        else:
            print(f"✗ WARNING: Point is OUTSIDE NEW bounds!")
            print(f"  lat {lat:.6f} should be in [{south:.6f}, {north:.6f}]")
            print(f"  lon {lon:.6f} should be in [{west:.6f}, {east:.6f}]")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
