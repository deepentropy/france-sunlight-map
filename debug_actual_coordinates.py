"""
Debug script to check what coordinates view.py actually generates
Uses real NPY and ASC files from the project
"""

import numpy as np
import os
import glob

def read_asc_header(asc_path):
    """Read ASC header (same as view.py)"""
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
                    elif key == 'nodata_value':
                        header[key] = float(value)
        return header
    except Exception as e:
        print(f"Error reading {asc_path}: {e}")
        return None

def find_local_maxima(data, threshold=14, window_size=20):
    """Find local maxima (same as view.py)"""
    mask = data >= threshold

    if not np.any(mask):
        return []

    try:
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(data, size=window_size)
        maxima = (data == local_max) & mask

        coords = np.column_stack(np.where(maxima))
        values = [data[i, j] for i, j in coords]
        sorted_indices = np.argsort(values)[::-1]

        return coords[sorted_indices].tolist()
    except ImportError:
        print("Warning: scipy not available, using simple max")
        coords = np.where(mask)
        return [(coords[0][0], coords[1][0])]

def main():
    print("="*70)
    print("DEBUGGING ACTUAL COORDINATE GENERATION")
    print("="*70)

    # Paths
    RESULTS_DIR = "daylight_results"
    ASC_DIR = "RGEALTI/1_DONNEES_LIVRAISON_2021-10-00009/RGEALTI_MNT_5M_ASC_LAMB93_IGN69_D074"

    # Find NPY files
    npy_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_daylight.npy")))

    if not npy_files:
        print(f"\nERROR: No NPY files found in {RESULTS_DIR}")
        print("Cannot debug without actual data files")
        return

    print(f"\nFound {len(npy_files)} NPY files")

    # Test with the specific tile from user's example
    target_tile = "RGEALTI_FXX_0945_6560_MNT_LAMB93_IGN69"
    target_npy = None

    for npy_file in npy_files:
        if "0945_6560" in npy_file:
            target_npy = npy_file
            break

    if not target_npy:
        print(f"\nWarning: Target tile {target_tile} not found, using first tile")
        target_npy = npy_files[0]

    basename = os.path.basename(target_npy).replace('_daylight.npy', '')
    print(f"\nTesting with tile: {basename}")
    print(f"NPY file: {target_npy}")

    # Find corresponding ASC file
    asc_pattern = os.path.join(ASC_DIR, f"{basename}.asc")
    asc_files = glob.glob(asc_pattern)

    if not asc_files:
        print(f"\nERROR: No ASC file found matching {asc_pattern}")
        print(f"ASC directory exists: {os.path.exists(ASC_DIR)}")
        if os.path.exists(ASC_DIR):
            all_asc = glob.glob(os.path.join(ASC_DIR, "*.asc"))
            print(f"Available ASC files: {len(all_asc)}")
            if all_asc:
                print(f"Example: {os.path.basename(all_asc[0])}")
        return

    asc_file = asc_files[0]
    print(f"ASC file: {asc_file}")

    # Read header
    header = read_asc_header(asc_file)
    if not header:
        print("\nERROR: Could not read ASC header")
        return

    print(f"\nHeader:")
    for key, value in header.items():
        print(f"  {key}: {value}")

    # Load NPY data
    print(f"\nLoading NPY data...")
    daylight = np.load(target_npy)
    print(f"Data shape: {daylight.shape}")
    print(f"Data range: {daylight.min():.2f} to {daylight.max():.2f}")

    # Find maxima
    print(f"\nFinding local maxima (threshold=14h, window=20)...")
    maxima = find_local_maxima(daylight, threshold=14, window_size=20)
    print(f"Found {len(maxima)} maxima")

    # Test coordinates for first 5 maxima
    print("\n" + "="*70)
    print("COORDINATE CALCULATIONS (using view.py formulas)")
    print("="*70)

    for idx, (i, j) in enumerate(maxima[:5]):
        print(f"\n--- Marker {idx + 1} ---")
        print(f"Array indices: Row {i}, Col {j}")
        print(f"Daylight: {daylight[i, j]:.2f}h")

        # EXACT code from view.py lines 362-367
        x_lambert = header['xllcorner'] + (j + 0.5) * header['cellsize']
        y_lambert = header['yllcorner'] + (header['nrows'] - i - 0.5) * header['cellsize']

        print(f"\nCoordinate calculation:")
        print(f"  x_lambert = {header['xllcorner']} + ({j} + 0.5) * {header['cellsize']}")
        print(f"            = {x_lambert}")
        print(f"  y_lambert = {header['yllcorner']} + ({header['nrows']} - {i} - 0.5) * {header['cellsize']}")
        print(f"            = {y_lambert}")

        print(f"\nResult:")
        print(f"  Lambert93: {x_lambert:.1f}, {y_lambert:.1f}")

        # Special check for Row 0, Col 20 (user's example)
        if i == 0 and j == 20:
            print("\n*** THIS IS THE USER'S EXAMPLE MARKER ***")
            print(f"*** Expected: 945102.5, 6564997.5 ***")
            print(f"*** Got:      {x_lambert:.1f}, {y_lambert:.1f} ***")
            if abs(x_lambert - 945102.5) < 0.1 and abs(y_lambert - 6564997.5) < 0.1:
                print("*** ✓ CORRECT ***")
            else:
                print("*** ✗ WRONG ***")

    # Now check what's actually in the HTML
    print("\n" + "="*70)
    print("CHECKING ACTUAL HTML OUTPUT")
    print("="*70)

    html_file = "daylight_map.html"
    if os.path.exists(html_file):
        print(f"\n{html_file} exists")

        # Search for Lambert93 coordinates in HTML
        with open(html_file, 'r') as f:
            content = f.read()

        # Look for the popup format
        import re

        # Current format: <b>Lambert93:</b> X.X, Y.Y<br>
        lambert_pattern = r'<b>Lambert93:</b>\s*([\d.]+),\s*([\d.]+)<br>'
        matches = re.findall(lambert_pattern, content)

        if matches:
            print(f"\nFound {len(matches)} Lambert93 coordinates in HTML")
            print("\nFirst 5 coordinates in HTML:")
            for idx, (x, y) in enumerate(matches[:5]):
                print(f"  {idx + 1}. Lambert93: {x}, {y}")

            # Check if user's example is in there
            user_coord = ('945100.0', '6560000.0')
            correct_coord = ('945102.5', '6564997.5')

            if user_coord in matches:
                print(f"\n⚠ OLD coordinate 945100.0, 6560000.0 found in HTML!")
            if correct_coord in matches:
                print(f"\n✓ CORRECT coordinate 945102.5, 6564997.5 found in HTML!")
        else:
            # Try old format: just WGS84 coordinates
            wgs_pattern = r'<b>Coordinates:</b>.*?([\d.]+)°N,\s*([\d.]+)°E'
            wgs_matches = re.findall(wgs_pattern, content)
            if wgs_matches:
                print(f"\n⚠ HTML uses OLD format (no Lambert93 coordinates)")
                print(f"Found {len(wgs_matches)} WGS84 coordinates instead")
                print("This suggests view.py code is NOT the current version!")
            else:
                print("\n❌ No coordinates found in HTML!")
    else:
        print(f"\n❌ {html_file} does not exist")
        print("Run: python view.py")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
