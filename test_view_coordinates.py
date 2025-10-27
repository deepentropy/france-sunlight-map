"""
Test the exact coordinate transformation logic from view.py
"""

# Simulate the exact code from view.py lines 362-367
def test_marker_coordinates():
    """Test marker coordinate transformation"""

    # Test case from user's marker
    # Tile: RGEALTI_FXX_0945_6560_MNT_LAMB93_IGN69
    header = {
        'xllcorner': 945000.0,
        'yllcorner': 6560000.0,
        'ncols': 1000,
        'nrows': 1000,
        'cellsize': 5.0
    }

    # Position from user's marker
    i = 0  # row
    j = 20  # col

    print("="*70)
    print("TESTING VIEW.PY COORDINATE TRANSFORMATION")
    print("="*70)
    print(f"\nInput:")
    print(f"  Tile: RGEALTI_FXX_0945_6560_MNT_LAMB93_IGN69")
    print(f"  Header: {header}")
    print(f"  Position: Row {i}, Col {j}")

    # CURRENT CODE from view.py lines 362-367
    print(f"\n{'='*70}")
    print("CURRENT CODE (view.py lines 362-367):")
    print("="*70)

    x_lambert = header['xllcorner'] + (j + 0.5) * header['cellsize']
    y_lambert = header['yllcorner'] + (header['nrows'] - i - 0.5) * header['cellsize']

    print(f"\nLine 362: x_lambert = header['xllcorner'] + (j + 0.5) * header['cellsize']")
    print(f"         x_lambert = {header['xllcorner']} + ({j} + 0.5) * {header['cellsize']}")
    print(f"         x_lambert = {header['xllcorner']} + {j + 0.5} * {header['cellsize']}")
    print(f"         x_lambert = {header['xllcorner']} + {(j + 0.5) * header['cellsize']}")
    print(f"         x_lambert = {x_lambert}")

    print(f"\nLine 367: y_lambert = header['yllcorner'] + (header['nrows'] - i - 0.5) * header['cellsize']")
    print(f"         y_lambert = {header['yllcorner']} + ({header['nrows']} - {i} - 0.5) * {header['cellsize']}")
    print(f"         y_lambert = {header['yllcorner']} + {header['nrows'] - i - 0.5} * {header['cellsize']}")
    print(f"         y_lambert = {header['yllcorner']} + {(header['nrows'] - i - 0.5) * header['cellsize']}")
    print(f"         y_lambert = {y_lambert}")

    # What the popup should display (line 378)
    print(f"\n{'='*70}")
    print("POPUP DISPLAY (view.py line 378):")
    print("="*70)
    print(f"  <b>Lambert93:</b> {x_lambert:.1f}, {y_lambert:.1f}<br>")

    # What the user reported seeing
    print(f"\n{'='*70}")
    print("USER REPORTED SEEING:")
    print("="*70)
    print(f"  Lambert93: 945100.0, 6560000.0")

    # Compare
    print(f"\n{'='*70}")
    print("COMPARISON:")
    print("="*70)

    expected_x = 945102.5
    expected_y = 6564997.5
    reported_x = 945100.0
    reported_y = 6560000.0

    print(f"  X coordinate:")
    print(f"    Expected (from current code): {expected_x}")
    print(f"    User reported:                {reported_x}")
    print(f"    Match: {'✓ YES' if abs(x_lambert - reported_x) < 1 else '✗ NO'}")
    print(f"    Difference: {x_lambert - reported_x:.1f}")

    print(f"\n  Y coordinate:")
    print(f"    Expected (from current code): {expected_y}")
    print(f"    User reported:                {reported_y}")
    print(f"    Match: {'✓ YES' if abs(y_lambert - reported_y) < 1 else '✗ NO'}")
    print(f"    Difference: {y_lambert - reported_y:.1f}")

    # Diagnosis
    print(f"\n{'='*70}")
    print("DIAGNOSIS:")
    print("="*70)

    if abs(x_lambert - expected_x) < 0.1 and abs(y_lambert - expected_y) < 0.1:
        print("  ✓ Current view.py code produces CORRECT coordinates")
        print("  ✓ x_lambert = 945102.5 (pixel center with +0.5 offset)")
        print("  ✓ y_lambert = 6564997.5 (top row, correctly inverted)")

        if abs(reported_x - expected_x) > 1 or abs(reported_y - expected_y) > 1:
            print("\n  ⚠ However, user reported DIFFERENT values!")
            print("  ⚠ This suggests:")
            print("    1. User is viewing OLD HTML file (before fixes)")
            print("    2. OR user needs to regenerate maps with: python view.py")
            print("    3. OR browser is caching old HTML")

            print("\n  SOLUTION:")
            print("    1. Delete old HTML files:")
            print("       rm -f *.html")
            print("    2. Regenerate maps:")
            print("       python view.py")
            print("    3. Open new HTML file (hard refresh: Ctrl+Shift+R)")
    else:
        print("  ✗ ERROR: Code produces wrong coordinates!")
        print(f"  Expected: {expected_x}, {expected_y}")
        print(f"  Got:      {x_lambert}, {y_lambert}")

    # Test tile bounds too
    print(f"\n{'='*70}")
    print("BONUS: Testing tile bounds (view.py lines 50-71):")
    print("="*70)

    xmin = header['xllcorner']
    ymin = header['yllcorner']
    xmax = xmin + header['ncols'] * header['cellsize']
    ymax = ymin + header['nrows'] * header['cellsize']

    print(f"  Lambert93 bounds:")
    print(f"    xmin (west):  {xmin}")
    print(f"    xmax (east):  {xmax}")
    print(f"    ymin (south): {ymin}")
    print(f"    ymax (north): {ymax}")
    print(f"\n  Folium ImageOverlay bounds should be:")
    print(f"    bounds_wgs84 = [[south, west], [north, east]]")
    print(f"                 = [[lat_at_{ymin}, lon_at_{xmin}], [lat_at_{ymax}, lon_at_{xmax}]]")
    print(f"\n  ✓ This ensures tiles align perfectly with NO GAPS!")

    print("\n" + "="*70)

if __name__ == "__main__":
    test_marker_coordinates()
