"""
Test to verify image orientation in Folium ImageOverlay
"""

import numpy as np
from PIL import Image
import folium
from io import BytesIO
import base64

def create_test_map():
    """Create a test map to verify image orientation"""

    print("="*70)
    print("TESTING FOLIUM IMAGE OVERLAY ORIENTATION")
    print("="*70)

    # Create a test image with clear orientation
    # We'll make a gradient that goes from WHITE (top) to BLACK (bottom)
    # And RED on left, BLUE on right

    height, width = 100, 100
    test_array = np.zeros((height, width, 4), dtype=np.uint8)

    # Create gradient: top (row 0) = bright, bottom (row 99) = dark
    for i in range(height):
        brightness = int(255 * (1 - i / height))  # Row 0 = 255, Row 99 = 0
        for j in range(width):
            if j < width // 2:
                # Left half: RED gradient
                test_array[i, j] = [brightness, 0, 0, 255]
            else:
                # Right half: BLUE gradient
                test_array[i, j] = [0, 0, brightness, 255]

    # Add text markers at corners (using pixels)
    # Top-left corner (should be NW)
    test_array[0:10, 0:10] = [0, 255, 0, 255]  # GREEN

    # Top-right corner (should be NE)
    test_array[0:10, -10:] = [255, 255, 0, 255]  # YELLOW

    # Bottom-left corner (should be SW)
    test_array[-10:, 0:10] = [255, 0, 255, 255]  # MAGENTA

    # Bottom-right corner (should be SE)
    test_array[-10:, -10:] = [0, 255, 255, 255]  # CYAN

    print("\nTest image created:")
    print("  - Row 0 (top): BRIGHT (white)")
    print("  - Row 99 (bottom): DARK (black)")
    print("  - Left half: RED")
    print("  - Right half: BLUE")
    print("  - Corners marked with colors:")
    print("    * Top-left (row 0, col 0): GREEN (should be NW)")
    print("    * Top-right (row 0, col 99): YELLOW (should be NE)")
    print("    * Bottom-left (row 99, col 0): MAGENTA (should be SW)")
    print("    * Bottom-right (row 99, col 99): CYAN (should be SE)")

    # Convert to PIL Image
    img = Image.fromarray(test_array)

    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    image_data = f"data:image/png;base64,{img_str}"

    # Define test bounds (somewhere in France)
    # These are WGS84 coordinates
    south, north = 46.0, 46.1  # Latitude
    west, east = 6.0, 6.1  # Longitude

    bounds = [[south, west], [north, east]]

    print(f"\nBounds: [[{south}, {west}], [{north}, {east}]]")
    print(f"  South={south}° (bottom edge)")
    print(f"  North={north}° (top edge)")
    print(f"  West={west}° (left edge)")
    print(f"  East={east}° (right edge)")

    # Create map
    center_lat = (south + north) / 2
    center_lon = (west + east) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # Add test overlay
    folium.raster_layers.ImageOverlay(
        image=image_data,
        bounds=bounds,
        opacity=0.8,
        name='Test Orientation',
        overlay=True,
        control=True
    ).add_to(m)

    # Add corner markers to verify
    folium.Marker(
        location=[north, west],
        popup="NW Corner - should see GREEN",
        tooltip="NW (green)",
        icon=folium.Icon(color='green')
    ).add_to(m)

    folium.Marker(
        location=[north, east],
        popup="NE Corner - should see YELLOW",
        tooltip="NE (yellow)",
        icon=folium.Icon(color='orange')
    ).add_to(m)

    folium.Marker(
        location=[south, west],
        popup="SW Corner - should see MAGENTA",
        tooltip="SW (magenta)",
        icon=folium.Icon(color='purple')
    ).add_to(m)

    folium.Marker(
        location=[south, east],
        popup="SE Corner - should see CYAN",
        tooltip="SE (cyan)",
        icon=folium.Icon(color='lightblue')
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # Save
    output_file = "test_orientation.html"
    m.save(output_file)

    print(f"\n{'='*70}")
    print(f"Test map saved to: {output_file}")
    print(f"{'='*70}")
    print("\nHow to verify:")
    print("1. Open test_orientation.html in browser")
    print("2. Check if corner colors match the markers:")
    print("   - NW (top-left): should be GREEN")
    print("   - NE (top-right): should be YELLOW")
    print("   - SW (bottom-left): should be MAGENTA")
    print("   - SE (bottom-right): should be CYAN")
    print("3. Check gradient:")
    print("   - North side should be BRIGHT")
    print("   - South side should be DARK")
    print("   - West side should be RED")
    print("   - East side should be BLUE")
    print("\nIf colors DON'T match, there's an orientation problem!")
    print("="*70)

if __name__ == "__main__":
    create_test_map()
