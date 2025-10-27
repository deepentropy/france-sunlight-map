"""
Quick check to see how bounds are stored in the HTML
"""

html_file = "daylight_map.html"

try:
    with open(html_file, 'r') as f:
        content = f.read()

    print("Searching for 'bounds' in HTML...")

    # Find all lines containing 'bounds'
    lines = content.split('\n')
    bounds_lines = [line for line in lines if 'bounds' in line.lower()]

    print(f"\nFound {len(bounds_lines)} lines containing 'bounds'")

    if bounds_lines:
        print("\nFirst 10 examples:")
        for i, line in enumerate(bounds_lines[:10]):
            print(f"\n{i+1}. {line.strip()[:200]}")

    # Also search for ImageOverlay
    print("\n" + "="*70)
    print("Searching for 'ImageOverlay' in HTML...")
    overlay_lines = [line for line in lines if 'ImageOverlay' in line or 'imageOverlay' in line]

    print(f"\nFound {len(overlay_lines)} lines containing 'ImageOverlay'")

    if overlay_lines:
        print("\nFirst 5 examples:")
        for i, line in enumerate(overlay_lines[:5]):
            print(f"\n{i+1}. {line.strip()[:300]}")

    # Search for latitude/longitude numbers that look like bounds
    print("\n" + "="*70)
    print("Searching for coordinate patterns...")

    import re
    # Look for patterns like [[number, number], [number, number]]
    coord_pattern = r'\[\s*\[\s*[\d.]+\s*,\s*[\d.]+\s*\]\s*,\s*\[\s*[\d.]+\s*,\s*[\d.]+\s*\]\s*\]'
    matches = re.findall(coord_pattern, content)

    print(f"\nFound {len(matches)} coordinate arrays")

    if matches:
        print("\nFirst 10 examples:")
        for i, match in enumerate(matches[:10]):
            print(f"{i+1}. {match}")

except FileNotFoundError:
    print(f"ERROR: {html_file} not found")
