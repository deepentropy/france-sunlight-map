"""
Test coordinate transformations for debugging
"""

# Test case from user's marker:
# Tile: RGEALTI_FXX_0945_6560_MNT_LAMB93_IGN69
# Position: Row 0, Col 20
# Expected: Should be at UPPER-LEFT area (row 0 = top)

# ASC header values (inferred from tile name):
xllcorner = 945000.0  # 0945 * 1000
yllcorner = 6560000.0  # 6560 * 1000
ncols = 1000
nrows = 1000
cellsize = 5.0

# Test position
row = 0  # Top row
col = 20

print("="*60)
print("COORDINATE TRANSFORMATION TEST")
print("="*60)
print(f"\nTile: RGEALTI_FXX_0945_6560_MNT_LAMB93_IGN69")
print(f"xllcorner: {xllcorner}")
print(f"yllcorner: {yllcorner}")
print(f"nrows: {nrows}, ncols: {ncols}, cellsize: {cellsize}")

print(f"\nTest position: Row {row}, Col {col}")
print(f"  (Row 0 = TOP of raster image)")

# Calculate X (simple - columns go left to right)
x_edge = xllcorner + col * cellsize
x_center = xllcorner + (col + 0.5) * cellsize

print(f"\nX coordinate (column {col}):")
print(f"  Cell left edge:   {x_edge}")
print(f"  Cell center:      {x_center}")

# Calculate Y (complex - rows go top to bottom, but Y increases upward!)
# In ASC format:
# - First data line (row 0) = NORTH edge (highest Y)
# - Last data line (row nrows-1) = SOUTH edge (lowest Y = yllcorner)

ymax = yllcorner + nrows * cellsize  # Top of grid
ymin = yllcorner                     # Bottom of grid

print(f"\nY extents:")
print(f"  ymin (bottom): {ymin}")
print(f"  ymax (top):    {ymax}")

# For row 0 (top row):
y_top_edge = ymax  # Top edge of top row
y_center = ymax - (row + 0.5) * cellsize  # Center of row

print(f"\nY coordinate (row {row} = TOP row):")
print(f"  Top edge of grid: {ymax}")
print(f"  Cell top edge:    {y_top_edge}")
print(f"  Cell center:      {y_center}")

# What the user reported:
print(f"\nUser reported in marker popup:")
print(f"  Lambert93: 945100.0, 6560000.0")
print(f"\nExpected cell center:")
print(f"  Lambert93: {x_center}, {y_center}")

print(f"\nDISCREPANCY:")
print(f"  X: reported {945100.0} vs expected {x_center} (diff: {945100.0 - x_center})")
print(f"  Y: reported {6560000.0} vs expected {y_center} (diff: {6560000.0 - y_center})")

print(f"\nCONCLUSION:")
if abs(945100.0 - x_edge) < 0.1:
    print(f"  X is using EDGE not CENTER (missing +0.5)")
if abs(6560000.0 - ymin) < 0.1:
    print(f"  Y is using BOTTOM LEFT CORNER - WRONG!")
    print(f"  Row 0 should be at TOP ({y_center}) not BOTTOM ({ymin})")

print("="*60)
