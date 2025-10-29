# Neighbor-Aware ASC Processing

Process RGEALTI ASC files directly with proper edge effect handling using spatial indexing.

## Why This Approach?

### Problems with the NPY merge:
- ❌ 106GB merged file exceeds RAM
- ❌ Lost spatial relationships between tiles
- ❌ Edge effects: incorrect shadows at tile boundaries
- ❌ High disk space usage

### This solution:
- ✅ Processes ASC files directly (no merge needed)
- ✅ Loads only necessary tiles into memory
- ✅ Handles edge effects correctly with neighbor context
- ✅ Memory efficient (~1-2GB per tile with neighbors)
- ✅ Can resume if interrupted
- ✅ Parallelizable

## How It Works

```
1. Build Spatial Index (once)
   ↓
2. For each tile:
   - Load tile + neighbor edges (100px overlap)
   - Process with GPU (full context for shadows)
   - Save only center tile result
   - Clear memory
```

### Edge Effect Handling

Without neighbors:
```
┌─────────┐ ← Tile boundary
│  ████   │   ← Wrong shadow (missing relief from neighbor)
│    X    │   ← Point near edge
└─────────┘
```

With neighbors:
```
     ┌───┐     ← Load 100px from North neighbor
┌────┼───┼────┐
│    │   │    │ ← Load 100px from West/East
├────┼───┼────┤
│    │ X │    │ ← Center tile
├────┼───┼────┤
│    │   │    │ ← Load 100px from all 8 neighbors
└────┴───┴────┘
     └───┘     ← Load 100px from South neighbor

Now X sees relief up to 500m in any direction!
```

## Step 1: Build Spatial Index

Build the index once for all your ASC files:

```bash
# Index all ASC files under RGEALTI/
python3 build_spatial_index.py RGEALTI/

# Creates:
#   - asc_spatial_index.pkl  (for processing)
#   - asc_spatial_index.json (for inspection)
```

**What the index contains:**
- File path for each tile
- Spatial coordinates (xllcorner, yllcorner)
- Grid position
- Neighbor relationships
- Bounds and metadata

**Directory structure doesn't matter!** Keep your files organized:
```
RGEALTI/
├── D001_5M/
│   └── *.asc
├── D002_5M/
│   └── *.asc
└── ...
```

The index scans recursively and finds all tiles.

## Step 2: Process with Neighbors

Process all tiles with proper edge handling:

```bash
# Basic usage
python3 process_asc_with_neighbors.py asc_spatial_index.pkl

# Custom overlap (for mountainous terrain with longer shadows)
python3 process_asc_with_neighbors.py asc_spatial_index.pkl --overlap 200

# Custom output directory
python3 process_asc_with_neighbors.py asc_spatial_index.pkl --output-dir results/

# Test with first 10 tiles
python3 process_asc_with_neighbors.py asc_spatial_index.pkl --max-tiles 10
```

## Overlap Parameter

The `--overlap` parameter controls how many pixels are loaded from neighbors:

| Overlap | Distance | Use Case |
|---------|----------|----------|
| 50      | 250m     | Flat terrain, minimal shadows |
| 100     | 500m     | Default, good for most cases |
| 150     | 750m     | Hilly terrain |
| 200     | 1000m    | Mountainous terrain, long shadows |

**Formula:** `distance = overlap × pixel_size`

For 5M resolution: `100 pixels × 5m = 500m`

## Memory Usage

Per tile processing:
- Center tile: 4000×4000 × 4 bytes = ~64 MB
- With 100px overlap: 4200×4200 × 4 bytes = ~71 MB
- Neighbors loaded: ~8 edges × 100px × 4000px × 4 bytes = ~13 MB
- **Total per tile: ~100-150 MB**

Much more manageable than 106GB!

## Processing Time

**For ~27,000 tiles on RTX 4090:**
- Per tile: ~2-3 seconds (with neighbor loading)
- Total: ~15-20 hours

**Comparison:**
- Old method: Load all → 2 hours processing BUT requires 106GB RAM
- New method: ~20 hours BUT only needs 2GB RAM

Trade-off: Slower but much more resource-efficient and correct!

## Output

Each tile saves to a separate file:

```
daylight_results_tiles/
├── RGEALTI_..._D001_..._daylight.npy
├── RGEALTI_..._D002_..._daylight.npy
└── ...
```

**Format:** NumPy array, shape (4000, 4000), dtype float32
- Values: hours of daylight (0-16)
- NaN: NODATA areas

## Loading Results

```python
import numpy as np

# Load single tile result
daylight = np.load('daylight_results_tiles/RGEALTI_..._D001_..._daylight.npy')

print(f"Shape: {daylight.shape}")
print(f"Daylight range: {np.nanmin(daylight):.1f} - {np.nanmax(daylight):.1f} hours")
print(f"Mean: {np.nanmean(daylight):.1f} hours")
```

## Merging Results (Optional)

If you need all results in one file later:

```python
from pathlib import Path
import numpy as np

# Load all result files
result_files = sorted(Path('daylight_results_tiles').glob('*_daylight.npy'))

results = []
for file in result_files:
    data = np.load(file)
    results.append(data)

# Stack into single array
merged = np.stack(results, axis=0)

# Save
np.savez_compressed('all_daylight_results.npz', merged)
```

## Resume Interrupted Processing

The processing creates one output file per input tile. To resume:

```bash
# List already processed tiles
ls daylight_results_tiles/ > processed.txt

# Or run again - it will skip existing files (add skip logic if needed)
```

## Verification

Check for edge discontinuities:

```python
import numpy as np
from pathlib import Path
from build_spatial_index import load_spatial_index, find_neighbors

# Load index
index = load_spatial_index('asc_spatial_index.pkl')

# Find two neighboring tiles
tile_a = 0
tile_b = find_neighbors(index, tile_a)['E']  # East neighbor

if tile_b is not None:
    # Load results
    file_a = Path(index['files'][tile_a]).stem + '_daylight.npy'
    file_b = Path(index['files'][tile_b]).stem + '_daylight.npy'

    result_a = np.load(f'daylight_results_tiles/{file_a}')
    result_b = np.load(f'daylight_results_tiles/{file_b}')

    # Check edge values
    edge_a = result_a[:, -1]   # Right edge of tile A
    edge_b = result_b[:, 0]    # Left edge of tile B

    # Should be similar (not identical due to local terrain)
    diff = np.abs(edge_a - edge_b)
    print(f"Edge difference: mean={np.nanmean(diff):.2f}h, max={np.nanmax(diff):.2f}h")
```

If edges are smooth, the neighbor loading is working correctly!

## Advantages

✅ **Memory efficient:** Only loads what's needed
✅ **Disk efficient:** No intermediate merged file
✅ **Correct edge handling:** Shadows computed with full context
✅ **Resumable:** Can restart without losing progress
✅ **Scalable:** Can process millions of tiles
✅ **Flexible:** Adjust overlap based on terrain

## Limitations

⚠️ Slower than batch processing (more file I/O)
⚠️ Requires building spatial index first
⚠️ Tiles must have consistent coordinate system

## Troubleshooting

### "No neighbors found"
- Check that tiles are on a regular grid
- Verify coordinate system is consistent
- Inspect `asc_spatial_index.json`

### "Tile X has no East neighbor"
- Normal at dataset boundaries
- Extended array filled with NaN where no neighbor exists
- Processing continues correctly

### Memory issues
- Reduce `--overlap` parameter
- Process fewer tiles at once
- Clear GPU memory between runs

## See Also

- [build_spatial_index.py](build_spatial_index.py) - Index builder
- [process_asc_with_neighbors.py](process_asc_with_neighbors.py) - Main processor
- [compute.py](compute.py) - GPU computation engine
