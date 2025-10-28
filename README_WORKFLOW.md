# Complete RGEALTI Daylight Computation Workflow

This guide shows the complete workflow from downloading RGEALTI elevation data to computing daylight hours using GPU acceleration.

## Overview

```
RGEALTI Pages → Links → Download → ASC Files → NPY → Merged NPZ → Daylight NPZ
```

## Step 1: Extract Download Links

Extract RGEALTI file download links from IGN geoservices pages.

```bash
cd links
python3 extract_rgealti_from_files.py
```

See [README_RGEALTI.md](README_RGEALTI.md) for details.

## Step 2: Download Elevation Data

Download RGEALTI files filtered by department and resolution.

```bash
# Download 5M resolution data for department D001
python3 download_rgealti.py --department D001 --resolution 5M

# Download all departments at 5M resolution
python3 download_rgealti.py --resolution 5M
```

This creates organized directories like `RGEALTI/D001_5M/`.

## Step 3: Extract Archives

If files are compressed (`.7z`, `.7z.001`, etc.), extract them:

```bash
# Extract all 7z files in a directory
cd RGEALTI/D001_5M/
7z x *.7z
```

## Step 4: Convert ASC to NPY (Per Department)

Convert all ASC files from each department to a single NPY file.

```bash
# For each department
python3 -c "
from save_asc_to_npy import save_all_asc_to_single_npy
import numpy as np

# Convert D001 5M data
data = save_all_asc_to_single_npy('RGEALTI/D001_5M/', 'npy/D001_5M.npz')
print(f'D001: {data.shape}')

# Or save as compressed NPZ directly
np.savez_compressed('npy/D001_5M.npz', data)
"
```

Repeat for each department. This creates files in `npy/` directory:
- `npy/D001_5M.npz`
- `npy/D002_5M.npz`
- etc.

## Step 5: Merge All Departments

Merge all department NPZ files into one large file.

```bash
python3 -c "
from merge_npy_files import merge_npy_files

# Merge all NPZ files from npy/ directory
merged = merge_npy_files('npy/', 'merged.npz', pattern='D*.npz')
print(f'Merged shape: {merged.shape}')
"
```

This creates `merged.npz` containing all elevation data.

## Step 6: Compute Daylight Hours (GPU)

Compute daylight hours for all points using GPU acceleration.

### Option A: Using the Script

```bash
python3 compute_daylight_npz.py merged.npz daylight_results.npz
```

### Option B: Using Python API

```python
from compute import BatchProcessor
from datetime import datetime

processor = BatchProcessor()

daylight = processor.process_from_npz(
    input_npz='merged.npz',
    output_npz='daylight_results.npz',
    date=datetime(2024, 6, 21),  # Summer solstice
    batch_size=240,               # Adjust for your GPU
    pixel_size=5.0,               # 5 meters for 5M data
    lat_center=46.0,              # France center
    lon_center=2.0
)

print(f'Daylight data shape: {daylight.shape}')
```

### Option C: Using Main Script (Modified)

Edit configuration in `compute.py` and run:

```bash
python3 compute.py
```

## Step 7: Load and Analyze Results

```python
import numpy as np

# Load daylight results
with np.load('daylight_results.npz') as data:
    key = list(data.keys())[0]
    daylight = data[key]

print(f'Shape: {daylight.shape}')
print(f'Data type: {daylight.dtype}')

# Analyze first tile
tile_0 = daylight[0]
print(f'Daylight hours range: {tile_0.min():.1f} - {tile_0.max():.1f}h')
print(f'Mean daylight: {tile_0.mean():.1f}h')
```

## Complete Example Workflow

```bash
#!/bin/bash

# 1. Download data for France (D001-D095 at 5M resolution)
for dept in D001 D002 D003; do
    python3 download_rgealti.py --department $dept --resolution 5M
done

# 2. Extract archives (if needed)
find RGEALTI/ -name "*.7z" -exec 7z x {} -o{//} \;

# 3. Convert each department to NPZ
mkdir -p npy
for dept_dir in RGEALTI/D*_5M/; do
    dept=$(basename "$dept_dir" | cut -d_ -f1)
    python3 -c "
from save_asc_to_npy import save_all_asc_to_single_npy
import numpy as np
data = save_all_asc_to_single_npy('$dept_dir', 'npy/${dept}_5M.npz')
print('Processed $dept: shape', data.shape)
"
done

# 4. Merge all departments
python3 -c "
from merge_npy_files import merge_npy_files
merged = merge_npy_files('npy/', 'merged.npz', pattern='D*_5M.npz')
print('Merged all departments:', merged.shape)
"

# 5. Compute daylight (GPU required)
python3 compute_daylight_npz.py merged.npz daylight_results.npz

echo "Complete! Results saved to daylight_results.npz"
```

## File Structure

```
sunlight-extractor/
├── links/
│   ├── rgealti_links.txt          # Downloaded links
│   └── extract_rgealti_*.py       # Link extraction scripts
├── RGEALTI/                        # Downloaded elevation data
│   ├── D001_5M/
│   │   └── *.asc                   # ASCII elevation files
│   ├── D002_5M/
│   │   └── *.asc
│   └── ...
├── npy/                            # Per-department NPZ files
│   ├── D001_5M.npz
│   ├── D002_5M.npz
│   └── ...
├── merged.npz                      # All departments merged
├── daylight_results.npz            # Computed daylight hours
└── compute.py                      # GPU computation engine
```

## Performance Notes

### GPU Requirements
- NVIDIA GPU with CUDA support
- Recommended: RTX 4090 (24GB) or similar
- Minimum: 8GB VRAM for small datasets

### Batch Size
- Adjust `batch_size` based on GPU memory:
  - RTX 4090 (24GB): 240 tiles
  - RTX 3090 (24GB): 200 tiles
  - RTX 3080 (10GB): 80 tiles
  - RTX 3060 (12GB): 100 tiles

### Processing Speed
- ~240 tiles/batch on RTX 4090
- ~2-3 seconds per batch
- Total time depends on number of tiles

## Troubleshooting

### Out of GPU Memory
Reduce `batch_size` in computation step:
```python
processor.process_from_npz(..., batch_size=100)
```

### NPZ File Too Large
Process departments separately or use chunking:
```python
# Process first 1000 tiles only
elevation_data = elevation_data[:1000]
```

### Different Resolutions
For 1M resolution data, adjust `pixel_size`:
```python
processor.process_from_npz(..., pixel_size=1.0)
```

## See Also

- [README_RGEALTI.md](README_RGEALTI.md) - Download RGEALTI data
- [README_ASC_CONVERSION.md](README_ASC_CONVERSION.md) - ASC to NPY conversion
- [compute.py](compute.py) - GPU computation engine
