# ASC to NPY Conversion

Convert RGEALTI ASC (ASCII Grid) files to NumPy NPY format for faster processing and smaller file sizes.

## Files

- `asc_to_npy.py` - Main conversion script with CLI and Python API
- `example_asc_conversion.py` - Usage examples

## Features

- **Single or batch conversion** - Convert one file or entire directories
- **NODATA handling** - Replace NODATA values with NaN, 0, or keep as-is
- **Metadata preservation** - Save header information alongside data
- **Memory efficient** - Configurable dtype (float32, float16, etc.)
- **Statistics** - Automatic data validation and statistics

## Quick Start

### Command Line Usage

```bash
# Convert single file (NODATA → NaN)
python3 asc_to_npy.py input.asc output.npy

# Convert single file (NODATA → 0)
python3 asc_to_npy.py input.asc output.npy --nodata-replacement 0

# Batch convert all ASC files in directory
python3 asc_to_npy.py RGEALTI/D001_5M/ --batch

# Batch convert to different output directory
python3 asc_to_npy.py RGEALTI/D001_5M/ --batch --output-dir processed/

# Use 16-bit float for smaller files
python3 asc_to_npy.py input.asc --dtype float16
```

### Python API Usage

```python
from asc_to_npy import asc_to_npy, load_npy_with_metadata
import numpy as np

# Convert single file
data, header = asc_to_npy(
    'elevation.asc',
    'elevation.npy',
    nodata_replacement=np.nan,
    dtype='float32'
)

# Access metadata
print(f"Cell size: {header['cellsize']} meters")
print(f"Dimensions: {header['nrows']}x{header['ncols']}")
print(f"Corner: ({header['xllcorner']}, {header['yllcorner']})")

# Analyze data
valid_data = data[~np.isnan(data)]
print(f"Elevation range: {valid_data.min():.1f} - {valid_data.max():.1f} m")
print(f"Mean elevation: {valid_data.mean():.1f} m")
```

### Batch Conversion

```python
from asc_to_npy import batch_convert_asc_to_npy
import numpy as np

# Convert all ASC files in directory
count = batch_convert_asc_to_npy(
    input_dir='RGEALTI/D001_5M/',
    output_dir='RGEALTI/D001_5M/npy/',
    pattern='*.asc',
    nodata_replacement=np.nan,
    dtype='float32'
)

print(f"Converted {count} files")
```

### Load Converted Data

```python
from asc_to_npy import load_npy_with_metadata
import numpy as np

# Load NPY file with metadata
data, metadata = load_npy_with_metadata('elevation.npy')

# Data is ready to use
elevation_at_point = data[1000, 2000]
print(f"Elevation at [1000, 2000]: {elevation_at_point:.2f} m")
```

## ASC File Format

RGEALTI ASC files have a header followed by space-delimited elevation data:

```
ncols         4000
nrows         4000
xllcorner     100000.0
yllcorner     6200000.0
cellsize      1.0
NODATA_value  -99999
152.3 152.1 152.0 151.8 ...
151.9 151.7 151.5 151.3 ...
...
```

## Output Files

The conversion creates two files:

1. **`output.npy`** - Binary NumPy array with elevation data
2. **`output.metadata.npy`** - Pickled dictionary with header information

## NODATA Handling Options

- **NaN (default)** - Best for scientific computing, automatic handling by NumPy
  ```bash
  python3 asc_to_npy.py input.asc --nodata-replacement nan
  ```

- **Zero** - Useful when NaN is not supported
  ```bash
  python3 asc_to_npy.py input.asc --nodata-replacement 0
  ```

- **Keep original** - Preserve NODATA values as-is
  ```bash
  python3 asc_to_npy.py input.asc --keep-nodata
  ```

## Data Type Options

Choose dtype based on precision needs vs file size:

| dtype   | Precision | Range              | File Size |
|---------|-----------|-------------------|-----------|
| float16 | ~3 digits | ±65,504           | Smallest  |
| float32 | ~7 digits | ±3.4×10³⁸ (default)| Medium    |
| float64 | ~15 digits| ±1.8×10³⁰⁸        | Largest   |

For elevation data, `float32` is usually sufficient (precision ~0.0001m).

## Performance Benefits

### File Size Reduction
- ASC text file: ~100 MB
- NPY binary file: ~30-60 MB (40-60% reduction)
- NPY with float16: ~15-30 MB (70-85% reduction)

### Loading Speed
- ASC text file: ~5-10 seconds
- NPY binary file: ~0.1-0.5 seconds (10-100x faster)

## Integration with RGEALTI Download

After downloading RGEALTI files:

```bash
# 1. Download files for a department
python3 download_rgealti.py --department D001 --resolution 5M

# 2. Extract 7z archives (if needed)
7z x RGEALTI/D001_5M/*.7z -oRGEALTI/D001_5M/

# 3. Convert all ASC files to NPY
python3 asc_to_npy.py RGEALTI/D001_5M/ --batch --pattern "*.asc"

# 4. Use in your Python scripts
python3 compute.py
```

## Example: Processing Pipeline

```python
from asc_to_npy import load_npy_with_metadata
import numpy as np

# Load elevation data
elevation, metadata = load_npy_with_metadata('RGEALTI_D001_5M.npy')

# Remove NODATA (if any)
valid_mask = ~np.isnan(elevation)
elevation_clean = np.where(valid_mask, elevation, 0)

# Compute slope or other derived products
# ... your processing here ...

# Save processed data
np.save('processed_elevation.npy', elevation_clean)
```

## See Also

- `example_asc_conversion.py` - More detailed examples
- `download_rgealti.py` - Download RGEALTI source files
- IGN RGEALTI documentation: https://geoservices.ign.fr/rgealti
