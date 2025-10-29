# Memory Management for Large NPZ Files

## Problem

You have a 106GB NPZ file but only 64GB of RAM, causing `MemoryError` when trying to process the entire file.

## Solution

The code now uses **memory mapping** and **chunk processing** to handle files larger than RAM.

## How It Works

```
1. Open file with memory mapping (no RAM used)
   ↓
2. Load chunk 1 (e.g., 1000 tiles) into RAM
   ↓
3. Process chunk 1 in GPU batches (e.g., 240 tiles per batch)
   ↓
4. Write results to disk, clear RAM
   ↓
5. Repeat for chunks 2, 3, 4...
   ↓
6. Compress final output
```

## Usage

### Basic Usage (64GB RAM)

```bash
python3 compute_daylight_npz.py merged.npz daylight.npz
```

Default settings:
- `chunk_size=1000` tiles (≈15-20 GB RAM)
- `batch_size=240` tiles (GPU parallel processing)

### Adjust for Your RAM

```bash
# For 32GB RAM (reduce chunk size)
python3 compute_daylight_npz.py merged.npz daylight.npz --chunk-size 500

# For 128GB RAM (increase chunk size)
python3 compute_daylight_npz.py merged.npz daylight.npz --chunk-size 2000
```

## Memory Guidelines

### Chunk Size (RAM Usage)

| chunk_size | RAM Usage | Suitable for |
|-----------|-----------|--------------|
| 250       | ~4-5 GB   | 16GB system  |
| 500       | ~8-10 GB  | 32GB system  |
| 1000      | ~15-20 GB | 64GB system  |
| 2000      | ~30-40 GB | 128GB system |

**Formula:** Each tile ≈ 4000×4000 pixels × 4 bytes (float32) ≈ 64 MB
Chunk RAM ≈ chunk_size × 64 MB × 1.2 (overhead)

### Batch Size (GPU Memory)

| batch_size | GPU Memory | Suitable for       |
|-----------|------------|--------------------|
| 50        | ~2-3 GB    | RTX 3060 (12GB)    |
| 100       | ~4-5 GB    | RTX 3080 (10GB)    |
| 200       | ~8-10 GB   | RTX 3090 (24GB)    |
| 240       | ~10-12 GB  | RTX 4090 (24GB)    |

## Complete Example

### For 106GB file on 64GB RAM system with RTX 4090:

```bash
python3 compute_daylight_npz.py merged.npz daylight.npz \
    --chunk-size 1000 \
    --batch-size 240 \
    --pixel-size 5.0 \
    --lat-center 46.0 \
    --lon-center 2.0
```

### Progress Output

```
Opening elevation data from merged.npz (memory-mapped)
File contains 27000 tiles
Shape: (27000, 4000, 4000)
Data size: 102.00 GB
Processing in chunks of 1000 tiles (batch_size=240)

============================================================
CHUNK 1/27: Tiles 1-1000 (1000 tiles)
============================================================
Loading chunk into RAM...
Chunk loaded: (1000, 4000, 4000), 16.00 GB
  Batch 1/5: Processing tiles 1-240
  ✓ Batch 1/5 complete: 240 tiles in 45.2s (5.31 tiles/s)
  ...
Writing chunk results to disk...
✓ Chunk 1/27 complete in 380.5s

============================================================
CHUNK 2/27: Tiles 1001-2000 (1000 tiles)
============================================================
...
```

## Python API

```python
from compute import BatchProcessor
from datetime import datetime

processor = BatchProcessor()

processor.process_from_npz(
    input_npz='merged.npz',
    output_npz='daylight.npz',
    date=datetime(2024, 6, 21),
    batch_size=240,      # GPU batches
    chunk_size=1000,     # RAM chunks
    pixel_size=5.0,
    lat_center=46.0,
    lon_center=2.0
)
```

## Performance

### Expected Processing Time (27,000 tiles on RTX 4090):

- Processing: ~5 tiles/second
- Total GPU time: ~90 minutes
- Disk I/O overhead: ~10-20 minutes
- **Total: ~2 hours**

### Optimization Tips

1. **Maximize chunk_size** without exceeding RAM:
   - Larger chunks = less disk I/O overhead
   - Monitor RAM usage with `htop` or `top`
   - Leave 10-20GB free for system overhead

2. **Maximize batch_size** without exceeding GPU memory:
   - Larger batches = better GPU utilization
   - Monitor GPU with `nvidia-smi`
   - Leave 2-4GB free for CUDA overhead

3. **Use fast storage** for temporary files:
   - SSD >> HDD for temp file writes
   - Consider `--output-dir` on fastest drive

## Troubleshooting

### "MemoryError: Unable to allocate X GB"

**Solution:** Reduce `chunk_size`

```bash
# Try halving the chunk size
python3 compute_daylight_npz.py merged.npz daylight.npz --chunk-size 500
```

### "CUDA out of memory"

**Solution:** Reduce `batch_size`

```bash
# Try smaller batches
python3 compute_daylight_npz.py merged.npz daylight.npz --batch-size 100
```

### Slow processing

**Check:**
1. GPU utilization: `nvidia-smi` (should be ~95-100%)
2. RAM usage: `htop` (should be using most available RAM)
3. Increase `chunk_size` if RAM is underutilized
4. Increase `batch_size` if GPU is underutilized

### Disk space

**Required:** ~210 GB free
- Input: 106 GB (merged.npz)
- Temp file: 102 GB (uncompressed)
- Output: ~40-60 GB (daylight_results.npz, compressed)

## How Memory Mapping Works

```python
# Traditional (loads entire file into RAM - FAILS for 106GB on 64GB system)
data = np.load('merged.npz')['arr_0']  # MemoryError!

# Memory mapping (no RAM allocation - WORKS!)
npz = np.load('merged.npz', mmap_mode='r')
data = npz['arr_0']  # Only metadata loaded

# Access data on demand (only requested slice loaded into RAM)
chunk = data[0:1000]  # Loads only 1000 tiles into RAM
```

## Output Files

1. **Temporary file** (deleted automatically):
   - `daylight_results_temp.npy` (~102 GB uncompressed)
   - Used during processing, removed at end

2. **Final output**:
   - `daylight_results.npz` (~40-60 GB compressed)
   - Contains all daylight computation results

## See Also

- [compute.py](compute.py) - GPU computation engine
- [README_WORKFLOW.md](README_WORKFLOW.md) - Complete workflow
