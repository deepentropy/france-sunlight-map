## 2026-02-27 - GPU-Accelerated Effective Sunlight for Haute-Savoie (Alps)

### Goal
Compute terrain-aware effective sunlight hours using RGEALTI 5M DEM data, GPU horizon computation, and pvlib solar positions. Output as interactive Folium HTML map.

### Approach Taken
Single-file approach (`sunlight.py`):
1. Download RGEALTI 5M archive for Haute-Savoie (D074)
2. Extract .7z, mosaic all ASC tiles into single grid, downsample 5m → 100m
3. GPU horizon angles: CuPy RawKernel, 36 azimuths, 30km search
4. pvlib solar positions at 10-min intervals for summer/winter solstice
5. Compare sun elevation vs horizon angle → effective sunlight hours
6. Reproject Lambert-93 → WGS84, render as Folium ImageOverlay

### What Worked
- **Pre-compute horizon once, reuse for all timesteps** — key insight
- Alps show dramatic terrain effects: max horizon angle 77.3°
  - Summer: 1.3h min (deep valleys) → 15.7h max (exposed ridges)
  - Winter: 0.0h min (valleys never see sun!) → 8.5h max
- CuPy RawKernel trivially fast on RTX 4090 (DEM 4MB + horizons 130MB)
- rasterio.warp.reproject with nearest-neighbor preserves NaN mask correctly

### What Failed
- Brittany (flat terrain): minimal variation, not interesting for visualization
- First reprojection attempt used bilinear resampling: interpolated NaN→valid at edges, filling the whole bounding rectangle with color. Fix: set sunlight=NaN where grid=NaN *before* reprojecting, use nearest-neighbor resampling
- Lambert-93 → WGS84 corner-only bounds: 0.26° skew across grid, caused visible misalignment. Fix: full rasterio.warp.reproject

### Current State
- `sunlight.py` runs end-to-end for Haute-Savoie
- Output: `sunlight.tif` (801KB) + `sunlight_viewer.html` (12KB)
- Data cached in `data/`

### Key Decisions
- 100m resolution (downsample 20x from 5m) — acceptable for POC
- 36 azimuths (10° spacing), 30km search radius
- Single center point for pvlib (45.9°N, 6.4°E)
- rasterio.warp.reproject (nearest) for correct WGS84 alignment
- YlOrRd colormap with min-max range

---

## 2026-02-27 - Separated GeoTIFF + HTML viewer (scalability)

### Goal
Replace embedded-data HTML (4MB+) with separate GeoTIFF + lightweight HTML viewer, enabling GitHub Pages deployment for larger datasets (all of France).

### Approach Taken
- Replaced `make_data_png` / `_build_full_html` (base64-embedded PNGs + JSON points) with:
  - `save_geotiff()`: 2-band uint8 tiled GeoTIFF with overviews (band1=summer, band2=winter)
  - `write_viewer_html()`: standalone HTML using `geotiff.js` from CDN
- Click popup now reads pixel values directly from GeoTIFF bands (no embedded JSON)
- Removed dependencies: `folium`, `Pillow`, `pyproj` (import), `io`, `base64`

### What Worked
- GeoTIFF with deflate compression: 801KB (vs 4.1MB embedded HTML)
- HTML viewer: 12KB (vs 4.1MB)
- `geotiff.js` CDN loads and parses GeoTIFF correctly via `fetch()` + `fromArrayBuffer()`
- Canvas rendering with LUT + threshold hatching works identically to previous approach
- Click handler computes grid cell from lat/lon + reads band values — eliminates ~260MB JSON for full-France scale

### What Failed
- Nothing — clean migration

### Current State
- `sunlight.py` outputs `sunlight.tif` + `sunlight_viewer.html`
- Requires HTTP server for local testing (`python -m http.server`)
- Ready to scale to more departments

### Key Decisions
- uint8 encoding: 0=nodata, 1-255=scaled hours (vmin to vmax)
- Tiled GeoTIFF (256x256 blocks) with 4 overview levels + deflate compression
- `geotiff.js@2.1.3` from jsdelivr CDN — mature, handles tiled GeoTIFF well
- Full raster loaded on page open (fine for <50MB; for France-scale, can switch to COG range requests later)

---

## 2026-02-27 - Scaled to All of Metropolitan France (96 departments)

### Goal
Process all 96 metropolitan French departments (~40GB of RGEALTI 5M data, 27,322 ASC tiles) into a single all-France sunlight map.

### Approach Taken
- Changed from single department URL to reading all 96 `_5M_` URLs from `geoservices/rgealti_links.txt`
- Increased DOWNSAMPLE from 20 to 40 (200m resolution) to fit VRAM
- Adjusted MAX_DISTANCE_CELLS from 300 to 150 to maintain 30km horizon search at 200m
- Center point moved to France center (46.5°N, 2.0°E)
- Concurrent 8-thread downloads with retry + resume (Range headers)
- Parallelized mosaic I/O with 32-thread ThreadPoolExecutor

### What Worked
- **Parallel mosaic**: 27K ASC files in 4 minutes (vs ~65 min sequential) — 32 threads reading + downsampling, main thread placing into grid
- **GPU horizons on RTX 4090**: 30.8M cells × 36 azimuths, DEM 123MB + horizons 4.4GB VRAM — completes in seconds
- **Base64-embedded GeoTIFF**: HTML works via file:// without server (18.9MB self-contained)
- Results: Summer 2.3h-15.7h (mean 15.4h), Winter 0.0h-8.5h (mean 8.1h)
- Alps, Pyrenees, Massif Central show dramatic terrain shading; flat regions (Beauce, Landes) get near-max sunlight

### What Failed
- **Sequential downloads**: 96 archives at ~3MB/s sequentially would take ~4 hours. Fix: ThreadPoolExecutor(8)
- **Download resume**: Server drops connections on large files (>400MB). Fix: Range header resume, 10 retries, 1MB chunks
- **Silent crash after mosaic**: Process died without output after mosaic loop completed. Cause: Python output buffering — tqdm progress bars flushed but subsequent `print()` didn't. Fix: run with `python -u` (unbuffered)
- **Download function re-downloading complete files**: Server sometimes returns 200 instead of 416 for Range requests on complete files. Fix: check if local file exists before attempting download

### Current State
- `sunlight.py` runs end-to-end for all 96 metropolitan departments
- Output: `sunlight.tif` (14.1MB) + `sunlight_viewer.html` (18.9MB)
- Grid: 5750 × 5350 = 30.8M cells (12.4M valid), EPSG:4326
- Data: 96 .7z archives + 27,322 extracted ASC files in `data/`

### Key Decisions
- 200m resolution (DOWNSAMPLE=40) — fits 4.4GB horizons in 24GB VRAM
- 8 concurrent download threads, 32 concurrent mosaic I/O threads
- Base64 embedding instead of separate GeoTIFF file (file:// compatibility)
- Single pvlib center point (46.5°N, 2.0°E) — max error ~2° across France, acceptable
