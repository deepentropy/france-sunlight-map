<p align="center">
  <img src="logo.png" alt="France Sunlight Map" width="200">
</p>

# France Sunlight Map

GPU-accelerated terrain sunlight computation for France. Computes effective sunlight hours by combining a 5m DEM with horizon-angle analysis and solar position modelling, then produces an interactive map viewer.

## How it works

1. **Download** — Fetches IGN RGE ALTI 5m elevation tiles for all of France (~100 departments)
2. **Mosaic & downsample** — Stitches tiles into a single grid at 200m resolution
3. **Horizon angles** — CUDA kernel computes the maximum obstruction angle at 36 azimuths per cell (30km search radius)
4. **Solar positions** — pvlib computes sun azimuth/elevation every 10 minutes on summer and winter solstices
5. **Effective sunlight** — Counts minutes where sun elevation exceeds the local horizon angle
6. **Viewer** — Generates a standalone HTML file with Leaflet overlay, histogram-equalized turbo colormap, threshold filter, address search, and per-pixel percentile ranking

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA 12.x (tested on RTX 4090)
- ~16 GB disk for elevation data

## Setup

```bash
uv sync
```

## Usage

```bash
python sunlight.py
```

This runs the full pipeline and produces:
- `sunlight.tif` — 2-band GeoTIFF (summer + winter, EPSG:3857)
- `sunlight_viewer.html` — Standalone interactive map (open in browser)

## Viewer features

- **Season toggle** — Switch between summer solstice (Jun 21) and winter solstice (Dec 21)
- **Threshold slider** — Filter out areas below a minimum sunlight threshold
- **Click anywhere** — Shows sunlight hours, percentile rank vs. France, and reverse-geocoded address
- **Address search** — Jump to any French city or address (Nominatim geocoding)
