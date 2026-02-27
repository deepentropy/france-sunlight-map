"""
GPU-accelerated effective sunlight computation for Haute-Savoie (Alps).
Pre-computes horizon angles once with CuPy, then evaluates sunlight for any time steps.
Output: GeoTIFF + HTML viewer with interactive threshold filter.
"""

import base64
import concurrent.futures
import json
import math
import os
from pathlib import Path

import cupy as cp
import matplotlib
import numpy as np
import pvlib
import pandas as pd
import py7zr
import rasterio
import rasterio.warp
from rasterio.transform import from_bounds
import requests
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
LINKS_FILE = Path("geoservices/rgealti_links.txt")
OUTPUT_TIFF = "sunlight.tif"
OUTPUT_HTML = "sunlight_viewer.html"

DOWNSAMPLE = 40  # 5m * 40 = 200m output resolution
N_AZIMUTHS = 36  # every 10 degrees
MAX_DISTANCE_CELLS = 150  # 150 * 200m = 30km horizon search
NODATA = -99999.0

# France center for pvlib
CENTER_LAT, CENTER_LON = 46.5, 2.0


# ── Step 2: Download ──────────────────────────────────────────────────────────

def download_file(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = out_dir / os.path.basename(url)
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}

    for attempt in range(10):
        existing = local_path.stat().st_size if local_path.exists() else 0
        hdrs = {**ua}
        if existing:
            hdrs["Range"] = f"bytes={existing}-"

        try:
            r = requests.get(url, headers=hdrs, stream=True, timeout=(30, 300))
            if r.status_code == 416:  # Range not satisfiable = already complete
                print(f"  Already complete: {local_path.name}")
                return local_path
            r.raise_for_status()

            if r.status_code == 200:
                mode, initial = "wb", 0
                total = int(r.headers.get("content-length", 0)) or None
            else:  # 206 Partial Content
                mode, initial = "ab", existing
                remaining = int(r.headers.get("content-length", 0)) or None
                total = (existing + remaining) if remaining else None

            with open(local_path, mode) as f, tqdm(
                total=total, initial=initial,
                unit="B", unit_scale=True, desc=local_path.name
            ) as bar:
                for chunk in r.iter_content(1 << 20):  # 1MB chunks
                    f.write(chunk)
                    bar.update(len(chunk))
            r.close()
            return local_path

        except requests.exceptions.RequestException:
            size = local_path.stat().st_size if local_path.exists() else 0
            print(f"\n  Interrupted at {size / 1e6:.0f}MB, retry {attempt + 1}/10...")

    raise RuntimeError(f"Failed after 10 retries: {local_path.name}")


def download_all() -> list[Path]:
    urls = [line.strip() for line in LINKS_FILE.read_text().splitlines() if "_5M_" in line]
    print(f"=== Downloading RGEALTI 5M: {len(urls)} departments (8 concurrent) ===")

    # Skip downloads for archives that already exist locally
    to_download = []
    already_done = []
    for url in urls:
        local_path = DATA_DIR / os.path.basename(url)
        if local_path.exists():
            already_done.append(local_path)
        else:
            to_download.append(url)

    if already_done:
        print(f"  {len(already_done)} archives already downloaded, {len(to_download)} remaining")

    if to_download:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            new = list(pool.map(lambda url: download_file(url, DATA_DIR), to_download))
        already_done.extend(new)

    return already_done


# ── Step 3: Extract ───────────────────────────────────────────────────────────

def extract_archive(archive_path: Path) -> Path:
    extract_dir = DATA_DIR / archive_path.stem
    if extract_dir.exists() and list(extract_dir.rglob("*.asc")):
        print(f"  Already extracted: {extract_dir.name}")
        return extract_dir
    print(f"  Extracting {archive_path.name}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, "r") as z:
        z.extractall(extract_dir)
    return extract_dir


def extract_all(archives: list[Path]) -> list[Path]:
    print("\n=== Extracting archives ===")
    asc_files = []
    for a in archives:
        extract_dir = extract_archive(a)
        found = sorted(extract_dir.rglob("RGEALTI_FXX_*.asc"))
        print(f"  {extract_dir.name}: {len(found)} ASC files")
        asc_files.extend(found)
    print(f"Total ASC files: {len(asc_files)}")
    return asc_files


# ── Step 4: Mosaic & downsample ──────────────────────────────────────────────

def _read_bounds(f: Path) -> tuple:
    with rasterio.open(f) as ds:
        return ds.bounds, round(ds.transform.a, 2)


def _read_and_downsample(f: Path) -> tuple | None:
    with rasterio.open(f) as ds:
        data = ds.read(1).astype(np.float32)
        nodata_val = ds.nodatavals[0]
        if nodata_val is not None:
            data[data == nodata_val] = np.nan
        tile_left = ds.bounds.left
        tile_top = ds.bounds.top

    tile_h, tile_w = data.shape
    trim_h = (tile_h // DOWNSAMPLE) * DOWNSAMPLE
    trim_w = (tile_w // DOWNSAMPLE) * DOWNSAMPLE
    if trim_h == 0 or trim_w == 0:
        return None
    data = data[:trim_h, :trim_w]
    blocks = data.reshape(trim_h // DOWNSAMPLE, DOWNSAMPLE, trim_w // DOWNSAMPLE, DOWNSAMPLE)
    with np.errstate(all="ignore"):
        downsampled = np.nanmean(blocks, axis=(1, 3))
    return downsampled, tile_left, tile_top


def mosaic_and_downsample(asc_files: list[Path]) -> tuple[np.ndarray, dict]:
    n_workers = os.cpu_count() or 8

    print(f"\n=== Pass 1: Computing global bounds ({n_workers} threads) ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(tqdm(pool.map(_read_bounds, asc_files), total=len(asc_files), desc="Reading bounds"))

    res_set = set(r[1] for r in results)
    assert len(res_set) == 1, f"Mixed resolutions: {res_set}"
    src_res = res_set.pop()
    out_res = src_res * DOWNSAMPLE

    min_x = min(r[0].left for r in results)
    min_y = min(r[0].bottom for r in results)
    max_x = max(r[0].right for r in results)
    max_y = max(r[0].top for r in results)

    out_w = int(math.ceil((max_x - min_x) / out_res))
    out_h = int(math.ceil((max_y - min_y) / out_res))
    print(f"Source res: {src_res}m, output res: {out_res}m")
    print(f"Global extent: ({min_x:.0f}, {min_y:.0f}) - ({max_x:.0f}, {max_y:.0f})")
    print(f"Output grid: {out_w} x {out_h} = {out_w * out_h:,} cells ({out_w * out_h * 4 / 1e6:.1f} MB)")

    grid = np.full((out_h, out_w), np.nan, dtype=np.float32)

    print(f"\n=== Pass 2: Mosaicking & downsampling ({n_workers} threads) ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for result in tqdm(pool.map(_read_and_downsample, asc_files), total=len(asc_files), desc="Mosaicking"):
            if result is None:
                continue
            downsampled, tile_left, tile_top = result

            col_off = int(round((tile_left - min_x) / out_res))
            row_off = int(round((max_y - tile_top) / out_res))
            dh, dw = downsampled.shape

            if row_off < 0 or col_off < 0:
                continue
            if row_off + dh > out_h:
                dh = out_h - row_off
                downsampled = downsampled[:dh]
            if col_off + dw > out_w:
                dw = out_w - col_off
                downsampled = downsampled[:, :dw]

            target = grid[row_off:row_off + dh, col_off:col_off + dw]
            mask_both = ~np.isnan(target) & ~np.isnan(downsampled)
            mask_new = np.isnan(target) & ~np.isnan(downsampled)
            target[mask_both] = (target[mask_both] + downsampled[mask_both]) / 2
            target[mask_new] = downsampled[mask_new]

    valid_count = np.count_nonzero(~np.isnan(grid))
    print(f"Valid cells: {valid_count:,} / {out_h * out_w:,} ({100 * valid_count / (out_h * out_w):.1f}%)")

    meta = {
        "min_x": min_x, "max_y": max_y,
        "out_res": out_res, "out_h": out_h, "out_w": out_w,
        "crs": "EPSG:2154",
    }
    return grid, meta


# ── Step 5: GPU horizon angles ───────────────────────────────────────────────

HORIZON_KERNEL = r"""
extern "C" __global__
void horizon_kernel(
    const float* dem,       // (H, W) elevation grid
    float* horizons,        // (H, W, N_AZ) output horizon elevation angles in degrees
    int H, int W, int N_AZ,
    float res,              // grid cell size in meters
    int max_dist,           // max search distance in cells
    float nodata
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W;
    if (idx >= total) return;

    int r = idx / W;
    int c = idx % W;
    float z0 = dem[idx];
    if (z0 != z0 || z0 <= nodata) {  // NaN check or nodata
        for (int a = 0; a < N_AZ; a++)
            horizons[idx * N_AZ + a] = 0.0f;
        return;
    }

    float pi = 3.14159265f;
    for (int a = 0; a < N_AZ; a++) {
        float az_rad = (float)a * 2.0f * pi / (float)N_AZ;
        // Azimuth: 0=North, clockwise. In grid coords: North = -row, East = +col
        float dc = sinf(az_rad);   // column step (East component)
        float dr = -cosf(az_rad);  // row step (North = negative row)

        float max_angle = 0.0f;
        for (int k = 1; k <= max_dist; k++) {
            float fc = (float)c + (float)k * dc;
            float fr = (float)r + (float)k * dr;
            int ic = (int)(fc + 0.5f);
            int ir = (int)(fr + 0.5f);
            if (ic < 0 || ic >= W || ir < 0 || ir >= H) break;

            float zt = dem[ir * W + ic];
            if (zt != zt || zt <= nodata) continue;  // skip NaN/nodata

            float dz = zt - z0;
            float dist = (float)k * res;
            float angle = atan2f(dz, dist) * 180.0f / pi;
            if (angle > max_angle) max_angle = angle;
        }
        horizons[idx * N_AZ + a] = max_angle;
    }
}
"""


def compute_horizon_angles(grid: np.ndarray, meta: dict) -> np.ndarray:
    print("\n=== Computing horizon angles on GPU ===")
    H, W = grid.shape
    res = meta["out_res"]

    dem_gpu = grid.copy()
    dem_gpu[np.isnan(dem_gpu)] = NODATA

    d_dem = cp.asarray(dem_gpu, dtype=cp.float32)
    d_horizons = cp.zeros((H, W, N_AZIMUTHS), dtype=cp.float32)

    kernel = cp.RawKernel(HORIZON_KERNEL, "horizon_kernel")
    total_pixels = H * W
    block_size = 256
    grid_size = (total_pixels + block_size - 1) // block_size

    print(f"Grid: {H}x{W} = {total_pixels:,} cells, {N_AZIMUTHS} azimuths")
    print(f"GPU memory: DEM {d_dem.nbytes / 1e6:.0f}MB + horizons {d_horizons.nbytes / 1e6:.0f}MB")
    print(f"Max search: {MAX_DISTANCE_CELLS} cells = {MAX_DISTANCE_CELLS * res / 1000:.0f}km")

    kernel(
        (grid_size,), (block_size,),
        (d_dem, d_horizons,
         np.int32(H), np.int32(W), np.int32(N_AZIMUTHS),
         np.float32(res), np.int32(MAX_DISTANCE_CELLS), np.float32(NODATA))
    )
    cp.cuda.Stream.null.synchronize()

    horizons = cp.asnumpy(d_horizons)
    print(f"Horizon angles: min={np.nanmin(horizons):.1f}, max={np.nanmax(horizons):.1f} degrees")

    del d_dem, d_horizons
    cp.get_default_memory_pool().free_all_blocks()

    return horizons


# ── Step 6: Solar positions ──────────────────────────────────────────────────

def compute_solar_positions() -> dict[str, pd.DataFrame]:
    print("\n=== Computing solar positions (pvlib) ===")
    results = {}
    for label, date in [("summer", "2025-06-21"), ("winter", "2025-12-21")]:
        times = pd.date_range(f"{date} 00:00", f"{date} 23:50", freq="10min", tz="UTC")
        solpos = pvlib.solarposition.get_solarposition(times, CENTER_LAT, CENTER_LON)
        daytime = solpos[solpos["elevation"] > 0][["azimuth", "elevation"]].copy()
        print(f"  {label}: {len(daytime)} daytime samples, "
              f"sun up {len(daytime) * 10 / 60:.1f}h, "
              f"max elevation {daytime['elevation'].max():.1f} deg")
        results[label] = daytime
    return results


# ── Step 7: Effective sunlight ───────────────────────────────────────────────

def compute_sunlight(horizons: np.ndarray, solar: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    print("\n=== Computing effective sunlight ===")
    H, W, N_AZ = horizons.shape
    az_step = 360.0 / N_AZ

    results = {}
    for label, solpos in solar.items():
        sunlit_minutes = np.zeros((H, W), dtype=np.float32)

        for _, row in solpos.iterrows():
            sun_az = row["azimuth"]
            sun_elev = row["elevation"]

            az_idx = sun_az / az_step
            az_lo = int(az_idx) % N_AZ
            az_hi = (az_lo + 1) % N_AZ
            frac = az_idx - int(az_idx)

            h_angle = horizons[:, :, az_lo] * (1 - frac) + horizons[:, :, az_hi] * frac
            visible = sun_elev > h_angle
            sunlit_minutes += visible.astype(np.float32) * 10

        sunlit_hours = sunlit_minutes / 60.0
        results[label] = sunlit_hours
        valid = ~np.isnan(horizons[:, :, 0]) & (horizons[:, :, 0] >= 0)
        vals = sunlit_hours[valid]
        print(f"  {label}: mean={vals.mean():.1f}h, min={vals.min():.1f}h, max={vals.max():.1f}h")

    return results


# ── Step 8: GeoTIFF + HTML viewer ────────────────────────────────────────────

def reproject_to_webmercator(
    data: np.ndarray, meta: dict
) -> tuple[np.ndarray, dict]:
    """Reproject a 2D float32 array from Lambert-93 to Web Mercator (EPSG:3857).

    Web Mercator matches Leaflet's display projection, so imageOverlay aligns
    correctly with the base map tiles (no vertical distortion).
    """
    H, W = data.shape
    min_x = meta["min_x"]
    max_y = meta["max_y"]
    out_res = meta["out_res"]
    max_x = min_x + W * out_res
    min_y = max_y - H * out_res

    src_transform = from_bounds(min_x, min_y, max_x, max_y, W, H)

    # Reproject to Web Mercator (meters)
    mx_min, my_min, mx_max, my_max = rasterio.warp.transform_bounds(
        "EPSG:2154", "EPSG:3857", min_x, min_y, max_x, max_y
    )
    dst_transform = from_bounds(mx_min, my_min, mx_max, my_max, W, H)

    dst_data = np.full((H, W), np.nan, dtype=np.float32)
    rasterio.warp.reproject(
        source=data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs="EPSG:2154",
        dst_transform=dst_transform,
        dst_crs="EPSG:3857",
        resampling=rasterio.warp.Resampling.nearest,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    # Also compute lat/lon bounds for Leaflet
    lon_min, lat_min, lon_max, lat_max = rasterio.warp.transform_bounds(
        "EPSG:3857", "EPSG:4326", mx_min, my_min, mx_max, my_max
    )

    return dst_data, {
        "mx_min": mx_min, "my_min": my_min,
        "mx_max": mx_max, "my_max": my_max,
        "lon_min": lon_min, "lat_min": lat_min,
        "lon_max": lon_max, "lat_max": lat_max,
    }


def save_geotiff(
    sunlight: dict[str, np.ndarray],
    grid: np.ndarray,
    meta: dict,
) -> tuple[dict, dict, dict]:
    """Save sunlight as 2-band tiled GeoTIFF: band1=summer, band2=winter (uint8).
    Each band uses its own p1-p99 percentile range for better contrast."""
    print("\n=== Saving GeoTIFF ===")

    nodata_mask = np.isnan(grid)
    for label in sunlight:
        sunlight[label][nodata_mask] = np.nan

    print("  Reprojecting to Web Mercator...")
    grid_wm, wm_meta = reproject_to_webmercator(grid, meta)
    sunlight_wm = {}
    for label in sunlight:
        sunlight_wm[label], _ = reproject_to_webmercator(sunlight[label], meta)

    valid_mask = ~np.isnan(grid_wm)

    # Per-season percentile-based scaling for better contrast
    H, W = grid_wm.shape
    encoded = {}
    vranges = {}
    for label in ["summer", "winter"]:
        vals = sunlight_wm[label][valid_mask]
        vmin = float(np.percentile(vals, 1))
        vmax = float(np.percentile(vals, 99))
        vranges[label] = (vmin, vmax)
        print(f"  {label}: p1={vmin:.1f}h, p99={vmax:.1f}h")

        data = sunlight_wm[label]
        nd = np.isnan(data)
        safe = np.nan_to_num(data, nan=vmin)
        scaled = np.clip((safe - vmin) / (vmax - vmin), 0, 1) * 254 + 1
        u8 = scaled.astype(np.uint8)
        u8[nd] = 0
        encoded[label] = u8

    # Write tiled GeoTIFF with overviews (Web Mercator)
    transform = from_bounds(
        wm_meta["mx_min"], wm_meta["my_min"],
        wm_meta["mx_max"], wm_meta["my_max"], W, H
    )

    with rasterio.open(
        OUTPUT_TIFF, "w", driver="GTiff",
        height=H, width=W, count=2, dtype="uint8",
        crs="EPSG:3857", transform=transform, nodata=0,
        tiled=True, blockxsize=256, blockysize=256, compress="deflate",
    ) as dst:
        dst.write(encoded["summer"], 1)
        dst.write(encoded["winter"], 2)
        dst.set_band_description(1, "Summer solstice (Jun 21)")
        dst.set_band_description(2, "Winter solstice (Dec 21)")
        dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.nearest)
        dst.update_tags(ns="rio_overview", resampling="nearest")

    size_mb = os.path.getsize(OUTPUT_TIFF) / 1e6
    print(f"  Saved: {OUTPUT_TIFF} ({size_mb:.1f} MB)")
    return encoded, vranges, wm_meta


def write_viewer_html(
    encoded: dict[str, np.ndarray],
    vranges: dict[str, tuple[float, float]], wm_meta: dict,
) -> None:
    """Write standalone HTML viewer with embedded gzipped band data."""
    import gzip
    print("\n=== Writing HTML viewer ===")

    H, W = encoded["summer"].shape
    s_vmin, s_vmax = vranges["summer"]
    w_vmin, w_vmax = vranges["winter"]

    # Gzip raw band bytes
    raw = encoded["summer"].tobytes() + encoded["winter"].tobytes()
    compressed = gzip.compress(raw, compresslevel=9)
    data_b64 = base64.b64encode(compressed).decode()
    print(f"  Raw: {len(raw) / 1e6:.1f} MB -> gzipped b64: {len(data_b64) / 1e6:.1f} MB")

    # Generate YlOrRd 256-entry LUT as JS array
    cmap = matplotlib.colormaps["YlOrRd"]
    lut = []
    for i in range(256):
        if i == 0:
            lut.append([0, 0, 0, 0])
        else:
            t = (i - 1) / 254.0
            r, g, b, _ = cmap(t)
            lut.append([int(r * 255), int(g * 255), int(b * 255), 153])
    lut_js = json.dumps(lut)

    lon_min = wm_meta["lon_min"]
    lat_min = wm_meta["lat_min"]
    lon_max = wm_meta["lon_max"]
    lat_max = wm_meta["lat_max"]
    mx_min = wm_meta["mx_min"]
    my_min = wm_meta["my_min"]
    mx_max = wm_meta["mx_max"]
    my_max = wm_meta["my_max"]
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Effective Sunlight - France</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body {{ margin:0; padding:0; }}
  #map {{ position:absolute; top:0; bottom:0; width:100%; }}
  #controls {{
    position:fixed; top:10px; right:10px; z-index:9999;
    background:white; padding:12px 16px; border-radius:8px;
    box-shadow:0 2px 8px rgba(0,0,0,0.3); font-family:sans-serif; font-size:13px;
    min-width:220px;
  }}
  #controls label {{ font-weight:bold; display:block; margin-bottom:6px; }}
  #controls select, #controls input[type=range] {{ width:100%; margin-bottom:8px; }}
  #legend {{
    position:fixed; bottom:30px; left:30px; z-index:9999;
    background:white; padding:10px; border-radius:5px;
    box-shadow:0 0 5px rgba(0,0,0,0.3); font-family:sans-serif; font-size:12px;
  }}
  #loading {{
    position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
    z-index:99999; background:white; padding:20px 30px; border-radius:8px;
    box-shadow:0 2px 12px rgba(0,0,0,0.3); font-family:sans-serif; font-size:16px;
  }}
</style>
</head><body>
<div id="map"></div>
<div id="loading">Loading sunlight data...</div>

<div id="controls" style="display:none">
  <label>Season</label>
  <select id="season-select">
    <option value="summer" selected>Summer Solstice (Jun 21)</option>
    <option value="winter">Winter Solstice (Dec 21)</option>
  </select>
  <label>Minimum sunlight: <span id="threshold-label">0.0</span>h</label>
  <input type="range" id="threshold-slider" min="0" max="16" step="0.5" value="0">
</div>

<div id="legend" style="display:none">
  <b>Effective Sunlight (hours)</b><br>
  <div style="display:flex;align-items:center;margin-top:5px;">
    <span id="legend-min"></span>
    <div style="width:150px;height:15px;margin:0 5px;
                background:linear-gradient(to right,#ffffb2,#fed976,#feb24c,#fd8d3c,#f03b20,#bd0026);"></div>
    <span id="legend-max"></span>
  </div>
  <div style="margin-top:4px;font-size:11px;color:#666;">
    <span style="display:inline-block;width:12px;height:12px;background:repeating-linear-gradient(45deg,#ccc,#ccc 2px,#fff 2px,#fff 4px);vertical-align:middle;margin-right:4px;"></span>
    Below threshold
  </div>
</div>

<script>
var LUT = {lut_js};
var VRANGES = {{
  summer: [{s_vmin}, {s_vmax}],
  winter: [{w_vmin}, {w_vmax}]
}};
var W = {W}, H = {H};
var BBOX = [{lon_min}, {lat_min}, {lon_max}, {lat_max}];
var MX_MIN = {mx_min}, MY_MIN = {my_min}, MX_MAX = {mx_max}, MY_MAX = {my_max};
var BOUNDS = [[{lat_min}, {lon_min}], [{lat_max}, {lon_max}]];

var map = L.map('map').setView([{center_lat}, {center_lon}], 6);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);

var DATA_GZ_B64 = "{data_b64}";
var DATA = null;
var overlay = null;

async function loadData() {{
  var bin = atob(DATA_GZ_B64);
  var gz = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) gz[i] = bin.charCodeAt(i);

  var ds = new DecompressionStream('gzip');
  var writer = ds.writable.getWriter();
  writer.write(gz);
  writer.close();
  var raw = new Uint8Array(await new Response(ds.readable).arrayBuffer());

  var n = W * H;
  DATA = {{
    summer: raw.subarray(0, n),
    winter: raw.subarray(n, 2 * n)
  }};
  document.getElementById('loading').style.display = 'none';
  document.getElementById('controls').style.display = '';
  document.getElementById('legend').style.display = '';
  renderOverlay();
}}

function pxToHours(val, season) {{
  var r = VRANGES[season];
  return (val - 1) / 254 * (r[1] - r[0]) + r[0];
}}

function renderOverlay() {{
  if (!DATA) return;
  var season = document.getElementById('season-select').value;
  var threshold = parseFloat(document.getElementById('threshold-slider').value);
  var data = DATA[season];
  var r = VRANGES[season];

  document.getElementById('legend-min').textContent = r[0].toFixed(1) + 'h';
  document.getElementById('legend-max').textContent = r[1].toFixed(1) + 'h';

  var canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  var ctx = canvas.getContext('2d');
  var imgData = ctx.createImageData(W, H);
  var out = imgData.data;

  // Convert threshold (hours) to pixel value for this season
  var threshPx = Math.round(((threshold - r[0]) / (r[1] - r[0])) * 254 + 1);

  for (var i = 0; i < W * H; i++) {{
    var val = data[i];
    if (val === 0) {{
      out[i * 4 + 3] = 0;
    }} else if (val < threshPx) {{
      var row = Math.floor(i / W);
      var col = i % W;
      if ((row + col) % 4 < 2) {{
        out[i * 4] = 180; out[i * 4 + 1] = 180; out[i * 4 + 2] = 180; out[i * 4 + 3] = 100;
      }} else {{
        out[i * 4 + 3] = 0;
      }}
    }} else {{
      var c = LUT[val];
      out[i * 4] = c[0]; out[i * 4 + 1] = c[1]; out[i * 4 + 2] = c[2]; out[i * 4 + 3] = c[3];
    }}
  }}

  ctx.putImageData(imgData, 0, 0);
  var url = canvas.toDataURL('image/png');
  if (overlay) map.removeLayer(overlay);
  overlay = L.imageOverlay(url, BOUNDS).addTo(map);
}}

document.getElementById('season-select').addEventListener('change', renderOverlay);
document.getElementById('threshold-slider').addEventListener('input', function() {{
  document.getElementById('threshold-label').textContent = this.value;
  renderOverlay();
}});

map.on('click', function(e) {{
  if (!DATA) return;
  var lat = e.latlng.lat, lon = e.latlng.lng;
  if (lon < BBOX[0] || lon > BBOX[2] || lat < BBOX[1] || lat > BBOX[3]) return;

  var col = Math.floor((lon - BBOX[0]) / (BBOX[2] - BBOX[0]) * W);
  var my = Math.log(Math.tan(Math.PI/4 + lat * Math.PI/360)) * 6378137;
  var row = Math.floor((MY_MAX - my) / (MY_MAX - MY_MIN) * H);
  if (col < 0 || col >= W || row < 0 || row >= H) return;

  var idx = row * W + col;
  var sv = DATA.summer[idx], wv = DATA.winter[idx];
  if (sv === 0 && wv === 0) return;

  var sh = pxToHours(sv, 'summer').toFixed(1);
  var wh = pxToHours(wv, 'winter').toFixed(1);

  L.popup().setLatLng(e.latlng).setContent(
    '<b>Summer:</b> ' + sh + 'h<br>' +
    '<b>Winter:</b> ' + wh + 'h'
  ).openOn(map);
}});

loadData();
</script>
</body></html>"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = os.path.getsize(OUTPUT_HTML) / 1e6
    print(f"  Viewer saved: {OUTPUT_HTML} ({size_mb:.1f} MB)")


def build_map(
    sunlight: dict[str, np.ndarray],
    grid: np.ndarray,
    meta: dict,
) -> None:
    encoded, vranges, wm_meta = save_geotiff(sunlight, grid, meta)
    write_viewer_html(encoded, vranges, wm_meta)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    archives = download_all()
    asc_files = extract_all(archives)
    grid, meta = mosaic_and_downsample(asc_files)
    horizons = compute_horizon_angles(grid, meta)
    solar = compute_solar_positions()
    sunlight = compute_sunlight(horizons, solar)
    build_map(sunlight, grid, meta)


if __name__ == "__main__":
    main()
